import torch.nn as nn
import torch
import numpy as np
import os
import torch.nn.functional as F



def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)



class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        result = (input > 0).float()
        return result 
    
    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x
    
class PositionEncoding(nn.Module):
    def __init__(self, order):
        self.construct_pe_coeff(order)

    def construct_pe_coeff(self, order):
        out_dim = int(2 ** order)
        coeff = torch.ones(out_dim, 1)
        bias = torch.zeros(out_dim)
        for i in range(out_dim):
            coeff[i, 0] = coeff[i, 0] * (1.25 ** int(i // 2))
            if i % 2 !=0:
                bias[i] = 0.5 * np.pi
        self.pe_coeff = coeff
        self.pe_bias = bias

    def forward(self, x):
        pe_out = torch.nn.functional.linear(x, weight=self.pe_coeff, bias=self.pe_bias)
        pe_out = torch.sin(pe_out)
        return pe_out
                

class MLPModel(nn.Module):
    def __init__(self, args, in_dim, out_renorm=True):
        super(MLPModel, self).__init__()
        self.model = self.make_mlp_layers(args, in_dim) 
        self.out_renorm = out_renorm
        
    def make_mlp_layers(self, args, in_dim):
        if args.mlp_pe:
            self.use_pe = True 
            self.pe_module, input_dim = get_embedder(args.mlp_pe_order)
        else:
            self.use_pe = False
            input_dim = in_dim
        layers = []
        for i in range(args.mlp_layer):
            if i < (args.mlp_layer - 1):
                cur_layer = nn.Sequential(nn.Linear(input_dim, args.mlp_inter_dim), nn.LeakyReLU())
                input_dim = args.mlp_inter_dim
            else:
                cur_layer = nn.Sequential(nn.Linear(input_dim, args.mlp_out_dim), nn.Sigmoid())
            layers.append(cur_layer)
        model = nn.Sequential(*layers)        
        return model
    
    def parameters(self):
        return self.model.parameters()
        
    def forward(self, x):
        if self.use_pe:
            x = self.pe_module(x)
        out = self.model(x)
        if self.out_renorm:
            out = (out + 1.0) / 2.0 
        return out
    
    
    
class MaskNet(nn.Module):
    def __init__(self, num_img, H=400, W=600, img_embed=32, act_type='sigmoid'):
        super().__init__()
        self.num_img = num_img
        self.W, self.H = W, H

        self.img_embed_cnl = img_embed

        self.min_freq, self.max_freq, self.num_frequencies = 0.0, 3.0, 4

        self.embedding_camera = nn.Embedding(self.num_img, self.img_embed_cnl)

        self.mlp_base_mlp = torch.nn.Sequential(
            torch.nn.Conv2d(48 + 16, 20, 1, bias=False), torch.nn.ReLU(),
            )

        self.mlp_mask1 = torch.nn.Conv2d(20, 1, 1, bias=False)

        self.conv_rgb = torch.nn.Sequential(
            torch.nn.Conv2d(5,  48,  3,  padding=1), torch.nn.ReLU(), torch.nn.InstanceNorm2d(64),
            torch.nn.Conv2d(48, 16,  3,  padding=1)
            )
        
        self.act_type = act_type
        if act_type == 'sigmoid':
            self.act = torch.sigmoid 
        elif act_type == 'binary':
            self.act = STEFunction()
            

    def forward(self, img_idx, pos_enc, img, coc_alpha):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(coc_alpha.shape) == 3:
            coc_alpha = coc_alpha.unsqueeze(0)
            
        rgbd_coc_feat = self.conv_rgb(torch.cat([img, coc_alpha], dim=1))

        img_embed = self.embedding_camera(torch.LongTensor([img_idx]).cuda())[None, None]
        img_embed = img_embed.expand(pos_enc.shape[0],pos_enc.shape[1],pos_enc.shape[2],img_embed.shape[-1])
        inp = torch.cat([img_embed, pos_enc],-1).permute(0,3,1,2)

        feat = self.mlp_base_mlp(torch.cat([inp, rgbd_coc_feat],1))
        mask = self.mlp_mask1(feat)
        if self.act_type == 'sigmoid':
            mask = self.act(mask)
        elif self.act_type == 'binary':
            mask = self.act.apply(mask)
        mask = mask.squeeze(0)
        return mask
      

class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.mlp_pe = True
        self.mlp_pe_order = 10
        self.mlp_input_dim = 1
        self.mlp_inter_dim = 128
        self.mlp_layer = 3
        self.mlp_out_dim = 1
        
                
if __name__ == "__main__":
    config = Config()
    model = MLPModel(config)
    print(model.model)
    x = torch.randn(2,1)
    print(x)
    pe_out = model.pe_func(x)
    print(pe_out)
    output = model(x)
    print(output, output.shape)