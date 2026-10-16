import torch
from src.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from src.utils.general_utils import safe_state
from argparse import ArgumentParser
from src.utils.general_utils import import_class_from_file, load_yaml_file
import sys
from omegaconf import OmegaConf
import omegaconf
import copy
from src.metrics import compute_img_metric
import cv2 
import numpy as np
from functools import partial
import seaborn as sns 
import math


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

def get_2d_emb(batch_size, x, y, out_ch, device):
    out_ch = int(np.ceil(out_ch / 4) * 2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, out_ch, 2).float() / out_ch))
    pos_x = torch.arange(x, device=device).type(inv_freq.type())*2*np.pi/x
    pos_y = torch.arange(y, device=device).type(inv_freq.type())*2*np.pi/y
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    emb_x = get_emb(sin_inp_x).unsqueeze(1)
    emb_y = get_emb(sin_inp_y)
    emb = torch.zeros((x, y, out_ch * 2), device=device)
    emb[:, :, : out_ch] = emb_x
    emb[:, :, out_ch : 2 * out_ch] = emb_y
    return emb[None, :, :, :].repeat(batch_size, 1, 1, 1)


def save_grayimg(img_tensor, save_path, colored=True, cmap=None):
    img_np = img_tensor.squeeze().cpu().numpy()
    img_np = (img_np * 255.0).round().astype(np.uint8)
    if cmap is None:
        cmap = cv2.COLORMAP_JET
    if colored:
        color_img = cv2.applyColorMap(img_np, cmap) 
    else:
        color_img = img_np
    cv2.imwrite(save_path, color_img)
       

def merge_params(conf, args):
    for key, value in args.__dict__.items():
        if key in conf.__dict__['_content'].keys():
            setattr(args, key, getattr(conf, key))
        else:
            setattr(conf, key, value)
        # print(key, getattr(args, key))
    
    for key, value in conf.__dict__['_content'].items():
        if key not in args.__dict__.keys():
            setattr(args, key, value)
    return conf, args


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, tone_map, slit_sweep=()):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_0604")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    valid_F = torch.tensor([1.0], device='cuda') 
    valid_F.unsqueeze(0)
    valid_K = torch.tensor([0.0], device='cuda')
    valid_K.unsqueeze(0)

    psnr_all = 0
    ssim_all = 0
    count = 0

    writer = open(os.path.join(render_path, 'test_metrics.txt'), 'a')
    writer.write('=========================\n')

    if name == 'train':
        # under training mode, we have the in-focus localization network, which uses positional encoding for training views as input
        pos_enc = get_2d_emb(1, views[0].original_image.shape[-2], views[0].original_image.shape[-1], 16, torch.device("cuda:0"))

    for idx in tqdm(range(len(views)), desc="Rendering progress"):
        view = views[idx]
        # print(idx, view.image_name)
        if name == 'train':
            renders = render(view, gaussians, pipeline, background, select_view=idx)
            renders_fake = render(view, gaussians, pipeline, background, valid_F, valid_K)
            rendering = renders["render"]
            rendering_aif = renders_fake["render"]
            rendering_aif = tone_map(rendering_aif)
            if 'depth' in renders.keys():
                depth = renders["depth"]
                depth_aif = renders_fake["depth"]
            if 'coc_alpha' in renders.keys():
                coc_alpha = renders["coc_alpha"]
                coc_a_exp = torch.log1p(coc_alpha)
                coc_a_exp = 1.0 - (coc_a_exp - coc_a_exp.min()) / (coc_a_exp.max() - coc_a_exp.min())
                
                blur_depth = renders["depth"]
                norm_depth = (blur_depth - blur_depth.min()) / (blur_depth.max() - blur_depth.min())
                norm_coc = (coc_alpha - coc_alpha.min()) / (coc_alpha.max()- coc_alpha.min())
                coc_alpha = coc_alpha.squeeze() / coc_alpha.max()
                outfocus_mask = gaussians.mask_net(idx, pos_enc,
                                                    torch.cat([rendering, norm_depth], dim=-3).detach(), norm_coc.detach())
                infocus_mask = 1.0 - outfocus_mask

            # Slit aperture: re-render this view with the slit rotated to the requested angles.
            if gaussians.aperture_type == 'slit' and len(slit_sweep) > 0:
                for ang in slit_sweep:
                    sweep = render(view, gaussians, pipeline, background, select_view=idx,
                                   slit_theta=gaussians.slit_theta_for_angle(math.radians(ang)))
                    sweep_img = torch.clamp(sweep["render"], 0.0, 1.0)
                    torchvision.utils.save_image(
                        sweep_img, os.path.join(render_path, '{}_slit{:06.2f}.png'.format(view.image_name, ang)))
                
        else:
            renders = render(view, gaussians, pipeline, background, valid_F, valid_K)
            rendering = renders["render"]
            if 'depth' in renders.keys():
                depth = renders["depth"]

        
        rendering = torch.clamp(rendering, 0.0, 1.0)    
        gt = view.original_image[0:3, :, :]

        psnr_cur = compute_img_metric(rendering, gt, 'psnr')
        ssim_cur = compute_img_metric(rendering, gt, 'ssim')

        psnr_all += psnr_cur 
        ssim_all += ssim_cur 
        count += 1

        view_idx = view.image_name
        writer.write(f'Idx: {idx:d}, PSNR: {psnr_cur:.4f}, SSIM: {ssim_cur:.4f}\n')

        # save images
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{}_ours.png'.format(view_idx)))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{}.png'.format(view_idx)))
        # save depth image 
        if 'depth' in renders.keys():
            depth = depth.squeeze() / depth.max()
            save_grayimg(depth, os.path.join(render_path, "{}_depth.png".format(view_idx)))


        if name == 'train':
            torchvision.utils.save_image(rendering_aif, os.path.join(render_path, "{}-aif.png".format(view_idx)))
            if 'depth' in renders.keys():
                depth_aif = depth_aif.squeeze() / depth_aif.max()
                save_grayimg(depth_aif, os.path.join(render_path, "{}_depth-aif.png".format(view_idx)))
            if 'coc_alpha' in renders.keys():      
                save_grayimg(coc_alpha, os.path.join(render_path, "{}_coc_a.png".format(view_idx)))
                save_grayimg(coc_a_exp, os.path.join(render_path, '{}_coc_exp_normed.png'.format(view_idx)), colored=False)
                save_grayimg(infocus_mask, os.path.join(render_path, '{}_infocus_mask.png').format(view_idx), colored=True, cmap=None)
    
    avg_psnr = psnr_all / count 
    avg_ssim = ssim_all / count 
    
    print(f'Count: {count:d}, Avg PSNR: {avg_psnr:.4f}  Avg SSIM: {avg_ssim:.4f} \n')
    writer.write(f'Count: {count:d}, Avg PSNR: {avg_psnr:.4f}  Avg SSIM: {avg_ssim:.4f} \n')
    writer.close()




def render_sets(GaussianModel, dataset, iteration, pipeline, skip_train : bool, skip_test : bool, slit_sweep=()):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        print('dataset.hold ', dataset.hold)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians = scene.gaussians

        
        train_cams = scene.getTrainCameras().copy()
        #gaussians.compute_3D_filter(cameras=train_cams)
        gaussians.setup_defocus_params(int(len(scene.getTrainCameras().copy())), scene.cameras_translate, dataset.source_path)

        
        tonemap_type = getattr(dataset, 'tonemap_type', 'none')
        tone_map = ToneMapping(tonemap_type).cuda()
        print('initialized tonemap function')

        defocus_ckpt = os.path.join(str(dataset.model_path), 'defocus_params', f'iteration_{iteration:d}/defocus_params.pth')
        if os.path.isfile(defocus_ckpt):
            gaussians.load_defocus_params(defocus_ckpt)
            print('load defocus ckpt succesfully')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras().copy(), gaussians, pipeline, background, tone_map,
                       slit_sweep=slit_sweep)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras().copy(), gaussians, pipeline, background, tone_map)



if __name__ == "__main__":
    # Set up command line argument parser
    parser_config = ArgumentParser(description="argparser for the config file")
    parser_config.add_argument('--scene', nargs="+", type=str, required=True, default=['real_cake'], help='specified which scene to eval and test')
    parser_config.add_argument('--config', type=str, required=True, help='config file')
    args_config = parser_config.parse_args(sys.argv[1:5])  #parse command-line arguments
    
    # load the config file
    conf = OmegaConf.load(args_config.config)
    # print(conf)
    test_scene_list = args_config.scene 
    
    # import the needed argument and model classes 
    ModelParams = import_class_from_file(f'src/{conf.model_version}/arguments/__init__.py', 'ModelParams')
    PipelineParams = import_class_from_file(f'src/{conf.model_version}/arguments/__init__.py', 'PipelineParams')
    OptimizationParams = import_class_from_file(f'src/{conf.model_version}/arguments/__init__.py', 'OptimizationParams')
    get_combined_args = import_class_from_file(f'src/{conf.model_version}/arguments/__init__.py', 'get_combined_args')
        
    # load model classes
    GaussianModel = import_class_from_file(f'src/{conf.model_version}/gaussian_model.py', 'GaussianModel')
    ToneMapping = import_class_from_file(f'src/{conf.model_version}/renderer/__init__.py', 'ToneMapping')
    training = import_class_from_file(f'src/{conf.model_version}/training.py', 'training')
    global render
    render = import_class_from_file(f'src/{conf.model_version}/renderer/__init__.py', 'render')
        
    # load information for all scenes within the scene_list:
    scene_dict = load_yaml_file('config_files/scene_info.yaml')

    if len(test_scene_list) == 1 and test_scene_list[0] == 'all':
        test_scene_list = [item for item in scene_dict.keys() if 'bush' not in item]
    elif len(test_scene_list) == 1 and test_scene_list[0] == 'all_real':
        test_scene_list = [item for item in scene_dict.keys() if 'bush' not in item and 'synthetic' not in item]
    elif len(test_scene_list) == 1 and test_scene_list[0] == 'all_synthetic':
        test_scene_list = [item for item in scene_dict.keys() if 'bush' not in item and 'real' not in item]
    elif len(test_scene_list) == 1 and test_scene_list[0] == 'all_m360':
        test_scene_list = [item for item in scene_dict.keys() if 'm360' in item]
    else:
        pass
    

    # setup the remaining parameters
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # Slit aperture: extra slit angles (deg) at which every training view is re-rendered.
    parser.add_argument("--slit_sweep", type=float, nargs='*', default=[])
    args = parser.parse_args(sys.argv[5:])


    for idx, scene_name in enumerate(test_scene_list):        
        try:
            conf.source_path = scene_dict[scene_name]['source_path']
            hold = scene_dict[scene_name].get('holdout', 8)
        except:
            print("Scene not found")
        conf.model_path = os.path.join('./output', conf.model_version, conf.model_folder, scene_name)

        # merge arguments and params specified within the training-time config file   
        cur_args = copy.deepcopy(args) 
        setattr(conf, 'hold', hold)
        conf, cur_args = merge_params(conf, cur_args)
        cur_args = get_combined_args(cur_args)
    
    
        if cur_args.iteration == -1:
            test_iteration = int(str(cur_args.iterations)) 
        else:
            test_iteration = cur_args.iteration

        print(f"Rendering {scene_name} from: " + cur_args.model_path)

        # Initialize system state (RNG)
        safe_state(cur_args.quiet)

        # Get band_limit with proper type handling
        band_limit = getattr(cur_args, 'band_limit', False)
        band_limit = False if str(band_limit).lower() == 'false' else bool(band_limit)
        # Get kernel_size with proper default
        kernel_size = getattr(conf, 'kernel_size', 0.1)
        render = partial(render, band_limit=band_limit, kernel_size=float(kernel_size))
        
        # render and compute metrics
        render_sets(GaussianModel, model.extract(cur_args), test_iteration, pipeline.extract(cur_args), 
                    cur_args.skip_train, cur_args.skip_test,
                    slit_sweep=getattr(cur_args, 'slit_sweep', []) or [])