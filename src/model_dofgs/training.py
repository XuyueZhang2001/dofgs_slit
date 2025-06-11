import os
import torch
from random import randint
import random
from src.utils.loss_utils import l1_loss, ssim
from src.model_dofgs.renderer import render, network_gui, ToneMapping
import sys
from src.scene import Scene
import uuid
from tqdm import tqdm
from src.utils.general_utils import save_image, save_depth
from argparse import Namespace
from src.metrics import compute_img_metric
import imageio.v2 as imageio
import numpy as np
   
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    

def pearson_correlation_coefficient(tensor1, tensor2, eps=1e-8):
    mean1 = torch.mean(tensor1)
    mean2 = torch.mean(tensor2)
    
    var1 = torch.var(tensor1)
    var2 = torch.var(tensor2)
    
    if var1 < eps:
        var1 += eps
    if var2 < eps:
        var2 += eps
        
    cov = torch.mean((tensor1 - mean1) * (tensor2 - mean2))

    corr_coefficient = cov / (torch.sqrt(var1) * torch.sqrt(var2))
    return corr_coefficient


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


def pearson_mask_loss(img_src, img_target):
    scaled_target = img_target * (img_src.max() - img_src.min()) + img_src.min()
    loss = 1.0 - pearson_correlation_coefficient(scaled_target.reshape(-1),  img_src.reshape(-1))
    return loss
    

def generate_F_cam_dict(view_stack):
    mapping_dict = {}
    views_sorted = sorted(view_stack.copy(), key = lambda x : x.image_name)
    for i in range(len(views_sorted)):
        cur_view = views_sorted[i]
        mapping_dict[cur_view.image_name] = i 
    return mapping_dict


def training(GaussianModel, dataset, opt, pipe, testing_iterations, saving_iterations, 
             checkpoint_iterations, add_point_iterations, checkpoint, debug_from, tb_writer, depth_estimator=None,
             device=torch.device('cuda:0')):
    
    random.seed(3467)
    torch.manual_seed(3467)
    
    first_iter = 0
    # if we are training on DoFNeRF dataset
    if 'DoF' in dataset.source_path:
        dataset.images = 'images/f11'
    
    # make folders for saving the images produced during training
    img_folders = {}
    for key in ['train', 'test']:
        save_folder = os.path.join(dataset.model_path, f'saved_images/{key}')
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        img_folders[key] = save_folder

    # define the gaussians
    gaussians = GaussianModel(dataset.sh_degree, device)
    scene = Scene(dataset, gaussians)

    viewpoint_stack = None
    trainCameras = scene.getTrainCameras().copy()
    # prepare the positional encoding in advance
    pos_enc = get_2d_emb(1, trainCameras[0].original_image.shape[-2], 
                        trainCameras[0].original_image.shape[-1], 16, device)
    
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
        mapping_dict  = generate_F_cam_dict(viewpoint_stack)
        print('\t'.join([item.image_name for item in viewpoint_stack]))
        
    gaussians.compute_3D_filter(cameras=trainCameras)
    gaussians.setup_defocus_params(len(viewpoint_stack), scene.cameras_translate, 
                                    dataset.source_path, mask_act='sigmoid', COC_MAX=dataset.coc_max)
    scene.gaussians = gaussians
    
    # Set up training stuffs, such as optimizers
    gaussians.training_setup(opt)
    gaussians.join_optim_mlp_params(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    
    ema_loss_for_log = 0.0
    total_iterations = opt.iterations
    progress_bar = tqdm(range(first_iter, int(total_iterations)), desc="Training progress")
    output_numbers = {}

    tone_map = ToneMapping(dataset.tonemap_type).to(device)
    print('initialized tonemap function')
    # Log and save
    valid_F = torch.tensor([1.0], device=device) 
    valid_F.unsqueeze(0)
    valid_K = torch.tensor([0.0], device=device)
    valid_K.unsqueeze(0)
    
    
    first_iter += 1
    for iteration in range(first_iter, total_iterations + 1):        
        iter_start.record()
        ori_iter = iteration
        
        cur_lr = gaussians.update_learning_rate(iteration)
        if ori_iter == opt.iter_defocus_start:
            gaussians.join_optim_defocus_params(opt)
                       

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # if the viewpoint stack has been traversed, update it, i.e., make sure that viewpoint stack is not None
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        
        # randomly pick an viewpoint
        select_index = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(select_index)   
        select_cam_index = mapping_dict[viewpoint_cam.image_name]
        
        # Render
        if (ori_iter - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device=device) if opt.random_background else background
        
        
        #print(f'selct_index: {select_cam_index:d} and view.image_name: {viewpoint_cam.image_name}')

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, select_view=select_cam_index, 
                            tone_map=tone_map, band_limit=opt.band_limit, kernel_size=opt.kernel_size)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], \
                            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.to(device)

        # if we start using in-focus loss to jointly optimize defocused and AiF images.
        if ori_iter > opt.infocus_start_iter:
            # render AiF image using an valid_K (aperture) that is set to zero.
            rendered_aif = render(viewpoint_cam, gaussians, pipe, bg, valid_F, valid_K, tone_map=tone_map, 
                                band_limit=opt.band_limit, kernel_size=opt.kernel_size)
            raii = rendered_aif['radii']
            coc_alpha = render_pkg['coc_alpha'] # get the coc map
            aif_img = rendered_aif['render'] # get the all-in-focus image
            aif_depth = rendered_aif['depth'] # fetch the all-in-focus depth image
            blur_depth = render_pkg['depth'] # fetch the depth rendered with coc > 0

            # normalization
            norm_depth = (blur_depth - blur_depth.min()) / (blur_depth.max() - blur_depth.min() + 1e-5)
            norm_coc = (coc_alpha - coc_alpha.min()) / (coc_alpha.max() - coc_alpha.min() + 1e-5)
            
            # predict out-of-focus mask
            infocus_mask = gaussians.mask_net(select_cam_index, pos_enc,
                                                torch.cat([image, norm_depth], dim=-3).detach(), norm_coc.detach())
            
            log_coc = torch.log1p(coc_alpha.detach())
            infocus_mask_coc = (log_coc - log_coc.min()) / (torch.abs(log_coc.max() - log_coc.min()) + 1e-5)    
            if opt.use_mask_loss:
                mask_loss = opt.lambda_mask_loss * (pearson_mask_loss(infocus_mask.squeeze(), infocus_mask_coc.squeeze())) 
                mask_reg_loss = (-1.0) * opt.lambda_mask_loss * 0.1 * ((infocus_mask * torch.log(infocus_mask + 1e-6)).mean())  # regularizer 
                maskloss = mask_loss + mask_reg_loss
            else:
                maskloss = 0.0
                                       
            blur_img1 = (1.0 - infocus_mask) * aif_img.detach() + infocus_mask * image
            loss_defocus = ((1.0 - opt.lambda_dssim) * l1_loss(blur_img1, gt_image) 
                    + opt.lambda_dssim * (1.0 - ssim(blur_img1, gt_image)))
            loss_defocus.backward(retain_graph=True) 
                
            blur_img2 = (1.0 - infocus_mask) * aif_img + infocus_mask * image.detach()   # this is the image blended with the rendered blurry image and aif rendering
            loss_aif = ((1.0 - opt.lambda_dssim) * l1_loss(blur_img2, gt_image) 
                    + opt.lambda_dssim * (1.0 - ssim(blur_img2, gt_image))) 
            loss = opt.lambda_infocus * loss_aif + maskloss
            loss.backward()

        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))  
            loss.backward()


        iter_end.record()

        ### print, test and save the results
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if ori_iter % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)

            if ori_iter == total_iterations:
                progress_bar.close()
            
            performance = training_report(tb_writer, ori_iter, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, 
                                        scene, render, (pipe, background, valid_F, valid_K, tone_map, opt.band_limit, opt.kernel_size), img_folders)
            output_numbers.update({f'Iter-{ori_iter:d}': performance})
            

            ### Save checkpoint
            if (ori_iter in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(ori_iter))
                # save optimized Gaussian points
                scene.save(ori_iter)
                #save the optimized defocus params (focal distance, aperture, and params for ILN)
                save_folder = f"{scene.model_path}/defocus_params/iteration_{ori_iter:d}"
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                gaussians.save_defocus_params(f"{save_folder}/defocus_params.pth")            

            

            ### Densification
            is_gs_densified = False   # a value that will specified as True after Gaussians are densified
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if (iteration > opt.opacity_reset_interval and opt.use_size_prune) else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold)
                    is_gs_densified = True   
                
                if opt.use_compatnes_density and iteration % opt.densify_compatnes_interval == 0:
                    gaussians.densify_by_shrink_then_compatness(opt.shrink_factor, opt.compatnes_K)
                    is_gs_densified = True

                if is_gs_densified:
                    gaussians.compute_3D_filter(cameras=trainCameras)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    if opt.band_limit:
                        gaussians.reset_opacity_with_filter3D()
                    else:
                        gaussians.reset_opacity()

                        
            if iteration == opt.iter_stop_cam_gd:
                print(f'turn off the gradient for camera parameters at {iteration:d}-th iteration')
                gaussians.Fs.requires_grad_(False)
                gaussians.Ks.requires_grad_(False)
                    
            
            # manually adding gaussian points to quickly densify the scene
            if ori_iter in add_point_iterations:
                gaussians.add_extra_points(opt.num_extra_points, range_factor=opt.extra_points_factor)
                gaussians.update_learning_rate(iteration)
                gaussians.compute_3D_filter(cameras=trainCameras)
                print('Successfully added extra points')

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < (opt.iterations - 100):
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)
                                        
            # Optimizer step
            if ori_iter < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (ori_iter in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(ori_iter))
                torch.save((gaussians.capture(), ori_iter), scene.model_path + "/chkpnt" + str(ori_iter) + ".pth")
    
    del scene, gaussians
    return output_numbers
    



@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, 
                    scene : Scene, renderFunc, renderArgs, img_folders):
    Performance = {}
    
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set, visualize rendered images
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # for saving time, just render and save a few sampled training news
        train_cam_ids = [idx % len(scene.getTrainCameras()) for idx in [1, 3, 7, 9, 28, 20, 16]]  # randomly sampled training views
        mapping_dict = generate_F_cam_dict(scene.getTrainCameras())
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx] for idx in train_cam_ids]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                
                if config['name'] == 'train':
                    psnr_fake = 0.0
                    ssim_fake = 0.0
                
                pos_enc = get_2d_emb(1, config['cameras'][0].original_image.shape[-2], config['cameras'][0].original_image.shape[-1], 16, torch.device("cuda:0"))
                for idx, viewpoint in enumerate(config['cameras']):
                    # for training view, we render defocused image, aif image, coc maps, defocused_depth, aif_depth and out-of-focus mask
                    if config['name'] == 'train':
                        # for training views, apply the aperture and focal distance.
                        rendered = renderFunc(viewpoint, scene.gaussians, renderArgs[0], 
                                               renderArgs[1], select_view=mapping_dict[viewpoint.image_name], tone_map=renderArgs[-3], 
                                               band_limit=renderArgs[-2], kernel_size=renderArgs[-1])
                        image = torch.clamp(rendered['render'], 0.0, 1.0)
                        coc_alpha = rendered['coc_alpha']
                        rendered_aif = renderFunc(viewpoint, scene.gaussians, *renderArgs[:-3], tone_map=renderArgs[-3], 
                                                band_limit=renderArgs[-2], kernel_size=renderArgs[-1])
                        image_aif = torch.clamp(rendered_aif["render"], 0.0, 1.0)  
                        depth_aif = rendered_aif['depth']
                        depth_blur = rendered['depth']
                        
                        # normalization 
                        norm_depth = (depth_blur - depth_blur.min()) / (depth_blur.max() - depth_blur.min())
                        norm_coc = (coc_alpha - coc_alpha.min()) / (coc_alpha.max()- coc_alpha.min())
                        
                        # predict out-of-focus mask
                        infocus_mask = scene.gaussians.mask_net(mapping_dict[viewpoint.image_name], pos_enc,
                                                        torch.cat([image, norm_depth], dim=-3).detach(), norm_coc.detach())
                    else:
                        # for test_view, we just render aif image
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs[:-3], tone_map=renderArgs[-3], 
                                                       band_limit=renderArgs[-2],  kernel_size=renderArgs[-1])["render"], 0.0, 1.0)  
                        image_aif = None
                                              
                    
                    # visualize images within tensorboard
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer:
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), 
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), 
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    

                    # for training view, 'image' denotes defocused iamge, for test view, 'image' denotes aif image
                    psnr_test += compute_img_metric(image, gt_image, 'psnr')
                    ssim_test += compute_img_metric(image, gt_image, 'ssim')
                    
                    # save images
                    save_name_render = os.path.join(img_folders[config['name']], 
                                                    config['name'] + "_view_{}_render_step-{:d}.png".format(viewpoint.image_name, iteration))
                    save_name_gt = os.path.join(img_folders[config['name']], 
                                                    config['name'] + "_view_{}_gt.png".format(viewpoint.image_name))
                    save_image(save_name_render, image)
                    save_image(save_name_gt, gt_image)

                    # for training view, test the metrics beween rendered aif image and ground-truth defocused image
                    if config['name'] == 'train':
                        psnr_fake += compute_img_metric(image_aif, gt_image, 'psnr')
                        ssim_fake += compute_img_metric(image_aif, gt_image, 'ssim')

                        save_name_render = os.path.join(img_folders[config['name']], 
                                                        config['name'] + "_view_{}_render_step-{:d}-aif.png".format(viewpoint.image_name, iteration))
                        save_image(save_name_render, image_aif)
                        save_depth(str.replace(save_name_render, 'aif.png', 'coc-a.png'), coc_alpha)
                        save_depth(str.replace(save_name_render, 'aif.png', 'pred-mask.png'), infocus_mask)
                

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])

                if config['name'] == 'train':
                    psnr_fake /= len(config['cameras'])
                    ssim_fake /= len(config['cameras'])
                    Performance.update({config['name']+'_psnr-fake': psnr_fake})
                    Performance.update({config['name']+'_ssim-fake': ssim_fake})
                
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                Performance.update({config['name']+'_psnr': psnr_test})
                Performance.update({config['name']+'_ssim': ssim_test})
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
            
    return Performance


def merge_params(conf, args):
    for key, value in args.__dict__.items():
        if key in conf.__dict__['_content'].keys():
            print('In config file ', key, getattr(conf, key))
            setattr(args, key, getattr(conf, key))
        else:
            setattr(conf, key, value)
        print(key, getattr(args, key))
    
    for key, value in conf.__dict__['_content'].items():
        if key not in args.__dict__.keys():
            setattr(args, key, value)
    return conf, args


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



