#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
from omegaconf import OmegaConf

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.tonemap_type = 'gamma'
        self.eval = True
        self.coc_max = 15
        self.hold = 8
        # ---- rotating slit aperture ----
        # 'disk': original DOF-GS circular aperture; 'slit': centered rotatable slit.
        self.aperture_type = 'disk'
        # JSON {image_name: angle_deg}; empty -> <source_path>/slit_angles.json
        self.slit_angle_file = ''
        # Optional regex whose first group is the angle (deg) in the image name,
        # used for images missing from the angle file.
        self.slit_angle_regex = ''
        # Initial short-axis / long-axis aperture ratio of the slit.
        self.slit_width_ratio = 0.4
        # Share one focus distance / aperture scale across all views (same lens
        # setting for the whole capture). Set to false to learn them per view.
        self.slit_share_F = True
        self.slit_share_K = True
        # Learn a global offset between the nominal slit angles and the image axes.
        self.slit_learn_theta_offset = True
        # Optics of simulated datasets (<source_path>/sim_gt.json: focus distance F,
        # aperture K (K_L for a slit, K_D for a disk) and slit width ratio):
        #   'none'  : default; F and K are initialised from the scene depth and learned
        #   'init'  : ground-truth values as initialisation, still learned
        #   'fixed' : ground-truth values, frozen (theta offset fixed at 0)
        self.optics_from_sim_gt = 'none'
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_infocus = 1.0
        self.lambda_mask_loss = 1e-3
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.fdist_lr = 0.001
        self.kapert_lr = 0.01
        self.slit_ratio_lr = 0.01
        self.slit_theta_lr = 0.001
        # Each iteration renders all images that share one camera pose (e.g. the
        # n slit angles) and averages their losses. False: one image per
        # iteration, as in the original DOF-GS.
        self.slit_group_by_pose = False
        # Directional all-in-focus loss for slit captures (src/utils/directional_loss.py):
        # compare the AiF render and the capture across the short slit axis after
        # smoothing both along the long axis. Only used for aperture_type 'slit'.
        self.slit_dir_loss = 0.0            # weight; 0 disables it (default). If used: 0.005-0.03, see README
        self.slit_dir_start_iter = -1       # -1: same as infocus_start_iter (never earlier)
        self.slit_dir_warmup = 2000         # linear ramp of the weight after the start
        self.slit_dir_sigma_w_max = 1.0     # short-axis blur std threshold (px) of the mask
        self.slit_dir_sigma_l_max = 4.0     # largest long-axis blur std (px) absorbed by the smoothing
        self.slit_dir_smooth_mult = 2.5     # smoothing std along the long axis = mult * sigma_l_max
        self.random_background = False
        self.iter_defocus_start = 100
        self.num_extra_points = 100000
        self.use_scale_regularization = False
        self.max_gauss_ratio = 10.0
        self.infocus_start_iter = 3000
        self.densify_compatnes_interval = 1000
        self.shrink_factor = 1.3
        self.compatnes_K = 3
        self.compatnes_until_iter = 30000
        self.min_opacity = 0.005
        self.use_compatnes_density = False
        # self.set_mlp_params()
        self.set_mask_params()
        self.extra_points_factor = 0.1
        self.use_size_prune = False
        self.iter_stop_cam_gd = 15000 
        self.band_limit = False
        self.kernel_size = 0.1
        self.F_optimizer_type = 'adam'
        
        super().__init__(parser, "Optimization Parameters")

    def set_mlp_params(self):
        self.mlp_layer = 3
        self.mlp_inter_dim = 128
        self.mlp_pe = True
        self.mlp_pe_order = 10
        self.mlp_input_dim = 1
        self.mlp_inter_dim = 128
        self.mlp_out_dim = 1
    
    def set_mask_params(self):
        self.mask_loss_alpha = 0.0001
        self.use_mask_loss = True
        self.maskNet_start_iter = 1000
        self.mask_nn_lr = 5e-4
        
    
def get_combined_args(args_cmdline):
    print('args_cmdline.model_path', args_cmdline.model_path)
    args_cfgfile = {}
    cfgfilepath = os.path.join(args_cmdline.model_path, "config_runtime.yaml")
    
    try:
        print("Looking for config file in", cfgfilepath)
        args_cfgfile = OmegaConf.load(cfgfilepath)
    except TypeError:
        print("Config file not found at ", cfgfilepath)
        pass

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
