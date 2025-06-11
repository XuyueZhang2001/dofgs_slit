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

import os
import torch
from random import randint
import sys
from src.scene import Scene
from src.utils.general_utils import safe_state, import_class_from_file, load_yaml_file
import uuid
from tqdm import tqdm
from src.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf
import omegaconf
from importlib import import_module
import json
import shutil 
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    


def backup_code(conf):
    if os.path.isdir(os.path.join(conf.model_path, 'backup_src')):
        shutil.rmtree(os.path.join(conf.model_path, 'backup_src'))
    shutil.copytree('./src', os.path.join(conf.model_path, 'backup_src'))
    


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



if __name__ == "__main__":
    # Set up command line argument parser
    parser_config = ArgumentParser(description="argparser for the config file")
    parser_config.add_argument('--config', type=str, required=True, help='config file')
    args_config = parser_config.parse_args(sys.argv[1:3])  # sys.argv[1:3] contains the config file path
    
    # load the config file
    conf = OmegaConf.load(args_config.config)
    
    # import the needed argument and model classes 
    ModelParams = import_class_from_file(f'src/{conf.model_version}/arguments/__init__.py', 'ModelParams')
    PipelineParams = import_class_from_file(f'src/{conf.model_version}/arguments/__init__.py', 'PipelineParams')
    OptimizationParams = import_class_from_file(f'src/{conf.model_version}/arguments/__init__.py', 'OptimizationParams')
        
    # load model classes
    GaussianModel = import_class_from_file(f'src/{conf.model_version}/gaussian_model.py', 'GaussianModel')
    training = import_class_from_file(f'src/{conf.model_version}/training.py', 'training')
    
    # load_scene_info:
    scene_dict = load_yaml_file('config_files/scene_info.yaml')
        
    # setup the remaining parameters
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=9933)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--add_point_iterations', nargs="+", type=int, default=[2500])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 3_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 3_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[3:])
    
    if args.save_iterations[-1] != conf.iterations:
        args.save_iterations.append(conf.iterations)
    
    if args.test_iterations[-1] != conf.iterations:
        args.test_iterations.append(conf.iterations)
    

    if conf.scene_name == 'all':
        scene_list = [item for item in scene_dict.keys() if 'bush' not in item]
    elif conf.scene_name == 'all_synthetic':
        scene_list = [item for item in scene_dict.keys() if 'bush' not in item and 'real' not in item and 'synthetic_gt' not in item]
    elif conf.scene_name == 'all_real':
        scene_list = [item for item in scene_dict.keys() if 'bush' not in item and 'synthetic' not in item]
    else:
        scene_list = list(conf.scene_name) if isinstance(conf.scene_name, omegaconf.listconfig.ListConfig) else [conf.scene_name]           
    print('We will train following scenes:', '.'.join(scene_list))
 
    
    depth_estimator=None
    # create the table head for the final performance summary table
    md_table_head = "| Scene Name | Iteration | Train PSNR  | Train PSNR (fake)  | Train SSIM | Train SSIM (fake) | Test PSNR  | Test SSIM |\n|-------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|\n"
    markdown_table = md_table_head

    
    for idx, scene_name in enumerate(scene_list):
        print(f'Start training {scene_name}==============================')

        # table head for the current scene
        cur_md_table = md_table_head
        conf.model_path = os.path.join('./output', conf.model_version, conf.model_folder, scene_name)

        try:
            conf.source_path = scene_dict[scene_name]['source_path']
            llffhold = scene_dict[scene_name].get('holdout', 8)
            setattr(conf, 'hold', llffhold)
        except:
            print(f'Data source for {scene_name} is not found, which is ', conf.source_path)    
        
        # merge arguments and params specified within the config file    
        conf, args = merge_params(conf, args)
        if 'infocus_range_list' in conf.__dict__['_content'].keys():
            args.infocus_range_list = [list(item) for item in list(conf.infocus_range_list)]
        

        # prepare the output folder and tensorboard writer
        tb_writer = prepare_output_and_logger(lp.extract(args))
        # backup source code folder into the model path (for reproducibility)
        backup_code(conf)
        # save the runtime config file to model path
        OmegaConf.save(config=conf, f=os.path.join(conf.model_path, 'config_runtime.yaml'))
        print("Optimizing " + args.model_path)

        # Initialize system state (RNG)
        safe_state(args.quiet)
        # Start GUI server, configure and run training
        #network_gui.init(args.ip, args.port)
        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        
        start = time.time()
        return_numbers = training(GaussianModel, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, 
                args.checkpoint_iterations, args.add_point_iterations, args.start_checkpoint, args.debug_from, tb_writer, depth_estimator)
        end = time.time()
        elapsed = end - start 
        minutes = elapsed / 60.0
        print(f'Training of {scene_name} took {minutes:.4f} minutes')


        # summarize the performance into a markdown table
        for iter_idx, iter in enumerate(conf.test_iterations):
            key = f'Iter-{iter:d}'
            if key in return_numbers.keys():
                cur_train_psnr, cur_test_psnr = return_numbers[f'Iter-{iter:d}']['train_psnr'], return_numbers[f'Iter-{iter:d}']['test_psnr']
                cur_train_ssim, cur_test_ssim = return_numbers[f'Iter-{iter:d}']['train_ssim'], return_numbers[f'Iter-{iter:d}']['test_ssim']
                
                if 'train_psnr-fake' not in return_numbers[key].keys():
                    if iter_idx == 0:
                        cur_row = f"| {scene_name} | {iter:d} | {cur_train_psnr: .4f} | | {cur_train_ssim: .4f} | |{cur_test_psnr: .4f} | {cur_test_ssim: .4f} |\n"
                    else:
                        cur_row = f"|  | {iter:d} | {cur_train_psnr: .4f} |  | {cur_train_ssim: .4f} |  |{cur_test_psnr: .4f} | {cur_test_ssim: .4f}  |\n"
                else:
                    train_psnr_fake, train_ssim_fake = return_numbers[f'Iter-{iter:d}']['train_psnr-fake'], return_numbers[f'Iter-{iter:d}']['train_ssim-fake']
                    if iter_idx == 0:
                        cur_row = f"| {scene_name} | {iter:d} | {cur_train_psnr: .4f} |{train_psnr_fake: .4f} | {cur_train_ssim: .4f} | {train_ssim_fake: .4f}|{cur_test_psnr: .4f} | {cur_test_ssim: .4f} |\n"
                    else:
                        cur_row = f"|  | {iter:d} | {cur_train_psnr: .4f} | {train_psnr_fake: .4f}  | {cur_train_ssim: .4f} |{train_ssim_fake: .4f}  |{cur_test_psnr: .4f} | {cur_test_ssim: .4f}  |\n"
                
                markdown_table += cur_row
                cur_md_table += cur_row
              
        
        # Write into a markdown file
        with open(os.path.join(conf.model_path, "cur_perform.md"), "w") as f:
            f.write(cur_md_table)
         
        # All done
        print(f"\nTraining Scene {idx:d} complete.")
        