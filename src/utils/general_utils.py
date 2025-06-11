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

import torch
import sys
from datetime import datetime
import numpy as np
import random
import importlib
import yaml 
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def import_class_from_file(file_path, class_name):
    # Create a module specification from the given file path
    spec = importlib.util.spec_from_file_location("module.name", file_path)

    # Create a new module based on this specification
    module = importlib.util.module_from_spec(spec)

    # Load the module (execute the code in the module)
    spec.loader.exec_module(module)

    # Get the class from the module
    return getattr(module, class_name)


def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def to_numpy(img):
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img[0]
        img = img.squeeze()
        img = img.detach().cpu()
    
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        img = img.numpy()
    return img
        

def save_image(path, img):
    img = to_numpy(img)
    img = (img * 255).round().astype(np.uint8)
    imageio.imwrite(path, img)
        
def save_depth(save_path, img_tensor):
    img_tensor = img_tensor / (img_tensor.max() + 1e-6)
    img_np = img_tensor.squeeze().cpu().numpy()
    img_np = (img_np * 255.0).round().astype(np.uint8)
    color_img = cv2.applyColorMap(img_np, cv2.COLORMAP_JET) 
    cv2.imwrite(save_path, color_img)
       
       
def calculate_psnr_heatmap(image1, image2, alpha=0.6):
    image1 = to_numpy(image1)
    image2 = to_numpy(image2)
    
    # ensure that two images have the same resolution
    assert image1.shape == image2.shape, "Images must have the same dimensions."
    
    
    # Initialize PSNR heatmap array
    psnr_heatmap = np.zeros(image1.shape[:2])
    
    # Iterate through each pixel of the image
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            # Calculate PSNR for each pixel and store it in the heatmap array
            mse = np.mean((image1[i, j] - image2[i, j]) ** 2)
            psnr_heatmap[i, j] = mse * 10.0

    heatmap_mask = plt.cm.coolwarm(psnr_heatmap)  # Using coolwarm colormap
    mask_image = heatmap_mask[..., :3] * alpha + (1-alpha) * image2
        
    return psnr_heatmap, mask_image
    

def plot_heatmap(heatmap, blended_image, save_path):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im =  ax.imshow(heatmap, cmap='coolwarm', interpolation='nearest', vmin=np.min(heatmap), vmax=np.percentile(heatmap, 99))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title("PSNR Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    blended_image = (blended_image * 255.0).round().astype(np.uint8)
    imageio.imwrite(str.replace(save_path, 'heat_map', 'heat_map_masked'), blended_image)
    