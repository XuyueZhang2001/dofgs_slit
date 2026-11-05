"""
Forward-only check of the energy-conserving CoC (rasterizer option coc_energy).

The source DOF-GS scene of a simulated dataset is rendered with the ground-truth
optics of that dataset (per-pose focus, K, width ratio, slit angles) by the
rasterizer, for several values of coc_energy, and compared with the dataset's
defocused training captures. No training is involved.

With a layered dataset (--psf gaussian_layered, a physical convolution with the
same Gaussian PSF), this measures how close the splat-based defocus comes to a
convolution: the closer coc_energy = 1 is to the captures compared with
coc_energy = 0, the more of the model error the option removes. With a dataset
simulated by the rasterizer itself (--psf gaussian, optionally --coc_energy c),
the value used for the simulation reproduces the captures up to 8-bit
quantisation, which checks this script.

Reported per dataset and value: PSNR / SSIM of the render vs the capture (mean
over the images), and as a reference the all-in-focus render vs the capture.

Usage (from the repository root):
    python tools/check_coc_energy.py \
        --sim ./data/slit_sim/slitsim_real_cake_n4_a0.40_glayer ./data/slit_sim/disksim_real_cake_glayer
    python tools/check_coc_energy.py --sim ./data/slit_sim/slitsim_real_cake_n4_a0.40_glayer \
        --coc_energy 0 0.25 0.5 0.75 1 --max_poses 10 --save_images 3
"""
import argparse
import json
import math
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
from simulate_slit_dataset import build_dataset_args, p11  # also sets repo root as cwd / sys.path

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image

from src.scene import Scene
from src.metrics import compute_img_metric
from src.utils.general_utils import load_yaml_file
from src.model_dofgs.gaussian_model import GaussianModel
from src.model_dofgs.renderer import render, ToneMapping


def parse_args():
    p = argparse.ArgumentParser(description="Splat-based defocus vs simulated captures for several coc_energy values")
    p.add_argument("--sim", nargs="+", required=True, help="simulated dataset folder(s) (with sim_gt.json)")
    p.add_argument("--coc_energy", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    p.add_argument("--coc_alpha_min", type=float, default=-1.0,
                   help="alpha threshold of the renders; < 0: automatic (1/255 for 0, 1e-4 otherwise)")
    p.add_argument("--source_config", default=None,
                   help="config of the source DOF-GS scene (default config_files/config_<source_scene>.yaml)")
    p.add_argument("--max_poses", type=int, default=0, help="use at most this many training poses (0 = all)")
    p.add_argument("--save_images", type=int, default=0,
                   help="save strips (capture | AiF | one render per value) for this many images")
    p.add_argument("--out", default="./output/check_coc_energy")
    return p.parse_args()


def load_png(path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).cuda()


def load_source(sim_gt, source_config, scene_dict):
    """Source DOF-GS scene (ground truth of the simulation), as in simulate_slit_dataset.py."""
    src_scene, src_iter = sim_gt["source_scene"], sim_gt["source_iteration"]
    cfg = source_config or os.path.join("config_files", f"config_{src_scene}.yaml")
    if not os.path.isfile(cfg):
        raise SystemExit(f"Source config {cfg} not found; pass --source_config.")
    conf = OmegaConf.load(cfg)
    dataset, pipe = build_dataset_args(conf, scene_dict, src_scene)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=src_iter, shuffle=False)
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    tone_map = ToneMapping(getattr(dataset, "tonemap_type", "none")).cuda()
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    kw = dict(tone_map=tone_map, band_limit=str(getattr(conf, "band_limit", False)).lower() == "true",
              kernel_size=float(getattr(conf, "kernel_size", 0.1)))
    return gaussians, cams, pipe, bg, kw


@torch.no_grad()
def check_dataset(sim_dir, a, scene_dict, out_dir):
    with open(os.path.join(sim_dir, "sim_gt.json")) as f:
        gt = json.load(f)
    with open(os.path.join(sim_dir, "slit_angles.json")) as f:
        angles = json.load(f)                                   # capture name -> nominal angle (deg)
    groups_path = os.path.join(sim_dir, "focus_groups.json")
    if os.path.isfile(groups_path):
        with open(groups_path) as f:
            pose_of = json.load(f)                              # capture name -> pose
    else:
        pose_of = {n: n.rsplit("_a", 1)[0] for n in angles}
    focus_of = gt.get("focus_per_group") or {}
    aperture = gt.get("aperture", "slit")
    K_L, K_W = gt["K_long"], gt["K_short"]
    bias = gt.get("theta_bias_deg", 0.0)

    gaussians, cams, pipe, bg, kw = load_source(gt, a.source_config, scene_dict)
    gaussians.aperture_coeff = gt["aperture_coeff"]
    print(f"\n=== {sim_dir}: {aperture}, psf {gt.get('psf', 'gaussian')}, simulated coc_energy "
          f"{gt.get('coc_energy', 0.0):g}, K_L = {K_L:.1f}, K_W = {K_W:.1f}, coeff {gt['aperture_coeff']:.4f}")

    names = sorted(angles)
    if a.max_poses > 0:
        poses = sorted({pose_of[n] for n in names})
        keep = set(poses[:: max(1, len(poses) // a.max_poses)][: a.max_poses])
        names = [n for n in names if pose_of[n] in keep]

    values = list(a.coc_energy)
    res = {v: {"psnr": [], "ssim": []} for v in values}
    res_aif = {"psnr": [], "ssim": []}
    saved = 0
    for name in names:
        pose = pose_of[name]
        cam = cams.get(pose)
        if cam is None:
            print(f"[skip] {name}: pose {pose} not among the source training cameras")
            continue
        target = load_png(os.path.join(sim_dir, "images", f"{name}.png"))
        F = focus_of.get(pose, gt["focus_distance"])
        theta = math.radians(angles[name] + bias) if aperture == "slit" else 0.0
        Kw = K_W if aperture == "slit" else K_L

        gaussians.set_coc_energy(0.0)
        aif = render(cam, gaussians, pipe, bg, p11(1.0), p11(0.0), select_view=0,
                     K_w=p11(0.0), slit_theta=p11(0.0), **kw)["render"].clamp(0, 1)
        res_aif["psnr"].append(float(compute_img_metric(aif, target, "psnr")))
        res_aif["ssim"].append(float(compute_img_metric(aif, target, "ssim")))
        strip = [target, aif]
        for v in values:
            gaussians.set_coc_energy(v, a.coc_alpha_min)
            img = render(cam, gaussians, pipe, bg, p11(F), p11(K_L), select_view=0,
                         K_w=p11(Kw), slit_theta=p11(theta), **kw)["render"].clamp(0, 1)
            res[v]["psnr"].append(float(compute_img_metric(img, target, "psnr")))
            res[v]["ssim"].append(float(compute_img_metric(img, target, "ssim")))
            strip.append(img)
        if saved < a.save_images:
            os.makedirs(out_dir, exist_ok=True)
            torchvision.utils.save_image(torch.cat(strip, dim=2), os.path.join(out_dir, f"{name}.png"))
            saved += 1

    n = len(res_aif["psnr"])
    rows = [("all-in-focus render (reference)", res_aif)] + [(f"coc_energy = {v:g}", res[v]) for v in values]
    print(f"{n} captures; render vs capture (mean):")
    print(f"  {'':34s} {'PSNR':>8s} {'SSIM':>8s}")
    for label, r in rows:
        print(f"  {label:34s} {np.mean(r['psnr']):8.2f} {np.mean(r['ssim']):8.4f}")
    if saved:
        print(f"  strips (capture | AiF | " + " | ".join(f"{v:g}" for v in values) + f") in {out_dir}")
    return dict(dataset=sim_dir, aperture=aperture, psf=gt.get("psf", "gaussian"), n_images=n,
                aif=dict(psnr=float(np.mean(res_aif["psnr"])), ssim=float(np.mean(res_aif["ssim"]))),
                coc_energy={f"{v:g}": dict(psnr=float(np.mean(res[v]["psnr"])), ssim=float(np.mean(res[v]["ssim"])))
                            for v in values})


def main():
    a = parse_args()
    for v in a.coc_energy:
        if not 0.0 <= v <= 1.0:
            raise SystemExit(f"--coc_energy values must be in [0, 1], got {v}")
    scene_dict = load_yaml_file("config_files/scene_info.yaml")
    results = [check_dataset(d, a, scene_dict, os.path.join(a.out, os.path.basename(os.path.normpath(d))))
               for d in a.sim]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {os.path.join(a.out, 'summary.json')}")


if __name__ == "__main__":
    main()
