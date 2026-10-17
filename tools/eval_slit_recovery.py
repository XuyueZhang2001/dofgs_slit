"""
Evaluate how well a DOF-GS model (slit or circular aperture) trained on a
simulated dataset (tools/simulate_slit_dataset.py) recovers the all-in-focus
(AIF) scene.

For every pose the recovered AIF render is compared with the simulation ground
truth, together with two baselines that need no reconstruction:
  * input   : each slit-blurred input image vs AIF GT (averaged over angles)
  * mean    : pixel-wise mean of the n slit images of a pose vs AIF GT
and the fitting quality of the defocus model:
  * refit   : slit image re-rendered with the learned F / K / Kw / theta vs the input

Metrics: PSNR, SSIM (same implementation as DOF-GS, src/metrics) and LPIPS if
the `lpips` package is installed (pip install lpips).

Usage (from the repository root):
    python tools/eval_slit_recovery.py --config ./config_files/config_slitsim_real_cake.yaml \
        --scene slitsim_real_cake --iteration 40000 --save_images
"""
import argparse
import csv
import re
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate_slit_dataset import build_dataset_args, p11  # also sets repo root as cwd / sys.path

import torchvision
from src.scene import Scene
from src.metrics import compute_img_metric
from src.utils.slit_utils import group_views_by_pose
from aperture_theory import summarize as theory_summary
from src.utils.general_utils import load_yaml_file
from src.model_dofgs.gaussian_model import GaussianModel
from src.model_dofgs.renderer import render, ToneMapping


def parse_args():
    p = argparse.ArgumentParser(description="AIF recovery evaluation on a simulated slit dataset")
    p.add_argument("--config", required=True, help="config used to train the slit model")
    p.add_argument("--scene", required=True, help="simulated scene key, e.g. slitsim_real_cake")
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    p.add_argument("--save_images", action="store_true",
                   help="save GT | input | mean | recovered strips for every pose")
    return p.parse_args()


def load_png(path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).cuda()


class Metrics:
    def __init__(self, lpips_net):
        try:
            import lpips
            self.lpips = lpips.LPIPS(net=lpips_net, verbose=False).cuda().eval()
            self.lpips_name = f"lpips_{lpips_net}"
        except ImportError:
            print("[INFO] `lpips` not installed, LPIPS skipped (pip install lpips).")
            self.lpips, self.lpips_name = None, None

    @torch.no_grad()
    def __call__(self, img, gt):
        img, gt = img.clamp(0, 1), gt.clamp(0, 1)
        out = {"psnr": float(compute_img_metric(img, gt, "psnr")),
               "ssim": float(compute_img_metric(img, gt, "ssim"))}
        if self.lpips is not None:
            out[self.lpips_name] = float(self.lpips(img[None] * 2 - 1, gt[None] * 2 - 1).item())
        return out


def mean_dict(ds):
    keys = ds[0].keys()
    return {k: float(np.mean([d[k] for d in ds])) for k in keys}


def main():
    a = parse_args()
    conf = OmegaConf.load(a.config)
    scene_dict = load_yaml_file("config_files/scene_info.yaml")
    dataset, pipe = build_dataset_args(conf, scene_dict, a.scene)
    sim_dir = dataset.source_path
    with open(os.path.join(sim_dir, "sim_gt.json")) as f:
        sim_gt = json.load(f)

    # ---- load the trained slit model and its optical parameters ----
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=a.iteration, shuffle=False)
    train_cams = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)  # == view-index order
    test_cams = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    gaussians.setup_defocus_params(len(train_cams), scene.cameras_translate, dataset.source_path)
    ckpt = os.path.join(dataset.model_path, "defocus_params", f"iteration_{a.iteration}", "defocus_params.pth")
    gaussians.load_defocus_params(ckpt)
    print(f"Model aperture: {gaussians.aperture_type}  |  simulated aperture: {sim_gt.get('aperture', 'slit')}")

    tone_map = ToneMapping(getattr(dataset, "tonemap_type", "none")).cuda()
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    band_limit = str(getattr(conf, "band_limit", False)).lower() == "true"
    kernel_size = float(getattr(conf, "kernel_size", 0.1))
    kw = dict(tone_map=tone_map, band_limit=band_limit, kernel_size=kernel_size)
    metrics = Metrics(a.lpips_net)

    out_dir = os.path.join(dataset.model_path, f"eval_slit_recovery/iteration_{a.iteration}")
    img_dir = os.path.join(out_dir, "strips")
    os.makedirs(img_dir if a.save_images else out_dir, exist_ok=True)

    # group the images that share a camera pose (the n slit angles, or disk copies);
    # the simulator names them <pose>_aXX and stores the AiF ground truth as aif/<pose>.png
    poses = {}
    for group in group_views_by_pose(train_cams):
        stem = re.sub(r"_a\d+$", "", group[0][1].image_name)
        poses[stem] = group

    rows, per_split = [], defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        # ---- training poses: AIF GT from <sim>/aif ----
        for stem, views in poses.items():
            gt = load_png(os.path.join(sim_dir, "aif", f"{stem}.png"))
            aif = render(views[0][1], gaussians, pipe, bg, p11(1.0), p11(0.0), **kw)["render"].clamp(0, 1)
            inputs = [cam.original_image.cuda() for _, cam in views]
            mean_in = torch.stack(inputs).mean(0)

            m_rec = metrics(aif, gt)
            m_in = mean_dict([metrics(x, gt) for x in inputs])
            m_mean = metrics(mean_in, gt)
            m_fit = mean_dict([metrics(render(cam, gaussians, pipe, bg, select_view=i, **kw)["render"], cam.original_image.cuda())
                               for i, cam in views])
            for name, m in (("recovered", m_rec), ("input", m_in), ("mean", m_mean), ("refit", m_fit)):
                per_split["train"][name].append(m)
                rows.append(dict(split="train", pose=stem, method=name, **m))
            if a.save_images:
                torchvision.utils.save_image(torch.cat([gt, inputs[0], mean_in, aif], dim=2),
                                             os.path.join(img_dir, f"train_{stem}.png"))

        # ---- test poses: AIF GT is the test image itself ----
        for cam in test_cams:
            gt = cam.original_image.cuda()
            aif = render(cam, gaussians, pipe, bg, p11(1.0), p11(0.0), **kw)["render"].clamp(0, 1)
            m = metrics(aif, gt)
            per_split["test"]["recovered"].append(m)
            rows.append(dict(split="test", pose=cam.image_name, method="recovered", **m))
            if a.save_images:
                torchvision.utils.save_image(torch.cat([gt, aif], dim=2), os.path.join(img_dir, f"test_{cam.image_name}.png"))

    # ---- report ----
    summary = {split: {name: mean_dict(ms) for name, ms in d.items()} for split, d in per_split.items()}
    keys = list(rows[0].keys())[3:]
    print(f"\nScene {a.scene}, iteration {a.iteration}  |  sim: {sim_gt.get('aperture', 'slit')}, max_blur {sim_gt['max_blur_px']:.2f} px, "
          f"width ratio {sim_gt['width_ratio']}, {len(sim_gt['angles_deg'])} angles, noise {sim_gt['noise_std']}")
    print(f"{'split':6s} {'compared':38s} " + " ".join(f"{k:>10s}" for k in keys))
    labels = {"recovered": "recovered AIF  vs AIF GT", "input": "slit input     vs AIF GT (baseline)",
              "mean": "mean of angles vs AIF GT (baseline)", "refit": "re-rendered slit vs input (fit)"}
    for split in ("train", "test"):
        for name in ("recovered", "input", "mean", "refit"):
            if name in summary.get(split, {}):
                s = summary[split][name]
                print(f"{split:6s} {labels[name]:38s} " + " ".join(f"{s[k]:10.4f}" for k in keys))
    if "train" in summary:
        gain = summary["train"]["recovered"]["psnr"] - summary["train"]["input"]["psnr"]
        print(f"\nAIF PSNR gain over the blurred inputs (train poses): {gain:+.2f} dB")

    with open(os.path.join(out_dir, "per_view.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        design = dict(aperture=sim_gt.get("aperture", "slit"), n_angles=len(set(sim_gt["angles_deg"])),
                      width_ratio=sim_gt["width_ratio"], images_per_pose=len(sim_gt["angles_deg"]),
                      group_by_pose=bool(getattr(conf, "slit_group_by_pose", False)),
                      iterations=int(getattr(conf, "iterations", a.iteration)), model_folder=conf.model_folder)
        if design["aperture"] == "slit":
            theory = sim_gt.get("theory") or theory_summary(sim_gt["width_ratio"], sim_gt["angles_deg"])
        else:
            theory = dict(worst_var_ratio=1.0, dof_gain_worst=1.0, mean_var_ratio=1.0, dof_gain_mean=1.0)
        json.dump(dict(scene=a.scene, iteration=a.iteration, design=design, theory=theory,
                       sim_gt=sim_gt, summary=summary), f, indent=2)
    print(f"Saved per-view metrics and summary to {out_dir}")


if __name__ == "__main__":
    main()
