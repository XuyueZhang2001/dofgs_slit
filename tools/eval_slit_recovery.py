"""
Evaluate how well a DOF-GS model (slit or circular aperture) trained on a
simulated dataset (tools/simulate_slit_dataset.py) recovers the all-in-focus
(AIF) scene, in image space and in 3D.

Image metrics (PSNR, SSIM as in DOF-GS / src/metrics, LPIPS if `pip install lpips`):
  * recovered : AIF render of the trained model vs AIF ground truth
  * input     : each blurred input image vs AIF GT (baseline, averaged over angles)
  * mean      : pixel-wise mean of the images of a pose vs AIF GT (baseline)
  * refit     : blurred image re-rendered with the learned F / K / Kw / theta vs the input

3D metrics (ground truth = the source DOF-GS scene the dataset was simulated from):
  A. Depth: AIF depth of the trained model vs depth of the source scene, per view
     (AbsRel, RMSE, inverse-depth RMSE, delta < 1.25, coverage). Depth is the
     alpha-normalised camera distance rendered by the rasterizer. AbsRel is also
     reported per defocus bin, |1/z - 1/F| / d_max, with z from the GT depth.
  B. Point cloud: centres of the opaque (opacity > 0.5), visible Gaussians of the
     trained model vs those of the source scene. Both live in the same COLMAP frame,
     so no alignment is needed. Accuracy (rec -> GT), completeness (GT -> rec),
     Chamfer distance and precision / recall / F-score at thresholds given as
     fractions of the GT bounding-box diagonal.

Usage (from the repository root):
    python tools/eval_slit_recovery.py --config ./config_files/config_slitsim_real_cake_n4_a0.40.yaml \
        --scene slitsim_real_cake_n4_a0.40 --iteration 40000 --save_images
"""
import argparse
import csv
import re
import json
import math
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

# relative defocus bins: |1/z - 1/F| / d_max  (d_max = defocus at the 5-95% depth extremes)
DEFOCUS_BINS = [0.0, 0.25, 0.5, 1.0, float("inf")]


def parse_args():
    p = argparse.ArgumentParser(description="AIF and 3D recovery evaluation on a simulated slit/disk dataset")
    p.add_argument("--config", required=True, help="config used to train the model")
    p.add_argument("--scene", required=True, help="simulated scene key, e.g. slitsim_real_cake_n4_a0.40")
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    p.add_argument("--save_images", action="store_true",
                   help="save GT | input | mean | recovered strips for every pose")
    p.add_argument("--no_3d", action="store_true", help="skip the depth and point-cloud evaluation")
    p.add_argument("--source_config", default=None,
                   help="config of the source DOF-GS scene (default: config_files/config_<source_scene>.yaml)")
    p.add_argument("--opacity_thresh", type=float, default=0.5, help="opacity threshold for point-cloud metrics")
    p.add_argument("--pc_max_points", type=int, default=500_000,
                   help="random subsample of each point cloud for the nearest-neighbour search")
    p.add_argument("--fscore_tau", type=float, nargs="+", default=[0.002, 0.005, 0.01],
                   help="F-score thresholds as fractions of the GT bounding-box diagonal")
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


# ---------------------------------------------------------------------------
# 3D helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def aif_depth(cam, gaussians, pipe, band_limit, kernel_size):
    """Alpha-normalised AIF depth (camera distance) and accumulated opacity of one view."""
    zero = torch.zeros(3, device="cuda")
    kw = dict(band_limit=band_limit, kernel_size=kernel_size, tone_map=None)
    out = render(cam, gaussians, pipe, zero, p11(1.0), p11(0.0), **kw)
    ones = torch.ones_like(gaussians.get_xyz)
    alpha = render(cam, gaussians, pipe, zero, p11(1.0), p11(0.0), override_color=ones, **kw)["render"][0]
    depth = out["depth"].reshape(alpha.shape) / alpha.clamp_min(1e-6)
    return depth, alpha


def distance_to_z(dist, cam):
    """Convert per-pixel camera distance to view-space depth z using the pinhole ray directions."""
    H, W = dist.shape
    fx = W / (2.0 * math.tan(cam.FoVx * 0.5))
    fy = H / (2.0 * math.tan(cam.FoVy * 0.5))
    ys, xs = torch.meshgrid(torch.arange(H, device=dist.device, dtype=torch.float32),
                            torch.arange(W, device=dist.device, dtype=torch.float32), indexing="ij")
    rx, ry = (xs + 0.5 - W / 2.0) / fx, (ys + 0.5 - H / 2.0) / fy
    return dist / torch.sqrt(1.0 + rx ** 2 + ry ** 2)


class DepthMetrics:
    """Per-view depth errors plus pooled AbsRel per relative-defocus bin."""

    def __init__(self, focus, d_max):
        self.focus, self.d_max = focus, d_max
        self.bin_sum = defaultdict(lambda: np.zeros(len(DEFOCUS_BINS) - 1))
        self.bin_cnt = defaultdict(lambda: np.zeros(len(DEFOCUS_BINS) - 1))

    def __call__(self, split, cam, d_rec, a_rec, d_gt, a_gt):
        valid_gt = (a_gt > 0.5) & torch.isfinite(d_gt) & (d_gt > 1e-6)
        valid = valid_gt & (a_rec > 0.5) & torch.isfinite(d_rec) & (d_rec > 1e-6)
        n_gt = int(valid_gt.sum())
        if int(valid.sum()) == 0:
            return None
        p, g = d_rec[valid], d_gt[valid]
        rel = (p - g).abs() / g
        ratio = torch.maximum(p / g, g / p)
        m = dict(abs_rel=float(rel.mean()),
                 rmse=float(torch.sqrt(((p - g) ** 2).mean())),
                 rmse_inv=float(torch.sqrt(((1.0 / p - 1.0 / g) ** 2).mean())),
                 delta1=float((ratio < 1.25).float().mean()),
                 coverage=float(valid.sum()) / max(n_gt, 1))
        # defocus of every valid pixel from the GT depth: |1/z - 1/F| / d_max
        z = distance_to_z(d_gt, cam)[valid]
        rd = ((1.0 / z - 1.0 / self.focus).abs() / self.d_max).cpu().numpy()
        r = rel.cpu().numpy()
        idx = np.clip(np.digitize(rd, DEFOCUS_BINS) - 1, 0, len(DEFOCUS_BINS) - 2)
        self.bin_sum[split] += np.bincount(idx, weights=r, minlength=len(DEFOCUS_BINS) - 1)
        self.bin_cnt[split] += np.bincount(idx, minlength=len(DEFOCUS_BINS) - 1)
        return m

    def binned(self, split):
        out = {}
        for i in range(len(DEFOCUS_BINS) - 1):
            lo, hi = DEFOCUS_BINS[i], DEFOCUS_BINS[i + 1]
            label = f"[{lo:g},{hi:g})" if math.isfinite(hi) else f">={lo:g}"
            cnt = self.bin_cnt[split][i]
            out[label] = dict(abs_rel=float(self.bin_sum[split][i] / cnt) if cnt > 0 else None,
                              pixel_fraction=float(cnt / max(self.bin_cnt[split].sum(), 1)))
        return out


@torch.no_grad()
def visible_opaque_points(gaussians, cams, opacity_thresh, max_points, seed=0):
    """Centres of Gaussians with opacity > thresh that project inside at least one camera."""
    xyz = gaussians.get_xyz
    keep = gaussians.get_opacity.squeeze(-1) > opacity_thresh
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    seen = torch.zeros_like(keep)
    for cam in cams:
        z = (xyz_h @ cam.world_view_transform)[:, 2]
        clip = xyz_h @ cam.full_proj_transform
        ndc = clip[:, :2] / (clip[:, 3:4] + 1e-7)
        seen |= (z > 0.2) & (ndc.abs() < 1.0).all(dim=1)
    pts = xyz[keep & seen]
    if pts.shape[0] > max_points:
        g = torch.Generator(device="cuda").manual_seed(seed)
        pts = pts[torch.randperm(pts.shape[0], generator=g, device="cuda")[:max_points]]
    return pts.contiguous()


@torch.no_grad()
def nn_dist(src, dst, chunk=8192):
    """Distance from every point in src to its nearest neighbour in dst."""
    try:
        import torch_cluster
        idx = torch_cluster.knn(dst, src, k=1)          # row: src index, col: dst index
        d = torch.full((src.shape[0],), float("inf"), device=src.device)
        d[idx[0]] = (src[idx[0]] - dst[idx[1]]).norm(dim=1)
        return d
    except Exception:
        out = []
        for i in range(0, src.shape[0], chunk):
            best = torch.full((min(chunk, src.shape[0] - i),), float("inf"), device=src.device)
            for j in range(0, dst.shape[0], 65536):
                best = torch.minimum(best, torch.cdist(src[i:i + chunk], dst[j:j + 65536]).min(dim=1).values)
            out.append(best)
        return torch.cat(out)


@torch.no_grad()
def pointcloud_metrics(rec, gt, taus_frac):
    sub = gt[torch.randperm(gt.shape[0], device=gt.device)[:200_000]]   # torch.quantile has a size limit
    lo, hi = torch.quantile(sub, 0.01, dim=0), torch.quantile(sub, 0.99, dim=0)
    diag = float((hi - lo).norm())
    d_rg, d_gr = nn_dist(rec, gt), nn_dist(gt, rec)
    acc, comp = float(d_rg.mean()), float(d_gr.mean())
    out = dict(n_rec=int(rec.shape[0]), n_gt=int(gt.shape[0]), bbox_diag=diag,
               accuracy=acc, completeness=comp, chamfer=0.5 * (acc + comp),
               accuracy_median=float(d_rg.median()), completeness_median=float(d_gr.median()))
    for t in taus_frac:
        tau = t * diag
        prec = float((d_rg < tau).float().mean())
        rec_ = float((d_gr < tau).float().mean())
        f = 2 * prec * rec_ / (prec + rec_) if prec + rec_ > 0 else 0.0
        out[f"tau{t:g}"] = dict(tau=tau, precision=prec, recall=rec_, fscore=f)
    return out


def load_source_model(a, sim_gt, scene_dict):
    """Ground-truth scene: the source DOF-GS model the dataset was simulated from."""
    src_scene, src_iter = sim_gt["source_scene"], sim_gt["source_iteration"]
    cfg = a.source_config or os.path.join("config_files", f"config_{src_scene}.yaml")
    if not os.path.isfile(cfg):
        raise SystemExit(f"Source config {cfg} not found; pass --source_config (or --no_3d).")
    src_conf = OmegaConf.load(cfg)
    src_dataset, _ = build_dataset_args(src_conf, scene_dict, src_scene)
    ply = os.path.join(src_dataset.model_path, "point_cloud", f"iteration_{src_iter}", "point_cloud.ply")
    if not os.path.isfile(ply):
        raise SystemExit(f"Source model {ply} not found; pass --source_config (or --no_3d).")
    g = GaussianModel(src_dataset.sh_degree)
    g.load_ply(ply)
    g.aperture_coeff = 0.25          # irrelevant for K = 0 (AIF) renders
    band_limit = str(getattr(src_conf, "band_limit", False)).lower() == "true"
    kernel_size = float(getattr(src_conf, "kernel_size", 0.1))
    print(f"Ground-truth scene for 3D metrics: {ply}")
    return g, band_limit, kernel_size


# ---------------------------------------------------------------------------
def main():
    a = parse_args()
    conf = OmegaConf.load(a.config)
    scene_dict = load_yaml_file("config_files/scene_info.yaml")
    dataset, pipe = build_dataset_args(conf, scene_dict, a.scene)
    sim_dir = dataset.source_path
    with open(os.path.join(sim_dir, "sim_gt.json")) as f:
        sim_gt = json.load(f)

    # ---- load the trained model and its optical parameters ----
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

    # ---- ground-truth scene for the 3D metrics ----
    do_3d = not a.no_3d
    if do_3d:
        gt_gauss, gt_band_limit, gt_kernel = load_source_model(a, sim_gt, scene_dict)
        d_max = sim_gt.get("d_max") or sim_gt["max_blur_px"] / sim_gt["K_long"]
        depth_metrics = DepthMetrics(sim_gt["focus_distance"], d_max)

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
    depth_rows, depth_split = [], defaultdict(list)

    def eval_depth(split, name, cam):
        d_rec, a_rec = aif_depth(cam, gaussians, pipe, band_limit, kernel_size)
        d_gt, a_gt = aif_depth(cam, gt_gauss, pipe, gt_band_limit, gt_kernel)
        m = depth_metrics(split, cam, d_rec, a_rec, d_gt, a_gt)
        if m is not None:
            depth_split[split].append(m)
            depth_rows.append(dict(split=split, pose=name, **m))

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
            if do_3d:
                eval_depth("train", stem, views[0][1])
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
            if do_3d:
                eval_depth("test", cam.image_name, cam)
            if a.save_images:
                torchvision.utils.save_image(torch.cat([gt, aif], dim=2), os.path.join(img_dir, f"test_{cam.image_name}.png"))

        # ---- point cloud ----
        geometry = None
        if do_3d:
            cams = train_cams + test_cams
            rec_pts = visible_opaque_points(gaussians, cams, a.opacity_thresh, a.pc_max_points)
            gt_pts = visible_opaque_points(gt_gauss, cams, a.opacity_thresh, a.pc_max_points)
            geometry = pointcloud_metrics(rec_pts, gt_pts, a.fscore_tau)
            geometry["opacity_thresh"] = a.opacity_thresh

    # ---- report ----
    summary = {split: {name: mean_dict(ms) for name, ms in d.items()} for split, d in per_split.items()}
    if do_3d:
        for split, ms in depth_split.items():
            summary[split]["depth"] = mean_dict(ms)
            summary[split]["depth_by_defocus"] = depth_metrics.binned(split)
        summary["geometry"] = geometry

    keys = list(rows[0].keys())[3:]
    print(f"\nScene {a.scene}, iteration {a.iteration}  |  sim: {sim_gt.get('aperture', 'slit')}, max_blur {sim_gt['max_blur_px']:.2f} px, "
          f"width ratio {sim_gt['width_ratio']}, {len(sim_gt['angles_deg'])} angles, noise {sim_gt['noise_std']}")
    print(f"{'split':6s} {'compared':38s} " + " ".join(f"{k:>10s}" for k in keys))
    labels = {"recovered": "recovered AIF  vs AIF GT", "input": "blurred input  vs AIF GT (baseline)",
              "mean": "mean of angles vs AIF GT (baseline)", "refit": "re-rendered blur vs input (fit)"}
    for split in ("train", "test"):
        for name in ("recovered", "input", "mean", "refit"):
            if name in summary.get(split, {}):
                s = summary[split][name]
                print(f"{split:6s} {labels[name]:38s} " + " ".join(f"{s[k]:10.4f}" for k in keys))
    if "train" in summary:
        gain = summary["train"]["recovered"]["psnr"] - summary["train"]["input"]["psnr"]
        print(f"\nAIF PSNR gain over the blurred inputs (train poses): {gain:+.2f} dB")

    if do_3d:
        dkeys = ["abs_rel", "rmse", "rmse_inv", "delta1", "coverage"]
        print(f"\n{'split':6s} {'depth (AIF render) vs GT scene depth':38s} " + " ".join(f"{k:>10s}" for k in dkeys))
        for split in ("train", "test"):
            if "depth" in summary.get(split, {}):
                s = summary[split]["depth"]
                print(f"{split:6s} {'depth':38s} " + " ".join(f"{s[k]:10.4f}" for k in dkeys))
        bins = list(summary.get("test", summary.get("train", {})).get("depth_by_defocus", {}).keys())
        if bins:
            print(f"\n{'split':6s} {'depth AbsRel by defocus |1/z-1/F|/d_max':38s} " + " ".join(f"{b:>10s}" for b in bins))
            for split in ("train", "test"):
                if "depth_by_defocus" in summary.get(split, {}):
                    s = summary[split]["depth_by_defocus"]
                    vals = " ".join(f"{s[b]['abs_rel']:10.4f}" if s[b]["abs_rel"] is not None else f"{'-':>10s}"
                                    for b in bins)
                    fr = " ".join(f"{100 * s[b]['pixel_fraction']:9.1f}%" for b in bins)
                    print(f"{split:6s} {'AbsRel':38s} {vals}")
                    print(f"{'':6s} {'pixel fraction':38s} {fr}")
        g = geometry
        print(f"\nPoint cloud (opacity > {g['opacity_thresh']:g}, visible): {g['n_rec']} reconstructed vs "
              f"{g['n_gt']} GT Gaussians, GT bbox diagonal {g['bbox_diag']:.4f}")
        print(f"  accuracy (rec->GT) {g['accuracy']:.5f}   completeness (GT->rec) {g['completeness']:.5f}   "
              f"chamfer {g['chamfer']:.5f}")
        for t in a.fscore_tau:
            s = g[f"tau{t:g}"]
            print(f"  tau = {t:g} x diag ({s['tau']:.5f}):  precision {s['precision']:.4f}   recall {s['recall']:.4f}   "
                  f"F-score {s['fscore']:.4f}")

    with open(os.path.join(out_dir, "per_view.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if depth_rows:
        with open(os.path.join(out_dir, "per_view_depth.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(depth_rows[0].keys()))
            w.writeheader()
            w.writerows(depth_rows)
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
