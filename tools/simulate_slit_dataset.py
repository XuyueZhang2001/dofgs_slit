"""
Simulate a rotating-slit-aperture dataset from a scene already reconstructed
by the original DOF-GS (circular aperture).

The trained DOF-GS model is used as the ground-truth scene. For every training
pose, the scene is rendered through a centered slit at `n_angles` orientations
with a known focus distance F, long-axis aperture K_L and short-axis aperture
K_W = width_ratio * K_L. Test poses are rendered all-in-focus (AIF), so the test
metrics of the slit model directly measure AIF recovery.

Output layout (a regular COLMAP scene that train.py / render.py can read):
    <out>/images/<pose>_aXX.png    slit-blurred training images
    <out>/images/<pose>.png        AIF test images
    <out>/aif/<pose>.png           AIF renders of the training poses (reference only)
    <out>/defocus/<pose>.npy       ground-truth defocus map: full long-axis blur length (px)
                                   per pixel, AIF-composited; NaN where nothing is rendered
    <out>/sparse/0/                cameras.txt, images.txt (poses duplicated per angle), points3D.*
    <out>/slit_angles.json         nominal slit angle (deg) per training image
    <out>/split.json               explicit test list
    <out>/sim_gt.json              ground-truth optical parameters
It also writes config_files/config_<new_scene>.yaml and registers <new_scene>
in config_files/scene_info.yaml.

Circular-aperture baseline (--aperture disk): renders the same scene through a
centered circular aperture instead. With --match_slit <slit sim folder> it uses
the same focus distance and an aperture of the SAME AREA as that slit:
    pi * (K_D / 2)^2 = K_L * K_W   ->   K_D = 2 * sqrt(K_L * K_W / pi)
(blur lengths are K * |1/z - 1/F|, so equal K-areas mean equal aperture areas).
The disk PSF is approximated by a Gaussian with the variance of a uniform disk
(r^2 / 4), consistent with the variance-matched box used for the slit.

Note: the slit images are produced with the same Gaussian PSF approximation
that the slit model uses for reconstruction, so this is a self-consistency
test of the optimisation (no model mismatch). Use --noise_std and
--theta_bias_deg to make it harder.

Usage (from the repository root):
    python tools/simulate_slit_dataset.py --config ./config_files/config_real_cake.yaml \
        --scene real_cake --iteration 15000 --n_angles 4 --max_blur_px 20 --width_ratio 0.1
"""
import argparse
import json
import math
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch
import torchvision
from omegaconf import OmegaConf

from src.scene import Scene
from src.scene.colmap_loader import (read_extrinsics_binary, read_extrinsics_text,
                                     read_intrinsics_binary, read_intrinsics_text)
from src.utils.general_utils import load_yaml_file
from src.model_dofgs.arguments import ModelParams, PipelineParams
from src.model_dofgs.gaussian_model import GaussianModel, SLIT_APERTURE_COEFF

# Variance coefficient of a uniform disk of radius r (per axis): r^2 / 4.
DISK_UNIFORM_COEFF = 0.25
from src.model_dofgs.renderer import render, ToneMapping

# Lower bound of the long-axis aperture representable by the current K
# parametrisation: K = clamp(Ks, 0.01, 200) * 100 + 100.
K_MIN_REPRESENTABLE = 101.0


def parse_args():
    p = argparse.ArgumentParser(description="Simulate a slit-aperture dataset from a trained DOF-GS scene")
    p.add_argument("--config", required=True, help="config used to train the source DOF-GS scene")
    p.add_argument("--scene", required=True, help="source scene key in config_files/scene_info.yaml")
    p.add_argument("--iteration", type=int, required=True, help="iteration of the trained source model")
    p.add_argument("--n_angles", type=int, default=4, help="number of slit orientations per pose")
    p.add_argument("--full_circle", action="store_true",
                   help="spread angles over 360 deg instead of 180 deg (theta and theta+180 give identical PSFs)")
    p.add_argument("--max_blur_px", type=float, default=20.0,
                   help="full length (px) of the long-axis PSF at the most defocused depth (5-95%% depth range)")
    p.add_argument("--width_ratio", type=float, default=0.1, help="ground-truth short/long aperture ratio K_W/K_L")
    p.add_argument("--focus_quantile", type=float, default=0.5,
                   help="focus distance as a quantile of the inverse scene depth (0 = far, 1 = near)")
    p.add_argument("--focus_depth", type=float, default=None, help="explicit focus distance, overrides --focus_quantile")
    p.add_argument("--theta_bias_deg", type=float, default=0.0,
                   help="systematic offset between rendered and recorded angles (tests theta-offset learning)")
    p.add_argument("--noise_std", type=float, default=0.0, help="additive Gaussian noise std on [0, 1] images")
    p.add_argument("--new_scene", default=None, help="name of the simulated scene (default: slitsim_<scene>)")
    p.add_argument("--out", default=None, help="output folder (default: ./data/slit_sim/<new_scene>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aperture", choices=["slit", "disk"], default="slit",
                   help="simulate the rotating slit, or a circular-aperture baseline")
    p.add_argument("--match_slit", default=None,
                   help="(disk) slit simulation folder: reuse its focus distance and match its aperture area")
    p.add_argument("--disk_copies", type=int, default=1,
                   help="(disk) images per pose; >1 only makes sense with noise (equal number of captures)")
    p.add_argument("--defocus_only", action="store_true",
                   help="only (re)write <out>/defocus/*.npy for an existing simulated dataset, "
                        "using the optics stored in its sim_gt.json")
    return p.parse_args()


def build_dataset_args(conf, scene_dict, scene):
    """Reproduce the ModelParams / PipelineParams that train.py builds for `scene`."""
    parser = argparse.ArgumentParser()
    mp, pp = ModelParams(parser), PipelineParams(parser)
    args = parser.parse_args([])
    for k, v in OmegaConf.to_container(conf, resolve=False).items():
        setattr(args, k, v)
    args.source_path = scene_dict[scene]["source_path"]
    args.hold = scene_dict[scene].get("holdout", 8)
    args.model_path = os.path.join("./output", conf.model_version, conf.model_folder, scene)
    return mp.extract(args), pp.extract(args)


def read_colmap(sparse_dir):
    try:
        extr = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
        intr = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    except Exception:
        extr = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
        intr = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))
    by_stem = {os.path.basename(e.name).split(".")[0]: e for e in extr.values()}
    return by_stem, intr


def pinhole_params(cam):
    """Express a COLMAP camera as PINHOLE (fx, fy, cx, cy), matching how dataset_readers uses it."""
    if cam.model == "PINHOLE":
        return list(cam.params[:4])
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        f, cx, cy = cam.params[:3]
        return [f, f, cx, cy]
    raise ValueError(f"Unsupported COLMAP camera model {cam.model}")


@torch.no_grad()
def visible_inverse_depths(gaussians, cameras, max_per_view=200_000, seed=0):
    """Inverse view-space depths of opaque Gaussians visible in the given cameras."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    xyz = gaussians.get_xyz
    opaque = gaussians.get_opacity.squeeze(-1) > 0.5
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    inv = []
    for cam in cameras:
        z = (xyz_h @ cam.world_view_transform)[:, 2]
        clip = xyz_h @ cam.full_proj_transform
        ndc = clip[:, :2] / (clip[:, 3:4] + 1e-7)
        m = opaque & (z > 0.2) & (ndc.abs() < 1.0).all(dim=1)
        zi = 1.0 / z[m]
        if zi.numel() > max_per_view:
            zi = zi[torch.randperm(zi.numel(), generator=g, device="cuda")[:max_per_view]]
        inv.append(zi)
    return torch.cat(inv)


def p11(v):
    return torch.tensor([[float(v)]], device="cuda")


@torch.no_grad()
def defocus_map(cam, gaussians, pipe, F, K_L, band_limit, kernel_size):
    """
    Ground-truth per-pixel defocus: full PSF length along the slit long axis,
    K_L * |1/z - 1/F| (px), composited with the all-in-focus alpha weights.
    """
    xyz = gaussians.get_xyz
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    z = (xyz_h @ cam.world_view_transform)[:, 2].clamp_min(1e-7)
    blur = (K_L * (1.0 / z - 1.0 / F).abs())[:, None].repeat(1, 3)
    zero_bg = torch.zeros(3, device="cuda")
    kw = dict(band_limit=band_limit, kernel_size=kernel_size, tone_map=None)
    num = render(cam, gaussians, pipe, zero_bg, p11(1.0), p11(0.0), override_color=blur, **kw)["render"][0]
    acc = render(cam, gaussians, pipe, zero_bg, p11(1.0), p11(0.0),
                 override_color=torch.ones_like(blur), **kw)["render"][0]
    out = num / acc.clamp_min(1e-4)
    out[acc < 0.5] = float("nan")
    return out.cpu().numpy().astype("float32")


def write_defocus_maps(out, cams, gaussians, pipe, F, K_L, band_limit, kernel_size):
    import numpy as np
    d_dir = os.path.join(out, "defocus")
    os.makedirs(d_dir, exist_ok=True)
    for cam in cams:
        np.save(os.path.join(d_dir, f"{cam.image_name}.npy"),
                defocus_map(cam, gaussians, pipe, F, K_L, band_limit, kernel_size))
    print(f"Wrote {len(cams)} ground-truth defocus maps to {d_dir}")


def main():
    a = parse_args()
    torch.manual_seed(a.seed)

    conf = OmegaConf.load(a.config)
    scene_dict = load_yaml_file("config_files/scene_info.yaml")
    dataset, pipe = build_dataset_args(conf, scene_dict, a.scene)

    prefix = "slitsim_" if a.aperture == "slit" else "disksim_"
    new_scene = a.new_scene or prefix + a.scene.replace("synthetic", "syn")
    out = a.out or os.path.join("./data/slit_sim", new_scene)
    # dataset_readers / training switch behaviour on these substrings of the data path.
    for bad in ("synthetic", "DoF"):
        if bad in out:
            raise ValueError(f"Output path must not contain '{bad}' (dataset loaders special-case it): {out}")

    # ---- load the trained source scene ----
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=a.iteration, shuffle=False)
    train_cams = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
    test_cams = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    # PSF variance coefficient used for the simulation
    gaussians.aperture_coeff = SLIT_APERTURE_COEFF if a.aperture == "slit" else DISK_UNIFORM_COEFF

    tonemap_type = getattr(dataset, "tonemap_type", "none")
    if tonemap_type == "learn":
        print("[WARN] tonemap_type 'learn' is not saved with the model; simulating with an untrained tone map.")
    tone_map = ToneMapping(tonemap_type).cuda()
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    band_limit = str(getattr(conf, "band_limit", False)).lower() == "true"
    kernel_size = float(getattr(conf, "kernel_size", 0.1))

    if a.defocus_only:
        with open(os.path.join(out, "sim_gt.json")) as f:
            gt = json.load(f)
        write_defocus_maps(out, train_cams + test_cams, gaussians, pipe, gt["focus_distance"], gt["K_long"],
                           band_limit, kernel_size)
        return

    # ---- choose ground-truth optics ----
    inv = visible_inverse_depths(gaussians, train_cams, seed=a.seed)
    sub = inv[torch.randperm(inv.numel(), device="cuda")[:5_000_000]]
    inv_far, inv_near = torch.quantile(sub, 0.05).item(), torch.quantile(sub, 0.95).item()
    slit_gt = None
    if a.match_slit:
        with open(os.path.join(a.match_slit, "sim_gt.json")) as f:
            slit_gt = json.load(f)
        F = slit_gt["focus_distance"]
    else:
        F = a.focus_depth if a.focus_depth is not None else 1.0 / torch.quantile(sub, a.focus_quantile).item()
    d_max = max(abs(inv_near - 1.0 / F), abs(1.0 / F - inv_far))
    print(f"Scene depth (5-95%): {1/inv_near:.4f} .. {1/inv_far:.4f}, focus F = {F:.4f}")

    if a.aperture == "slit":
        K_L = a.max_blur_px / d_max      # full PSF length = K * |1/z - 1/F|
        K_W = a.width_ratio * K_L
        max_blur = a.max_blur_px
        print(f"Ground truth: K_L = {K_L:.2f}, K_W = {K_W:.2f} (ratio {a.width_ratio}), "
              f"max blur {max_blur:.1f} px x {max_blur * a.width_ratio:.2f} px")
        span = 360.0 if a.full_circle else 180.0
        angles = [k * span / a.n_angles for k in range(a.n_angles)]
    else:
        if slit_gt is not None:
            K_L = 2.0 * math.sqrt(slit_gt["K_long"] * slit_gt["K_short"] / math.pi)
            if a.noise_std != slit_gt["noise_std"]:
                print(f"[WARN] noise_std {a.noise_std} differs from the slit simulation ({slit_gt['noise_std']}); "
                      f"an equal-area comparison should use the same noise level.")
        else:
            K_L = a.max_blur_px / d_max
        K_W = K_L
        max_blur = K_L * d_max
        print(f"Ground truth (disk): K_D = {K_L:.2f}, max blur diameter {max_blur:.2f} px"
              + (f"  [equal area to slit {slit_gt['K_long']:.1f} x {slit_gt['K_short']:.1f}]" if slit_gt else ""))
        angles = [0.0] * a.disk_copies
    if K_L < K_MIN_REPRESENTABLE:
        print(f"[WARN] K = {K_L:.1f} is below the smallest K the model can represent ({K_MIN_REPRESENTABLE:.0f}); "
              f"the model cannot fit it. Increase the blur (--max_blur_px, or a larger slit in --match_slit).")

    # ---- output folders ----
    if os.path.exists(out):
        shutil.rmtree(out)
    img_dir, aif_dir, sp_dir = (os.path.join(out, d) for d in ("images", "aif", "sparse/0"))
    for d in (img_dir, aif_dir, sp_dir):
        os.makedirs(d)

    src_sparse = os.path.join(dataset.source_path, "sparse/0")
    extr_by_stem, intr = read_colmap(src_sparse)

    def render_img(cam, K, Kw, theta):
        pkg = render(cam, gaussians, pipe, bg, p11(F if K > 0 else 1.0), p11(K), select_view=0,
                     tone_map=tone_map, band_limit=band_limit, kernel_size=kernel_size,
                     K_w=p11(Kw), slit_theta=p11(theta))
        return pkg["render"]

    def save(img, path, noisy):
        if noisy and a.noise_std > 0:
            img = img + a.noise_std * torch.randn_like(img)
        torchvision.utils.save_image(img.clamp(0.0, 1.0), path)

    image_lines, slit_angles, test_names, used_cams = [], {}, [], set()

    def add_image(name, stem):
        e = extr_by_stem[stem]
        used_cams.add(e.camera_id)
        q, t = e.qvec, e.tvec
        image_lines.append(f"{len(image_lines) + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} "
                           f"{e.camera_id} {name}.png\n\n")

    with torch.no_grad():
        for cam in train_cams:
            stem = cam.image_name
            save(render_img(cam, 0.0, 0.0, 0.0), os.path.join(aif_dir, f"{stem}.png"), noisy=False)
            for k, ang in enumerate(angles):
                name = f"{stem}_a{k:02d}"
                theta = math.radians(ang + a.theta_bias_deg)
                save(render_img(cam, K_L, K_W, theta), os.path.join(img_dir, f"{name}.png"), noisy=True)
                add_image(name, stem)
                slit_angles[name] = ang  # nominal angle, without the simulated bias
        for cam in test_cams:
            stem = cam.image_name
            save(render_img(cam, 0.0, 0.0, 0.0), os.path.join(img_dir, f"{stem}.png"), noisy=False)
            add_image(stem, stem)
            test_names.append(stem)

    write_defocus_maps(out, train_cams + test_cams, gaussians, pipe, F, K_L, band_limit, kernel_size)

    # ---- COLMAP model (text) ----
    with open(os.path.join(sp_dir, "cameras.txt"), "w") as f:
        for cid in sorted(used_cams):
            c = intr[cid]
            f.write(f"{cid} PINHOLE {c.width} {c.height} " + " ".join(str(v) for v in pinhole_params(c)) + "\n")
    with open(os.path.join(sp_dir, "images.txt"), "w") as f:
        f.writelines(image_lines)
    n_pts = 0
    for fn in ("points3D.bin", "points3D.txt", "points3D.ply"):
        if os.path.isfile(os.path.join(src_sparse, fn)):
            shutil.copy(os.path.join(src_sparse, fn), os.path.join(sp_dir, fn))
            n_pts += 1
    assert n_pts > 0, f"no points3D file found in {src_sparse}"

    with open(os.path.join(out, "slit_angles.json"), "w") as f:
        json.dump(slit_angles, f, indent=2)
    with open(os.path.join(out, "split.json"), "w") as f:
        json.dump({"test": test_names}, f, indent=2)
    gt = dict(aperture=a.aperture, source_scene=a.scene, source_iteration=a.iteration, focus_distance=F,
              K_long=K_L, K_short=K_W, width_ratio=K_W / K_L, angles_deg=angles,
              theta_bias_deg=a.theta_bias_deg if a.aperture == "slit" else 0.0,
              max_blur_px=max_blur, noise_std=a.noise_std, aperture_coeff=gaussians.aperture_coeff,
              aperture_area_K2=(K_L * K_W if a.aperture == "slit" else math.pi * K_L ** 2 / 4),
              matched_slit=a.match_slit, n_train_poses=len(train_cams), n_test_poses=len(test_cams))
    with open(os.path.join(out, "sim_gt.json"), "w") as f:
        json.dump(gt, f, indent=2)

    # ---- training config + scene registration ----
    new_conf = OmegaConf.load(a.config)
    new_conf.scene_name = [new_scene]
    new_conf.model_folder = "slit_sim"
    new_conf.resolution = 1          # images are already rendered at the training resolution
    new_conf.images = "images"
    new_conf.coc_max = float(max_blur)
    if a.aperture == "slit":
        new_conf.aperture_type = "slit"
        new_conf.slit_angle_file = ""    # -> <source_path>/slit_angles.json
        new_conf.slit_width_ratio = 0.2  # deliberately different from the ground truth
        new_conf.slit_share_F = True
        new_conf.slit_share_K = True
        new_conf.slit_learn_theta_offset = True
    else:
        new_conf.aperture_type = "disk"  # original DOF-GS model
        for k in [k for k in new_conf.keys() if str(k).startswith("slit_")]:
            new_conf.pop(k)
    cfg_path = os.path.join("config_files", f"config_{new_scene}.yaml")
    OmegaConf.save(new_conf, cfg_path)

    if new_scene not in scene_dict:
        with open("config_files/scene_info.yaml", "a") as f:
            f.write(f"\n{new_scene}:\n  source_path: {out}\n  holdout: 8\n")

    print(f"\nWrote {len(train_cams) * len(angles)} {a.aperture} images ({len(train_cams)} poses x {len(angles)}) "
          f"and {len(test_cams)} AIF test images to {out}")
    print(f"Config: {cfg_path}   (scene '{new_scene}' registered in config_files/scene_info.yaml)")
    print(f"Next:   python train.py --config {cfg_path}")


if __name__ == "__main__":
    main()
