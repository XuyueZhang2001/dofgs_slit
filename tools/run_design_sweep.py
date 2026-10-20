"""
Run simulate -> train -> evaluate for a set of equal-area aperture designs
(n slit angles, width ratio a) plus the equal-area circular-aperture baseline,
then collect the results into one table.

All designs reuse the focus distance and aperture AREA of a reference
simulated dataset (--match_area, default ./data/slit_sim/slitsim_<scene>).
Steps that are already done (dataset, final checkpoint, evaluation summary)
are skipped, so an interrupted sweep can simply be restarted.

Training modes
  per_view (default)  one image per iteration, exactly as the original DOF-GS;
                      for n = 4 the budget equals the earlier experiments.
  per_pose            all images of one pose per iteration (slit_group_by_pose).
--scale_iterations    multiply the iteration schedule of slit designs by n / ref_n
                      (default ref_n = 4) so every image gets the same number of
                      updates as in an n = ref_n run (off by default; the disk
                      baseline always keeps the base schedule).

Sensor noise (off by default)
  --noise_std s       additive Gaussian noise with std s on every defocused training
                      capture (images in [0, 1], e.g. 0.01 = 1 % of full scale),
                      independent per image; all-in-focus references and test views
                      stay clean. Noisy runs get the suffix _noise<s>.
  --disk_copies       captures per pose of the disk baseline. Default: with noise,
                      the slit n (same number of captures as the slit, each copy with
                      its own noise); without noise, 1 (identical copies add nothing,
                      so the existing noise-free disk run is reused).

Exposure (only matters with noise; see tools/simulate_slit_dataset.py)
  --slit_exposure e   relative exposure time of every slit capture (default 1)
  --disk_exposure e   relative exposure time of every disk capture (default 1)
  --noise_scaling     shot (s / sqrt(e), default) or read (s / e) growth of the noise
  --equal_total_exposure
                      equal total exposure time per pose: every slit design is captured
                      n times at exposure 1 / n, the disk once at exposure 1
                      (default disk captures 1 in this mode). Exposure-reduced runs
                      get the suffix _e<e>_<scaling>.

Directional AiF loss (slit only; src/utils/directional_loss.py)
  --dir_loss w [w ...]
                      weight(s) of the directional all-in-focus loss. Default: the
                      default of OptimizationParams.slit_dir_loss. Every value is trained
                      and evaluated on the same simulated dataset; a weight > 0 adds the
                      suffix _dir<w> to the model_folder (e.g. slit_sim_dir0.1), and
                      0 reproduces the original model in the unsuffixed folder.

Usage (from the repository root, e.g. inside tmux):
    python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
        --scene real_cake --iteration 15000 --gpu 1                    # default design 4:0.4 + disk
    python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
        --scene real_cake --iteration 15000 --gpu 1 --designs 4:0.1 4:0.4 8:0.2 12:0.13
    python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
        --scene real_cake --iteration 15000 --gpu 1 --noise_std 0.01   # slit n=4 a=0.4 vs disk x4, noisy
    python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
        --scene real_cake --iteration 15000 --gpu 1 --noise_std 0.01 \
        --equal_total_exposure --noise_scaling read                     # slit 4 x 0.25 vs disk 1 x 1.0
    python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
        --scene real_cake --iteration 15000 --gpu 1 --no_disk --dir_loss 0.03 0.1 0.3
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from omegaconf import OmegaConf

# Iteration-valued schedule entries (OptimizationParams and train.py arguments).
SCHEDULE_SCALARS = [
    "iterations", "position_lr_max_steps", "densification_interval", "opacity_reset_interval",
    "densify_from_iter", "densify_until_iter", "iter_defocus_start", "infocus_start_iter",
    "densify_compatnes_interval", "compatnes_until_iter", "iter_stop_cam_gd", "maskNet_start_iter",
]
SCHEDULE_LISTS = ["test_iterations", "save_iterations", "add_point_iterations"]
TRAIN_PY_LIST_DEFAULTS = {"test_iterations": [1_000, 3_000, 7_000, 30_000],
                          "save_iterations": [1_000, 3_000, 7_000, 30_000],
                          "add_point_iterations": [2500]}


def parse_args():
    p = argparse.ArgumentParser(description="Equal-area aperture design sweep")
    p.add_argument("--config", required=True, help="config of the source DOF-GS scene (ground truth)")
    p.add_argument("--scene", required=True, help="source scene key, e.g. real_cake")
    p.add_argument("--iteration", type=int, required=True, help="iteration of the source DOF-GS model")
    p.add_argument("--match_area", default=None,
                   help="reference simulated dataset whose area / focus all designs share "
                        "(default ./data/slit_sim/slitsim_<scene>)")
    p.add_argument("--designs", nargs="+", default=["4:0.4"], help="designs as n:a, e.g. 4:0.4 8:0.2")
    p.add_argument("--no_disk", action="store_true", help="skip the equal-area disk baseline")
    p.add_argument("--disk_copies", type=int, default=None,
                   help="captures per pose for the disk baseline; default: the slit n when --noise_std > 0 "
                        "(equal number of captures), otherwise 1")
    p.add_argument("--noise_std", type=float, default=0.0,
                   help="std of additive Gaussian noise on the [0, 1] training captures "
                        "(default 0 = noise-free; e.g. 0.01 = 1%% of full scale)")
    p.add_argument("--slit_exposure", type=float, default=None,
                   help="relative exposure time of every slit capture (default 1)")
    p.add_argument("--disk_exposure", type=float, default=1.0,
                   help="relative exposure time of every disk capture (default 1)")
    p.add_argument("--noise_scaling", choices=["read", "shot"], default="shot",
                   help="noise growth for exposure e < 1: read -> s / e, shot -> s / sqrt(e)")
    p.add_argument("--equal_total_exposure", action="store_true",
                   help="slit: n captures at exposure 1/n; disk: 1 capture at exposure 1 (equal total time)")
    p.add_argument("--dir_loss", type=float, nargs="+", default=None,
                   help="weight(s) of the directional slit AiF loss; default: OptimizationParams default, "
                        "0 = original model")
    p.add_argument("--train_mode", choices=["per_view", "per_pose"], default="per_view")
    p.add_argument("--scale_iterations", action="store_true",
                   help="scale the iteration schedule by images-per-pose / ref_n (off by default)")
    p.add_argument("--ref_n", type=int, default=4)
    p.add_argument("--iterations", type=int, default=None, help="override the number of training iterations")
    p.add_argument("--gpu", default="0", help="value for CUDA_VISIBLE_DEVICES")
    p.add_argument("--steps", nargs="+", default=["simulate", "train", "eval"],
                   choices=["simulate", "train", "eval"])
    p.add_argument("--save_images", action="store_true", help="pass --save_images to the evaluation")
    p.add_argument("--dry_run", action="store_true", help="print the commands without running them")
    a = p.parse_args()
    if a.noise_std < 0:
        p.error("--noise_std must be >= 0")
    if a.disk_copies is not None and a.disk_copies < 1:
        p.error("--disk_copies must be >= 1")
    if a.equal_total_exposure and a.slit_exposure is not None:
        p.error("--equal_total_exposure sets the slit exposure to 1/n; do not pass --slit_exposure as well")
    for k in ("slit_exposure", "disk_exposure"):
        v = getattr(a, k)
        if v is not None and v <= 0:
            p.error(f"--{k} must be > 0")
    if a.noise_std == 0 and (a.equal_total_exposure or (a.slit_exposure or 1.0) != 1.0 or a.disk_exposure != 1.0):
        print("[WARN] exposure settings have no effect without noise (--noise_std 0); they are ignored.")
    return a


def slit_exposure_for(a, n):
    """Relative exposure of every capture of a slit design with n angles."""
    if a.noise_std == 0:
        return 1.0
    if a.equal_total_exposure:
        return 1.0 / n
    return a.slit_exposure if a.slit_exposure is not None else 1.0


def exposure_tag(a, e):
    """Name suffix of an exposure-reduced (or -increased) noisy run."""
    return f"_e{e:.4g}_{a.noise_scaling}" if (a.noise_std > 0 and e != 1.0) else ""


def resolve_disk_copies(a, slit_ns):
    """Captures per pose of the disk baseline (see the module docstring)."""
    if a.disk_copies is not None:
        if a.disk_copies > 1 and a.noise_std == 0:
            print(f"[WARN] {a.disk_copies} noise-free disk copies per pose are identical images.")
        return a.disk_copies
    if a.noise_std == 0 or a.equal_total_exposure:
        return 1
    ns = sorted(set(slit_ns))
    if len(ns) == 1:
        return ns[0]
    print(f"[INFO] designs use different n {ns}; the disk is captured ref_n = {a.ref_n} times per pose "
          f"(pass --disk_copies to choose another count).")
    return a.ref_n


# ---------------------------------------------------------------------------
def run(cmd, log_path, env_extra, dry_run):
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}" + (f"   > {log_path}" if log_path else ""))
    if dry_run:
        return
    env = dict(os.environ, PYTHONUNBUFFERED="1", **env_extra)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Forward raw bytes: text mode would turn tqdm's '\r' (redraw the same line)
    # into newlines and print every progress-bar refresh on its own line.
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        while True:
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
        proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with exit code {proc.returncode}; see {log_path}")


def optimization_defaults():
    from src.model_dofgs.arguments import OptimizationParams
    return vars(OptimizationParams(argparse.ArgumentParser()))


def build_variant_config(base_cfg_path, a, images_per_pose, is_slit, dir_loss=None):
    """Write the training config of this variant; returns (config path, model_folder, final iteration)."""
    conf = OmegaConf.load(base_cfg_path)
    defaults = optimization_defaults()
    parts = []
    conf.slit_group_by_pose = a.train_mode == "per_pose"
    if a.train_mode == "per_pose":
        parts.append("perpose")
    if is_slit and dir_loss is not None:
        # explicit weight in the config, so the run does not depend on the code default
        conf.slit_dir_loss = float(dir_loss)
        if dir_loss > 0:
            parts.append(f"dir{dir_loss:g}")

    base_iters = int(conf.get("iterations", defaults["iterations"]))
    scale = 1.0
    if a.iterations is not None:
        # explicit iteration count: stretch the whole schedule to it
        scale = a.iterations / base_iters
        parts.append(f"it{a.iterations}")
    elif a.scale_iterations and is_slit and images_per_pose != a.ref_n:
        # n = ref_n is identical to the unscaled run and shares its outputs
        scale = images_per_pose / a.ref_n
        parts.append(f"scaled{a.ref_n}")

    if scale != 1.0:
        for k in SCHEDULE_SCALARS:
            v = conf.get(k, defaults.get(k))
            if v is not None:
                conf[k] = max(1, int(round(float(v) * scale)))
        for k in SCHEDULE_LISTS:
            v = conf.get(k, TRAIN_PY_LIST_DEFAULTS[k])
            conf[k] = sorted({max(1, int(round(float(x) * scale))) for x in v})
    final_iter = int(conf.get("iterations", defaults["iterations"]))
    for k in ("test_iterations", "save_iterations"):   # make sure the final checkpoint is saved / tested
        v = list(conf.get(k, TRAIN_PY_LIST_DEFAULTS[k]))
        if final_iter not in v:
            conf[k] = sorted(v + [final_iter])

    conf.model_folder = "slit_sim" + ("_" + "_".join(parts) if parts else "")
    if not parts:
        out_path = base_cfg_path
    else:
        # e.g. config_slitsim_real_cake_n4_a0.40_dir0.1.yaml
        out_path = base_cfg_path.replace(".yaml", f"_{'_'.join(parts)}.yaml")
    OmegaConf.save(conf, out_path)
    return out_path, conf.model_folder, final_iter


def area_of(sim_dir):
    with open(os.path.join(sim_dir, "sim_gt.json")) as f:
        gt = json.load(f)
    return gt.get("aperture_area_K2") or gt["K_long"] * gt["K_short"]


def main():
    a = parse_args()
    py = sys.executable
    tag = f"_noise{a.noise_std:g}" if a.noise_std > 0 else ""
    ref = a.match_area or os.path.join("./data/slit_sim", f"slitsim_{a.scene}")
    if not os.path.isfile(os.path.join(ref, "sim_gt.json")):
        raise SystemExit(f"Reference dataset {ref} not found; pass --match_area <simulated dataset folder>.")
    ref_area = area_of(ref)
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(a.gpu)}
    dir_values = a.dir_loss if a.dir_loss is not None else [float(optimization_defaults().get("slit_dir_loss", 0.0))]
    if any(w < 0 for w in dir_values):
        raise SystemExit("--dir_loss weights must be >= 0")
    print(f"Directional AiF loss weight(s) for slit runs: {', '.join(f'{w:g}' for w in dir_values)}")

    # (scene name, simulate arguments, images per pose)
    jobs, slit_ns, slit_totals = [], [], []
    for d in a.designs:
        n_str, r_str = d.split(":")
        n, r = int(n_str), float(r_str)
        e = slit_exposure_for(a, n)
        slit_ns.append(n)
        slit_totals.append(n * e)
        name = f"slitsim_{a.scene}_n{n}_a{r:.2f}{tag}{exposure_tag(a, e)}"
        jobs.append((name, ["--n_angles", str(n), "--width_ratio", str(r), "--exposure", repr(e),
                            "--noise_scaling", a.noise_scaling], n))
        if a.noise_std > 0:
            print(f"slit n={n}, a={r:g}: {n} captures x exposure {e:.4g} (total {n * e:.4g} per pose)")
    if not a.no_disk:
        copies = resolve_disk_copies(a, slit_ns)
        e_d = a.disk_exposure if a.noise_std > 0 else 1.0
        if a.noise_std > 0:
            print(f"disk: {copies} capture(s) x exposure {e_d:.4g} (total {copies * e_d:.4g} per pose), "
                  f"noise std {a.noise_std:g}, {a.noise_scaling} scaling")
            if any(abs(t - copies * e_d) > 1e-6 for t in slit_totals):
                print(f"[WARN] the total exposure per pose differs between the slit designs "
                      f"({', '.join(f'{t:.4g}' for t in slit_totals)}) and the disk ({copies * e_d:.4g}).")
        name = f"disksim_{a.scene}" + (f"_c{copies}" if copies > 1 else "") + tag + exposure_tag(a, e_d)
        jobs.append((name, ["--aperture", "disk", "--disk_copies", str(copies), "--exposure", repr(e_d),
                            "--noise_scaling", a.noise_scaling], copies))

    for name, sim_args, per_pose in jobs:
        is_slit = name.startswith("slitsim_")
        print(f"\n================ {name} ================")
        data_dir = os.path.join("./data/slit_sim", name)
        base_cfg = os.path.join("config_files", f"config_{name}.yaml")
        log_dir = os.path.join("logs", "design_sweep")

        # ---- simulate ----
        have_data = os.path.isfile(os.path.join(data_dir, "sim_gt.json")) and os.path.isfile(base_cfg)
        if have_data and abs(area_of(data_dir) - ref_area) > 1e-3 * ref_area:
            raise SystemExit(f"{data_dir} exists but its aperture area differs from {ref}; "
                             f"delete it (and its outputs) or use another --match_area.")
        if "simulate" in a.steps and not have_data:
            run([py, "tools/simulate_slit_dataset.py", "--config", a.config, "--scene", a.scene,
                 "--iteration", str(a.iteration), "--match_area", ref, "--noise_std", str(a.noise_std),
                 "--new_scene", name] + sim_args,
                os.path.join(log_dir, f"{name}_simulate.log"), env, a.dry_run)
        elif have_data:
            print(f"[skip] dataset {data_dir} exists")
        if a.dry_run and not os.path.isfile(base_cfg):
            print("[dry run] config not generated yet; later steps shown with placeholders")
            continue

        # one training run per directional-loss weight for slit datasets, one run for the disk
        for dir_loss in (dir_values if is_slit else [None]):
            cfg, model_folder, final_iter = build_variant_config(base_cfg, a, per_pose, is_slit, dir_loss)
            model_path = os.path.join("./output/model_dofgs", model_folder, name)
            tag_run = f"{name}_{model_folder}"
            if dir_loss is not None:
                print(f"--- directional AiF loss weight {dir_loss:g} -> {model_folder}")

            # ---- train ----
            ckpt = os.path.join(model_path, "point_cloud", f"iteration_{final_iter}", "point_cloud.ply")
            if "train" in a.steps and not os.path.isfile(ckpt):
                run([py, "train.py", "--config", cfg], os.path.join(log_dir, f"{tag_run}_train.log"), env,
                    a.dry_run)
            elif os.path.isfile(ckpt):
                print(f"[skip] trained model {ckpt} exists")

            # ---- evaluate ----
            summary = os.path.join(model_path, "eval_slit_recovery", f"iteration_{final_iter}", "summary.json")
            if "eval" in a.steps and not os.path.isfile(summary):
                run([py, "tools/eval_slit_recovery.py", "--config", cfg, "--scene", name,
                     "--iteration", str(final_iter)] + (["--save_images"] if a.save_images else []),
                    os.path.join(log_dir, f"{tag_run}_eval.log"), env, a.dry_run)
            elif os.path.isfile(summary):
                print(f"[skip] evaluation {summary} exists")

    if not a.dry_run:
        run([py, "tools/collect_design_sweep.py", "--source_scene", a.scene],
            os.path.join("logs", "design_sweep", f"collect_{a.scene}.log"), {}, False)


if __name__ == "__main__":
    main()
