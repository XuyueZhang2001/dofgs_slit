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

Usage (from the repository root, e.g. inside tmux):
    python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
        --scene real_cake --iteration 15000 --gpu 1                    # default design 4:0.4 + disk
    python tools/run_design_sweep.py --config ./config_files/config_real_cake.yaml \
        --scene real_cake --iteration 15000 --gpu 1 --designs 4:0.1 4:0.4 8:0.2 12:0.13
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
    p.add_argument("--disk_copies", type=int, default=1,
                   help="images per pose for the disk (use n when noise is added, equal number of captures)")
    p.add_argument("--noise_std", type=float, default=0.0)
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
    return p.parse_args()


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


def build_variant_config(base_cfg_path, a, images_per_pose, is_slit):
    """Write the training config of this variant; returns (config path, model_folder, final iteration)."""
    conf = OmegaConf.load(base_cfg_path)
    defaults = optimization_defaults()
    parts = []
    conf.slit_group_by_pose = a.train_mode == "per_pose"
    if a.train_mode == "per_pose":
        parts.append("perpose")

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
        out_path = base_cfg_path.replace(".yaml", f"__{'_'.join(parts)}.yaml")
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
    if a.noise_std > 0 and a.disk_copies == 1 and not a.no_disk:
        print("[WARN] noise is added but the disk uses 1 image per pose; use --disk_copies n for equal captures.")

    # (scene name, simulate arguments, images per pose)
    jobs = []
    for d in a.designs:
        n_str, r_str = d.split(":")
        n, r = int(n_str), float(r_str)
        name = f"slitsim_{a.scene}_n{n}_a{r:.2f}{tag}"
        jobs.append((name, ["--n_angles", str(n), "--width_ratio", str(r)], n))
    if not a.no_disk:
        name = f"disksim_{a.scene}" + (f"_c{a.disk_copies}" if a.disk_copies > 1 else "") + tag
        jobs.append((name, ["--aperture", "disk", "--disk_copies", str(a.disk_copies)], a.disk_copies))

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

        cfg, model_folder, final_iter = build_variant_config(base_cfg, a, per_pose, is_slit)
        model_path = os.path.join("./output/model_dofgs", model_folder, name)
        tag_run = f"{name}__{model_folder}"

        # ---- train ----
        ckpt = os.path.join(model_path, "point_cloud", f"iteration_{final_iter}", "point_cloud.ply")
        if "train" in a.steps and not os.path.isfile(ckpt):
            run([py, "train.py", "--config", cfg], os.path.join(log_dir, f"{tag_run}_train.log"), env, a.dry_run)
        elif os.path.isfile(ckpt):
            print(f"[skip] trained model {ckpt} exists")

        # ---- evaluate ----
        summary = os.path.join(model_path, "eval_slit_recovery", f"iteration_{final_iter}", "summary.json")
        if "eval" in a.steps and not os.path.isfile(summary):
            run([py, "tools/eval_slit_recovery.py", "--config", cfg, "--scene", name, "--iteration", str(final_iter)]
                + (["--save_images"] if a.save_images else []),
                os.path.join(log_dir, f"{tag_run}_eval.log"), env, a.dry_run)
        elif os.path.isfile(summary):
            print(f"[skip] evaluation {summary} exists")

    if not a.dry_run:
        run([py, "tools/collect_design_sweep.py", "--source_scene", a.scene],
            os.path.join("logs", "design_sweep", f"collect_{a.scene}.log"), {}, False)


if __name__ == "__main__":
    main()
