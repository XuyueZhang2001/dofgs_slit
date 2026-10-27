"""
Collect the evaluation summaries of all simulated aperture designs of one
source scene into a single table, and plot the measured AIF gain over the
equal-area disk against the theoretical depth-of-field gain.

It scans <root>/<model_folder>/<scene>/eval_slit_recovery/iteration_*/summary.json
(written by tools/eval_slit_recovery.py), keeps the latest iteration of every run
and compares each run with the disk baseline of the same source scene and noise
setting and the same PSF model of the simulation (preferring the disk trained
in the same model_folder). With noise, the
disk must also have the same total exposure per pose (captures x exposure) and,
when its exposure is reduced, the same noise scaling; without noise, exposure
does not matter.

Usage (from the repository root):
    python tools/collect_design_sweep.py --source_scene real_cake
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aperture_theory import summarize as theory_summary

# dataviz reference palette (light surface); text stays in neutral ink
SERIES = ["#2a78d6", "#eb6834"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def parse_args():
    p = argparse.ArgumentParser(description="Collect equal-area aperture design results")
    p.add_argument("--source_scene", required=True, help="source DOF-GS scene the datasets were simulated from")
    p.add_argument("--root", default="./output/model_dofgs")
    p.add_argument("--out", default=None, help="default ./output/design_sweep/<source_scene>")
    p.add_argument("--all_iterations", action="store_true", help="keep every evaluated iteration, not only the latest")
    return p.parse_args()


def load_runtime_config(model_path):
    path = os.path.join(model_path, "config_runtime.yaml")
    if not os.path.isfile(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_rows(a):
    rows = []
    for path in glob.glob(os.path.join(a.root, "*", "*", "eval_slit_recovery", "iteration_*", "summary.json")):
        with open(path) as f:
            s = json.load(f)
        gt = s.get("sim_gt", {})
        if gt.get("source_scene") != a.source_scene:
            continue
        model_path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
        model_folder = os.path.basename(os.path.dirname(model_path))
        aperture = gt.get("aperture", "slit")
        cfg = load_runtime_config(model_path)
        # runs trained before the directional loss existed have no key: weight 0
        dir_loss = float(cfg.get("slit_dir_loss", 0.0)) if aperture == "slit" else None
        design = s.get("design")
        if design is None:   # summaries written before designs were stored
            design = dict(aperture=aperture, n_angles=len(set(gt["angles_deg"])), width_ratio=gt["width_ratio"],
                          images_per_pose=len(gt["angles_deg"]),
                          group_by_pose=bool(cfg.get("slit_group_by_pose", False)),
                          iterations=int(cfg.get("iterations", s["iteration"])), model_folder=model_folder)
        theory = s.get("theory")
        if theory is None:
            theory = (theory_summary(gt["width_ratio"], gt["angles_deg"]) if aperture == "slit"
                      else dict(worst_var_ratio=1.0, dof_gain_worst=1.0))
        summ = s["summary"]
        test = summ.get("test", {}).get("recovered", {})
        train = summ.get("train", {})
        rows.append(dict(
            model_folder=model_folder, scene=s["scene"], iteration=s["iteration"], aperture=aperture,
            n=design["n_angles"] if aperture == "slit" else "-",
            a=round(design["width_ratio"], 3) if aperture == "slit" else "-",
            images_per_pose=design["images_per_pose"],
            train_mode="per_pose" if design.get("group_by_pose") else "per_view",
            dir_loss=dir_loss,
            psf=psf_label(gt),
            noise=gt.get("noise_std", 0.0),
            # datasets simulated before exposure control used exposure 1 for every capture
            exposure=gt.get("exposure", 1.0),
            noise_scaling=gt.get("noise_scaling", "shot"),
            noise_eff=gt.get("noise_std_effective", gt.get("noise_std", 0.0)),
            total_exposure=gt.get("total_exposure_per_pose",
                                  gt.get("exposure", 1.0) * design["images_per_pose"]),
            test_psnr=test.get("psnr"), test_ssim=test.get("ssim"),
            test_lpips=next((v for k, v in test.items() if k.startswith("lpips")), None),
            train_psnr=train.get("recovered", {}).get("psnr"),
            input_psnr=train.get("input", {}).get("psnr"),
            theory_dof_gain=theory.get("dof_gain_worst"), theory_worst_var=theory.get("worst_var_ratio"),
            summary_path=path,
        ))
    if not a.all_iterations:
        latest = {}
        for r in rows:
            key = (r["model_folder"], r["scene"])
            if key not in latest or r["iteration"] > latest[key]["iteration"]:
                latest[key] = r
        rows = list(latest.values())
    return rows


def psf_label(gt):
    """PSF model of a simulated dataset: gaussian (default), box, gaussian_layered, '+diff' with diffraction."""
    label = gt.get("psf", "gaussian")
    if (gt.get("diffraction") or {}).get("enabled"):
        label += "+diff"
    return label


def same_capture_budget(d, r):
    """True if disk run d is a fair reference for run r (same PSF model, noise setting and total exposure)."""
    if d["psf"] != r["psf"]:
        return False
    if d["noise"] != r["noise"]:
        return False
    if r["noise"] == 0:
        return True   # without noise, exposure and the number of captures do not change the images
    if abs(d["total_exposure"] - r["total_exposure"]) > 1e-6:
        return False
    # the noise scaling only matters for a disk whose exposure is not 1
    return d["exposure"] == 1.0 or d["noise_scaling"] == r["noise_scaling"]


def base_model_folder(folder):
    """model_folder without the directional-loss suffix (slit_sim_dir0.1 -> slit_sim)."""
    return re.sub(r"_dir[0-9.eE+-]+", "", folder)


def add_disk_reference(rows):
    disks = [r for r in rows if r["aperture"] == "disk"]
    for r in rows:
        cands = [d for d in disks if same_capture_budget(d, r)]
        # prefer the disk of the same folder, then of the same folder without the _dir suffix
        same = [d for d in cands if d["model_folder"] == r["model_folder"]]
        base = [d for d in cands if d["model_folder"] == base_model_folder(r["model_folder"])]
        ref = (same or base or cands or [None])[0]
        r["disk_ref"] = f"{ref['model_folder']}/{ref['scene']}" if ref else ""
        ok = ref is not None and r["test_psnr"] is not None and ref["test_psnr"] is not None
        r["delta_test_psnr_vs_disk"] = (r["test_psnr"] - ref["test_psnr"]) if ok else None
        r["gain_over_input"] = ((r["train_psnr"] - r["input_psnr"])
                                if r["train_psnr"] is not None and r["input_psnr"] is not None else None)


def fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_table(rows, out_dir):
    for r in rows:
        r["captures"] = f"{r['images_per_pose']} x {r['exposure']:.4g}"
    cols = [("run", None), ("aperture", "aperture"), ("n", "n"), ("a", "a"), ("train mode", "train_mode"),
            ("λ_dir", "dir_loss"), ("PSF", "psf"), ("iter", "iteration"), ("noise σ", "noise"), ("captures/pose x exposure", "captures"),
            ("total exposure", "total_exposure"), ("σ_eff", "noise_eff"), ("test PSNR", "test_psnr"), ("test SSIM", "test_ssim"),
            ("test LPIPS", "test_lpips"), ("gain over input (dB)", "gain_over_input"),
            ("Δ PSNR vs disk (dB)", "delta_test_psnr_vs_disk"), ("theory DOF gain", "theory_dof_gain"),
            ("theory worst σ²/disk", "theory_worst_var")]
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        vals = []
        for c, k in cols:
            if k is None:
                vals.append(f"{r['model_folder']}/{r['scene']}")
            else:
                vals.append(fmt(r[k], 2 if "psnr" in k or "gain" in k or "delta" in k else 3))
        lines.append("| " + " | ".join(vals) + " |")
    md = "\n".join(lines)
    with open(os.path.join(out_dir, "design_sweep.md"), "w") as f:
        f.write(md + "\n")
    keys = [k for k in rows[0].keys()]
    with open(os.path.join(out_dir, "design_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return md


def plot(rows, out_dir, source_scene):
    pts = [r for r in rows if r["delta_test_psnr_vs_disk"] is not None]
    if not pts:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)

    ax.axhline(0.0, color=MUTED, linewidth=1.2, linestyle="--")
    ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
    modes = sorted({r["train_mode"] for r in pts if r["aperture"] == "slit"})
    for i, mode in enumerate(modes):
        sel = [r for r in pts if r["aperture"] == "slit" and r["train_mode"] == mode]
        ax.scatter([r["theory_dof_gain"] for r in sel], [r["delta_test_psnr_vs_disk"] for r in sel],
                   s=64, color=SERIES[i % 2], edgecolors="white", linewidths=2, zorder=3,
                   label=f"slit ({mode.replace('_', '-')})")
        for r in sel:
            label = f"n={r['n']}, a={r['a']:g}"
            if r["noise"]:
                label += f", σ={r['noise']:g}"
            if r["exposure"] != 1.0:
                label += f", e={r['exposure']:.3g} ({r['noise_scaling']})"
            if r["dir_loss"]:
                label += f", λ_dir={r['dir_loss']:g}"
            if r["psf"] != "gaussian":
                label += f", {r['psf']}"
            ax.annotate(label, (r["theory_dof_gain"], r["delta_test_psnr_vs_disk"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=8.5, color=INK)
    disk = [r for r in pts if r["aperture"] == "disk"]
    if disk:
        ax.scatter([1.0], [0.0], s=64, color=MUTED, edgecolors="white", linewidths=2, zorder=3,
                   label="equal-area disk")
    ax.margins(x=0.15, y=0.15)
    ax.set_xlabel("theoretical depth-of-field gain vs disk (worst direction)", color=INK)
    ax.set_ylabel("measured test PSNR − disk (dB)", color=INK)
    ax.set_title(f"Measured vs predicted, {source_scene}", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "measured_vs_theory.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    a = parse_args()
    out_dir = a.out or os.path.join("./output/design_sweep", a.source_scene)
    os.makedirs(out_dir, exist_ok=True)
    rows = load_rows(a)
    if not rows:
        raise SystemExit(f"No evaluation summaries found for source scene '{a.source_scene}' under {a.root}. "
                         f"Run tools/eval_slit_recovery.py first.")
    add_disk_reference(rows)
    rows.sort(key=lambda r: (r["aperture"] != "disk", r["train_mode"],
                             r["n"] if isinstance(r["n"], int) else 0, r["a"] if isinstance(r["a"], float) else 0))
    md = write_table(rows, out_dir)
    print(md)
    try:
        p = plot(rows, out_dir, a.source_scene)
        if p:
            print(f"\nSaved plot {p}")
    except ImportError:
        print("[INFO] matplotlib not available, plot skipped")
    print(f"Saved table to {os.path.join(out_dir, 'design_sweep.md')} and design_sweep.csv")


if __name__ == "__main__":
    main()
