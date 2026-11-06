"""
Summarise the slit vs equal-area disk comparison over several source scenes
(produced by tools/run_all_scenes.sh / tools/run_design_sweep.py).

For every source scene it reads the latest evaluation summary
(<root>/<model_folder>/<scene>/eval_slit_recovery/iteration_*/summary.json) of
    slit (densmax+L_dir)  <new_folder>/slitsim_<scene>_<design><suffix>
                          (max_angle densification + directional AiF loss)
    slit (orig)           <orig_folder>/slitsim_<scene>_<design><suffix>  (original model; optional)
    disk                  <disk_folder>/disksim_<scene><suffix>
and reports, on the test split, the AiF PSNR / SSIM / LPIPS, the rendered depth
AbsRel / RMSE, and the point-cloud Chamfer distance and F-score (tau = 0.005 x bbox
diagonal). The Chamfer distance is given in % of the GT bounding-box diagonal so
that scenes of different scale can be averaged; depth RMSE stays in scene units
(scale dependent, so its mean over scenes is only indicative).

Default names follow tools/run_design_sweep.py for the given --coc_energy (default
1, energy-conserving CoC) and --psf (default gaussian), with per-pose focus:
    dataset suffix  _pf[_glayer|_box][_ec<lambda>]   (_ec only for --psf gaussian)
    new_folder      slit_sim_dir0.005[_ec<lambda>]_densmax
    orig_folder     slit_sim[_ec<lambda>]
    disk_folder     slit_sim[_ec<lambda>]
Every name can be given explicitly. --orig_folder none skips the original slit
model (also skipped when it is the same folder as --new_folder).

Outputs (default ./output/scene_summary/): scenes.md (mean over scenes, per-scene
tables, paired differences and win counts) and scenes.csv (one row per scene and
method). Means use only the scenes where all methods were evaluated. Warnings are
listed for runs whose settings differ from the expected ones (noise, coc_energy,
directional loss / densification).

Usage (from the repository root):
    python tools/summarize_scenes.py                                   # Gaussian PSF, coc_energy 1
    python tools/summarize_scenes.py --psf gaussian_layered --orig_folder none
    python tools/summarize_scenes.py --coc_energy 0                    # earlier runs (original CoC)
    python tools/summarize_scenes.py --coc_energy 0 --suffix ""        # earlier global-focus datasets
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

# method labels used in the report
NEW, ORIG, DISK = "slit (densmax+L_dir)", "slit (orig)", "disk"

REAL_SCENES = ["real_cake", "real_caps", "real_cisco", "real_coral", "real_cupcake",
               "real_cups", "real_daisy", "real_sausage", "real_seal", "real_tools"]

# dataset-name tag of the PSF model (as in run_design_sweep.py)
PSF_TAG = {"gaussian": "", "box": "_box", "gaussian_layered": "_glayer"}
DIR_LOSS_DEFAULT = 0.005   # OptimizationParams.slit_dir_loss

# (key, header, higher is better, decimals)
METRICS = [
    ("psnr", "PSNR", True, 2),
    ("ssim", "SSIM", True, 4),
    ("lpips", "LPIPS", False, 4),
    ("abs_rel", "depth AbsRel", False, 4),
    ("rmse", "depth RMSE*", False, 2),
    ("chamfer_pct", "Chamfer (% diag)", False, 3),
    ("fscore", "F@0.005", True, 4),
]


def default_names(coc_energy, psf="gaussian", focus_suffix="_pf", dir_loss=DIR_LOSS_DEFAULT):
    """Default dataset suffix and model folders of run_design_sweep.py for a coc_energy / PSF."""
    ec = f"_ec{coc_energy:g}" if coc_energy > 0 else ""
    suffix = focus_suffix + PSF_TAG[psf] + (ec if psf == "gaussian" else "")
    return dict(suffix=suffix,
                new_folder=f"slit_sim_dir{dir_loss:g}{ec}_densmax",
                orig_folder=f"slit_sim{ec}",
                disk_folder=f"slit_sim{ec}")


def parse_args():
    p = argparse.ArgumentParser(description="Slit vs disk summary over several scenes")
    p.add_argument("--scenes", nargs="+", default=REAL_SCENES, help="source scenes (default: the 10 real scenes)")
    p.add_argument("--design", default="n4_a0.40", help="slit design suffix of the dataset names")
    p.add_argument("--coc_energy", type=float, default=1.0,
                   help="energy-conserving CoC of the runs (default 1); sets the default names and is "
                        "checked against each run (0 = original CoC, the earlier runs)")
    p.add_argument("--psf", choices=sorted(PSF_TAG), default="gaussian",
                   help="PSF model of the simulated datasets (sets the default suffix)")
    p.add_argument("--suffix", default=None,
                   help="dataset-name suffix (default: _pf + PSF tag [+ _ec<coc_energy> for gaussian]); "
                        "'' = earlier global-focus datasets")
    p.add_argument("--root", default="./output/model_dofgs")
    p.add_argument("--new_folder", "--ours_folder", dest="new_folder", default=None,
                   help="model_folder of the slit model with max_angle densification + directional AiF loss")
    p.add_argument("--orig_folder", default=None,
                   help="model_folder of the original slit model; 'none' skips it")
    p.add_argument("--disk_folder", default=None, help="model_folder of the disk model")
    p.add_argument("--out", default="./output/scene_summary")
    a = p.parse_args()
    d = default_names(a.coc_energy, a.psf)
    for k in ("suffix", "new_folder", "orig_folder", "disk_folder"):
        if getattr(a, k) is None:
            setattr(a, k, d[k])
    if str(a.orig_folder).lower() == "none" or a.orig_folder == a.new_folder:
        if a.orig_folder == a.new_folder:
            print(f"[INFO] --orig_folder is the same as --new_folder ({a.new_folder}): slit (orig) skipped")
        a.orig_folder = None
    return a


def latest_summary(model_path):
    best = None
    for path in glob.glob(os.path.join(model_path, "eval_slit_recovery", "iteration_*", "summary.json")):
        with open(path) as f:
            s = json.load(f)
        if best is None or s["iteration"] > best["iteration"]:
            best = s
    return best


def runtime_config(model_path):
    try:
        import yaml
        with open(os.path.join(model_path, "config_runtime.yaml")) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def extract(s):
    """Metrics of one evaluation summary (None where missing)."""
    summ = s.get("summary", {})
    test = summ.get("test", {})
    rec, depth, geo = test.get("recovered", {}), test.get("depth", {}), summ.get("geometry") or {}
    chamfer, diag = geo.get("chamfer"), geo.get("bbox_diag")
    return dict(
        psnr=rec.get("psnr"), ssim=rec.get("ssim"),
        lpips=next((v for k, v in rec.items() if k.startswith("lpips")), None),
        abs_rel=depth.get("abs_rel"), rmse=depth.get("rmse"),
        chamfer_pct=(100.0 * chamfer / diag if chamfer is not None and diag else None),
        fscore=(geo.get("tau0.005") or {}).get("fscore"),
        input_psnr=summ.get("train", {}).get("input", {}).get("psnr"),
        n_gaussians=geo.get("n_rec"), iteration=s.get("iteration"),
    )


def check_variant(name, cfg, sim_gt, coc_energy):
    """Warnings if a run does not have the expected settings."""
    w = []
    if sim_gt.get("noise_std", 0.0):
        w.append(f"{name}: dataset has noise (std {sim_gt['noise_std']:g})")
    # runs trained before the option have no key: original CoC (0)
    run_ec = float(cfg.get("coc_energy", 0.0) or 0.0)
    if abs(run_ec - coc_energy) > 1e-9:
        w.append(f"{name}: trained with coc_energy {run_ec:g}, expected {coc_energy:g}")
    if name == NEW and (float(cfg.get("slit_dir_loss", 0.0)) <= 0
                        or cfg.get("slit_densify_stat", "mean") != "max_angle"):
        w.append(f"{name}: expected slit_dir_loss > 0 and slit_densify_stat max_angle")
    if name == ORIG and (float(cfg.get("slit_dir_loss", 0.0)) > 0
                         or cfg.get("slit_densify_stat", "mean") != "mean"):
        w.append(f"{name}: expected slit_dir_loss 0 and slit_densify_stat mean")
    if cfg.get("optics_from_sim_gt", "none") != "none":
        w.append(f"{name}: optics_from_sim_gt is {cfg.get('optics_from_sim_gt')}")
    return w


def fmt(v, nd):
    return "–" if v is None else f"{v:.{nd}f}"


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return m, sd


def main():
    a = parse_args()
    methods = [(NEW, a.new_folder, "slitsim_{s}_" + a.design + a.suffix)]
    if a.orig_folder:
        methods.append((ORIG, a.orig_folder, "slitsim_{s}_" + a.design + a.suffix))
    methods.append((DISK, a.disk_folder, "disksim_{s}" + a.suffix))
    names = [m[0] for m in methods]
    has_orig = ORIG in names

    data, warnings, missing = {}, [], []
    for s in a.scenes:
        for name, folder, pattern in methods:
            model_path = os.path.join(a.root, folder, pattern.format(s=s))
            summ = latest_summary(model_path)
            if summ is None:
                missing.append(f"{s}: {name} ({model_path})")
                continue
            data[(s, name)] = extract(summ)
            warnings += [f"{s} {w}" for w in check_variant(name, runtime_config(model_path),
                                                            summ.get("sim_gt", {}), a.coc_energy)]

    complete = [s for s in a.scenes if all((s, n) in data for n in names)]
    if not data:
        sys.exit(f"No evaluation summaries found under {a.root} for scenes {a.scenes} "
                 f"(suffix '{a.suffix}', folders {', '.join(m[1] for m in methods)})")

    folders = ", ".join(f"{n}: `{f}`" for n, f, _ in methods)
    lines = [f"# Slit vs equal-area disk over {len(complete)} scene(s)", "",
             f"Scenes with all methods: {', '.join(complete) or 'none'}  ",
             f"Test split; slit n = 4, a = 0.4 (`{a.design}`); PSF `{a.psf}`; coc_energy {a.coc_energy:g}; "
             f"dataset suffix `{a.suffix}`.  ",
             f"Model folders: {folders}.  ",
             "*Depth RMSE is in scene units (scale dependent); Chamfer is in % of the GT bbox diagonal.", ""]

    # ---- mean over the complete scenes ----
    lines += ["## Mean ± std over scenes", "",
              "| method | " + " | ".join(h + (" ↑" if hb else " ↓") for _, h, hb, _ in METRICS) + " |",
              "|---" * (len(METRICS) + 1) + "|"]
    for n in names:
        cells = []
        for k, _, _, nd in METRICS:
            m, sd = mean_std([data[(s, n)][k] for s in complete])
            cells.append("–" if m is None else f"{m:.{nd}f} ± {sd:.{nd}f}")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- paired differences and win counts ----
    pairs = [(NEW, DISK)] + ([(ORIG, DISK), (NEW, ORIG)] if has_orig else [])
    lines += ["## Paired comparison over scenes", "",
              "Mean difference (first − second) and the number of scenes where the first is better.", "",
              "| comparison | " + " | ".join(h for _, h, _, _ in METRICS) + " |",
              "|---" * (len(METRICS) + 1) + "|"]
    for x, y in pairs:
        cells = []
        for k, _, hb, nd in METRICS:
            diffs = [(data[(s, x)][k], data[(s, y)][k]) for s in complete]
            diffs = [(u, v) for u, v in diffs if u is not None and v is not None]
            if not diffs:
                cells.append("–")
                continue
            d = sum(u - v for u, v in diffs) / len(diffs)
            wins = sum((u > v) if hb else (u < v) for u, v in diffs)
            cells.append(f"{d:+.{nd}f} ({wins}/{len(diffs)})")
        lines.append(f"| {x} vs {y} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- per-scene tables ----
    for n in names:
        lines += [f"## {n}", "",
                  "| scene | " + " | ".join(h for _, h, _, _ in METRICS) + " | input PSNR | Gaussians | iter |",
                  "|---" * (len(METRICS) + 4) + "|"]
        for s in a.scenes:
            r = data.get((s, n))
            if r is None:
                lines.append(f"| {s} | " + " | ".join("–" for _ in METRICS) + " | – | – | – |")
                continue
            lines.append(f"| {s} | " + " | ".join(fmt(r[k], nd) for k, _, _, nd in METRICS)
                         + f" | {fmt(r['input_psnr'], 2)} | {r['n_gaussians'] or '–'} | {r['iteration']} |")
        lines.append("")

    # ---- PSNR side by side ----
    diff_cols = [(NEW, DISK, "densmax+L_dir − disk")]
    if has_orig:
        diff_cols += [(ORIG, DISK, "orig − disk"), (NEW, ORIG, "densmax+L_dir − orig")]
    lines += ["## Test PSNR per scene", "",
              "| scene | " + " | ".join(names) + " | " + " | ".join(h for _, _, h in diff_cols) + " |",
              "|---" * (len(names) + len(diff_cols) + 1) + "|"]
    for s in a.scenes:
        v = {n: (data.get((s, n)) or {}).get("psnr") for n in names}
        d = lambda x, y: fmt(v[x] - v[y], 2) if v[x] is not None and v[y] is not None else "–"
        lines.append(f"| {s} | " + " | ".join(fmt(v[n], 2) for n in names)
                     + " | " + " | ".join(d(x, y) for x, y, _ in diff_cols) + " |")
    lines.append("")

    if missing:
        lines += ["## Missing runs", ""] + [f"- {m}" for m in missing] + [""]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]

    md = "\n".join(lines)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "scenes.md"), "w") as f:
        f.write(md + "\n")
    keys = [k for k, _, _, _ in METRICS] + ["input_psnr", "n_gaussians", "iteration"]
    with open(os.path.join(a.out, "scenes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "method"] + keys)
        for s in a.scenes:
            for n in names:
                if (s, n) in data:
                    w.writerow([s, n] + [data[(s, n)][k] for k in keys])
    print(md)
    print(f"\nSaved {os.path.join(a.out, 'scenes.md')} and scenes.csv")


if __name__ == "__main__":
    main()
