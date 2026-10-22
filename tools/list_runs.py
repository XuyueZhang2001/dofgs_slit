"""
List the evaluation results of every run (image, depth and point-cloud metrics)
in one table, read from
    <root>/<model_folder>/<scene>/eval_slit_recovery/iteration_*/summary.json
(written by tools/eval_slit_recovery.py). Only the latest evaluated iteration
of each run is shown.

Usage (from the repository root):
    python tools/list_runs.py                       # all runs
    python tools/list_runs.py --filter slitsim_real_cake_n4_a0.40
    python tools/list_runs.py --csv runs.csv        # also save as CSV
"""
import argparse
import csv
import glob
import json
import os

COLUMNS = [  # (header, key)
    ("model_folder", "model_folder"), ("scene", "scene"), ("iter", "iteration"), ("λ_dir", "dir_loss"),
    ("noise", "noise"), ("PSNR", "psnr"), ("SSIM", "ssim"), ("LPIPS", "lpips"),
    ("abs_rel", "abs_rel"), ("rmse", "rmse"), ("delta1", "delta1"),
    ("acc", "accuracy"), ("comp", "completeness"), ("chamfer", "chamfer"),
    ("acc_med", "accuracy_median"), ("comp_med", "completeness_median"),
    ("F@.002", "f0.002"), ("F@.005", "f0.005"), ("F@.01", "f0.01"), ("#pts", "n_rec"),
]


def runtime_dir_loss(model_path):
    """slit_dir_loss from config_runtime.yaml (0 for runs trained before it existed)."""
    path = os.path.join(model_path, "config_runtime.yaml")
    try:
        import yaml
        with open(path) as f:
            return float((yaml.safe_load(f) or {}).get("slit_dir_loss", 0.0))
    except Exception:
        return None


def load(root, flt):
    latest = {}
    for path in glob.glob(os.path.join(root, "*", "*", "eval_slit_recovery", "iteration_*", "summary.json")):
        model_path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
        key = (os.path.basename(os.path.dirname(model_path)), os.path.basename(model_path))
        if flt and not any(flt in k for k in key):
            continue
        with open(path) as f:
            s = json.load(f)
        if key in latest and latest[key]["iteration"] >= s["iteration"]:
            continue
        gt, summ = s.get("sim_gt", {}), s.get("summary", {})
        test = summ.get("test", {})
        rec, depth, geo = test.get("recovered", {}), test.get("depth", {}), summ.get("geometry", {})
        aperture = gt.get("aperture", "slit")
        row = dict(model_folder=key[0], scene=key[1], iteration=s["iteration"],
                   dir_loss=runtime_dir_loss(model_path) if aperture == "slit" else None,
                   noise=gt.get("noise_std"), psnr=rec.get("psnr"), ssim=rec.get("ssim"),
                   lpips=next((v for k, v in rec.items() if k.startswith("lpips")), None),
                   abs_rel=depth.get("abs_rel"), rmse=depth.get("rmse"), delta1=depth.get("delta1"))
        for k in ("accuracy", "completeness", "chamfer", "accuracy_median", "completeness_median", "n_rec"):
            row[k] = geo.get(k)
        for t in ("0.002", "0.005", "0.01"):
            row[f"f{t}"] = geo.get(f"tau{t}", {}).get("fscore")
        latest[key] = row
    return sorted(latest.values(), key=lambda r: (r["scene"], r["model_folder"]))


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def main():
    p = argparse.ArgumentParser(description="Table of all evaluated runs")
    p.add_argument("--root", default="./output/model_dofgs")
    p.add_argument("--filter", default="", help="only runs whose model_folder or scene contains this text")
    p.add_argument("--csv", default="", help="also write the table to this CSV file")
    a = p.parse_args()

    rows = load(a.root, a.filter)
    if not rows:
        raise SystemExit(f"No summary.json found under {a.root}")
    cells = [[h for h, _ in COLUMNS]] + [[fmt(r[k]) for _, k in COLUMNS] for r in rows]
    widths = [max(len(c[i]) for c in cells) for i in range(len(COLUMNS))]
    for j, c in enumerate(cells):
        print("  ".join(x.ljust(w) if i < 2 else x.rjust(w) for i, (x, w) in enumerate(zip(c, widths))))
        if j == 0:
            print("  ".join("-" * w for w in widths))
    print("\nTest split. abs_rel/rmse/delta1: rendered depth; acc/comp/chamfer: point-cloud distances "
          "(lower is better); F@tau: point-cloud F-score (higher is better).")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[k for _, k in COLUMNS])
            w.writeheader()
            w.writerows({k: r[k] for _, k in COLUMNS} for r in rows)
        print(f"Saved {a.csv}")


if __name__ == "__main__":
    main()
