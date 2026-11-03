"""
Diagnose why the slit advantage over the equal-area disk differs between scenes
(e.g. real_sausage: no gain; real_cake: an unusually large gain).

For every scene it combines
  (1) dataset statistics, read from the simulated datasets (no GPU needed):
      - relative depth range d_max * F (max |F/z - 1| at the reference focus)
      - per-pixel blur of the training poses, from <dataset>/defocus/<pose>.npy
        (disk: blur-disk diameter in px; slit: PSF length along the long axis in px),
        pooled over all training poses: median / p90, fraction of pixels above
        --blur_thresh px, and the background fraction (pixels without geometry)
      - image detail of the AiF ground truth (<dataset>/aif/<pose>.png): mean
        gradient magnitude, and the "blurred detail" = the share of the gradient
        energy that lies in pixels whose disk blur exceeds --blur_thresh px
        (detail the disk destroys and the slit could keep)
  (2) evaluation results of the three models (latest eval_slit_recovery summary):
      - test PSNR (novel views) and its per-view spread
      - train-pose PSNR of the recovered AiF and of the blurred input; the train
        Δ(slit - disk) measures deblurring, the test Δ also includes novel-view
        generalisation
  (3) within-scene check: per training pose, Δ PSNR of the recovered AiF
      (slit - disk) against the mean disk blur of that pose (Spearman rho, and
      the mean Δ in the lowest / highest blur third of the poses)
  (4) across scenes: Spearman rho between the test Δ PSNR and each statistic
      (10 scenes: |rho| > ~0.65 is needed for p < 0.05, so treat it as a hint).

Output: printed and saved as <out>/diagnose.md (+ diagnose.csv).

Usage (from the repository root):
    python tools/diagnose_scenes.py
    python tools/diagnose_scenes.py --scenes real_sausage real_cake real_cisco --blur_thresh 4
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

NEW, ORIG, DISK = "slit (densmax+L_dir)", "slit (orig)", "disk"
REAL_SCENES = ["real_cake", "real_caps", "real_cisco", "real_coral", "real_cupcake",
               "real_cups", "real_daisy", "real_sausage", "real_seal", "real_tools"]


def parse_args():
    p = argparse.ArgumentParser(description="Per-scene diagnosis of the slit vs disk gain")
    p.add_argument("--scenes", nargs="+", default=REAL_SCENES)
    p.add_argument("--design", default="n4_a0.40")
    p.add_argument("--suffix", default="_pf", help="dataset suffix (_pf = per-pose focus, '' = global focus)")
    p.add_argument("--data_root", default="./data/slit_sim")
    p.add_argument("--root", default="./output/model_dofgs")
    p.add_argument("--new_folder", default="slit_sim_dir0.03_densmax")
    p.add_argument("--orig_folder", default="slit_sim")
    p.add_argument("--disk_folder", default="slit_sim")
    p.add_argument("--blur_thresh", type=float, default=4.0,
                   help="disk blur diameter (px) above which a pixel counts as strongly blurred")
    p.add_argument("--max_poses", type=int, default=0, help="use at most this many training poses (0 = all)")
    p.add_argument("--out", default="./output/scene_summary")
    return p.parse_args()


# ---------------------------------------------------------------- helpers
def rankdata(x):
    """Ranks with ties averaged (as scipy.stats.rankdata)."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f"{v:.{nd}f}"


def latest_eval_dir(model_path):
    best, best_it = None, -1
    for path in glob.glob(os.path.join(model_path, "eval_slit_recovery", "iteration_*", "summary.json")):
        with open(path) as f:
            it = json.load(f).get("iteration", -1)
        if it > best_it:
            best, best_it = os.path.dirname(path), it
    return best


def load_eval(model_path):
    """Summary metrics and per-pose recovered PSNR of one run (None if not evaluated)."""
    d = latest_eval_dir(model_path)
    if d is None:
        return None
    with open(os.path.join(d, "summary.json")) as f:
        s = json.load(f)
    summ = s.get("summary", {})
    tr, te = summ.get("train", {}), summ.get("test", {})
    train_psnr, test_psnr = {}, []
    pv = os.path.join(d, "per_view.csv")
    if os.path.isfile(pv):
        with open(pv) as f:
            for r in csv.DictReader(f):
                if r.get("method") != "recovered" or not r.get("psnr"):
                    continue
                if r["split"] == "train":
                    train_psnr[r["pose"]] = float(r["psnr"])
                elif r["split"] == "test":
                    test_psnr.append(float(r["psnr"]))
    return dict(
        test_psnr=te.get("recovered", {}).get("psnr"),
        test_min=min(test_psnr) if test_psnr else None,
        test_std=float(np.std(test_psnr)) if len(test_psnr) > 1 else None,
        train_rec=tr.get("recovered", {}).get("psnr"),
        train_in=tr.get("input", {}).get("psnr"),
        abs_rel=te.get("depth", {}).get("abs_rel"),
        n_gauss=(summ.get("geometry") or {}).get("n_rec"),
        per_pose=train_psnr,
    )


def gray(path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def grad_mag(g):
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, :-1] = g[:, 1:] - g[:, :-1]
    gy[:-1, :] = g[1:, :] - g[:-1, :]
    return np.sqrt(gx ** 2 + gy ** 2)


def dataset_stats(ds_dir, blur_thresh, max_poses):
    """Blur and detail statistics of the training poses of one simulated dataset."""
    gt_path = os.path.join(ds_dir, "sim_gt.json")
    if not os.path.isfile(gt_path):
        return None
    with open(gt_path) as f:
        gt = json.load(f)
    poses = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ds_dir, "aif", "*.png")))
    if max_poses > 0:
        poses = poses[:: max(1, len(poses) // max_poses)][:max_poses]
    blur_all, bg_frac, g_all, g_blur, pose_blur = [], [], 0.0, 0.0, {}
    n_pix = 0
    for p in poses:
        dpath = os.path.join(ds_dir, "defocus", f"{p}.npy")
        if not os.path.isfile(dpath):
            continue
        b = np.load(dpath)
        g = grad_mag(gray(os.path.join(ds_dir, "aif", f"{p}.png")))
        if g.shape != b.shape:  # defocus map at a different resolution: skip the detail terms
            g = None
        valid = np.isfinite(b)
        bg_frac.append(1.0 - valid.mean())
        bv = b[valid]
        blur_all.append(bv[:: max(1, bv.size // 200_000)])  # subsample for the percentiles
        pose_blur[p] = float(bv.mean()) if bv.size else float("nan")
        if g is not None:
            e = g ** 2
            g_all += float(e.sum())
            g_blur += float(e[valid & (np.nan_to_num(b, nan=0.0) > blur_thresh)].sum())
            n_pix += g.size
    if not blur_all:
        return dict(gt=gt, poses=poses, missing_defocus=True)
    bl = np.concatenate(blur_all)
    return dict(
        gt=gt, poses=poses,
        rel_depth=gt["d_max"] * gt["focus_distance"],
        blur_med=float(np.median(bl)), blur_p90=float(np.percentile(bl, 90)),
        frac_blur=float((bl > blur_thresh).mean()),
        bg_frac=float(np.mean(bg_frac)),
        detail=(math.sqrt(g_all / n_pix) if n_pix else None),
        blurred_detail=(g_blur / g_all if g_all > 0 else None),
        pose_blur=pose_blur,
    )


# ---------------------------------------------------------------- main
def main():
    a = parse_args()
    rows, within = [], []
    for s in a.scenes:
        slit_ds = os.path.join(a.data_root, f"slitsim_{s}_{a.design}{a.suffix}")
        disk_ds = os.path.join(a.data_root, f"disksim_{s}{a.suffix}")
        dst = dataset_stats(disk_ds, a.blur_thresh, a.max_poses)
        if dst is None or dst.get("missing_defocus"):
            print(f"[skip] {s}: no disk dataset / defocus maps in {disk_ds}")
            continue
        sst = dataset_stats(slit_ds, a.blur_thresh, a.max_poses)   # only its long-axis blur is reported
        ev = {NEW: load_eval(os.path.join(a.root, a.new_folder, f"slitsim_{s}_{a.design}{a.suffix}")),
              ORIG: load_eval(os.path.join(a.root, a.orig_folder, f"slitsim_{s}_{a.design}{a.suffix}")),
              DISK: load_eval(os.path.join(a.root, a.disk_folder, f"disksim_{s}{a.suffix}"))}

        def d(key, x, y=DISK):
            u, v = ev[x], ev[y]
            return u[key] - v[key] if u and v and u[key] is not None and v[key] is not None else None

        row = dict(scene=s, n_poses=len(dst["poses"]),
                   rel_depth=dst["rel_depth"],
                   disk_blur_med=dst["blur_med"], disk_blur_p90=dst["blur_p90"],
                   frac_blur=dst["frac_blur"], bg_frac=dst["bg_frac"],
                   detail=dst["detail"], blurred_detail=dst["blurred_detail"],
                   slit_blur_med=(sst or {}).get("blur_med"),
                   test_disk=(ev[DISK] or {}).get("test_psnr"),
                   test_d_orig=d("test_psnr", ORIG), test_d_new=d("test_psnr", NEW),
                   train_d_orig=d("train_rec", ORIG), train_d_new=d("train_rec", NEW),
                   in_disk=(ev[DISK] or {}).get("train_in"), in_slit=(ev[ORIG] or {}).get("train_in"),
                   gap_disk=((ev[DISK]["train_rec"] - ev[DISK]["test_psnr"])
                             if ev[DISK] and ev[DISK]["train_rec"] is not None and ev[DISK]["test_psnr"] is not None
                             else None))
        for name, key in ((NEW, "new"), (ORIG, "orig"), (DISK, "disk")):
            e = ev[name] or {}
            row[f"test_min_{key}"] = e.get("test_min")
            row[f"test_std_{key}"] = e.get("test_std")
            row[f"absrel_{key}"] = e.get("abs_rel")
            row[f"ngauss_{key}"] = e.get("n_gauss")
        rows.append(row)

        # ---- within scene: per-pose deblurring gain vs pose blur ----
        for name in (ORIG, NEW):
            if not (ev[name] and ev[DISK]):
                continue
            pa, pb = ev[name]["per_pose"], ev[DISK]["per_pose"]
            common = [p for p in pa if p in pb and p in dst["pose_blur"] and math.isfinite(dst["pose_blur"][p])]
            if len(common) < 6:
                continue
            blur = np.array([dst["pose_blur"][p] for p in common])
            gain = np.array([pa[p] - pb[p] for p in common])
            order = np.argsort(blur)
            third = max(1, len(common) // 3)
            within.append(dict(scene=s, method=name, n=len(common), rho=spearman(blur, gain),
                               blur_lo=float(blur[order[:third]].mean()), gain_lo=float(gain[order[:third]].mean()),
                               blur_hi=float(blur[order[-third:]].mean()), gain_hi=float(gain[order[-third:]].mean())))

    if not rows:
        sys.exit("No scenes with datasets found.")

    t = a.blur_thresh
    L = [f"# Per-scene diagnosis of the slit vs disk gain ({len(rows)} scenes)", "",
         f"Datasets `slitsim_<scene>_{a.design}{a.suffix}` / `disksim_<scene>{a.suffix}`; models "
         f"`{a.new_folder}` (densmax+L_dir), `{a.orig_folder}` (orig slit), `{a.disk_folder}` (disk).", "",
         "## 1. Scene statistics (training poses) and the gain", "",
         "rel. depth = d_max·F (max |F/z − 1| at the reference focus); disk blur = blur-disk diameter (px) "
         f"over pixels with geometry; >{t:g}px = share of those pixels; bg = pixels without geometry; "
         "detail = RMS image gradient of the AiF GT; blurred detail = share of the gradient energy in pixels "
         f"with disk blur > {t:g} px. Δ = slit − disk PSNR (dB); train Δ = recovered AiF of the training poses.", "",
         "| scene | rel. depth | disk blur med / p90 | >" + f"{t:g}px" + " | bg | detail | blurred detail "
         "| input PSNR disk / slit | test disk | test Δ orig / new | train Δ orig / new | disk train−test |",
         "|---" * 12 + "|"]
    for r in rows:
        L.append(f"| {r['scene']} | {fmt(r['rel_depth'], 3)} | {fmt(r['disk_blur_med'])} / {fmt(r['disk_blur_p90'])} "
                 f"| {fmt(100 * r['frac_blur'], 0)}% | {fmt(100 * r['bg_frac'], 0)}% | {fmt(r['detail'], 4)} "
                 f"| {fmt(100 * r['blurred_detail'], 0) + '%' if r['blurred_detail'] is not None else '–'} "
                 f"| {fmt(r['in_disk'])} / {fmt(r['in_slit'])} | {fmt(r['test_disk'])} "
                 f"| {fmt(r['test_d_orig'])} / {fmt(r['test_d_new'])} | {fmt(r['train_d_orig'])} / {fmt(r['train_d_new'])} "
                 f"| {fmt(r['gap_disk'])} |")
    L.append("")

    L += ["## 2. Test-view spread and model size", "",
          "Lowest test-view PSNR / std over the test views; a single bad view (floater) shows up as a low minimum.", "",
          "| scene | test min new / orig / disk | test std new / orig / disk | Gaussians new / orig / disk |",
          "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['scene']} | {fmt(r['test_min_new'])} / {fmt(r['test_min_orig'])} / {fmt(r['test_min_disk'])} "
                 f"| {fmt(r['test_std_new'])} / {fmt(r['test_std_orig'])} / {fmt(r['test_std_disk'])} "
                 f"| {r['ngauss_new'] or '–'} / {r['ngauss_orig'] or '–'} / {r['ngauss_disk'] or '–'} |")
    L.append("")

    L += ["## 3. Within each scene: deblurring gain vs pose blur", "",
          "Per training pose: gain = recovered AiF PSNR (slit − disk); blur = mean disk blur of the pose (px). "
          "rho > 0: the slit helps more on the more defocused poses.", "",
          "| scene | method | poses | Spearman rho | low-blur third: blur / gain | high-blur third: blur / gain |",
          "|---|---|---|---|---|---|"]
    for w in within:
        L.append(f"| {w['scene']} | {w['method']} | {w['n']} | {fmt(w['rho'])} | {fmt(w['blur_lo'])} / {fmt(w['gain_lo'])} "
                 f"| {fmt(w['blur_hi'])} / {fmt(w['gain_hi'])} |")
    L.append("")

    stats = [("rel_depth", "rel. depth"), ("disk_blur_med", "disk blur median"), ("disk_blur_p90", "disk blur p90"),
             ("frac_blur", f"> {t:g} px share"), ("bg_frac", "background share"), ("detail", "detail"),
             ("blurred_detail", "blurred detail"), ("in_disk", "disk input PSNR"), ("test_disk", "disk test PSNR"),
             ("gap_disk", "disk train − test PSNR")]
    L += ["## 4. Across scenes: Spearman rho with the gain", "",
          f"{len(rows)} scenes; |rho| > ~0.65 for p < 0.05 at n = 10.", "",
          "| statistic | vs test Δ orig | vs test Δ new | vs train Δ orig |", "|---|---|---|---|"]
    for k, h in stats:
        x = [r[k] if r[k] is not None else float("nan") for r in rows]
        cols = []
        for yk in ("test_d_orig", "test_d_new", "train_d_orig"):
            y = [r[yk] if r[yk] is not None else float("nan") for r in rows]
            cols.append(fmt(spearman(x, y)))
        L.append(f"| {h} | " + " | ".join(cols) + " |")
    L.append("")

    md = "\n".join(L)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "diagnose.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(a.out, "diagnose.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(md)
    print(f"\nSaved {os.path.join(a.out, 'diagnose.md')} and diagnose.csv")


if __name__ == "__main__":
    main()
