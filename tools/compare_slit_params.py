"""
Compare the optical parameters recovered by a model (slit or disk) with the
simulation ground truth.

Focus distance: one value per parameter slot (share_F: global / group / view).
For per-pose-focus datasets (sim_gt focus_per_group) every slot is compared with
the focus of its pose; otherwise with the single focus distance. The relative
errors are summarised by their mean absolute value and maximum.

Usage (from the repository root):
    python tools/compare_slit_params.py \
        --sim ./data/slit_sim/slitsim_real_cake_n4_a0.40_pf \
        --params ./output/model_dofgs/slit_sim_dir0.005_densmax/slitsim_real_cake_n4_a0.40_pf/defocus_params/iteration_40000/defocus_params.pth
"""
import argparse
import json
import math
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.focus_utils import pose_stem


def rel(est, ref):
    return (est - ref) / ref * 100.0 if ref != 0 else float("nan")


def row(name, est, ref):
    print(f"{name:22s} est {est:12.4f}   gt {ref:12.4f}   rel.err {rel(est, ref):7.2f} %")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", required=True, help="simulated dataset folder (contains sim_gt.json)")
    p.add_argument("--params", required=True, help="defocus_params.pth saved during training")
    p.add_argument("--per_slot", action="store_true", help="print every focus slot")
    a = p.parse_args()

    with open(os.path.join(a.sim, "sim_gt.json")) as f:
        gt = json.load(f)
    st = torch.load(a.params, map_location="cpu")
    ap = st.get("aperture") or {}
    aperture = ap.get("aperture_type", "disk")

    # Same mapping as GaussianModel.get_defocus_params() (one value per slot).
    F = (torch.clamp(st["focal_distance"], 0.01, 1000) * 10.0).reshape(-1)
    K = (torch.clamp(st["K_aperture"], 0.01, 200) * 100.0 + 100.0).reshape(-1)
    f_mode = ap.get("share_F_mode") or ("global" if F.numel() == 1 else "view")
    k_mode = ap.get("share_K_mode") or ("global" if K.numel() == 1 else "view")
    names = list(ap.get("f_slot_names") or [])
    print(f"aperture {aperture}: share_F={f_mode} ({F.numel()} value(s)), share_K={k_mode} ({K.numel()} value(s))")

    # ---- focus distance ----
    per_group = gt.get("focus_per_group") or {}
    if per_group and f_mode != "global" and len(names) == F.numel():
        errs = []
        for n, est in zip(names, F.tolist()):
            ref = per_group.get(n, per_group.get(pose_stem(n)))
            if ref is None:
                continue
            errs.append(rel(est, ref))
            if a.per_slot:
                print(f"  F[{n}]  est {est:10.4f}   gt {ref:10.4f}   rel.err {errs[-1]:7.2f} %")
        if errs:
            mae = sum(abs(e) for e in errs) / len(errs)
            print(f"{'focus distance F':22s} {len(errs)} slots: mean |rel.err| {mae:6.2f} %, "
                  f"max |rel.err| {max(abs(e) for e in errs):6.2f} %, mean signed {sum(errs) / len(errs):+6.2f} %")
    elif per_group:
        vals = list(per_group.values())
        print(f"{'focus distance F':22s} est {F.mean().item():12.4f}   gt (per pose) {min(vals):.4f} .. {max(vals):.4f} "
              f"(a single estimated F cannot match per-pose focus)")
    else:
        row("focus distance F", F.mean().item(), gt["focus_distance"])
        if F.numel() > 1:
            print(f"  per-slot spread: std {F.std().item():.4f}")

    # ---- aperture ----
    if aperture == "slit":
        ratio = torch.sigmoid(ap["slit_ratio_raw"]).item()
        row("K long", K.mean().item(), gt["K_long"])
        row("K short", K.mean().item() * ratio, gt["K_short"])
        row("width ratio", ratio, gt["width_ratio"])
        offset = math.degrees(ap["theta_offset"].item())
        print(f"{'theta offset (deg)':22s} est {offset:12.4f}   gt {gt.get('theta_bias_deg', 0.0):12.4f}")
    else:
        row("K (disk diameter)", K.mean().item(), gt["K_long"])
    if K.numel() > 1:
        print(f"  K per-view spread: std {K.std().item():.4f}")


if __name__ == "__main__":
    main()
