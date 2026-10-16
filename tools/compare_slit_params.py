"""
Compare the optical parameters recovered by the slit model with the simulation
ground truth.

Usage (from the repository root):
    python tools/compare_slit_params.py \
        --sim ./data/slit_sim/slitsim_real_cake \
        --params ./output/model_dofgs/slit_sim/slitsim_real_cake/defocus_params/iteration_40000/defocus_params.pth
"""
import argparse
import json
import math
import os

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", required=True, help="simulated dataset folder (contains sim_gt.json)")
    p.add_argument("--params", required=True, help="defocus_params.pth saved during training")
    a = p.parse_args()

    with open(os.path.join(a.sim, "sim_gt.json")) as f:
        gt = json.load(f)
    st = torch.load(a.params, map_location="cpu")
    ap = st.get("aperture")
    if ap is None or ap["aperture_type"] != "slit":
        raise SystemExit("These parameters were not trained with aperture_type: slit")

    # Same mapping as GaussianModel.get_defocus_params().
    F = torch.clamp(st["focal_distance"], 0.01, 1000) * 10.0
    K = torch.clamp(st["K_aperture"], 0.01, 200) * 100.0 + 100.0
    ratio = torch.sigmoid(ap["slit_ratio_raw"]).item()
    offset = math.degrees(ap["theta_offset"].item())

    def row(name, est, ref):
        err = (est - ref) / ref * 100 if ref != 0 else float("nan")
        print(f"{name:22s} est {est:12.4f}   gt {ref:12.4f}   rel.err {err:7.2f} %")

    print(f"views: {F.numel() if not ap['share_F'] else 'shared'} F, {K.numel() if not ap['share_K'] else 'shared'} K")
    row("focus distance F", F.mean().item(), gt["focus_distance"])
    row("K long", K.mean().item(), gt["K_long"])
    row("K short", K.mean().item() * ratio, gt["K_short"])
    row("width ratio", ratio, gt["width_ratio"])
    print(f"{'theta offset (deg)':22s} est {offset:12.4f}   gt {gt['theta_bias_deg']:12.4f}")
    if not ap["share_F"] or not ap["share_K"]:
        print(f"per-view spread: F std {F.std().item():.4f}, K std {K.std().item():.4f}")


if __name__ == "__main__":
    main()
