"""
Equal-area design theory for a rotating slit (rectangular) aperture versus a
circular aperture, under the Gaussian PSF approximation used by the renderer.

Setting
-------
All apertures have the same area A (same light per capture). Blur sizes are
expressed at one defocus level, in pixels; every length scales with |1/z - 1/F|.

  slit  : full lengths L x W, width ratio a = W / L, so L^2 = A / a, W^2 = A * a.
          Gaussian variances along the long / short axis: s_L = c_s L^2 / 4,
          s_W = c_s W^2 / 4   (c_s = 1/3, uniform box)
  disk  : radius R with pi R^2 = A, variance per axis s_D = c_d R^2 = c_d A / pi
          (c_d = 1/4, uniform disk)

Capture k has its slit long axis at angle theta_k. Along a spatial/frequency
direction phi its blur variance is
      s_k(phi) = s_L cos^2(phi - theta_k) + s_W sin^2(phi - theta_k).

Single-best-capture analysis (independent of A and of the frequency):
      R(phi) = min_k s_k(phi) / s_D
             = (pi c_s / (4 c_d)) * min_k [ cos^2(phi - theta_k) / a + a sin^2(phi - theta_k) ]
The worst direction is half-way between two neighbouring orientations,
Delta_max = (largest orientation gap) / 2 = 90 deg / n for n evenly spaced angles over 180 deg.
Minimising the worst case over a gives
      a* = tan(Delta_max),     R_worst(a*) = (pi c_s / (4 c_d)) * sin(2 Delta_max).
Depth-of-field gain at equal light (worst direction):  1 / sqrt(R_worst).
Light needed by a disk to match that depth of field:     R_worst  (fraction of A).

Multi-capture fusion at a frequency f (needs an absolute scale, --area_px2):
with Gaussian MTFs M = exp(-2 pi^2 s f^2) and independent noise of equal level
in every capture, the information along phi is proportional to sum_k M_k^2.
Against a disk taking the same number of captures (equal total light):
      SNR gain(phi, f) = 10 log10( sum_k M_k(phi, f)^2 / (n * M_D(f)^2) )   [dB]

Usage:
    python tools/aperture_theory.py                       # table + plots, default n list
    python tools/aperture_theory.py --n 4 8 --ratios 0.1 0.2 0.41 --area_px2 40 --freqs 0.05 0.1 0.2
"""
import argparse
import csv
import json
import math
import os

import numpy as np

SLIT_COEFF = 1.0 / 3.0   # variance of a uniform box of full length L: L^2 / 12 = (1/3) (L/2)^2
DISK_COEFF = 1.0 / 4.0   # variance of a uniform disk of radius R (per axis): R^2 / 4
PREFACTOR = math.pi * SLIT_COEFF / (4.0 * DISK_COEFF)   # = pi / 3


# ---------------------------------------------------------------------------
# Geometry helpers (also imported by simulate_slit_dataset.py)
# ---------------------------------------------------------------------------
def evenly_spaced_angles(n, full_circle=False):
    span = 360.0 if full_circle else 180.0
    return [k * span / n for k in range(n)]


def unique_orientations(angles_deg):
    """Slit orientations modulo 180 deg (theta and theta + 180 give the same PSF)."""
    o = sorted({round(a % 180.0, 6) for a in angles_deg})
    return o


def max_offset_deg(angles_deg):
    """Largest angle between any direction and the nearest slit long axis' normal = half the largest gap."""
    o = unique_orientations(angles_deg)
    if len(o) == 1:
        return 90.0
    gaps = [o[i + 1] - o[i] for i in range(len(o) - 1)] + [o[0] + 180.0 - o[-1]]
    return max(gaps) / 2.0


def optimal_ratio(angles_deg):
    """Width ratio a* = tan(Delta_max) that minimises the worst-direction blur at fixed area."""
    return min(1.0, math.tan(math.radians(max_offset_deg(angles_deg))))


def worst_variance_ratio_closed_form(a, angles_deg):
    d = math.radians(max_offset_deg(angles_deg))
    # worst direction is offset d from the short axis of the nearest capture
    return PREFACTOR * (a * math.cos(d) ** 2 + math.sin(d) ** 2 / a)


# ---------------------------------------------------------------------------
# Numerical evaluation over all directions
# ---------------------------------------------------------------------------
def directional_profile(a, angles_deg, n_dirs=1800):
    """phi (deg), and per-capture variance ratios s_k(phi) / s_D, shape (n_dirs, n_captures)."""
    phi = np.linspace(0.0, 180.0, n_dirs, endpoint=False)
    th = np.asarray(angles_deg, dtype=np.float64)
    d = np.radians(phi[:, None] - th[None, :])
    r = PREFACTOR * (np.cos(d) ** 2 / a + a * np.sin(d) ** 2)
    return phi, r


def summarize(a, angles_deg, area_px2=None, freqs=()):
    phi, r = directional_profile(a, angles_deg)
    best = r.min(axis=1)
    out = dict(
        n=len(angles_deg),
        ratio=a,
        delta_max_deg=max_offset_deg(angles_deg),
        worst_var_ratio=float(best.max()),
        mean_var_ratio=float(best.mean()),
        dof_gain_worst=float(1.0 / math.sqrt(best.max())),
        dof_gain_mean=float(np.mean(1.0 / np.sqrt(best))),
        disk_light_fraction_for_same_dof=float(best.max()),
    )
    if area_px2 is not None:
        s_disk = DISK_COEFF * area_px2 / math.pi
        out["slit_full_length_px"] = math.sqrt(area_px2 / a)
        out["slit_full_width_px"] = math.sqrt(area_px2 * a)
        out["disk_diameter_px"] = 2.0 * math.sqrt(area_px2 / math.pi)
        for f in freqs:
            m2_slit = np.exp(-4.0 * math.pi ** 2 * (r * s_disk) * f ** 2).sum(axis=1)   # sum_k M_k^2
            m2_disk = len(angles_deg) * math.exp(-4.0 * math.pi ** 2 * s_disk * f ** 2)
            gain_db = 10.0 * np.log10(np.maximum(m2_slit, 1e-300) / max(m2_disk, 1e-300))
            out[f"snr_gain_db_worst_f{f:g}"] = float(gain_db.min())
            out[f"snr_gain_db_mean_f{f:g}"] = float(gain_db.mean())
    return out


# ---------------------------------------------------------------------------
# Plots (dataviz reference palette; validated for 4 series on light surface)
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def make_plots(out_dir, n_plot, full_circle, eval_ratios):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) worst-direction variance ratio vs width ratio, one line per n
    a_grid = np.logspace(math.log10(0.02), 0, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    _style(ax)
    for i, n in enumerate(n_plot[:4]):
        ang = evenly_spaced_angles(n, full_circle)
        y = [summarize(a, ang)["worst_var_ratio"] for a in a_grid]
        ax.plot(a_grid, y, color=SERIES[i], linewidth=2, label=f"n = {n}")
        a_star = optimal_ratio(ang)
        y_star = summarize(a_star, ang)["worst_var_ratio"]
        ax.plot([a_star], [y_star], "o", color=SERIES[i], markersize=8, markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"n={n}: a*={a_star:.2f}", (a_star, y_star), textcoords="offset points", xytext=(6, -14),
                    fontsize=8.5, color=INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle="--")
    ax.text(0.021, 1.05, "equal-area disk", color=MUTED, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_yticks([0.2, 0.3, 0.5, 1, 2, 3, 5, 8])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo * 0.85, hi)
    ax.set_xlabel("slit width ratio a = W / L", color=INK)
    ax.set_ylabel("worst-direction blur variance / disk", color=INK)
    ax.set_title("Equal-area slit vs disk: lower is better", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "worst_variance_vs_ratio.png")
    fig.savefig(p1)
    plt.close(fig)

    # (2) depth-of-field gain vs number of angles, at a* and at fixed ratios
    ns = np.arange(2, 17)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    _style(ax)
    series = [("optimal a* = tan(90°/n)", None)] + [(f"a = {r:g}", r) for r in eval_ratios[:3]]
    for i, (label, r) in enumerate(series):
        y = []
        for n in ns:
            ang = evenly_spaced_angles(int(n), full_circle)
            y.append(summarize(r if r is not None else optimal_ratio(ang), ang)["dof_gain_worst"])
        ax.plot(ns, y, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=label)
    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle="--")
    ax.text(16.3, 1.03, "equal-area disk", color=MUTED, fontsize=8.5, ha="right")
    ax.set_xlim(1.5, 16.5)
    ax.set_xlabel("number of slit orientations n (over 180°)", color=INK)
    ax.set_ylabel("depth-of-field gain (worst direction)", color=INK)
    ax.set_title("Depth of field at equal light, relative to the disk", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xticks(ns)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "dof_gain_vs_n.png")
    fig.savefig(p2)
    plt.close(fig)
    return [p1, p2]


def main():
    p = argparse.ArgumentParser(description="Equal-area slit vs disk theory (Gaussian PSF approximation)")
    p.add_argument("--n", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12, 16], help="numbers of slit angles")
    p.add_argument("--full_circle", action="store_true", help="angles spread over 360 deg instead of 180 deg")
    p.add_argument("--ratios", type=float, nargs="*", default=[0.1, 0.4],
                   help="extra fixed width ratios to evaluate besides the optimum")
    p.add_argument("--area_px2", type=float, default=None,
                   help="aperture area in px^2 at the reference defocus (e.g. 20 x 2 = 40), enables SNR gains")
    p.add_argument("--freqs", type=float, nargs="*", default=[0.05, 0.1, 0.2], help="frequencies (cycles/px)")
    p.add_argument("--out", default="./output/aperture_theory")
    p.add_argument("--no_plots", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for n in a.n:
        ang = evenly_spaced_angles(n, a.full_circle)
        a_star = optimal_ratio(ang)
        for label, r in [("optimal", a_star)] + [("fixed", r) for r in a.ratios]:
            s = summarize(r, ang, a.area_px2, a.freqs)
            s["choice"] = label
            s["closed_form_worst"] = worst_variance_ratio_closed_form(r, ang)
            rows.append(s)

    hdr = f"{'n':>3} {'choice':>8} {'a':>6} {'Δmax°':>6} {'worst σ²/disk':>14} {'DOF gain':>9} {'mean σ²/disk':>13} {'disk light for same DOF':>24}"
    print("\nEqual-area rotating slit vs disk (single best capture per direction, Gaussian PSFs)")
    print(hdr)
    for s in rows:
        print(f"{s['n']:>3} {s['choice']:>8} {s['ratio']:6.3f} {s['delta_max_deg']:6.2f} {s['worst_var_ratio']:14.3f} "
              f"{s['dof_gain_worst']:9.2f} {s['mean_var_ratio']:13.3f} {100 * s['disk_light_fraction_for_same_dof']:23.0f}%")
    print("  worst σ²/disk < 1 and DOF gain > 1 mean the slit beats the equal-area disk in every direction.")

    if a.area_px2 is not None:
        print(f"\nMulti-capture SNR gain over a disk with the same number of captures (area {a.area_px2:g} px^2)")
        cols = [f"f={f:g}" for f in a.freqs]
        print(f"{'n':>3} {'choice':>8} {'a':>6} {'L x W (px)':>14} " + " ".join(f"{c + ' worst/mean dB':>22}" for c in cols))
        for s in rows:
            dims = f"{s['slit_full_length_px']:.1f}x{s['slit_full_width_px']:.1f}"
            vals = " ".join(f"{s[f'snr_gain_db_worst_f{f:g}']:10.2f}/{s[f'snr_gain_db_mean_f{f:g}']:<10.2f}"
                            for f in a.freqs)
            print(f"{s['n']:>3} {s['choice']:>8} {s['ratio']:6.3f} {dims:>14} {vals}")
        print(f"  equal-area disk diameter: {rows[0]['disk_diameter_px']:.2f} px")

    csv_path = os.path.join(a.out, "aperture_theory.csv")
    with open(csv_path, "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "aperture_theory.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved table to {csv_path}")

    if not a.no_plots:
        n_plot = [n for n in (4, 6, 8, 12) if n in a.n] or a.n[:4]
        for path in make_plots(a.out, n_plot, a.full_circle, a.ratios):
            print(f"Saved plot {path}")

    print("\nSimulate a chosen design at the area of an existing slit dataset, e.g.:")
    for n in a.n:
        if n in (4, 8):
            print(f"  python tools/simulate_slit_dataset.py --config ./config_files/config_real_cake.yaml "
                  f"--scene real_cake --iteration 15000 --n_angles {n} "
                  f"--match_area ./data/slit_sim/slitsim_real_cake      # a* = {optimal_ratio(evenly_spaced_angles(n, a.full_circle)):.3f}")


if __name__ == "__main__":
    main()
