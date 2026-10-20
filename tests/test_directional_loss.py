"""
CPU sanity checks for src/utils/directional_loss.py (no GPU / rasterizer needed).

    python tests/test_directional_loss.py

1. Axis convention: a horizontal ramp has zero derivative across the slit for
   theta = 0 (u horizontal) and a full derivative for theta = 90 deg.
2. Invariance to the long-axis blur: a random texture blurred along u
   (sigma_L = 3 px) gives a much smaller loss than the same texture blurred
   along v by the same amount, for several angles.
3. The mask follows the short-axis blur: pixels with large r_L get weight ~0.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.directional_loss import (directional_derivative, short_axis_mask, slit_directional_aif_loss,
                                        smooth_along)

COEFF = 1.0 / 3.0


def texture(h=160, w=160, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(3, h, w, generator=g)
    # mild isotropic smoothing so the texture is not pure pixel noise
    img = smooth_along(img, 0.0, 1.0)
    return smooth_along(img, math.pi / 2, 1.0)


def test_axis_convention():
    h, w = 64, 64
    ramp = torch.linspace(0, 1, w).expand(3, h, w).clone()          # varies along x only
    d0 = directional_derivative(ramp, 0.0)[:, 8:-8, 8:-8]           # v = (0, 1): across rows
    d90 = directional_derivative(ramp, math.pi / 2)[:, 8:-8, 8:-8]  # v = (-1, 0): along x
    assert d0.abs().max() < 1e-6, d0.abs().max()
    assert abs(d90.abs().mean().item() - 1.0 / (w - 1)) < 1e-4, d90.abs().mean()
    print("axis convention ok")


def test_long_axis_invariance():
    x = texture()
    h, w = x.shape[-2:]
    sigma_l = 3.0                                                     # long-axis blur std (px)
    r_l = torch.full((1, h, w), sigma_l / math.sqrt(COEFF))          # matching half length r_L
    for deg in (0.0, 45.0, 90.0, 135.0, 30.0):
        th = math.radians(deg)
        along_u = smooth_along(x, th, sigma_l)                        # what a slit capture looks like
        along_v = smooth_along(x, th + math.pi / 2, sigma_l)          # wrong direction
        l_u, cov = slit_directional_aif_loss(x, along_u, r_l, th, ratio=0.0, coeff=COEFF)
        l_v, _ = slit_directional_aif_loss(x, along_v, r_l, th, ratio=0.0, coeff=COEFF)
        print(f"theta {deg:5.1f} deg: loss(blur along u) {l_u.item():.5f}  "
              f"loss(blur along v) {l_v.item():.5f}  coverage {cov:.2f}")
        assert l_u.item() < 0.25 * l_v.item(), (deg, l_u.item(), l_v.item())
    print("long-axis invariance ok")


def test_mask():
    r_l = torch.tensor([[[0.0, 2.0, 20.0]]])                          # r_L in px
    m = short_axis_mask(r_l, ratio=0.4, coeff=COEFF, sigma_w_max=1.0, sigma_l_max=4.0)
    assert m[0, 0, 0] > 0.99 and m[0, 0, 1] > 0.9 and m[0, 0, 2] < 0.01, m
    print("mask ok")


def test_gradient_flows():
    x = texture(64, 64).requires_grad_(True)
    target = smooth_along(x.detach(), 0.3, 2.0)
    r_l = torch.full((1, 64, 64), 2.0)
    loss, _ = slit_directional_aif_loss(x, target, r_l, 0.3, ratio=0.4, coeff=COEFF,
                                        sigma_l_max=2.0, smooth_mult=2.0)
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("gradient ok")


if __name__ == "__main__":
    test_axis_convention()
    test_long_axis_invariance()
    test_mask()
    test_gradient_flows()
    print("all directional-loss checks passed")
