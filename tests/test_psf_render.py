"""
CPU checks for tools/psf_render.py (no GPU / rasterizer / dataset needed).

    python tests/test_psf_render.py

1. Kernels are normalised and have the expected second moments
   (box: L^2/12, disk: D^2/16, Gaussian: sigma^2; plus 1/12 px^2 from the pixel
   integration of the supersampling), and the slit long axis follows theta.
2. Diffraction kernels are normalised; the slit sinc^2 is wider across the slit.
3. Kernel composition adds variances.
4. 'Same' convolution keeps a constant image constant and a delta kernel is the identity.
5. Back-to-front compositing: an opaque in-focus layer replaces the background,
   a transparent layer leaves it unchanged.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import psf_render as pr

PIX = 1.0 / 12.0   # variance of the pixel footprint


def moments(k):
    k = k.double()
    n = k.shape[-1]
    c = torch.arange(n, dtype=torch.float64) - n // 2
    y, x = torch.meshgrid(c, c, indexing="ij")
    return float((k * x * x).sum()), float((k * y * y).sum()), float(k.sum())


def close(a, b, rel=0.06):
    return abs(a - b) <= rel * max(abs(b), 1e-9)


def box_var_1d(width):
    """
    Exact variance (about pixel centres) of a 1D box of the given width after
    integration over pixels: pixel i gets the overlap of [i-0.5, i+0.5] with the box.
    Unlike the smooth-density rule var + 1/12, this is exact when the box edges fall
    on pixel boundaries (e.g. width 3 -> three equal pixels, variance 2/3).
    """
    r = int(math.ceil(width / 2)) + 1
    w = [max(0.0, min(i + 0.5, width / 2) - max(i - 0.5, -width / 2)) for i in range(-r, r + 1)]
    return sum(i * i * wi for i, wi in zip(range(-r, r + 1), w)) / sum(w)


def test_box_and_orientation():
    L, W = 12.0, 3.0
    vx, vy, s = moments(pr.box_kernel(L, W, 0.0))
    assert close(s, 1.0, 1e-6)
    assert close(vx, box_var_1d(L), 0.01) and close(vy, box_var_1d(W), 0.01), (vx, vy)
    # edges inside pixels (on the 1/8 supersampling grid, so the expectation stays exact)
    vx2, vy2, _ = moments(pr.box_kernel(12.5, 3.25, 0.0))
    assert close(vx2, box_var_1d(12.5), 0.01) and close(vy2, box_var_1d(3.25), 0.01), (vx2, vy2)
    vx90, vy90, _ = moments(pr.box_kernel(L, W, math.pi / 2))
    assert close(vx90, vy) and close(vy90, vx), (vx90, vy90)
    print("box kernel ok")


def test_disk_and_gaussian():
    D = 10.0
    vx, vy, s = moments(pr.disk_kernel(D))
    assert close(s, 1.0, 1e-6) and close(vx, D * D / 16 + PIX) and close(vy, vx, 0.02), (vx, vy)
    sl, sw = 3.0, 1.0
    gx, gy, s = moments(pr.gaussian_kernel(sl, sw, 0.0))
    assert close(s, 1.0, 1e-6) and close(gx, sl * sl + PIX) and close(gy, sw * sw + PIX), (gx, gy)
    # gaussian_layered uses the rasterizer variance c r^2 (c = 1/3): same variance as the box
    r_l = 6.0
    bx, _, _ = moments(pr.box_kernel(2 * r_l, 0.0, 0.0))
    gx2, _, _ = moments(pr.gaussian_kernel(math.sqrt(1 / 3) * r_l, 0.0, 0.0))
    assert close(bx, gx2, 0.05), (bx, gx2)
    print("disk / gaussian kernels ok")


def test_diffraction():
    k = pr.slit_diffraction_kernel(10.0, 4.0, 0.0, 550.0, 50.0, 4.0)
    vx, vy, s = moments(k)
    assert close(s, 1.0, 1e-6) and vy > vx, (vx, vy)     # narrow side (4 mm) diffracts more
    a = pr.disk_diffraction_kernel(7.1, 550.0, 50.0, 4.0)
    ax, ay, s = moments(a)
    assert close(s, 1.0, 1e-6) and close(ax, ay, 0.02), (ax, ay)
    print("diffraction kernels ok")


def test_compose():
    g = pr.box_kernel(12.0, 3.0, 0.0)
    o = torch.stack([pr.gaussian_kernel(2.0, 2.0, 0.0)] * 3)
    c = pr.compose_kernels(g, o)
    vx, vy, s = moments(c[0])
    gx, gy, _ = moments(g)
    ox, oy, _ = moments(o[0])
    assert close(s, 1.0, 1e-5) and close(vx, gx + ox, 0.03) and close(vy, gy + oy, 0.03), (vx, gx + ox)
    print("kernel composition ok")


def test_conv_and_composite():
    img = torch.rand(3, 40, 50)
    assert torch.allclose(pr.conv_same(img, pr.delta_kernel("cpu")), img)
    const = torch.full((3, 40, 50), 0.3)
    out = pr.conv_same(const, pr.box_kernel(9.0, 3.0, 0.7))
    assert out.shape == const.shape and torch.allclose(out, const, atol=1e-5)

    acc = torch.full((3, 40, 50), 0.2)
    color = torch.rand(3, 40, 50)
    opaque = torch.ones(1, 40, 50)
    res = pr.composite_over(acc, color * opaque, opaque, pr.delta_kernel("cpu"))
    assert torch.allclose(res, color)
    res = pr.composite_over(acc, torch.zeros_like(color), torch.zeros(1, 40, 50), pr.box_kernel(9.0, 3.0, 0.3))
    assert torch.allclose(res, acc, atol=1e-6)
    print("convolution / compositing ok")


if __name__ == "__main__":
    test_box_and_orientation()
    test_disk_and_gaussian()
    test_diffraction()
    test_compose()
    test_conv_and_composite()
    print("all psf_render checks passed")
