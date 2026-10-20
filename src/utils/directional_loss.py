"""
Directional all-in-focus (AiF) constraint for the rotating slit aperture.

A slit capture at angle theta is blurred strongly along the long slit axis
    u = (cos theta, sin theta)
and only weakly along the short axis
    v = (-sin theta, cos theta)
(image coordinates: x to the right, y downwards; same convention as
computeSlitCoC in cuda_rasterizer/auxiliary.h). Where the short-axis blur is
small, the capture still carries the scene detail across v, even if the pixel
is far from focus along u.

Comparing the AiF render with the capture directly would force the AiF to be
blurred along u. Both images are therefore first smoothed along u with the same
1D Gaussian G_u that is wider than the long-axis blur (G_u * h_u ~= G_u), and
only their derivatives across v are compared:

    L_dir = || M_W . d_v( G_u * I_aif - G_u * I_theta ) ||_1

M_W selects pixels whose short-axis blur sigma_W is below a threshold and whose
long-axis blur sigma_L is small enough to be absorbed by G_u. The loss does not
depend on the exact long-axis blur, i.e. on the estimated depth / K / F, only
on an upper bound of it. Averaged over the n slit angles, it supervises the AiF
render along n directions.

Blur sizes follow the slit model of the rasterizer: the rendered CoC map holds
the alpha-composited long-axis half length r_L (pixels), the Gaussian variance
along u is coeff * r_L^2 (coeff = 1/3 for the box-variance-matched slit), and
the short axis is r_W = ratio * r_L.
"""
import math

import torch
import torch.nn.functional as F

# Sobel kernels normalised to a per-pixel derivative (central difference with smoothing).
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0],
                         [-2.0, 0.0, 2.0],
                         [-1.0, 0.0, 1.0]]) / 8.0
_SOBEL_Y = _SOBEL_X.t().contiguous()


def slit_axes(theta):
    """Unit vectors (u, v) of the long and short slit axes in image coordinates."""
    c, s = math.cos(theta), math.sin(theta)
    return (c, s), (-s, c)


def smooth_along(img, theta, sigma, max_taps=49):
    """
    1D Gaussian smoothing of img (C, H, W) along u = (cos theta, sin theta).

    Implemented as a weighted sum of bilinearly shifted copies (grid_sample),
    which works for any angle and keeps gradients w.r.t. img. The tap spacing
    is 1 px unless more than max_taps taps would be needed.
    """
    if sigma <= 0:
        return img
    C, H, W = img.shape
    half = 3.0 * sigma
    n_half = int(math.ceil(half))
    if 2 * n_half + 1 > max_taps:
        n_half = (max_taps - 1) // 2
    offsets = torch.linspace(-half, half, 2 * n_half + 1, device=img.device, dtype=img.dtype)
    weights = torch.exp(-0.5 * (offsets / sigma) ** 2)
    weights = weights / weights.sum()

    ys, xs = torch.meshgrid(torch.arange(H, device=img.device, dtype=img.dtype),
                            torch.arange(W, device=img.device, dtype=img.dtype), indexing="ij")
    ux, uy = math.cos(theta), math.sin(theta)
    sx, sy = 2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)   # pixel -> normalised coordinates
    inp = img.unsqueeze(0)
    out = torch.zeros_like(inp)
    for t, w in zip(offsets.tolist(), weights.tolist()):
        grid = torch.stack(((xs + t * ux) * sx - 1.0, (ys + t * uy) * sy - 1.0), dim=-1).unsqueeze(0)
        out = out + w * F.grid_sample(inp, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(0)


def directional_derivative(img, theta):
    """Derivative of img (C, H, W) along the short slit axis v = (-sin theta, cos theta)."""
    C = img.shape[0]
    kx = _SOBEL_X.to(img.device, img.dtype).expand(C, 1, 3, 3)
    ky = _SOBEL_Y.to(img.device, img.dtype).expand(C, 1, 3, 3)
    x = F.pad(img.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(x, kx, groups=C)
    gy = F.conv2d(x, ky, groups=C)
    (_, _), (vx, vy) = slit_axes(theta)
    return (vx * gx + vy * gy).squeeze(0)


def _soft_below(x, thresh, sharpness=8):
    """Smooth step: ~1 for x << thresh, 0.5 at thresh, ~0 for x >> thresh."""
    return 1.0 / (1.0 + (x / thresh).clamp(min=0.0) ** sharpness)


@torch.no_grad()
def short_axis_mask(coc_long, ratio, coeff, sigma_w_max, sigma_l_max, margin=0):
    """
    Soft mask M_W (1, H, W) of pixels that are sharp along the short slit axis.

    coc_long: rendered long-axis half length r_L in pixels, (H, W) or (1, H, W).
    sigma_w_max: threshold on the short-axis blur std sigma_W (pixels).
    sigma_l_max: upper bound on the long-axis blur std sigma_L (pixels) that the
                 smoothing G_u is designed to absorb.
    margin: pixels excluded at the image border (where G_u sees padding).
    """
    r_l = coc_long.reshape(1, *coc_long.shape[-2:]).clamp(min=0.0)
    sigma_l = math.sqrt(coeff) * r_l
    sigma_w = ratio * sigma_l
    m = _soft_below(sigma_w, sigma_w_max) * _soft_below(sigma_l, sigma_l_max)
    if margin > 0:
        m[..., :margin, :] = 0.0
        m[..., -margin:, :] = 0.0
        m[..., :, :margin] = 0.0
        m[..., :, -margin:] = 0.0
    return m


def slit_directional_aif_loss(aif, target, coc_long, theta, ratio, coeff,
                              sigma_w_max=1.0, sigma_l_max=4.0, smooth_mult=2.5):
    """
    L_dir for one slit capture.

    aif:      rendered all-in-focus image (C, H, W), gradients flow through it.
    target:   captured slit image (C, H, W), treated as a constant.
    coc_long: rendered long-axis half length r_L (pixels) of this view, constant.
    theta:    slit angle of this view in radians (float).
    ratio:    short / long slit ratio K_W / K_L.
    coeff:    PSF variance coefficient of the slit model (1/3).
    Returns (loss, mask coverage). The loss is an L1 mean over the masked
    pixels, so its scale does not depend on how much of the image is masked.
    """
    sigma_u = smooth_mult * sigma_l_max
    margin = int(math.ceil(3.0 * sigma_u))
    mask = short_axis_mask(coc_long, ratio, coeff, sigma_w_max, sigma_l_max, margin=margin)
    with torch.no_grad():
        d_target = directional_derivative(smooth_along(target, theta, sigma_u), theta)
    d_aif = directional_derivative(smooth_along(aif, theta, sigma_u), theta)
    n_ch = aif.shape[0]
    denom = (mask.sum() * n_ch).clamp(min=1.0)
    loss = (mask * (d_aif - d_target).abs()).sum() / denom
    return loss, float(mask.mean())
