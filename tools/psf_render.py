"""
Layered defocus rendering with physically shaped PSFs, used by
tools/simulate_slit_dataset.py (--psf box / gaussian_layered).

The rasterizer approximates every defocus PSF by a Gaussian added to the 2D
covariance of each splat. To simulate captures with more realistic optics, the
scene is instead
  1. split into depth layers, uniform in inverse depth, by Gaussian centre;
  2. rendered layer by layer without defocus (premultiplied colour + alpha);
     the full 3D model also provides content hidden behind foreground layers;
  3. each layer is convolved with the PSF of its defocus d = |1/z - 1/F|:
       slit:  box of size (K_L d) x (K_W d) rotated by theta
       disk:  uniform disk of diameter K_D d
       gaussian_layered: the rasterizer's Gaussian (variance check of the pipeline)
     optionally convolved with an in-focus Fraunhofer diffraction PSF;
  4. composited back to front: I <- k*(alpha C) + (1 - k*alpha) I.
Blur lengths follow the slit model of the rasterizer (full length = K * d,
image coordinates with x to the right and y downwards, long axis
u = (cos theta, sin theta)). Kernels are supersampled for anti-aliasing and
normalised to unit sum. Convolutions use FFTs with replicate padding.
"""
import math

import torch
import torch.nn.functional as F

_KERNEL_RADIUS_MAX = 256  # safety cap for very defocused outlier layers (pixels)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def _subpixel_grid(radius, ss, device):
    """Supersampled coordinates (pixels) of a (2r+1)^2 kernel, centre pixel at 0."""
    n = 2 * radius + 1
    c = (torch.arange(n * ss, device=device, dtype=torch.float64) + 0.5) / ss - n / 2.0
    y, x = torch.meshgrid(c, c, indexing="ij")
    return x, y, n


def _pool(values, n, ss):
    """Average the ss x ss subsamples of every pixel and normalise to unit sum."""
    k = values.reshape(n, ss, n, ss).mean(dim=(1, 3))
    s = k.sum()
    if s <= 0:
        k = torch.zeros_like(k)
        k[n // 2, n // 2] = 1.0
        return k
    return k / s


def delta_kernel(device):
    return torch.ones(1, 1, device=device, dtype=torch.float32)


def is_delta(kernel):
    return kernel.shape[-1] == 1 and kernel.shape[-2] == 1


def box_kernel(length, width, theta, ss=8, device="cpu"):
    """Uniform rectangle (full length x full width, pixels) with its long side along theta."""
    if length < 1.0 / ss and width < 1.0 / ss:
        return delta_kernel(device)
    length, width = max(length, 1.0 / ss), max(width, 1.0 / ss)   # keep sub-pixel lines visible
    radius = min(int(math.ceil(0.5 * math.hypot(length, width))) + 1, _KERNEL_RADIUS_MAX)
    x, y, n = _subpixel_grid(radius, ss, device)
    c, s = math.cos(theta), math.sin(theta)
    along = x * c + y * s
    across = -x * s + y * c
    inside = ((along.abs() <= 0.5 * length) & (across.abs() <= 0.5 * width)).double()
    return _pool(inside, n, ss).float()


def disk_kernel(diameter, ss=8, device="cpu"):
    """Uniform disk of the given diameter (pixels)."""
    if diameter < 1.0 / ss:
        return delta_kernel(device)
    diameter = max(diameter, 1.0 / ss)
    radius = min(int(math.ceil(0.5 * diameter)) + 1, _KERNEL_RADIUS_MAX)
    x, y, n = _subpixel_grid(radius, ss, device)
    inside = ((x * x + y * y) <= (0.5 * diameter) ** 2).double()
    return _pool(inside, n, ss).float()


def gaussian_kernel(sigma_long, sigma_short, theta, ss=8, device="cpu"):
    """Anisotropic Gaussian (stds in pixels) with its long axis along theta."""
    if max(sigma_long, sigma_short) < 1e-3:
        return delta_kernel(device)
    # at least half a subsample, so that a very narrow axis is still sampled (becomes a line)
    sl, sw = max(sigma_long, 0.5 / ss), max(sigma_short, 0.5 / ss)
    radius = min(int(math.ceil(4.0 * max(sl, sw))) + 1, _KERNEL_RADIUS_MAX)
    x, y, n = _subpixel_grid(radius, ss, device)
    c, s = math.cos(theta), math.sin(theta)
    along = x * c + y * s
    across = -x * s + y * c
    g = torch.exp(-0.5 * ((along / sl) ** 2 + (across / sw) ** 2))
    return _pool(g, n, ss).float()


def slit_diffraction_kernel(long_mm, short_mm, theta, wavelength_nm, focal_mm, pixel_um,
                            ss=4, n_lobes=6, device="cpu"):
    """
    In-focus Fraunhofer PSF of a rectangular aperture (long_mm x short_mm) at
    distance ~ focal_mm: sinc^2 along both aperture axes, rotated by theta.
    The first zero across the short axis lies at lambda * f / short (sensor units).
    """
    lam_um = wavelength_nm * 1e-3
    z_long = lam_um * focal_mm / long_mm / pixel_um     # first-zero distances in pixels
    z_short = lam_um * focal_mm / short_mm / pixel_um
    radius = min(int(math.ceil(n_lobes * max(z_long, z_short))) + 1, _KERNEL_RADIUS_MAX)
    x, y, n = _subpixel_grid(radius, ss, device)
    c, s = math.cos(theta), math.sin(theta)
    along = x * c + y * s
    across = -x * s + y * c
    k = torch.sinc(along / z_long) ** 2 * torch.sinc(across / z_short) ** 2
    return _pool(k, n, ss).float()


def disk_diffraction_kernel(diameter_mm, wavelength_nm, focal_mm, pixel_um, ss=4, n_lobes=6, device="cpu"):
    """In-focus Airy pattern of a circular aperture; first zero at 1.22 lambda f / D."""
    lam_um = wavelength_nm * 1e-3
    unit = lam_um * focal_mm / diameter_mm / pixel_um      # lambda f / D in pixels
    radius = min(int(math.ceil(n_lobes * 1.22 * unit)) + 1, _KERNEL_RADIUS_MAX)
    x, y, n = _subpixel_grid(radius, ss, device)
    arg = math.pi * torch.sqrt(x * x + y * y) / unit
    safe = torch.where(arg < 1e-8, torch.ones_like(arg), arg)
    airy = torch.where(arg < 1e-8, torch.ones_like(arg), (2.0 * torch.special.bessel_j1(safe) / safe) ** 2)
    return _pool(airy, n, ss).float()


def compose_kernels(geometric, optical):
    """
    Convolve a geometric kernel (h1, w1) with per-channel optical kernels
    (C, h2, w2); returns (C, h1 + h2 - 1, w1 + w2 - 1). All kernels are symmetric.
    """
    g = geometric.reshape(1, 1, *geometric.shape[-2:])
    h2, w2 = optical.shape[-2:]
    g = F.pad(g, (w2 - 1, w2 - 1, h2 - 1, h2 - 1))
    out = F.conv2d(g, optical.reshape(-1, 1, h2, w2))[0]
    return out / out.sum(dim=(-2, -1), keepdim=True)


# ---------------------------------------------------------------------------
# convolution and compositing
# ---------------------------------------------------------------------------
def conv_same(img, kernel):
    """
    'Same'-size convolution of img (C, H, W) with kernel (Ck, kh, kw), Ck in {1, C},
    odd kh / kw, via FFT with replicate padding.
    """
    if is_delta(kernel):
        return img
    C, H, W = img.shape
    kh, kw = kernel.shape[-2:]
    ph, pw = kh // 2, kw // 2
    x = F.pad(img.unsqueeze(0), (pw, pw, ph, ph), mode="replicate")[0]
    hs, ws = x.shape[-2] + kh - 1, x.shape[-1] + kw - 1
    k = kernel.reshape(-1, kh, kw).to(img.dtype)
    if k.shape[0] == 1 and C > 1:
        k = k.expand(C, kh, kw)
    y = torch.fft.irfft2(torch.fft.rfft2(x, s=(hs, ws)) * torch.fft.rfft2(k, s=(hs, ws)), s=(hs, ws))
    return y[:, 2 * ph:2 * ph + H, 2 * pw:2 * pw + W]


def composite_over(acc, premult, alpha, kernel):
    """
    Composite one blurred layer over the accumulated image (back to front).
    premult: (3, H, W) premultiplied colour, alpha: (1, H, W), kernel: (1 or 3, kh, kw).
    """
    if is_delta(kernel):
        return premult + (1.0 - alpha) * acc
    if kernel.reshape(-1, *kernel.shape[-2:]).shape[0] == 1:
        bc = conv_same(premult, kernel)
        ba = conv_same(alpha, kernel)
    else:   # per-channel (wavelength-dependent) kernels: blur alpha per channel too
        bc = conv_same(premult, kernel)
        ba = conv_same(alpha.expand(3, -1, -1).contiguous(), kernel)
    return bc + (1.0 - ba.clamp(0.0, 1.0)) * acc


# ---------------------------------------------------------------------------
# layered rendering
# ---------------------------------------------------------------------------
@torch.no_grad()
def layer_edges(gaussians, cam, n_layers, near=0.2, q=(0.001, 0.999), max_samples=2_000_000):
    """
    Inverse-depth layer edges for one view (far -> near) and the layer index of
    every Gaussian centre. Centres outside the [q0, q1] quantile range of the
    visible centres go to the first / last layer.
    """
    xyz = gaussians.get_xyz
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    z = (xyz_h @ cam.world_view_transform)[:, 2]
    clip = xyz_h @ cam.full_proj_transform
    ndc = clip[:, :2] / (clip[:, 3:4] + 1e-7)
    vis = (z > near) & (ndc.abs() < 1.0).all(dim=1)
    inv = 1.0 / z.clamp_min(near)
    sample = inv[vis] if vis.any() else inv
    if sample.numel() > max_samples:
        sample = sample[torch.randperm(sample.numel(), device=sample.device)[:max_samples]]
    lo, hi = torch.quantile(sample.float(), q[0]).item(), torch.quantile(sample.float(), q[1]).item()
    if hi <= lo:
        hi = lo + 1e-6
    edges = torch.linspace(lo, hi, n_layers + 1, device=inv.device)
    idx = torch.bucketize(inv, edges[1:-1])          # 0 = farthest layer
    return edges, idx


@torch.no_grad()
def render_layered(cam, gaussians, render_aif, focus_distance, thetas, kernel_fn, n_layers, background,
                   tone_map=None):
    """
    Render one view through n slit angles (or one disk) with layered PSF convolution.

    render_aif(cam, bg) -> (3, H, W): render without defocus and without tone mapping.
    thetas: slit angles in radians (one entry per output image).
    kernel_fn(d, theta) -> kernel (1 or 3, kh, kw) for defocus d = |1/z - 1/F|.
    background: (3,) background colour.
    Returns (list of (3, H, W) images, info dict).
    """
    edges, idx = layer_edges(gaussians, cam, n_layers)
    raw = gaussians._opacity.data.clone()
    black = torch.zeros(3, device=raw.device)
    white = torch.ones(3, device=raw.device)
    outs, total = None, None
    try:
        for layer in range(n_layers):                         # far -> near
            sel = idx == layer
            if not bool(sel.any()):
                continue
            gaussians._opacity.data.copy_(raw)
            gaussians._opacity.data[~sel] = -1e4             # sigmoid -> 0: hide other layers
            cb = render_aif(cam, black)
            cw = render_aif(cam, white)
            alpha = (1.0 - (cw - cb)).mean(dim=0, keepdim=True).clamp(0.0, 1.0)
            if outs is None:
                bg = background.reshape(3, 1, 1).expand_as(cb).contiguous()
                outs = [bg.clone() for _ in thetas]
                total = torch.zeros_like(alpha)
            inv_c = 0.5 * float(edges[layer] + edges[layer + 1])
            d = abs(inv_c - 1.0 / focus_distance)
            for j, th in enumerate(thetas):
                outs[j] = composite_over(outs[j], cb, alpha, kernel_fn(d, th))
            total = alpha + (1.0 - alpha) * total
    finally:
        gaussians._opacity.data.copy_(raw)
    if outs is None:   # nothing visible
        bg = background.reshape(3, 1, 1).expand(3, cam.image_height, cam.image_width).contiguous()
        outs = [bg.clone() for _ in thetas]
    if tone_map is not None:
        outs = [tone_map(o) for o in outs]
    info = dict(inv_depth_min=float(edges[0]), inv_depth_max=float(edges[-1]),
                layer_step=float(edges[1] - edges[0]))
    return outs, info


def make_kernel_fn(psf, aperture, K_long, K_short, coeff, ss=8, optical=None):
    """
    kernel_fn(d, theta) for render_layered.

    psf: 'box' (box slit / uniform disk) or 'gaussian_layered' (rasterizer's Gaussian).
    aperture: 'slit' or 'disk' (for a disk K_short = K_long = K_D).
    coeff: Gaussian variance coefficient (1/3 slit, 1/4 disk) for 'gaussian_layered'.
    optical: optional callable theta -> (3, h, w) in-focus diffraction kernels.
    Kernels are cached per (d, theta) rounded to 1e-6.
    """
    cache, opt_cache = {}, {}

    def fn(d, theta):
        th = theta if aperture == "slit" else 0.0
        key = (round(d, 6), round(th, 6))
        if key in cache:
            return cache[key]
        length, width = K_long * d, K_short * d
        if psf == "box":
            g = box_kernel(length, width, th, ss) if aperture == "slit" else disk_kernel(length, ss)
        elif psf == "gaussian_layered":
            sl = math.sqrt(coeff) * 0.5 * length          # same variances as the rasterizer
            sw = math.sqrt(coeff) * 0.5 * width
            g = gaussian_kernel(sl, sw, th, ss)
        else:
            raise ValueError(f"unknown psf '{psf}'")
        if optical is not None:
            okey = round(th, 6)
            if okey not in opt_cache:
                opt_cache[okey] = optical(th)
            g = compose_kernels(g, opt_cache[okey])
        cache[key] = g.cuda() if torch.cuda.is_available() else g
        return cache[key]

    return fn
