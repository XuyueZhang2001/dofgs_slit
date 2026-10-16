"""
Smoke test for the slit-aperture DOF-GS rasterizer (needs a CUDA GPU, no dataset).

Run from the repository root after reinstalling the rasterizer:
    python tests/test_slit_rasterizer.py

Checks
  1. Regression: Kw == K gives exactly the original isotropic result for any theta.
  2. Anisotropy: a defocused Gaussian is elongated along x at theta = 0 and
     along y at theta = 90 deg.
  3. Gradients: analytic gradients w.r.t. K, Kw, theta, F match finite differences.
"""
import math
import os
import torch
from diff_dof_gs_rasterizer import GaussianRasterizationSettings, GaussianRasterizer

dev = "cuda"
# DEBUG=1 makes the rasterizer synchronize after every kernel and raise the real CUDA error.
DEBUG = os.environ.get("DEBUG", "0") == "1"
H = W = 128
FOV = math.radians(50.0)
SLIT_COEFF = 1.0 / 3.0
DISK_COEFF = 0.36067376022224085


def projection_matrix(znear, zfar, fovx, fovy):
    # Same convention as 3DGS getProjectionMatrix (row-major, transposed later).
    t, r = math.tan(fovy / 2) * znear, math.tan(fovx / 2) * znear
    P = torch.zeros(4, 4)
    P[0, 0] = znear / r
    P[1, 1] = znear / t
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def make_scene(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    xyz = torch.empty(n, 3)
    xyz[:, 2] = torch.rand(n, generator=g) * 4.0 + 2.0          # depth in [2, 6]
    xyz[:, :2] = (torch.rand(n, 2, generator=g) - 0.5) * xyz[:, 2:3] * 0.6
    return dict(
        means3D=xyz.to(dev),
        colors=torch.rand(n, 3, generator=g).to(dev),
        opacities=torch.full((n, 1), 0.8).to(dev),
        scales=(torch.rand(n, 3, generator=g) * 0.02 + 0.01).to(dev),
        rotations=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1).to(dev),
    )


def render(scene, F, K, Kw=None, theta=None, coeff=SLIT_COEFF):
    view = torch.eye(4, device=dev)
    proj = projection_matrix(0.01, 100.0, FOV, FOV).T.to(dev)
    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=math.tan(FOV / 2), tanfovy=math.tan(FOV / 2),
        kernel_size=0.1,
        subpixel_offset=torch.zeros(H, W, 2, device=dev),
        bg=torch.zeros(3, device=dev), scale_modifier=1.0,
        viewmatrix=view, projmatrix=view @ proj,
        sh_degree=0, campos=torch.zeros(3, device=dev),
        prefiltered=False, debug=DEBUG, vIdx=0, coc2z_coeff=0.0,
        aperture_coeff=coeff)
    means2D = torch.zeros_like(scene["means3D"], requires_grad=True)
    out = GaussianRasterizer(settings)(
        means3D=scene["means3D"], means2D=means2D, opacities=scene["opacities"],
        colors_precomp=scene["colors"], scales=scene["scales"], rotations=scene["rotations"],
        F=F, K=K, Kw=Kw, theta=theta)
    return out[0]


def p(v, grad=False):
    return torch.tensor([[float(v)]], device=dev, requires_grad=grad)


# 1. Regression: Kw == K -> isotropic, identical to the original path, for any theta.
scene = make_scene(200)
ref = render(scene, p(3.0), p(40.0), coeff=DISK_COEFF)
for th in [0.0, 0.7, 2.0]:
    img = render(scene, p(3.0), p(40.0), Kw=p(40.0), theta=p(th), coeff=DISK_COEFF)
    assert torch.allclose(img, ref, atol=1e-5), f"isotropic mismatch at theta={th}"
print("1. regression (Kw == K reproduces original DOF-GS): OK")


# 2. Anisotropy: one Gaussian on the optical axis, far from the focal plane.
one = make_scene(1)
one["means3D"] = torch.tensor([[0.0, 0.0, 5.0]], device=dev)
one["opacities"] = torch.full((1, 1), 0.99, device=dev)
ys, xs = torch.meshgrid(torch.arange(H, device=dev).float(), torch.arange(W, device=dev).float(), indexing="ij")


def spread(img):
    w = img.sum(0)
    w = w / w.sum()
    cx, cy = (w * xs).sum(), (w * ys).sum()
    return (w * (xs - cx) ** 2).sum().item(), (w * (ys - cy) ** 2).sum().item()


vx0, vy0 = spread(render(one, p(2.0), p(200.0), Kw=p(20.0), theta=p(0.0)))
vx90, vy90 = spread(render(one, p(2.0), p(200.0), Kw=p(20.0), theta=p(math.pi / 2)))
print(f"   theta=0 : var_x={vx0:.1f} var_y={vy0:.1f}   theta=90: var_x={vx90:.1f} var_y={vy90:.1f}")
assert vx0 > 3 * vy0 and vy90 > 3 * vx90, "PSF is not elongated along the slit direction"
print("2. anisotropy (PSF follows the slit orientation): OK")


# 3. Gradient check against central finite differences.
scene = make_scene(150, seed=1)
wgt = torch.rand(3, H, W, device=dev, generator=torch.Generator(dev).manual_seed(2))
base = dict(F=3.0, K=60.0, Kw=15.0, theta=0.6)


def loss(vals, grad=False):
    t = {k: p(v, grad) for k, v in vals.items()}
    img = render(scene, t["F"], t["K"], Kw=t["Kw"], theta=t["theta"])
    return (img * wgt).sum(), t


L, t = loss(base, grad=True)
L.backward()
eps = dict(F=1e-2, K=0.5, Kw=0.5, theta=1e-2)
ok = True
for k in base:
    up, dn = dict(base), dict(base)
    up[k] += eps[k]
    dn[k] -= eps[k]
    with torch.no_grad():
        num = (loss(up)[0] - loss(dn)[0]).item() / (2 * eps[k])
    ana = t[k].grad.item()
    rel = abs(ana - num) / (abs(num) + 1e-3)
    flag = "ok" if rel < 0.05 else "MISMATCH"
    ok &= rel < 0.05
    print(f"   d/d{k:5s}: analytic={ana: .5e}  numeric={num: .5e}  rel.err={rel:.3f}  {flag}")
assert ok, "gradient mismatch (see above)"
print("3. gradients (K, Kw, theta, F): OK")
