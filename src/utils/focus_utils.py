"""
Focus groups and sharing of the defocus parameters.

A *focus group* is a set of images captured with one lens focus setting, e.g.
the n slit angles taken at one camera position without refocusing, or the
capture(s) of a disk at one position. The focus distance F is shared within a
focus group; the aperture scale K belongs to the lens and is shared by all
images (by default).

Sharing modes
    share_F: 'global' (one F for all images), 'group' (one F per focus group,
             default) or 'view' (one F per image, the original DOF-GS behaviour)
    share_K: 'global' (one K for all images, default) or 'view' (one K per image)

Each mode is turned into an index tensor that maps every image (view index) to
its parameter slot; GaussianModel stores one parameter per slot and gathers
them with the index.
"""
import json
import os
import re
import zlib

import torch

F_MODES = ("global", "group", "view")
K_MODES = ("global", "view")

_ANGLE_SUFFIX = re.compile(r"_a\d+$")


def pose_stem(image_name):
    """Name of the camera pose of a simulated image (<pose>_aXX -> <pose>)."""
    return _ANGLE_SUFFIX.sub("", os.path.splitext(image_name)[0])


def resolve_share_modes(share_F, share_K, legacy_F=None, legacy_K=None):
    """
    Effective (F mode, K mode). The legacy booleans slit_share_F / slit_share_K
    of older configs take precedence when set (True -> 'global', False -> 'view'),
    so that runs trained with them are reproduced.
    """
    f = ("global" if legacy_F else "view") if isinstance(legacy_F, bool) else str(share_F).lower()
    k = ("global" if legacy_K else "view") if isinstance(legacy_K, bool) else str(share_K).lower()
    if f not in F_MODES:
        raise ValueError(f"share_F must be one of {F_MODES}, got '{share_F}'")
    if k not in K_MODES:
        raise ValueError(f"share_K must be one of {K_MODES}, got '{share_K}'")
    return f, k


def load_focus_groups(path):
    """{image name (without extension) -> focus group name} from a JSON file, or None."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {os.path.splitext(str(k))[0]: str(v) for k, v in raw.items()}


def build_view_groups(view_names, group_of_name):
    """
    Contiguous focus-group ids (0..G-1, in order of first appearance) for the
    views in view-index order. Returns (LongTensor (N_view,), list of group names).
    """
    names, ids, index = [], [], {}
    missing = [n for n in view_names if n not in group_of_name]
    if missing:
        raise KeyError(f"{len(missing)} image(s) have no focus group, e.g. {missing[:3]}")
    for n in view_names:
        g = group_of_name[n]
        if g not in index:
            index[g] = len(names)
            names.append(g)
        ids.append(index[g])
    return torch.tensor(ids, dtype=torch.long), names


def param_index(mode, groups):
    """(N_view,) parameter slot of every view for a sharing mode."""
    n = groups.numel()
    if mode == "global":
        return torch.zeros(n, dtype=torch.long)
    if mode == "group":
        return groups.clone().long()
    if mode == "view":
        return torch.arange(n, dtype=torch.long)
    raise ValueError(f"unknown sharing mode '{mode}'")


def slot_means(values, index):
    """Mean of per-view values over the views of every slot (index from param_index)."""
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    n = int(index.max()) + 1
    sums = torch.zeros(n, dtype=torch.float64).index_add_(0, index, values)
    cnts = torch.zeros(n, dtype=torch.float64).index_add_(0, index, torch.ones_like(values))
    return (sums / cnts.clamp_min(1)).float()


def pose_focus_fraction(scene, pose, seed, lo, hi):
    """
    Deterministic quantile in [lo, hi] for the focus of one pose. It depends only
    on (scene, pose name, seed), so a slit and a disk dataset of the same scene
    get identical focus distances.
    """
    h = zlib.crc32(f"{scene}|{pose}|{seed}".encode()) & 0xFFFFFFFF
    return lo + (hi - lo) * (h / 0xFFFFFFFF)
