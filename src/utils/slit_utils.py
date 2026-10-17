"""
Helpers for the rotating slit-aperture extension of DOF-GS.

Angle convention
----------------
Angles are the orientation of the slit LONG axis as it appears in the image:
x to the right, y downwards, measured from +x towards +y. Because the slit
rotates about the optical axis, the PSF is centrally symmetric, so theta and
theta + 180 deg give the same PSF.

Angle file
----------
A JSON file mapping image names (file name without extension, i.e. the same
`image_name` used by the Camera objects) to angles in degrees:

    {
        "IMG_0001": 0.0,
        "IMG_0002": 22.5,
        ...
    }
"""

import json
import math
import os
import re

import numpy as np
import torch


def default_slit_angle_file(source_path):
    return os.path.join(source_path, "slit_angles.json")


def _angle_from_name(image_name, regex):
    m = re.search(regex, image_name)
    if m is None:
        return None
    return float(m.group(1))


def load_slit_angles(image_names, angle_file="", source_path="", name_regex=""):
    """
    Build the per-view slit angle table, in the same order as `image_names`
    (which must follow the view-index order used for F / K, i.e. the order of
    `generate_F_cam_dict`).

    Angles are read from `angle_file` (JSON, degrees). If no file is given,
    `<source_path>/slit_angles.json` is tried. Images missing from the file can
    fall back to `name_regex`, whose first capture group must be the angle in
    degrees (e.g. r"_ang(-?\\d+(?:\\.\\d+)?)").

    Returns a (N_view, 1) float tensor of angles in radians (on CPU).
    """
    if not angle_file and source_path:
        angle_file = default_slit_angle_file(source_path)

    table = {}
    if angle_file and os.path.isfile(angle_file):
        with open(angle_file, "r") as f:
            table = {str(k): float(v) for k, v in json.load(f).items()}

    angles_deg = []
    missing = []
    for name in image_names:
        if name in table:
            angles_deg.append(table[name])
            continue
        ang = _angle_from_name(name, name_regex) if name_regex else None
        if ang is None:
            missing.append(name)
            angles_deg.append(0.0)
        else:
            angles_deg.append(ang)

    if missing:
        raise ValueError(
            f"Slit angle not found for {len(missing)} view(s), e.g. {missing[:5]}. "
            f"Provide them in '{angle_file}' or set slit_angle_regex."
        )

    thetas = torch.tensor([[math.radians(a)] for a in angles_deg], dtype=torch.float32)
    return thetas


def _pose_key(cam, decimals=5):
    """Hashable key of a camera's extrinsics (R, T), robust to float noise."""
    R = np.round(np.asarray(cam.R, dtype=np.float64), decimals) + 0.0   # + 0.0 turns -0.0 into 0.0
    T = np.round(np.asarray(cam.T, dtype=np.float64), decimals) + 0.0
    return tuple(R.reshape(-1).tolist()) + tuple(T.reshape(-1).tolist())


def group_views_by_pose(cameras):
    """
    Group cameras that share the same pose (identical R, T), e.g. the n slit
    orientations captured from one viewpoint. Grouping uses the extrinsics, not
    the file names, so it works for any naming scheme.

    Returns a list of groups in first-appearance order; each group is a list of
    (index_in_input, camera) pairs.
    """
    groups = {}
    for i, cam in enumerate(cameras):
        groups.setdefault(_pose_key(cam), []).append((i, cam))
    return list(groups.values())
