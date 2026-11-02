"""
CPU checks for src/utils/focus_utils.py (no GPU / dataset needed).

    python tests/test_focus_groups.py

1. Legacy flags: slit_share_F / slit_share_K booleans override share_F / share_K.
2. Focus groups: contiguous ids in order of first appearance; missing images fail.
3. Parameter indices for global / group / view and per-slot means of a ground truth.
4. Gathering one F per group gives every slit angle of a pose the same focus.
5. The per-pose focus fraction is deterministic, lies in the range, and is the
   same for a slit and a disk dataset of the same scene (depends on scene + pose).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.focus_utils import (build_view_groups, param_index, pose_focus_fraction, pose_stem,
                                   resolve_share_modes, slot_means)


def test_modes():
    assert resolve_share_modes("group", "global") == ("group", "global")
    assert resolve_share_modes("group", "global", True, True) == ("global", "global")      # old slit configs
    assert resolve_share_modes("group", "global", False, False) == ("view", "view")
    assert resolve_share_modes("view", "view", "", "") == ("view", "view")                 # unset legacy flags
    for bad in (("pose", "global"), ("group", "group")):
        try:
            resolve_share_modes(*bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    print("sharing modes ok")


def test_groups_and_indices():
    poses = ["p0", "p1", "p2"]
    views = [f"{p}_a{k:02d}" for p in poses for k in range(4)]          # 3 poses x 4 slit angles
    views = views[5:] + views[:5]                                       # arbitrary view order
    group_of = {v: pose_stem(v) for v in views}
    ids, names = build_view_groups(views, group_of)
    assert names[0] == pose_stem(views[0]) and len(names) == 3 and int(ids.max()) == 2
    for v, g in zip(views, ids.tolist()):
        assert names[g] == pose_stem(v)
    try:
        build_view_groups(views + ["extra_a00"], group_of)
        raise AssertionError("missing image accepted")
    except KeyError:
        pass

    assert param_index("global", ids).tolist() == [0] * len(views)
    assert param_index("group", ids).tolist() == ids.tolist()
    assert param_index("view", ids).tolist() == list(range(len(views)))

    # ground truth per view -> per-slot initial values
    gt = {"p0": 50.0, "p1": 60.0, "p2": 80.0}
    f_view = [gt[pose_stem(v)] for v in views]
    per_group = slot_means(f_view, param_index("group", ids))
    assert torch.allclose(per_group, torch.tensor([gt[n] for n in names]))
    assert abs(float(slot_means(f_view, param_index("global", ids))[0]) - sum(f_view) / len(f_view)) < 1e-5

    # gathering: all angles of a pose see the same F
    Fs = per_group.reshape(-1, 1)
    F_view = Fs[param_index("group", ids)].reshape(-1)
    for v, f in zip(views, F_view.tolist()):
        assert abs(f - gt[pose_stem(v)]) < 1e-5
    print("focus groups / indices ok")


def test_focus_fraction():
    lo, hi = 0.2, 0.8
    us = [pose_focus_fraction("real_cake", f"pose{i:03d}", 0, lo, hi) for i in range(200)]
    assert all(lo <= u <= hi for u in us)
    assert len(set(round(u, 9) for u in us)) > 190                      # varies between poses
    assert pose_focus_fraction("real_cake", "pose007", 0, lo, hi) == us[7]   # deterministic
    assert pose_focus_fraction("real_cake", "pose007", 1, lo, hi) != us[7]   # depends on the seed
    assert pose_focus_fraction("real_caps", "pose007", 0, lo, hi) != us[7]   # and on the scene
    assert pose_stem("1_a03") == "1" and pose_stem("img_a0_a12.png") == "img_a0"
    print("per-pose focus ok")


if __name__ == "__main__":
    test_modes()
    test_groups_and_indices()
    test_focus_fraction()
    print("all focus-group checks passed")
