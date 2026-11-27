"""Tests for build_track_dataset.py and the track-target loss in lerobot_relative_pose.py.

* The builder's cleaning, filling, shifting and layout are pure numpy.
* The loss patch is checked against lerobot 0.6.1's own multi_task_dit loss
  functions (skipped without lerobot): with weight 1 and every point visible
  it returns the same number, and a dataset without track targets installs
  nothing at all.
* One end-to-end build on a small lerobot dataset made here (skipped without
  lerobot).

Run: python -m pytest tests/test_track_targets.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "crisp_gym" / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


btd = _load("build_track_dataset")
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_LEROBOT = importlib.util.find_spec("lerobot") is not None
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
needs_lerobot = pytest.mark.skipif(not HAVE_LEROBOT, reason="needs lerobot")


# ── cleaning ──────────────────────────────────────────────────────────────────


def test_a_jump_hides_the_point_for_pad_frames_either_side():
    T = 20
    tracks = np.zeros((T, 2, 2))
    tracks[:, 0, 0] = np.arange(T)  # smooth: 1 px a frame
    tracks[:, 1, 0] = np.arange(T)
    tracks[10:, 1, 0] += 50  # point 1 jumps 51 px at frame 10
    visible = np.ones((T, 2), bool)
    out, hidden = btd.hide_jumps(tracks, visible, image_w=640, fraction=0.05, pad=3)  # limit 32 px
    assert out[:, 0].all()
    assert np.flatnonzero(~out[:, 1]).tolist() == [7, 8, 9, 10, 11, 12, 13]
    assert hidden[:, 1].sum() == 7 and not hidden[:, 0].any()


def test_a_jump_between_frames_already_hidden_is_not_counted():
    tracks = np.zeros((6, 1, 2))
    tracks[3:, 0, 0] = 100
    visible = np.ones((6, 1), bool)
    visible[3, 0] = False
    out, hidden = btd.hide_jumps(tracks, visible, image_w=640, fraction=0.05, pad=1)
    assert not hidden.any()


def test_a_flicker_breaks_rigidity_and_both_points_are_hidden_there():
    T = 30
    t = np.arange(T, dtype=float)
    a = np.stack([100 + 3 * t, 200 + t], 1)
    b = a + np.stack([10 + 0.3 * t, np.zeros(T)], 1)  # distance grows smoothly 10 -> 18.7 px
    b[15] += [25.0, 0.0]  # one bad frame
    tracks = np.stack([a, b], 1)
    out, hidden = btd.hide_rigid_breaks(
        tracks, np.ones((T, 2), bool), [[0, 1]], rel=0.35, abs_px=8.0
    )
    assert np.flatnonzero(~out[:, 0]).tolist() == [15] and np.flatnonzero(~out[:, 1]).tolist() == [
        15
    ]


def test_a_smooth_change_of_distance_is_not_a_break():
    T = 40
    t = np.arange(T, dtype=float)
    a = np.stack([100 + 2 * t, 200 + t], 1)
    b = a + np.stack([20 + 1.0 * t, np.zeros(T)], 1)  # the camera closes in: 20 -> 59 px
    out, _ = btd.hide_rigid_breaks(np.stack([a, b], 1), np.ones((T, 2), bool), [[0, 1]], 0.35, 8.0)
    assert out.all()


def test_points_are_grouped_by_the_object_named_before_the_last_underscore():
    assert btd.default_groups(["button_a", "button_b", "door_a", "door_b", "lid"]) == [
        [0, 1],
        [2, 3],
    ]
    assert btd.default_groups(["button", "door_a"]) == []


# ── the track dims of one episode ─────────────────────────────────────────────


def test_a_hidden_point_holds_its_last_visible_position():
    xy = np.stack([np.arange(8.0), np.zeros(8)], 1)
    vis = np.array([0, 0, 1, 1, 0, 0, 1, 0], bool)
    out = btd.fill_hidden(xy, vis, centre=(99, 99))
    assert out[:, 0].tolist() == [
        2,
        2,
        2,
        3,
        3,
        3,
        6,
        6,
    ]  # before first seen: the first visible one
    assert btd.fill_hidden(xy, np.zeros(8, bool), (5.0, 7.0)).tolist() == [[5.0, 7.0]] * 8


def test_track_dims_are_normalized_shifted_one_row_and_clamped_at_the_end():
    T, H, W = 5, 401, 641
    tracks = np.zeros((T, 1, 2))
    tracks[:, 0] = np.stack([np.arange(T) * 160.0, np.arange(T) * 100.0], 1)
    vis = np.ones((T, 1), bool)
    vis[2, 0] = False
    dims = btd.episode_track_dims(tracks, vis, (H, W), row_shift=1)
    assert dims.shape == (5, 3) and dims.dtype == np.float32
    np.testing.assert_allclose(dims[0], [160 / 640, 100 / 400, 1.0])  # row 0 = image 1
    np.testing.assert_allclose(
        dims[1], [160 / 640, 100 / 400, 0.0]
    )  # image 2 hidden: held at image 1
    np.testing.assert_allclose(dims[3], [640 / 640, 400 / 400, 1.0])  # row 3 = image 4
    np.testing.assert_allclose(dims[4], dims[3])  # last row repeats the last image
    np.testing.assert_allclose(
        btd.episode_track_dims(tracks, vis, (H, W), 0)[1], [160 / 640, 100 / 400, 1.0]
    )


def test_the_action_layout_keeps_pose_first_and_gripper_last():
    names = ["x", "y", "z"] + [f"rot6d_{i}" for i in range(6)] + ["gripper"]
    lay = btd.action_layout(names, ["button_a", "door_a"])
    assert lay["action_names"][:9] == names[:9] and lay["action_names"][-1] == "gripper"
    assert lay["action_names"][9:15] == [
        "button_a_x",
        "button_a_y",
        "button_a_visible",
        "door_a_x",
        "door_a_y",
        "door_a_visible",
    ]
    assert lay["track_dims"] == list(range(9, 15)) and lay["gripper_dim"] == 15
    assert lay["position_dims"] == {"button_a": [9, 10], "door_a": [12, 13]}
    assert lay["visibility_dims"] == {"button_a": 11, "door_a": 14}
    with pytest.raises(SystemExit, match="10-D action"):
        btd.action_layout(names + ["extra"], ["p"])


def test_stats_have_the_keys_lerobot_stores():
    v = np.arange(20, dtype=float).reshape(10, 2)
    s = btd.stats_of(v, ["min", "max", "mean", "std", "count", "q01", "q50", "q99"])
    assert (
        s["count"].tolist() == [10]
        and s["min"].tolist() == [0, 1]
        and s["max"].tolist() == [18, 19]
    )
    np.testing.assert_allclose(s["q50"], np.median(v, 0))


def _write_tracks(folder: Path, episodes, T=12, hw=(400, 640), points=("button_a", "button_b")):
    folder.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        tracks = np.zeros((T, len(points), 2), np.float32)
        tracks[:, 0] = np.stack([np.arange(T) * 10.0 + ep, np.full(T, 200.0)], 1)
        tracks[:, 1] = tracks[:, 0] + [20.0, 0.0]
        visible = np.ones((T, len(points)), bool)
        np.savez(
            folder / f"episode_{ep:06d}.npz",
            tracks=tracks,
            visible=visible,
            visibility=visible.astype(np.float32),
            query_frames=np.zeros(len(points), np.int32),
            query_xy=tracks[0],
            points=np.asarray(points),
            image_size=np.asarray(hw),
        )


def test_new_actions_are_placed_by_row_id_whatever_the_row_order(tmp_path):
    _write_tracks(tmp_path / "tracks", [0, 1])
    T = 12
    ep = np.repeat([0, 1], T)
    fr = np.tile(np.arange(T), 2)
    idx = np.arange(2 * T)
    action = np.tile(np.arange(10, dtype=float), (2 * T, 1)) + idx[:, None] * 100
    order = np.random.default_rng(0).permutation(2 * T)  # rows shuffled
    cols = {
        "action": action[order],
        "episode_index": ep[order],
        "frame_index": fr[order],
        "index": idx[order],
    }
    new, layout, report, points, hw = btd.compute_new_actions(
        cols,
        tmp_path / "tracks",
        row_shift=1,
        jump_fraction=0.05,
        jump_pad=3,
        rigid=True,
        rigid_groups=None,
        rigid_rel=0.35,
        rigid_abs=8.0,
    )
    assert new.shape == (24, 16) and points == ["button_a", "button_b"] and hw == (400, 640)
    np.testing.assert_array_equal(new[:, :9], action[:, :9])
    np.testing.assert_array_equal(new[:, -1], action[:, 9])
    row = 12 + 3  # episode 1, row 3 -> image 4
    np.testing.assert_allclose(new[row, 9:12], [(4 * 10.0 + 1) / 639, 200 / 399, 1.0], rtol=1e-6)
    assert report["button_a"]["frames"] == 24 and report["button_a"]["visible_in_target"] == 24


def test_episodes_without_tracks_are_refused_by_name(tmp_path):
    _write_tracks(tmp_path / "tracks", [0])
    cols = {
        "action": np.zeros((24, 10)),
        "episode_index": np.repeat([0, 1], 12),
        "frame_index": np.tile(np.arange(12), 2),
        "index": np.arange(24),
    }
    with pytest.raises(SystemExit, match=r"no tracks for episode\(s\) \[1\]"):
        btd.compute_new_actions(
            cols,
            tmp_path / "tracks",
            row_shift=1,
            jump_fraction=0.05,
            jump_pad=3,
            rigid=True,
            rigid_groups=None,
            rigid_rel=0.35,
            rigid_abs=8.0,
        )


# ── the loss patch ────────────────────────────────────────────────────────────

D = 16  # [pose 9, button_a x y vis, button_b x y vis, gripper]
STAMP = {
    "action_names": [f"a{i}" for i in range(D)],
    "points": ["button_a", "button_b"],
    "track_dims": list(range(9, 15)),
    "position_dims": {"button_a": [9, 10], "button_b": [12, 13]},
    "visibility_dims": {"button_a": 11, "button_b": 14},
}
UNIT = {"min": np.zeros(D), "max": np.ones(D)}


@pytest.fixture
def lrp():
    if not HAVE_TORCH:
        pytest.skip("needs torch")
    mod = _load("lerobot_relative_pose")
    yield mod
    mod.uninstall_track_loss()


@needs_torch
def test_visibility_is_read_back_through_lerobots_min_max(lrp):
    import torch

    tl = lrp.TrackLoss(STAMP, np.zeros(D), np.ones(D), 0.5)
    a = torch.full((1, 2, D), -1.0)  # normalized -1 = raw 0 = hidden
    a[0, 1, 11] = 1.0  # button_a visible in the second step
    assert tl.visible(a).tolist() == [[[False, False, False, False], [True, True, False, False]]]
    # a visible dim that is constant over the dataset: lerobot maps it to -1, its value decides
    always = lrp.TrackLoss(STAMP, np.r_[np.zeros(11), 1, 0, 0, 0, 0], np.ones(D), 0.5)
    assert always.visible(torch.full((1, 1, D), -1.0))[0, 0].tolist() == [True, True, False, False]


@needs_torch
def test_the_loss_weights_track_dims_and_skips_hidden_positions(lrp):
    import torch

    w = 0.25
    tl = lrp.TrackLoss(STAMP, *UNIT.values(), w)
    loss = torch.rand(2, 3, D)
    a = torch.full((2, 3, D), 1.0)
    a[..., 14] = -1.0  # button_b hidden everywhere
    weights = torch.ones(D)
    weights[9:15] = w
    keep = torch.ones(D)
    keep[[12, 13]] = 0.0  # its x, y out; its visible dim stays in
    expect = (loss * weights * keep).sum() / (weights * keep).sum().mul(6)
    torch.testing.assert_close(tl.reduce(loss, a), expect)
    # a hidden position's value cannot move the loss
    loss2 = loss.clone()
    loss2[..., [12, 13]] = 123.0
    torch.testing.assert_close(tl.reduce(loss2, a), tl.reduce(loss, a))


@needs_torch
def test_with_weight_one_and_everything_visible_it_is_the_plain_mean(lrp):
    import torch

    tl = lrp.TrackLoss(STAMP, *UNIT.values(), 1.0)
    loss = torch.rand(3, 4, D)
    a = torch.ones(3, 4, D)
    torch.testing.assert_close(tl.reduce(loss, a), loss.mean())
    pad = torch.zeros(3, 4, dtype=torch.bool)
    pad[0, 2:] = True
    m = (~pad).unsqueeze(-1).float()
    torch.testing.assert_close(tl.reduce(loss, a, pad), (loss * m).sum() / (m.sum() * D))


@needs_lerobot
@pytest.mark.parametrize("objective", ["DiffusionObjective", "FlowMatchingObjective"])
@pytest.mark.parametrize("prediction", ["epsilon", "sample"])
@pytest.mark.parametrize("pad_mask", [False, True])
def test_the_patched_loss_equals_lerobots_own_with_weight_one(lrp, objective, prediction, pad_mask):
    import torch
    from lerobot.policies.multi_task_dit import modeling_multi_task_dit as m
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig

    if objective == "FlowMatchingObjective" and prediction == "sample":
        pytest.skip("prediction type is a diffusion setting")
    cfg = MultiTaskDiTConfig(prediction_type=prediction)
    obj = getattr(m, objective)(cfg, D, 8, do_mask_loss_for_padding=pad_mask)
    net = torch.nn.Linear(D, D)
    model = lambda x, t, conditioning_vec=None: net(x)  # noqa: E731
    torch.manual_seed(0)
    act = torch.rand(3, 8, D) * 2 - 1
    act[..., [11, 14]] = 1.0
    pad = torch.zeros(3, 8, dtype=torch.bool)
    pad[1, 5:] = True
    batch = {"action": act, "action_is_pad": pad}
    torch.manual_seed(7)
    ref = obj.compute_loss(model, batch, None)
    lrp.install_track_loss("multi_task_dit", STAMP, UNIT, 1.0)
    assert (
        getattr(m, objective).compute_loss
        is not lrp._ORIGINAL_LOSSES[(getattr(m, objective), "compute_loss")]
    )
    torch.manual_seed(7)
    got = obj.compute_loss(model, batch, None)
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-7)


class _DS:
    def __init__(self, root):
        self.root = root


@needs_lerobot
def test_a_dataset_without_track_targets_installs_nothing(lrp, tmp_path, caplog):
    from lerobot.policies.multi_task_dit import modeling_multi_task_dit as m

    before = (m.DiffusionObjective.compute_loss, m.FlowMatchingObjective.compute_loss)
    stats = {"action": {"min": np.zeros(10), "max": np.ones(10)}}
    assert lrp.configure_track_targets(_DS(tmp_path), stats, "multi_task_dit", None, True) is None
    assert (m.DiffusionObjective.compute_loss, m.FlowMatchingObjective.compute_loss) == before
    assert lrp._TRACK_LOSS is None
    # flags given for such a dataset are ignored, loudly
    assert lrp.configure_track_targets(_DS(tmp_path), stats, "multi_task_dit", 0.5, True) is None
    assert "ignored" in caplog.text
    assert (m.DiffusionObjective.compute_loss, m.FlowMatchingObjective.compute_loss) == before


def _stamped(tmp_path):
    (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
    (tmp_path / "meta" / "track_targets.json").write_text(json.dumps(STAMP))
    return _DS(tmp_path)


@needs_lerobot
def test_a_stamped_dataset_installs_the_loss_and_records_it(lrp, tmp_path):
    from lerobot.policies.multi_task_dit import modeling_multi_task_dit as m

    rec = lrp.configure_track_targets(
        _stamped(tmp_path), {"action": UNIT}, "multi_task_dit", None, True
    )
    assert rec["loss_patch"] is True and rec["loss_weight"] == lrp.DEFAULT_TRACK_LOSS_WEIGHT
    assert m.DiffusionObjective.compute_loss is lrp._mtdit_diffusion_loss
    assert lrp._TRACK_LOSS.weight == lrp.DEFAULT_TRACK_LOSS_WEIGHT
    lrp.uninstall_track_loss()
    assert m.DiffusionObjective.compute_loss is not lrp._mtdit_diffusion_loss


@needs_lerobot
def test_the_patch_can_be_left_off_and_other_policies_are_refused(lrp, tmp_path):
    from lerobot.policies.multi_task_dit import modeling_multi_task_dit as m

    before = m.DiffusionObjective.compute_loss
    rec = lrp.configure_track_targets(
        _stamped(tmp_path), {"action": UNIT}, "multi_task_dit", None, False
    )
    assert rec["loss_patch"] is False and m.DiffusionObjective.compute_loss is before
    with pytest.raises(RuntimeError, match="implemented for multi_task_dit and groot, not 'diffusion'"):
        lrp.configure_track_targets(_stamped(tmp_path), {"action": UNIT}, "diffusion", None, True)
    with pytest.raises(RuntimeError, match="describes 16 action dims, the dataset's action has 10"):
        lrp.configure_track_targets(
            _stamped(tmp_path),
            {"action": {"min": np.zeros(10), "max": np.ones(10)}},
            "multi_task_dit",
            None,
            True,
        )


@needs_torch
def test_our_flags_are_taken_out_of_lerobots_arguments(lrp):
    assert lrp._pop_track_flags(["--steps=3", "--track-loss-weight", "0.5", "--x"]) == (
        0.5,
        True,
        ["--steps=3", "--x"],
    )
    assert lrp._pop_track_flags(["--track-loss-weight=0.1", "--no-track-loss-patch"]) == (
        0.1,
        False,
        [],
    )
    assert lrp._pop_track_flags(["--a"]) == (None, True, ["--a"])
    with pytest.raises(SystemExit):
        lrp._pop_track_flags(["--track-loss-weight"])


# ── end to end ────────────────────────────────────────────────────────────────


@needs_lerobot
def test_a_built_copy_loads_in_lerobot_with_the_wider_action(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = ["x", "y", "z"] + [f"rot6d_{i}" for i in range(6)]
    features = {
        "observation.state.cartesian": {"dtype": "float32", "shape": (9,), "names": names},
        "observation.state": {"dtype": "float32", "shape": (10,), "names": names + ["gripper"]},
        "action": {"dtype": "float32", "shape": (10,), "names": names + ["gripper"]},
    }
    src_root = tmp_path / "src"
    ds = LeRobotDataset.create(
        "local/src", fps=15, features=features, root=src_root, use_videos=False
    )
    T = 12
    for ep in range(2):
        for t in range(T):
            pose = np.r_[[0.1 * t + ep, 0.0, 0.3], [1, 0, 0, 0, 1, 0]].astype(np.float32)
            ds.add_frame(
                {
                    "observation.state.cartesian": pose,
                    "observation.state": np.r_[pose, 0.5].astype(np.float32),
                    "action": np.r_[pose, 0.25].astype(np.float32),
                    "task": "t",
                }
            )
        ds.save_episode()
    ds.finalize()
    (src_root / "meta" / "record_config.json").write_text("{}")
    _write_tracks(tmp_path / "tracks", [0, 1], T=T)

    stamp = btd.build(src_root, tmp_path / "tracks", tmp_path / "out")
    new = LeRobotDataset("local/out", root=tmp_path / "out")
    assert len(new) == 2 * T and list(new.meta.features["action"]["shape"]) == [16]
    assert new.meta.features["action"]["names"][9] == "button_a_x"
    row = new[T + 3]  # episode 1, row 3 -> image 4
    np.testing.assert_allclose(
        row["action"].numpy()[9:12], [(4 * 10.0 + 1) / 639, 200 / 399, 1.0], rtol=1e-5
    )
    assert float(row["action"][-1]) == pytest.approx(0.25)
    stats = json.loads((tmp_path / "out" / "meta" / "stats.json").read_text())["action"]
    assert len(stats["min"]) == 16
    assert (tmp_path / "out" / "meta" / "record_config.json").exists()
    on_disk = json.loads((tmp_path / "out" / "meta" / "track_targets.json").read_text())
    assert on_disk["track_dims"] == list(range(9, 15)) == stamp["track_dims"]
    with pytest.raises(SystemExit, match="exists already"):
        btd.build(src_root, tmp_path / "tracks", tmp_path / "out")


# ── GR00T N1.7 ────────────────────────────────────────────────────────────────


def _tiny_groot_head():
    """lerobot's real GR00T N1.7 action head, at a tiny random size."""
    import copy

    from lerobot.policies.groot import groot_n1_7 as g

    d = copy.deepcopy(g.GR00T_N1_7_DEFAULTS)
    small = dict(d)
    small["diffusion_model_cfg"] = {**d["diffusion_model_cfg"], "num_layers": 2, "num_attention_heads": 2,
                                    "attention_head_dim": 16, "output_dim": 64}
    small.update(hidden_size=64, input_embedding_dim=32, backbone_embedding_dim=48)
    if d.get("vl_self_attention_cfg"):
        small["vl_self_attention_cfg"] = {**d["vl_self_attention_cfg"], "num_layers": 1,
                                          "num_attention_heads": 2, "attention_head_dim": 24}
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    return g.GR00TN17ActionHead(g.GR00TN17Config(**small)).train(), small


def _groot_inputs(small, visible_b=True):
    import torch
    from transformers.feature_extraction_utils import BatchFeature

    B, L, H, Dmax = 2, 7, small["action_horizon"], small["max_action_dim"]
    torch.manual_seed(1)
    bb = BatchFeature({"backbone_features": torch.randn(B, L, 48),
                       "backbone_attention_mask": torch.ones(B, L, dtype=torch.bool),
                       "image_mask": torch.zeros(B, L, dtype=torch.bool)})
    act = torch.zeros(B, H, Dmax)
    act[..., :D] = torch.rand(B, H, D) * 2 - 1
    act[..., 11] = 1.0                                   # button_a visible
    act[..., 14] = 1.0 if visible_b else -1.0            # button_b visible or hidden
    mask = torch.zeros(B, H, Dmax)
    mask[:, :16, :D] = 1.0                               # GR00T's own horizon / dim padding
    ai = BatchFeature({"state": torch.randn(B, 1, small["max_state_dim"]), "action": act,
                       "action_mask": mask, "embodiment_id": torch.zeros(B, dtype=torch.long)})
    return bb, ai


@needs_lerobot
def test_the_groot_loss_with_weight_one_is_groots_own(lrp):
    """GR00T's head is not repeatable call to call even with one seed, so the
    comparison is within one call: GR00T's own loss, kept beside the new one."""
    import torch

    head, small = _tiny_groot_head()
    bb, ai = _groot_inputs(small)
    lrp.install_track_loss("groot", STAMP, UNIT, 1.0)
    out = head(bb, ai)
    torch.testing.assert_close(out["loss"], out["loss_unweighted"], rtol=0, atol=0)
    # and that kept number is GR00T's formula on its own outputs
    torch.testing.assert_close(out["loss_unweighted"],
                               out["action_loss"].sum() / (out["action_mask"].sum() + 1e-6), rtol=0, atol=0)


@needs_lerobot
def test_the_groot_loss_weights_track_dims_and_skips_hidden_positions(lrp):
    import torch

    head, small = _tiny_groot_head()
    bb, ai = _groot_inputs(small, visible_b=False)
    lrp.install_track_loss("groot", STAMP, UNIT, 0.25)
    torch.manual_seed(5)
    out = head(bb, ai)
    w = torch.ones(small["max_action_dim"])
    w[9:15] = 0.25
    keep = torch.ones_like(out["action_loss"])
    keep[..., [12, 13]] = 0.0                            # button_b hidden: its x, y out
    expect = (out["action_loss"] * w * keep).sum() / ((out["action_mask"] * w * keep).sum() + 1e-6)
    torch.testing.assert_close(out["loss"], expect)
    assert float(out["loss"]) != pytest.approx(float(out["action_loss"].sum() / (out["action_mask"].sum() + 1e-6)))


@needs_lerobot
def test_groot_with_its_own_relative_actions_is_refused_and_uninstall_restores(lrp):
    import types

    from lerobot.policies.groot import groot_n1_7 as g

    before = g.GR00TN17ActionHead.forward
    with pytest.raises(RuntimeError, match="use_relative_actions=false"):
        lrp.install_track_loss("groot", STAMP, UNIT, 0.25, policy_cfg=types.SimpleNamespace(use_relative_actions=True))
    assert g.GR00TN17ActionHead.forward is before
    lrp.install_track_loss("groot", STAMP, UNIT, 0.25, policy_cfg=types.SimpleNamespace(use_relative_actions=False))
    assert g.GR00TN17ActionHead.forward is not before
    lrp.uninstall_track_loss()
    assert g.GR00TN17ActionHead.forward is before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
