"""check_policy_openloop.py: the point-track report and --track-sheets.

A checkpoint trained on a build_track_dataset.py copy predicts each point's
(x, y, visible) between pose and gripper. These tests check that the script
finds that layout, compares the tracks at the right timesteps (the copy stores
image t + row_shift in action row t), and draws them. Only numpy is needed;
the drawing tests also need opencv and are skipped without it.

The full script was run end to end on CPU with real lerobot checkpoints, with
and without tracks (see the commit message).

Run: python -m pytest tests/test_openloop_tracks.py -v
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "crisp_gym" / "scripts"


def _load_openloop():
    stub = types.ModuleType("lerobot_relative_pose")
    stub.POSE_ACTION_KEY = "action"
    saved = sys.modules.get("lerobot_relative_pose")
    sys.modules["lerobot_relative_pose"] = stub
    try:
        return SourceFileLoader(
            "check_policy_openloop_tracks_under_test", str(SCRIPTS / "check_policy_openloop.py")
        ).load_module()
    finally:
        if saved is None:
            sys.modules.pop("lerobot_relative_pose", None)
        else:
            sys.modules["lerobot_relative_pose"] = saved


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_track_dataset", SCRIPTS / "build_track_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_track_dataset", mod)
    spec.loader.exec_module(mod)
    return mod


ol = _load_openloop()
btd = _load_builder()

needs_cv2 = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None, reason="drawing needs opencv (lerobot brings it)"
)

POINTS = ["button_a", "door_a"]
IMAGE_HW = (400, 640)
LAYOUT = {
    **btd.action_layout([f"p{i}" for i in range(9)] + ["gripper"], POINTS),
    "points": POINTS,
    "image_size": list(IMAGE_HW),
    "row_shift": 1,
    "camera": "observation.images.oakw_cam",
}
D = len(LAYOUT["action_names"])  # 9 + 3 * 2 + 1 = 16


def _rows(xy, vis):
    """Action rows [K, D] holding pixel positions xy [K, N, 2] and visibility vis [K, N]."""
    H, W = IMAGE_HW
    rows = np.zeros((len(xy), D))
    for i, p in enumerate(POINTS):
        ix, iy = LAYOUT["position_dims"][p]
        rows[:, ix] = xy[:, i, 0] / (W - 1)
        rows[:, iy] = xy[:, i, 1] / (H - 1)
        rows[:, LAYOUT["visibility_dims"][p]] = vis[:, i]
    return rows


# ── finding the layout ────────────────────────────────────────────────────────


def _checkpoint(tmp_path, pose_repr):
    out = tmp_path / "run"
    ckpt = out / "checkpoints" / "000500" / "pretrained_model"
    ckpt.mkdir(parents=True)
    if pose_repr is not None:
        (out / "pose_repr.json").write_text(json.dumps(pose_repr))
    return ckpt


def test_the_layout_is_read_from_pose_repr_at_the_training_root(tmp_path):
    record = {k: LAYOUT[k] for k in ("points", "action_names", "position_dims",
                                     "visibility_dims", "image_size", "row_shift", "camera")}
    found = ol.find_track_layout(_checkpoint(tmp_path, {"action": {}, "track_targets": record}))
    assert found["points"] == POINTS and found["row_shift"] == 1
    # json turns the dict keys into strings; the dims stay usable as indices
    assert found["position_dims"]["door_a"] == LAYOUT["position_dims"]["door_a"]


@pytest.mark.parametrize(
    "pose_repr",
    [None, {"action": {"pose_repr": "relative"}}, {"track_targets": {"points": POINTS}}],
    ids=["no pose_repr.json", "a model without tracks", "an incomplete record"],
)
def test_no_layout_means_no_track_report(tmp_path, pose_repr):
    assert ol.find_track_layout(_checkpoint(tmp_path, pose_repr)) is None


def test_row_shift_defaults_to_the_builders_default(tmp_path):
    record = {k: LAYOUT[k] for k in ("points", "action_names", "position_dims",
                                     "visibility_dims", "image_size")}
    assert ol.find_track_layout(_checkpoint(tmp_path, {"track_targets": record}))["row_shift"] == 1


@pytest.mark.parametrize(
    ("model_dim", "dataset_dim", "stamp", "says"),
    [
        (D, D, {"action_names": LAYOUT["action_names"]}, None),
        (10, D, {"action_names": LAYOUT["action_names"]}, "predicts 10 action dims"),
        (D, 10, None, "point --repo-id at the track copy"),
        (D, D, None, "not a track copy"),
        (D, D, {"action_names": list(reversed(LAYOUT["action_names"]))}, "differs"),
    ],
    ids=["track copy", "model mismatch", "source dataset", "no stamp", "other layout"],
)
def test_the_report_runs_only_on_the_track_copy_it_was_trained_on(model_dim, dataset_dim, stamp, says):
    problem = ol.track_layout_problem(LAYOUT, model_dim, dataset_dim, stamp)
    assert problem is None if says is None else says in problem


def test_dataset_track_stamp_reads_the_copys_meta(tmp_path):
    (tmp_path / "meta").mkdir()
    assert ol.dataset_track_stamp(types.SimpleNamespace(root=tmp_path)) is None
    (tmp_path / "meta" / "track_targets.json").write_text(json.dumps({"action_names": ["x"]}))
    assert ol.dataset_track_stamp(types.SimpleNamespace(root=tmp_path)) == {"action_names": ["x"]}


class _Column:
    def __init__(self, values):
        self._values = values

    def to_pylist(self):
        return list(self._values)


class _FakeHF:
    def __init__(self, table_rows, full_rows):
        self.data = types.SimpleNamespace(column=lambda name: _Column(table_rows))
        self._full = full_rows

    def with_format(self, _):
        return {"action": self._full}


def test_dataset_column_reads_the_arrow_column_or_falls_back():
    rows = [[0.0, 1.0], [2.0, 3.0]]
    ds = type("DS", (), {"hf_dataset": _FakeHF(rows, None), "__len__": lambda self: 2})()
    np.testing.assert_array_equal(ol.dataset_column(ds, "action"), rows)
    # an Arrow table that does not line up with the rows: the HF column is used
    ds = type("DS", (), {"hf_dataset": _FakeHF(rows * 3, rows), "__len__": lambda self: 2})()
    np.testing.assert_array_equal(ol.dataset_column(ds, "action"), rows)


# ── the comparison ────────────────────────────────────────────────────────────


def test_track_pixels_undo_the_builders_normalisation():
    xy = np.array([[[0.0, 0.0], [639.0, 399.0]]])
    got_xy, got_vis = ol.track_pixels(_rows(xy, np.array([[1.0, 0.0]])), LAYOUT)
    np.testing.assert_allclose(got_xy, xy)
    np.testing.assert_array_equal(got_vis, [[1.0, 0.0]])


def test_chunk_and_current_position_line_up_with_the_images_the_builder_stored():
    """Action row t holds image t + 1: the chunk from row t is images t+1.., 'now' is image t."""
    T, t, K = 40, 10, 5
    # point 0 sits at (image index, 2 x image index) in every image
    tracks = np.stack([np.stack([np.arange(T), 2 * np.arange(T)], -1)] * 2, 1).astype(float)
    dims = btd.episode_track_dims(tracks, np.ones((T, 2), bool), IMAGE_HW, row_shift=1)
    episode = np.zeros((T, D))
    episode[:, LAYOUT["track_dims"]] = dims
    now_xy, now_vis = ol.track_now(episode, t, LAYOUT)
    np.testing.assert_allclose(now_xy[0], [t, 2 * t], atol=1e-3)
    assert now_vis.all()
    rec_xy, _ = ol.track_pixels(episode[t:t + K], LAYOUT)
    np.testing.assert_allclose(rec_xy[:, 0, 0], np.arange(t + 1, t + K + 1), atol=1e-3)
    # the recorded motion from 'now' at step k is k + 1 images
    m = ol.track_chunk_metrics(episode[t:t + K], episode[t:t + K], now_xy, LAYOUT)
    np.testing.assert_allclose(m["move"][0], np.hypot(1, 2) * np.arange(1, K + 1), atol=1e-2)
    np.testing.assert_allclose(m["err"], 0.0, atol=1e-9)


def test_the_first_frames_of_an_episode_have_no_current_position():
    episode = np.zeros((10, D))
    assert ol.track_now(episode, 0, LAYOUT) == (None, None)
    assert ol.track_now(episode, 1, LAYOUT)[0] is not None


def test_errors_count_only_where_the_tracker_saw_the_point():
    K = 4
    rec_xy = np.tile(np.array([[100.0, 100.0], [200.0, 50.0]]), (K, 1, 1))
    rec_vis = np.ones((K, 2))
    rec_vis[2, 1] = 0.0  # the tracker lost door_a at step 2
    pred_xy = rec_xy + np.array([3.0, 4.0])  # 5 px off everywhere
    pred_vis = np.ones((K, 2))
    pred_vis[2, 1] = 0.9  # ... but the model says it is visible
    pred_vis[3, 0] = 0.2  # and calls button_a hidden at step 3
    now = np.array([[100.0, 90.0], [200.0, 50.0]])
    m = ol.track_chunk_metrics(_rows(pred_xy, pred_vis), _rows(rec_xy, rec_vis), now, LAYOUT)
    assert m["err"].shape == (2, K)
    np.testing.assert_allclose(m["err"][0], 5.0, atol=1e-6)
    assert np.isnan(m["err"][1, 2]) and np.isnan(m["move"][1, 2])
    np.testing.assert_allclose(m["move"][0], 10.0, atol=1e-6)
    np.testing.assert_array_equal(m["agree"], [[1, 1, 1, 0], [1, 1, 0, 1]])
    np.testing.assert_array_equal(m["seen"], [[1, 1, 1, 1], [1, 1, 0, 1]])
    # without a current position nothing enters err / move; visibility still counts
    m = ol.track_chunk_metrics(_rows(pred_xy, pred_vis), _rows(rec_xy, rec_vis), None, LAYOUT)
    assert np.isnan(m["err"]).all() and np.isnan(m["move"]).all()
    assert np.isfinite(m["agree"]).all()


def test_the_summary_averages_err_and_move_over_the_same_entries():
    def frame(err, move, K):
        return {"err": np.full((2, K), err), "move": np.full((2, K), move),
                "agree": np.ones((2, K)), "seen": np.ones((2, K))}

    a, b = frame(2.0, 4.0, 3), frame(6.0, 8.0, 2)
    b["move"][1, 0] = np.nan
    b["err"][1, 0] = np.nan
    stacked = ol.stack_track_metrics([a, b])
    assert stacked["err"].shape == (2, 2, 3) and np.isnan(stacked["err"][1, :, 2]).all()
    s = ol.track_step_summary(stacked, 0)
    assert s["counted"] == 3
    assert s["err"] == pytest.approx((2 + 2 + 6) / 3) and s["move"] == pytest.approx((4 + 4 + 8) / 3)
    assert s["ratio"] == pytest.approx(s["err"] / s["move"])
    one = ol.track_step_summary(stacked, 0, point=1)
    assert one["counted"] == 1 and one["err"] == pytest.approx(2.0)
    # only the shorter chunk is missing at step 2
    assert ol.track_step_summary(stacked, 2)["counted"] == 2


def test_the_report_lists_every_step_and_point(caplog):
    m = {"err": np.full((2, 4), 3.0), "move": np.full((2, 4), 6.0),
         "agree": np.ones((2, 4)), "seen": np.ones((2, 4))}
    with caplog.at_level(logging.INFO):
        ol.log_track_report(ol.stack_track_metrics([m, m]), LAYOUT, exec_steps=3)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "2 points (button_a, door_a), in pixels of the 640x400 camera frame" in text
    assert text.count("<- re-observe") == 1
    assert "per point at step 2" in text and "button_a" in text and "door_a" in text
    assert "0.50" in text  # err/move


# ── drawing ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "shape"),
    [
        (np.ones((3, 8, 10)) * 0.5, (8, 10, 3)),             # CHW float, as lerobot gives it
        (np.zeros((2, 3, 8, 10), np.uint8), (8, 10, 3)),     # an n_obs window: the latest frame
        (np.zeros((8, 10, 3), np.uint8), (8, 10, 3)),        # already HWC
        (np.zeros((1, 8, 10)), (8, 10, 3)),                  # grey
    ],
)
def test_camera_frames_become_hwc_uint8(value, shape):
    img = ol.frame_rgb_uint8(value)
    assert img.shape == shape and img.dtype == np.uint8


def test_float_frames_are_scaled_to_0_255():
    assert ol.frame_rgb_uint8(np.ones((3, 4, 4)))[0, 0, 0] == 255


def _chunk(K=6):
    xy = np.stack([np.stack([100 + 20 * np.arange(K), np.full(K, 100.0)], -1),
                   np.stack([np.full(K, 500.0), 300 - 10 * np.arange(K)], -1)], 1)
    vis = np.ones((K, 2))
    return xy, vis


@needs_cv2
def test_a_tile_shows_the_recorded_and_predicted_paths():
    K = 6
    rec_xy, rec_vis = _chunk(K)
    pred_xy, pred_vis = rec_xy + np.array([0.0, 40.0]), rec_vis.copy()
    pred_vis[:, 1] = 0.1  # door_a predicted hidden throughout
    image = np.zeros((*IMAGE_HW, 3), np.uint8)
    now = rec_xy[0] - np.array([20.0, 0.0])
    tile = ol.draw_track_tile(image, _rows(pred_xy, pred_vis), _rows(rec_xy, rec_vis), now,
                              np.array([True, True]), LAYOUT, exec_steps=3, title="episode 0")
    H, W = IMAGE_HW
    assert tile.shape == (H + 60, W, 3)  # title bar above, two legend lines below
    body = tile[20:20 + H]
    colour = np.array(ol.TRACK_COLOURS[0])
    # button_a's recorded path runs along y = 100, its predicted one along y = 140
    assert (np.abs(body[100, 130:170].astype(int) - colour).sum(-1) < 60).any()
    assert (np.abs(body[140, 130:170].astype(int) - colour).sum(-1) < 60).any()
    # door_a is predicted hidden: grey crosses, no dashed line in its colour
    grey = np.array(ol.TRACK_HIDDEN_COLOUR)
    near = body[330:350, 490:510].astype(int)
    assert (np.abs(near - grey).sum(-1) < 30).any()


@needs_cv2
def test_small_frames_are_enlarged_and_points_stay_in_place():
    rec_xy, rec_vis = _chunk()
    small = np.zeros((96, 96, 3), np.uint8)
    tile = ol.draw_track_tile(small, _rows(rec_xy, rec_vis), _rows(rec_xy, rec_vis), None, None,
                              LAYOUT, exec_steps=2, title="t")
    assert tile.shape[1] == ol.TRACK_TILE_MIN_WIDTH
    # the first recorded point (100, 100) of a 640x400 frame, on a 480x480 tile
    x = round(100 * (480 - 1) / (640 - 1))
    y = round(100 * (480 - 1) / (400 - 1)) + 20
    patch = tile[y - 3:y + 4, x - 3:x + 4].astype(int)
    assert (np.abs(patch - np.array(ol.TRACK_COLOURS[0])).sum(-1) < 60).any()


@needs_cv2
def test_sheets_and_videos_are_written_per_episode(tmp_path):
    import cv2

    rec_xy, rec_vis = _chunk()
    rows = _rows(rec_xy, rec_vis)
    png = cv2.imencode(".png", np.zeros((*IMAGE_HW, 3), np.uint8))[1].tobytes()
    frames = [
        {"episode": ep, "frame": f, "row": 100 * ep + f, "image": png, "pred": rows, "rec": rows,
         "now_xy": rec_xy[0], "now_vis": np.array([True, False])}
        for ep in (0, 3) for f in range(0, 50, 5)
    ]
    written = ol.write_track_sheets(tmp_path / "sheets", frames, LAYOUT, exec_steps=3,
                                    per_sheet=4, video_fps=3.0)
    names = sorted(Path(w).name for w in written)
    assert "episode_000_tracks.png" in names and "episode_003_tracks.png" in names
    sheet = cv2.imread(str(tmp_path / "sheets" / "episode_003_tracks.png"))
    H, W = IMAGE_HW
    assert sheet.shape == (H + 60, 4 * W, 3)  # 4 of the 10 frames, one row
    for mp4 in (w for w in written if w.endswith(".mp4")):
        cap = cv2.VideoCapture(mp4)
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
        cap.release()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
