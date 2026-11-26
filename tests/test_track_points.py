"""Tests for crisp_gym/scripts/track_points.py.

The tracker is replaced by a stand-in with RoboTracker's contract (positions
in tracking pixels, visibility zero before each query frame), and the video by
an in-memory one, so neither torch, lerobot nor a GPU is needed. The real
RoboTracker on a real lerobot dataset was run by hand; see the commit message.

Run: python -m pytest tests/test_track_points.py -v
"""

import csv
import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "track_points", REPO / "crisp_gym" / "scripts" / "track_points.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tp = _load()

IMAGE_HW = (800, 1280)
TRACK_HW = tp.TRACK_SIZE
POINTS = ["button_a", "door_a", "lid"]


def fake_track(velocity=(2.0, 1.0)):
    """RoboTracker's contract: [T, n, 2] in tracking pixels, visibility 0 before each query frame."""
    calls = []

    def track(frames, query_xy, query_frames):
        calls.append((len(frames), np.array(query_xy), np.array(query_frames)))
        T, n = len(frames), len(query_frames)
        pos = np.zeros((T, n, 2), np.float32)
        vis = np.zeros((T, n), np.float32)
        for j in range(n):
            q = int(query_frames[j])
            steps = np.clip(np.arange(T) - q, 0, None)[:, None]
            pos[:, j] = query_xy[j] + steps * np.asarray(velocity, np.float32)
            vis[q:, j] = 0.9
        return pos, vis

    track.calls = calls
    return track


def labels(button_frame=0, door_frame=4):
    return {
        "button_a": {"frame": button_frame, "x": 640.0, "y": 400.0},
        "door_a": {"frame": door_frame, "x": 1279.0, "y": 0.0},
        "lid": None,
    }


# ── coordinates ───────────────────────────────────────────────────────────────


def test_scaling_maps_first_and_last_pixel_centres_onto_each_other():
    corners = np.array([[0, 0], [1279, 799], [639.5, 399.5]], np.float32)
    small = tp.scale_xy(corners, IMAGE_HW, TRACK_HW)
    np.testing.assert_allclose(small[:2], [[0, 0], [511, 383]], atol=1e-4)
    np.testing.assert_allclose(small[2], [255.5, 191.5], atol=1e-4)  # the centre stays the centre
    np.testing.assert_allclose(tp.scale_xy(small, TRACK_HW, IMAGE_HW), corners, atol=1e-3)


def test_queries_keep_the_point_order_and_mark_not_visible_points():
    xy, frames = tp.episode_queries(labels(), POINTS)
    assert frames.tolist() == [0, 4, -1]
    np.testing.assert_array_equal(xy[:2], [[640, 400], [1279, 0]])
    assert np.isnan(xy[2]).all()


# ── one episode ───────────────────────────────────────────────────────────────


def test_tracks_come_back_in_camera_pixels():
    frames = np.zeros((10, *TRACK_HW, 3), np.uint8)
    xy, qf = tp.episode_queries(labels(), POINTS)
    track = fake_track(velocity=(2.0, 1.0))
    out = tp.track_episode(frames, xy, qf, IMAGE_HW, track)
    # the tracker saw the two visible points only, at the tracking size
    ((_, q_trk, q_frames),) = track.calls
    assert q_frames.tolist() == [0, 4]
    np.testing.assert_allclose(q_trk, tp.scale_xy(xy[:2], IMAGE_HW, TRACK_HW), atol=1e-4)
    # at the query frame the track is the click, in camera pixels
    np.testing.assert_allclose(out["tracks"][0, 0], [640, 400], atol=1e-3)
    np.testing.assert_allclose(out["tracks"][4, 1], [1279, 0], atol=1e-3)
    # one step later it moved by the tracker's step, scaled back to camera pixels
    step = tp.scale_xy(np.array([[2.0, 1.0]]), TRACK_HW, IMAGE_HW)[0]
    np.testing.assert_allclose(out["tracks"][1, 0] - out["tracks"][0, 0], step, atol=1e-3)
    assert out["tracks"].dtype == np.float32 and out["tracks"].shape == (10, 3, 2)


def test_before_its_click_a_point_is_not_visible_and_sits_at_the_click():
    frames = np.zeros((10, *TRACK_HW, 3), np.uint8)
    xy, qf = tp.episode_queries(labels(door_frame=4), POINTS)
    out = tp.track_episode(frames, xy, qf, IMAGE_HW, fake_track())
    assert not out["visible"][:4, 1].any() and out["visible"][4:, 1].all()
    np.testing.assert_allclose(out["tracks"][:4, 1], np.tile([1279, 0], (4, 1)), atol=1e-3)


def test_a_point_not_visible_in_the_episode_has_no_position():
    frames = np.zeros((6, *TRACK_HW, 3), np.uint8)
    xy, qf = tp.episode_queries(labels(), POINTS)
    out = tp.track_episode(frames, xy, qf, IMAGE_HW, fake_track())
    assert np.isnan(out["tracks"][:, 2]).all()
    assert not out["visible"][:, 2].any() and out["query_frames"][2] == -1


def test_an_episode_with_no_visible_point_never_calls_the_tracker():
    track = fake_track()
    frames = np.zeros((6, *TRACK_HW, 3), np.uint8)
    xy, qf = tp.episode_queries({p: None for p in POINTS}, POINTS)
    out = tp.track_episode(frames, xy, qf, IMAGE_HW, track)
    assert track.calls == [] and not out["visible"].any()


def test_a_click_past_the_end_of_the_episode_is_refused():
    frames = np.zeros((4, *TRACK_HW, 3), np.uint8)
    xy, qf = tp.episode_queries(labels(door_frame=4), POINTS)
    with pytest.raises(ValueError, match="past the end"):
        tp.track_episode(frames, xy, qf, IMAGE_HW, fake_track())


def test_the_summary_counts_visibility_from_the_click_and_flags_jumps():
    T = 10
    result = {
        "tracks": np.zeros((T, 2, 2), np.float32),
        "visible": np.ones((T, 2), bool),
        "query_frames": np.array([2, -1]),
    }
    result["visible"][:2, 0] = False  # before the click: not counted
    result["visible"][6:8, 0] = False  # two lost frames after it
    result["tracks"][5:, 0] = [300.0, 0.0]  # a jump of 300 px between frames 4 and 5
    rows = tp.summarize(result, ["button_a", "lid"], IMAGE_HW, jump_fraction=0.05)  # limit 64 px
    assert rows[0] == {"point": "button_a", "query_frame": 2, "visible_share": 0.75, "jumps": 1}
    assert rows[1] == {"point": "lid", "query_frame": -1, "visible_share": 0.0, "jumps": 0}


def test_an_episode_file_holds_the_documented_arrays(tmp_path):
    frames = np.zeros((5, *TRACK_HW, 3), np.uint8)
    xy, qf = tp.episode_queries(labels(), POINTS)
    out = tp.track_episode(frames, xy, qf, IMAGE_HW, fake_track())
    path = tmp_path / "episode_000012.npz"
    tp.save_episode(
        path,
        out,
        points=POINTS,
        image_hw=IMAGE_HW,
        track_hw=TRACK_HW,
        checkpoint="robotracker",
        episode=12,
    )
    with np.load(path) as d:
        assert set(d.files) == {
            "tracks",
            "visible",
            "visibility",
            "query_frames",
            "query_xy",
            "points",
            "image_size",
            "track_size",
            "checkpoint",
            "episode",
        }
        assert d["points"].tolist() == POINTS and d["image_size"].tolist() == list(IMAGE_HW)
        assert int(d["episode"]) == 12 and str(d["checkpoint"]) == "robotracker"
        assert d["visible"].dtype == bool
    assert list(tmp_path.iterdir()) == [path], "no temporary file left behind"


# ── the run ───────────────────────────────────────────────────────────────────


class FakeVideo:
    """Episodes 0..2 of 6 frames, already at whatever size is asked for."""

    def __init__(self, hw=IMAGE_HW):
        self.decoded = []
        self.source = types.SimpleNamespace(
            ranges={0: (0, 6), 1: (6, 12), 2: (12, 18)},
            frame=lambda ep, i: np.zeros((*hw, 3), np.uint8),
        )

    def frames(self, episode, sizes):
        self.decoded.append(episode)
        return [np.zeros((6, h, w, 3), np.uint8) for h, w in sizes]


def queries(episodes):
    return {
        "version": 1,
        "camera": "observation.images.oakw_cam",
        "image_size": list(IMAGE_HW),
        "points": POINTS,
        "episodes": {str(e): v for e, v in episodes.items()},
    }


def test_a_run_tracks_the_labelled_episodes_and_writes_the_summary(tmp_path, caplog):
    video, track = FakeVideo(), fake_track()
    q = queries({0: labels(door_frame=2), 2: {p: None for p in POINTS}, 7: labels()})
    rows = tp.run(video, q, track, tmp_path, checkpoint="robotracker")
    assert video.decoded == [0, 2]  # 7 is labelled but not in the dataset
    assert sorted(p.name for p in tmp_path.glob("*.npz")) == [
        "episode_000000.npz",
        "episode_000002.npz",
    ]
    with open(tmp_path / "summary.csv") as f:
        table = list(csv.DictReader(f))
    assert len(table) == len(rows) == 6
    assert table[0] == {
        "episode": "0",
        "point": "button_a",
        "query_frame": "0",
        "visible_share": "1.0",
        "jumps": "0",
    }
    assert any("not in the dataset" in r.getMessage() for r in caplog.records)


def test_a_second_run_continues_without_tracking_again(tmp_path):
    q = queries({0: labels(door_frame=2), 1: labels(door_frame=1)})
    tp.run(FakeVideo(), q, fake_track(), tmp_path, episodes=[0])
    video, track = FakeVideo(), fake_track()
    rows = tp.run(video, q, track, tmp_path)
    assert video.decoded == [1] and len(track.calls) == 1
    assert {r["episode"] for r in rows} == {0, 1}  # the summary still covers both
    again = FakeVideo()
    tp.run(again, q, fake_track(), tmp_path, overwrite=True)
    assert again.decoded == [0, 1]


def test_episodes_without_clicks_are_skipped_by_name(tmp_path, caplog):
    video = FakeVideo()
    tp.run(video, queries({0: labels(door_frame=2)}), fake_track(), tmp_path, episodes=[0, 1])
    assert video.decoded == [0]
    assert any("no clicks for episode(s) [1]" in r.getMessage() for r in caplog.records)


def test_frames_of_another_size_than_the_clicks_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="clicks were made on 1280x800 frames"):
        tp.run(FakeVideo(hw=(400, 640)), queries({0: labels(door_frame=2)}), fake_track(), tmp_path)


def test_a_preview_is_drawn_for_the_episodes_asked_for(tmp_path):
    pytest.importorskip("cv2")
    q = queries({0: labels(door_frame=2), 1: labels(door_frame=1)})
    tp.run(FakeVideo(), q, fake_track(), tmp_path, preview=[1])
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "preview_episode_000001.mp4" in names and "preview_episode_000001.png" in names
    assert "preview_episode_000000.png" not in names
    import cv2

    sheet = cv2.imread(str(tmp_path / "preview_episode_000001.png"))
    assert sheet.shape == (2 * 400, 4 * 640, 3)  # 8 frames, 640 px wide, aspect kept


# ── decoding ──────────────────────────────────────────────────────────────────


def _fake_dataset(rows, query_result):
    frames = [np.full((3, 8, 10), (r % 7) / 7.0, np.float32) for r in range(rows)]

    class Hf:
        data = types.SimpleNamespace(
            column=lambda name: types.SimpleNamespace(
                to_numpy=lambda: np.arange(rows, dtype=np.float64) / 15.0
            )
        )

    ds_obj = type(
        "DS", (), {"__getitem__": lambda self, i: {"cam": frames[i]}, "hf_dataset": Hf()}
    )()
    queried = []

    def _query_videos(ts, ep):
        queried.append(len(ts["cam"]))
        return {"cam": query_result(ts["cam"])}

    ds_obj.reader = types.SimpleNamespace(_query_videos=_query_videos)
    source = types.SimpleNamespace(ds=ds_obj, camera="cam", ranges={0: (0, rows)})
    return source, frames, queried


def test_the_fast_decoder_is_used_when_it_matches_dataset_rows():
    pytest.importorskip("cv2")
    rows = 20
    source, frames, queried = _fake_dataset(
        rows,
        lambda ts: np.stack(
            [np.full((3, 8, 10), (round(t * 15) % 7) / 7.0, np.float32) for t in ts]
        ),
    )
    (out,) = tp.EpisodeVideo(source, chunk=8).frames(0, [(8, 10)])
    assert out.shape == (rows, 8, 10, 3)
    assert [int(f[0, 0, 0]) for f in out] == [round((r % 7) / 7 * 255) for r in range(rows)]
    assert queried == [3, 8, 8, 4]  # the 3-frame check, then chunks of 8


def test_a_fast_decoder_that_disagrees_with_dataset_rows_is_not_used(caplog):
    pytest.importorskip("cv2")
    rows = 9
    source, frames, queried = _fake_dataset(
        rows, lambda ts: np.zeros((len(ts), 3, 8, 10), np.float32)
    )
    (out,) = tp.EpisodeVideo(source, chunk=4).frames(0, [(8, 10)])
    assert [int(f[0, 0, 0]) for f in out] == [round((r % 7) / 7 * 255) for r in range(rows)]
    assert queried == [3]  # tried once, then row by row
    assert any("frame by frame" in r.getMessage() for r in caplog.records)


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-v"]))
