"""Tests for prefill_track_points.py and the click tool's --suggestions.

The tracker is a stand-in with RoboTracker's contract and the frame sources are
in memory, so neither torch, lerobot nor a GPU is needed. The real RoboTracker
was run by hand through prefill.run() on a real lerobot dataset (see the commit
message).

Run: python -m pytest tests/test_prefill_track_points.py -v
"""

import importlib.util
import json
import logging
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


ctp = _load("click_track_points")
pf = _load("prefill_track_points")
TRACK_HW = pf.TRACK_SIZE

POINTS = ["button_a", "door_a", "lid"]
REF_HW = (800, 1280)


class Source:
    def __init__(self, frames, hw):
        self._frames, self.hw, self.calls = frames, hw, []

    def episode_frames(self):
        return dict(self._frames)

    def frame(self, ep, i):
        self.calls.append((ep, i))
        return np.full((*self.hw, 3), (ep * 7 + i) % 256, np.uint8)


def shifting_tracker(shift=(10.0, -5.0), vis=0.9, calls=None):
    """Moves every point by `shift` tracking pixels between the clip's two frames."""

    def track(frames, query_xy, query_frames):
        if calls is not None:
            calls.append((len(frames), np.array(query_xy), np.array(query_frames)))
        q = np.asarray(query_xy, np.float32)
        pos = np.stack([q, q + np.asarray(shift, np.float32)])
        v = np.full((2, len(q)), vis, np.float32)
        v[0] = 1.0
        return pos, v

    return track


needs_cv2 = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None,
    reason="the pre-fill resizes frames with opencv (lerobot brings it)",
)

REF_LABELS = {
    "button_a": {"frame": 3, "x": 640.0, "y": 400.0},
    "door_a": {"frame": 0, "x": 100.0, "y": 50.0},
    "lid": None,
}


# ── the suggestion of one episode ─────────────────────────────────────────────


def test_reference_clicks_are_grouped_by_their_frame():
    groups = pf.reference_groups(REF_LABELS, POINTS)
    assert sorted(groups) == [0, 3]
    assert [i for i, _ in groups[3]] == [0] and [i for i, _ in groups[0]] == [1]


@needs_cv2
def test_each_reference_frame_is_one_two_image_clip_and_positions_come_back_in_target_pixels():
    calls = []
    run = pf.run(
        Source({0: 10}, REF_HW),
        7,
        REF_LABELS,
        Source({0: 20, 1: 20}, (400, 640)),
        POINTS,
        shifting_tracker(calls=calls),
        camera="cam",
        reference={"episode": 7},
    )
    # two reference frames x two episodes, each a clip of 2 at the tracking size
    assert len(calls) == 4 and all(n == 2 for n, _, _ in calls)
    assert all((qf == 0).all() for _, _, qf in calls)
    # the button: reference (640, 400) on 1280x800 -> tracking size -> +shift -> 640x400 target
    q_trk = pf.scale_xy(np.array([[640.0, 400.0]]), REF_HW, TRACK_HW)
    want = pf.scale_xy(q_trk + np.array([10.0, -5.0]), TRACK_HW, (400, 640))[0]
    got = run["episodes"]["0"]["button_a"]
    assert got["frame"] == 0 and got["visibility"] == pytest.approx(0.9)
    np.testing.assert_allclose([got["x"], got["y"]], want, atol=0.01)
    assert run["episodes"]["0"]["lid"] is None  # not clicked in the reference
    assert run["image_size"] == [400, 640] and run["points"] == POINTS
    assert run["kind"] == "suggestions" and run["reference"] == {"episode": 7}


@needs_cv2
def test_a_point_the_tracker_does_not_see_gets_no_suggestion():
    run = pf.run(
        Source({0: 10}, REF_HW),
        0,
        REF_LABELS,
        Source({0: 20}, REF_HW),
        POINTS,
        shifting_tracker(vis=0.3),
        camera="cam",
        min_visibility=0.6,
    )
    assert run["episodes"]["0"] == {"button_a": None, "door_a": None, "lid": None}


@needs_cv2
def test_a_suggestion_outside_the_image_is_dropped():
    run = pf.run(
        Source({0: 10}, REF_HW),
        0,
        REF_LABELS,
        Source({0: 20}, REF_HW),
        POINTS,
        shifting_tracker(shift=(-100.0, 0.0)),
        camera="cam",
    )
    assert run["episodes"]["0"]["door_a"] is None  # x 40 -> -60 in tracking pixels: off the image
    assert run["episodes"]["0"]["button_a"] is not None


@needs_cv2
def test_the_suggestions_go_on_the_requested_frame():
    tgt = Source({0: 20}, REF_HW)
    run = pf.run(
        Source({0: 10}, REF_HW),
        0,
        REF_LABELS,
        tgt,
        POINTS,
        shifting_tracker(),
        camera="cam",
        frame=5,
    )
    assert tgt.calls == [(0, 5)]
    assert run["episodes"]["0"]["button_a"]["frame"] == 5


@needs_cv2
def test_an_episode_shorter_than_the_frame_is_skipped(caplog):
    with caplog.at_level(logging.WARNING):
        run = pf.run(
            Source({0: 10}, REF_HW),
            0,
            REF_LABELS,
            Source({0: 3, 1: 20}, REF_HW),
            POINTS,
            shifting_tracker(),
            camera="cam",
            frame=5,
        )
    assert list(run["episodes"]) == ["1"]
    assert any("episode 0 has only 3 frames" in r.getMessage() for r in caplog.records)


@needs_cv2
def test_a_reference_without_any_click_is_refused():
    with pytest.raises(SystemExit, match="no clicked point"):
        pf.run(
            Source({0: 10}, REF_HW),
            0,
            {p: None for p in POINTS},
            Source({0: 20}, REF_HW),
            POINTS,
            shifting_tracker(),
            camera="cam",
        )


@needs_cv2
def test_episodes_not_in_the_dataset_are_refused():
    with pytest.raises(SystemExit, match=r"\[9\] are not in the dataset"):
        pf.run(
            Source({0: 10}, REF_HW),
            0,
            REF_LABELS,
            Source({0: 20}, REF_HW),
            POINTS,
            shifting_tracker(),
            camera="cam",
            episodes=[0, 9],
        )


# ── the click tool reading them ───────────────────────────────────────────────


def store(tmp_path, **kw):
    args = dict(
        dataset_root="/d",
        camera="cam",
        points=POINTS,
        image_size=REF_HW,
        episode_frames={0: 20, 1: 20},
    )
    args.update(kw)
    return ctp.LabelStore(tmp_path / "labels.json", **args)


def write(tmp_path, data, name="sugg.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def suggestions(**kw):
    data = {
        "version": 1,
        "kind": "suggestions",
        "camera": "cam",
        "image_size": list(REF_HW),
        "points": POINTS,
        "episodes": {
            "0": {
                "button_a": {"frame": 0, "x": 600.0, "y": 380.5, "visibility": 0.93},
                "door_a": None,
                "lid": None,
            },
            "1": {
                "button_a": {"frame": 30, "x": 1.0, "y": 1.0},
                "door_a": {"frame": 2, "x": 9999.0, "y": 1.0},
                "lid": {"frame": 0, "x": 5.0, "y": 6.0},
            },
            "7": {"button_a": {"frame": 0, "x": 1.0, "y": 1.0}, "door_a": None, "lid": None},
        },
    }
    data.update(kw)
    return data


def test_the_click_tool_takes_the_valid_suggestions_and_drops_the_rest(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        got = ctp.load_suggestions(write(tmp_path, suggestions()), store(tmp_path))
    assert set(got) == {"0", "1"}  # episode 7 is not in the dataset
    assert got["0"]["button_a"] == {"frame": 0, "x": 600.0, "y": 380.5}  # visibility not carried
    assert got["0"]["door_a"] is None
    assert got["1"]["button_a"] is None  # frame 30 of a 20-frame episode
    assert got["1"]["door_a"] is None  # outside the image
    assert got["1"]["lid"] == {"frame": 0, "x": 5.0, "y": 6.0}
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "3 suggestion(s) dropped" in message and "episode 7: not in this dataset" in message


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"points": ["door_a", "button_a", "lid"]}, "points"),
        ({"image_size": [400, 640]}, "image_size"),
        ({"camera": "other"}, "camera"),
    ],
)
def test_suggestions_made_for_other_labels_are_refused(tmp_path, change, field):
    with pytest.raises(ctp.LabelError, match=f"does not fit these labels:\\n  {field}"):
        ctp.load_suggestions(write(tmp_path, suggestions(**change)), store(tmp_path))


def test_a_labels_file_can_serve_as_suggestions(tmp_path):
    s = store(tmp_path)
    s.save(1, {"button_a": {"frame": 2, "x": 3.0, "y": 4.0}, "door_a": None, "lid": None})
    other = store(tmp_path / "other")
    got = ctp.load_suggestions(tmp_path / "labels.json", other)
    assert got == {"1": {"button_a": {"frame": 2, "x": 3.0, "y": 4.0}, "door_a": None, "lid": None}}


def test_the_page_receives_the_suggestions(tmp_path):
    import urllib.request

    s = store(tmp_path)
    sugg = ctp.load_suggestions(write(tmp_path, suggestions()), s)
    server = ctp.ClickServer(Source({0: 20, 1: 20}, REF_HW), s, port=0, suggestions=sugg).start()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{server.port}/api/state", timeout=5) as r:
            state = json.loads(r.read())
        with opener.open(f"http://127.0.0.1:{server.port}/", timeout=5) as r:
            page = r.read()
    finally:
        server.stop()
    assert state["suggestions"]["0"]["button_a"] == {"frame": 0, "x": 600.0, "y": 380.5}
    assert b"Enter accepts them" in page and b"redo" in page


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
