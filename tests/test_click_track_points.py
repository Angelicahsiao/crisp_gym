"""Tests for crisp_gym/scripts/click_track_points.py.

The labels file (validation, atomic writes, resuming) and the HTTP API run
against a stand-in frame source, so neither lerobot nor a dataset is needed.
The lerobot frame source itself is exercised with a real dataset by hand (see
the module docstring for the command).

Run: python -m pytest tests/test_click_track_points.py -v
"""

import importlib.util
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "click_track_points", REPO / "crisp_gym" / "scripts" / "click_track_points.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ctp = _load()

POINTS = ["button_a", "button_b", "door_a"]
H, W = 48, 64
FRAMES = {0: 30, 1: 12, 5: 20}          # episode -> number of frames (not contiguous ids)


def make_store(tmp_path, **kw):
    args = dict(dataset_root="/data/ds", camera="observation.images.oakw_cam", points=POINTS,
                image_size=(H, W), episode_frames=FRAMES)
    args.update(kw)
    return ctp.LabelStore(tmp_path / "labels.json", **args)


def good(frame=0):
    return {"button_a": {"frame": frame, "x": 10.0, "y": 20.0},
            "button_b": {"frame": frame + 1, "x": 63.0, "y": 47.0},
            "door_a": None}


# ── the labels file ───────────────────────────────────────────────────────────

def test_a_saved_episode_is_on_disk_at_once_in_the_documented_format(tmp_path):
    store = make_store(tmp_path)
    store.save(5, good(frame=3))
    data = json.loads((tmp_path / "labels.json").read_text())
    assert data["version"] == 1
    assert data["camera"] == "observation.images.oakw_cam"
    assert data["image_size"] == [H, W]
    assert data["points"] == POINTS
    assert data["episodes"] == {"5": {"button_a": {"frame": 3, "x": 10.0, "y": 20.0},
                                      "button_b": {"frame": 4, "x": 63.0, "y": 47.0},
                                      "door_a": None}}
    assert not (tmp_path / "labels.json.tmp").exists()


def test_every_point_must_be_clicked_or_marked_not_visible(tmp_path):
    store = make_store(tmp_path)
    partial = good()
    del partial["door_a"]
    with pytest.raises(ctp.LabelError, match=r"missing \['door_a'\]"):
        store.save(0, partial)
    with pytest.raises(ctp.LabelError, match=r"unknown \['lid'\]"):
        store.save(0, {**good(), "lid": None})
    assert not (tmp_path / "labels.json").exists(), "nothing is written for a refused label"


@pytest.mark.parametrize(("click", "match"), [
    ({"frame": 30, "x": 1, "y": 1}, "outside episode 0"),          # episode 0 has frames 0..29
    ({"frame": -1, "x": 1, "y": 1}, "outside episode 0"),
    ({"frame": 0, "x": 64, "y": 1}, "outside the 64x48 image"),    # x must be <= W - 1
    ({"frame": 0, "x": 1, "y": -0.5}, "outside the 64x48 image"),
    ({"frame": 0, "x": float("nan"), "y": 1}, "outside the 64x48 image"),
    ({"frame": 0, "x": 1}, "needs frame, x and y"),
    ("here", "needs frame, x and y"),
])
def test_a_click_outside_the_episode_or_the_image_is_refused(tmp_path, click, match):
    with pytest.raises(ctp.LabelError, match=match):
        make_store(tmp_path).save(0, {**good(), "button_a": click})


def test_an_episode_the_dataset_does_not_have_is_refused(tmp_path):
    with pytest.raises(ctp.LabelError, match="episode 2 is not in this dataset"):
        make_store(tmp_path).save(2, good())
    with pytest.raises(ctp.LabelError, match="must be a number"):
        make_store(tmp_path).save("first", good())


def test_saving_an_episode_again_replaces_it(tmp_path):
    store = make_store(tmp_path)
    store.save(0, good(frame=0))
    store.save(0, good(frame=7))
    assert store.get(0)["button_a"]["frame"] == 7
    assert store.labelled() == [0]


def test_a_restart_continues_the_same_file(tmp_path):
    make_store(tmp_path).save(1, good())
    again = make_store(tmp_path)
    assert again.labelled() == [1]
    assert again.get(1)["button_b"] == {"frame": 1, "x": 63.0, "y": 47.0}
    again.save(0, good())
    assert make_store(tmp_path).labelled() == [0, 1]


@pytest.mark.parametrize("change", [
    {"camera": "observation.images.other"},
    {"points": ["button_a", "door_a"]},
    {"points": ["button_b", "button_a", "door_a"]},       # order is part of the meaning
    {"image_size": (96, 128)},
])
def test_a_file_made_for_something_else_is_not_mixed_into(tmp_path, change):
    make_store(tmp_path).save(0, good())
    with pytest.raises(ctp.LabelError, match="made for something else"):
        make_store(tmp_path, **change)


def test_labels_for_episodes_not_in_the_dataset_are_kept_but_not_shown(tmp_path, caplog):
    make_store(tmp_path).save(5, good())
    with caplog.at_level(logging.WARNING):
        fewer = make_store(tmp_path, episode_frames={0: 30, 1: 12})
    assert fewer.labelled() == []
    assert any("does not have: [5]" in r.getMessage() for r in caplog.records)
    fewer.save(0, good())
    assert set(json.loads((tmp_path / "labels.json").read_text())["episodes"]) == {"0", "5"}


def test_point_names_must_be_unique_and_present(tmp_path):
    with pytest.raises(ctp.LabelError, match="unique"):
        make_store(tmp_path, points=["a", "a"])
    with pytest.raises(ctp.LabelError, match="at least one"):
        make_store(tmp_path, points=[])


# ── frames ────────────────────────────────────────────────────────────────────

def test_lerobot_frames_become_hwc_uint8():
    chw = np.zeros((3, 4, 5), np.float32)
    chw[0, 1, 2] = 1.0
    chw[2, 3, 4] = 0.5
    out = ctp.to_hwc_uint8(chw)
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8
    assert out[1, 2, 0] == 255 and out[3, 4, 2] == 128
    hwc = np.full((4, 5, 3), 7, np.uint8)
    np.testing.assert_array_equal(ctp.to_hwc_uint8(hwc), hwc)
    assert ctp.to_hwc_uint8(np.zeros((1, 4, 5), np.float32)).shape == (4, 5, 3)
    with pytest.raises(ValueError, match="3-D"):
        ctp.to_hwc_uint8(np.zeros((4, 5)))


def _decoder():
    try:
        import cv2

        return lambda data: cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    except ImportError:
        pass
    try:
        import io

        from PIL import Image

        return lambda data: np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except ImportError:
        return None


def test_a_frame_survives_jpeg_encoding():
    decode = _decoder()
    if decode is None:
        pytest.skip("needs opencv or pillow (the lerobot environment has both)")
    yy, xx = np.mgrid[0:H, 0:W]
    frame = np.stack([xx * 4, yy * 5, np.full_like(xx, 200)], -1).astype(np.uint8)
    back = decode(ctp.encode_jpeg(frame, quality=95))
    assert back.shape == frame.shape
    assert np.abs(back.astype(int) - frame.astype(int)).mean() < 3


# ── the HTTP API ──────────────────────────────────────────────────────────────

class FakeSource:
    def __init__(self):
        self.calls = []

    def episode_frames(self):
        return dict(FRAMES)

    def frame(self, episode, index):
        self.calls.append((episode, index))
        if not 0 <= index < FRAMES[episode]:
            raise IndexError(f"frame {index} is outside episode {episode}")
        return np.full((H, W, 3), (episode * 10 + index) % 256, np.uint8)


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setattr(ctp, "encode_jpeg", lambda frame, quality=90: b"JPEG" + bytes([int(frame[0, 0, 0])]))
    source, store = FakeSource(), make_store(tmp_path)
    server = ctp.ClickServer(source, store, host="127.0.0.1", port=0).start()
    base = f"http://127.0.0.1:{server.port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    yield base, opener, source, store
    server.stop()


def _get(opener, url):
    try:
        with opener.open(url, timeout=5) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def _post(opener, url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return _get(opener, req)


def test_the_page_is_served(served):
    base, opener, _, _ = served
    status, kind, body = _get(opener, base + "/")
    assert status == 200 and kind.startswith("text/html")
    assert b"Click track points" in body and b"/api/label" in body


def test_the_state_describes_the_dataset_and_the_labels(served):
    base, opener, _, store = served
    store.save(1, good())
    status, _, body = _get(opener, base + "/api/state")
    state = json.loads(body)
    assert status == 200
    assert state["points"] == POINTS
    assert state["image_size"] == [H, W]
    assert state["episodes"] == [0, 1, 5]
    assert state["n_frames"] == {"0": 30, "1": 12, "5": 20}
    assert list(state["labels"]) == ["1"]
    assert state["camera"] == "observation.images.oakw_cam"


def test_a_frame_is_decoded_once_and_then_cached(served):
    base, opener, source, _ = served
    status, kind, body = _get(opener, base + "/api/frame?ep=5&f=3")
    assert (status, kind, body) == (200, "image/jpeg", b"JPEG" + bytes([53]))
    _get(opener, base + "/api/frame?ep=5&f=3")
    assert source.calls == [(5, 3)]


@pytest.mark.parametrize("query", ["ep=2&f=0", "ep=0&f=30", "ep=0", "ep=x&f=0"])
def test_a_frame_that_does_not_exist_is_a_400_with_a_reason(served, query):
    base, opener, _, _ = served
    status, kind, body = _get(opener, base + "/api/frame?" + query)
    assert status == 400 and kind == "application/json"
    assert json.loads(body)["error"]


def test_a_label_posted_from_the_page_is_saved(served, tmp_path):
    base, opener, _, _ = served
    status, _, body = _post(opener, base + "/api/label", {"episode": 0, "points": good()})
    reply = json.loads(body)
    assert status == 200 and reply["ok"] and reply["labelled"] == [0]
    on_disk = json.loads((tmp_path / "labels.json").read_text())
    assert on_disk["episodes"]["0"]["door_a"] is None


def test_a_bad_label_is_refused_with_the_reason_and_nothing_is_saved(served, tmp_path):
    base, opener, _, _ = served
    status, _, body = _post(opener, base + "/api/label",
                            {"episode": 0, "points": {**good(), "button_a": {"frame": 99, "x": 1, "y": 1}}})
    assert status == 400
    assert "frame 99 is outside episode 0" in json.loads(body)["error"]
    assert not (tmp_path / "labels.json").exists()


def test_unknown_paths_are_404(served):
    base, opener, _, _ = served
    assert _get(opener, base + "/nope")[0] == 404
    assert _post(opener, base + "/api/other", {})[0] == 404


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
