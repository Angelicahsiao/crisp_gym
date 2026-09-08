"""Tests for the codec re-encode migration (crisp_gym/scripts/reencode_dataset_videos.py).

The script exists so a dataset recorded as av1 can be aggregated with one
recorded as h264. Its risky part is not the re-encode — lerobot's
`reencode_video` does that — but the VERIFICATION: in the v3.0 layout one mp4
holds many episodes and `meta/episodes/*.parquet` locates each by a timestamp
offset into that shared file, so a re-encode that changes the frame count or
shifts the timeline corrupts every later episode while the dataset still loads.
These tests pin the checks that refuse such a file.

Run:  python -m pytest tests/test_reencode_dataset_videos.py
"""

import sys
import types
from dataclasses import dataclass, field
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _install_stubs() -> None:
    """Stub av + lerobot so this file needs neither installed.

    Only the module-level symbols the script imports are required; every
    function under test is pure given its arguments.
    """

    def _mod(name: str):
        return sys.modules.setdefault(name, types.ModuleType(name))

    for name in (
        "av",
        "av.logging",
        "lerobot",
        "lerobot.configs",
        "lerobot.configs.video",
        "lerobot.datasets",
        "lerobot.datasets.video_utils",
        "lerobot.utils",
        "lerobot.utils.constants",
    ):
        _mod(name)

    sys.modules["av"].logging = sys.modules["av.logging"]
    sys.modules["lerobot.utils.constants"].HF_LEROBOT_HOME = Path("/tmp/lerobot_test_home")
    sys.modules["lerobot.datasets.video_utils"].reencode_video = lambda *a, **k: None
    sys.modules["lerobot.datasets.video_utils"].get_video_info = lambda *a, **k: {}
    sys.modules["lerobot.configs.video"].VIDEO_CODECS_ALIASES = {"av1": "libsvtav1"}

    @dataclass
    class _RGBEncoderConfig:
        """Mirrors the real dataclass closely enough for the bf=0 rule."""

        vcodec: str = "libsvtav1"
        pix_fmt: str = "yuv420p"
        crf: int | None = 30
        g: int | None = 2
        preset: Any = None
        extra_options: dict = field(default_factory=dict)

        def __post_init__(self) -> None:
            self.vcodec = {"av1": "libsvtav1"}.get(self.vcodec, self.vcodec)

        def get_codec_options(self, encoder_threads=None, as_strings=False):  # noqa: ANN001
            opts = {"g": self.g, "crf": self.crf, **self.extra_options}
            opts = {k: v for k, v in opts.items() if v is not None}
            return {k: str(v) for k, v in opts.items()} if as_strings else opts

    sys.modules["lerobot.configs.video"].RGBEncoderConfig = _RGBEncoderConfig


_install_stubs()

reencode = SourceFileLoader(
    "reencode_dataset_videos",
    str(REPO / "crisp_gym" / "scripts" / "reencode_dataset_videos.py"),
).load_module()

VideoProbe = reencode.VideoProbe


def _probe(frames: int = 500, fps: float = 15.0, width: int = 1280, height: int = 800, **kw):
    return VideoProbe(
        path=Path(kw.get("path", "/tmp/f.mp4")),
        codec=kw.get("codec", "h264"),
        pix_fmt=kw.get("pix_fmt", "yuv420p"),
        width=width,
        height=height,
        fps=fps,
        frames=frames,
        duration_s=kw.get("duration_s", frames / fps),
    )


# ── timeline verification ────────────────────────────────────────────────────


def test_identical_timeline_is_accepted():
    """A clean re-encode changes the codec and nothing else."""
    before = _probe(codec="av1")
    after = _probe(codec="h264")
    assert reencode.verify_timeline(before, after) == []


def test_a_single_dropped_frame_is_rejected():
    """One frame short shifts every episode after it in the same file."""
    problems = reencode.verify_timeline(_probe(frames=500), _probe(frames=499))
    assert any("frame count 500 -> 499" in p for p in problems), problems


def test_resolution_change_is_rejected():
    """concatenate_video_files compares width/height too, so a resize breaks the merge."""
    problems = reencode.verify_timeline(_probe(), _probe(width=640, height=400))
    assert any("size 1280x800 -> 640x400" in p for p in problems), problems


def test_fps_change_is_rejected():
    """Fps is what turns a stored timestamp back into a frame index."""
    problems = reencode.verify_timeline(_probe(fps=15.0), _probe(fps=30.0))
    assert any("fps 15.0 -> 30.0" in p for p in problems), problems


def test_duration_drift_under_one_frame_is_tolerated():
    """Container rounding moves the duration slightly; that is not corruption."""
    before = _probe(frames=500, fps=15.0)
    after = _probe(frames=500, fps=15.0, duration_s=before.duration_s + 0.02)
    assert reencode.verify_timeline(before, after) == []


def test_duration_drift_over_one_frame_is_rejected():
    before = _probe(frames=500, fps=15.0)
    after = _probe(frames=500, fps=15.0, duration_s=before.duration_s + 0.5)
    problems = reencode.verify_timeline(before, after)
    assert any("duration" in p for p in problems), problems


# ── episode offset verification ──────────────────────────────────────────────


def _offsets():
    """Three episodes packed into chunk 0 / file 0, back to back at 15 fps."""
    return [
        (0, 0, 0, 0.0, 33.3333),
        (1, 0, 0, 33.3333, 66.6666),
        (2, 0, 0, 66.6666, 100.0),
    ]


def test_offsets_pass_when_the_file_still_covers_them():
    after = _probe(frames=1500, fps=15.0)  # 100.0 s
    assert reencode.verify_episode_offsets(_offsets(), "cam", 0, 0, after) == []


def test_truncated_file_is_caught_by_the_last_episode():
    """The failure mode a frame count alone would miss on a per-file basis."""
    after = _probe(frames=1000, fps=15.0)  # 66.67 s — episode 2 no longer fits
    problems = reencode.verify_episode_offsets(_offsets(), "cam", 0, 0, after)
    assert any("episode 2" in p for p in problems), problems


def test_offsets_for_other_files_are_ignored():
    """Each mp4 is only responsible for the episodes stored inside it."""
    rows = [(7, 1, 0, 0.0, 10.0)]  # lives in chunk 1, not the file being checked
    after = _probe(frames=15, fps=15.0)  # 1.0 s — would fail if wrongly applied
    assert reencode.verify_episode_offsets(rows, "cam", 0, 0, after) == []


def test_last_episode_ending_exactly_at_the_duration_is_accepted():
    """to_timestamp == duration is the normal case, not an overrun."""
    after = _probe(frames=1500, fps=15.0)
    rows = [(0, 0, 0, 0.0, after.duration_s)]
    assert reencode.verify_episode_offsets(rows, "cam", 0, 0, after) == []


# ── encoder construction ─────────────────────────────────────────────────────


def test_nvenc_at_small_gop_gets_bf0():
    """NVENC at a small GOP must get bf=0, the same rule as recording.

    NVENC needs gop > b_frames + 1 and its presets enable B-frames, so
    lerobot's g=2 cannot open unless B-frames are off.
    """
    encoder = reencode.build_encoder("h264_nvenc", None, 21, 2, None, None)
    assert encoder.extra_options.get("bf") == 0


def test_nvenc_at_large_gop_is_left_alone():
    encoder = reencode.build_encoder("h264_nvenc", None, 21, 60, None, None)
    assert "bf" not in encoder.extra_options


def test_software_codec_never_gets_bf0():
    encoder = reencode.build_encoder("libsvtav1", None, 30, 2, None, None)
    assert "bf" not in encoder.extra_options


def test_explicit_bf_is_not_overridden():
    """An operator who passed --extra-options '{"bf": 2}' meant it."""
    encoder = reencode.build_encoder("h264_nvenc", None, 21, 2, None, {"bf": 2})
    assert encoder.extra_options["bf"] == 2


def test_stored_codec_name_av1_is_accepted_as_a_target():
    """A codec name copied out of an info.json must be accepted as a target.

    info.json stores 'av1' while the encoder is called 'libsvtav1'.
    """
    assert reencode.build_encoder("av1", None, 30, None, None, None).vcodec == "libsvtav1"


# ── metadata helpers ─────────────────────────────────────────────────────────


def test_video_keys_selects_only_video_features():
    info = {
        "features": {
            "observation.images.cam": {"dtype": "video"},
            "observation.state": {"dtype": "float32"},
            "action": {"dtype": "float32"},
        }
    }
    assert reencode.video_keys(info) == ["observation.images.cam"]


def test_compat_keys_are_the_ones_lerobot_actually_compares():
    """Guard against drift from lerobot.

    concatenate_video_files' compatibility check reads exactly height, width,
    fps, codec and pix_fmt.
    """
    assert set(reencode.COMPAT_KEYS) == {
        "video.codec",
        "video.pix_fmt",
        "video.width",
        "video.height",
        "video.fps",
    }


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
