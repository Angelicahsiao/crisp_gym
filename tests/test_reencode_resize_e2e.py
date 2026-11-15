"""End-to-end --resize on REAL video (crisp_gym/scripts/reencode_dataset_videos.py).

tests/test_reencode_dataset_videos.py stubs PyAV, so it cannot exercise the
part a resize adds: decoding, scaling and re-encoding real frames while keeping
the frame count and timeline intact. This builds a small AV1 dataset with real
PyAV, runs the script's main() with --resize, and checks the result by decoding
it again.

It runs in a SUBPROCESS: the stub-based test file replaces `av` in sys.modules,
and in the same interpreter that would leak into this one depending on test
order. The driver uses the real lerobot when it is installed (the pixi env) and
a minimal stand-in otherwise, so the script's own code is what is tested either
way. Skipped where PyAV itself is not installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "crisp_gym" / "scripts" / "reencode_dataset_videos.py"

DRIVER = textwrap.dedent(r'''
    import json, sys, types
    from dataclasses import dataclass, field
    from importlib.machinery import SourceFileLoader
    from pathlib import Path
    from typing import Any

    import av
    import numpy as np

    script, work = Path(sys.argv[1]), Path(sys.argv[2])
    codec = "libsvtav1" if "libsvtav1" in av.codecs_available else "libx264"

    # Real lerobot when installed; otherwise just the symbols the script imports.
    try:
        import lerobot.datasets.video_utils  # noqa: F401
        real_lerobot = True
    except ImportError:
        real_lerobot = False
        def _mod(name):
            return sys.modules.setdefault(name, types.ModuleType(name))
        for name in ("lerobot", "lerobot.configs", "lerobot.configs.video", "lerobot.datasets",
                     "lerobot.datasets.video_utils", "lerobot.utils", "lerobot.utils.constants"):
            _mod(name)

        @dataclass
        class RGBEncoderConfig:
            vcodec: str = "libsvtav1"
            pix_fmt: str = "yuv420p"
            crf: int | None = 30
            g: int | None = 2
            preset: Any = None
            extra_options: dict = field(default_factory=dict)
            def __post_init__(self):
                self.vcodec = {"av1": "libsvtav1", "h264": "libx264"}.get(self.vcodec, self.vcodec)
            def get_codec_options(self, encoder_threads=None, as_strings=False):
                opts = {"g": self.g, "crf": self.crf, **self.extra_options}
                if self.vcodec == "libsvtav1":
                    opts["preset"] = self.preset if self.preset is not None else 12
                opts = {k: v for k, v in opts.items() if v is not None}
                return {k: str(v) for k, v in opts.items()} if as_strings else opts

        def get_video_info(path, video_encoder=None):
            with av.open(str(path)) as c:
                s = c.streams.video[0]
                return {"video.height": s.height, "video.width": s.width,
                        "video.codec": s.codec.canonical_name, "video.pix_fmt": s.pix_fmt,
                        "video.fps": int(s.base_rate), "video.channels": 3}

        def reencode_video(*a, **k):
            raise AssertionError("--resize must not go through lerobot's reencode_video")

        sys.modules["lerobot.configs.video"].RGBEncoderConfig = RGBEncoderConfig
        sys.modules["lerobot.configs.video"].VIDEO_CODECS_ALIASES = {"av1": "libsvtav1", "h264": "libx264"}
        sys.modules["lerobot.datasets.video_utils"].get_video_info = get_video_info
        sys.modules["lerobot.datasets.video_utils"].reencode_video = reencode_video
        sys.modules["lerobot.utils.constants"].HF_LEROBOT_HOME = work / "hf_home"

    H, W, N, FPS = 160, 256, 30, 15
    KEY = "observation.images.cam"

    def make_dataset(root):
        """A one-file dataset whose frames each differ (a bar moving right)."""
        (root / "meta").mkdir(parents=True)
        video = root / "videos" / KEY / "chunk-000" / "file-000.mp4"
        video.parent.mkdir(parents=True)
        with av.open(str(video), "w") as c:
            s = c.add_stream(codec, FPS)
            s.width, s.height, s.pix_fmt = W, H, "yuv420p"
            if codec == "libsvtav1":
                s.options = {"preset": "12", "crf": "20", "g": "2"}
            for i in range(N):
                img = np.zeros((H, W, 3), np.uint8)
                img[:, :, 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
                x = 8 * i
                img[40:120, x:x + 24, :] = 255
                for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    c.mux(p)
            for p in s.encode():
                c.mux(p)
        with av.open(str(video)) as c:
            st = c.streams.video[0]
            src_codec = st.codec.canonical_name
        # Two episodes sharing the file, located by timestamp offsets: the
        # check that a resize must not move them is part of what runs.
        import pandas as pd
        ep_dir = root / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True)
        half = N // 2
        pd.DataFrame({
            "episode_index": [0, 1],
            f"videos/{KEY}/chunk_index": [0, 0],
            f"videos/{KEY}/file_index": [0, 0],
            f"videos/{KEY}/from_timestamp": [0.0, half / FPS],
            f"videos/{KEY}/to_timestamp": [half / FPS, N / FPS],
        }).to_parquet(ep_dir / "file-000.parquet")
        (root / "meta" / "info.json").write_text(json.dumps({
            "fps": FPS,
            "features": {KEY: {
                "dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channels"],
                "info": {"video.codec": src_codec, "video.pix_fmt": "yuv420p",
                         "video.height": H, "video.width": W, "video.fps": FPS},
                "video_info": {"video.height": H, "video.width": W},
            }},
        }))
        return root, video

    reencode = SourceFileLoader("reencode_dataset_videos", str(script)).load_module()
    src, src_video = make_dataset(work / "src")
    out = work / "out"
    code = reencode.main(["--root", str(src), "--resize", "80", "128", "--output-root", str(out),
                          "--verify-frames", "5"])

    result = {"code": code, "codec": codec, "real_lerobot": real_lerobot}
    if code == 0:
        new_video = out / "videos" / KEY / "chunk-000" / "file-000.mp4"
        before, after = reencode.probe(src_video), reencode.probe(new_video)
        info = json.loads((out / "meta" / "info.json").read_text())["features"][KEY]
        # Content check, independent of the script's own: frame 20 of the copy
        # must match frame 20 of the source scaled down, not a neighbour.
        def frame(path, idx, size=None):
            with av.open(str(path)) as c:
                for i, f in enumerate(c.decode(video=0)):
                    if i == idx:
                        if size:
                            f = f.reformat(width=size[0], height=size[1], interpolation="AREA")
                        return f.to_ndarray(format="rgb24").astype(np.int16)
        ref = frame(src_video, 20, (128, 80))
        same = float(np.abs(frame(new_video, 20) - ref).mean())
        neighbour = float(np.abs(frame(new_video, 21) - ref).mean())
        result.update({
            "before": [before.width, before.height, before.frames],
            "after": [after.width, after.height, after.frames],
            "shape": info["shape"],
            "info_hw": [info["info"]["video.height"], info["info"]["video.width"]],
            "video_info_hw": [info["video_info"]["video.height"], info["video_info"]["video.width"]],
            "codec_after": info["info"]["video.codec"],
            "diff_same_frame": same,
            "diff_neighbour_frame": neighbour,
            "src_untouched": [reencode.probe(src_video).width, reencode.probe(src_video).height],
        })
    print("RESULT " + json.dumps(result))
''')


def _run(tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    probe = subprocess.run([sys.executable, "-c", "import av, numpy, pandas, pyarrow"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("PyAV / numpy / pandas / pyarrow not installed for this interpreter")
    proc = subprocess.run(
        [sys.executable, str(driver), str(SCRIPT), str(tmp_path)],
        capture_output=True, text=True, timeout=600,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, f"driver failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr[-4000:]}"
    return json.loads(lines[-1][len("RESULT "):]), proc


def test_resize_end_to_end_on_real_video(tmp_path):
    res, proc = _run(tmp_path)
    assert res["code"] == 0, proc.stderr[-4000:]
    # Timeline intact, size as asked (PyAV reports width x height).
    assert res["before"] == [256, 160, 30]
    assert res["after"] == [128, 80, 30]
    # info.json now describes the new files, in every place it records size.
    assert res["shape"] == [80, 128, 3]
    assert res["info_hw"] == [80, 128]
    assert res["video_info_hw"] == [80, 128]
    # Codec kept from the source when none was asked for.
    assert res["codec_after"] == ("av1" if res["codec"] == "libsvtav1" else "h264")
    # Frame 20 of the copy IS frame 20 of the source (scaled), not a neighbour:
    # only encoding error separates them, while the next frame differs clearly.
    assert res["diff_same_frame"] < 8.0, res
    assert res["diff_neighbour_frame"] > 2 * res["diff_same_frame"], res
    # The source dataset is left alone.
    assert res["src_untouched"] == [256, 160]
