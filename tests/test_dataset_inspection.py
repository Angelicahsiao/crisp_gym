"""inspect_dataset.py, check_tactile_sync.py and scripts/code_metrics.py on synthetic data.

The two dataset scripts were inline commands run on real recordings first; these
tests pin what they report on a dataset whose answers are known: the stamp join
(int64 exact, float32 useless), the handheld signature (placeholder command,
joints that never move), and the cases the inline command crashed on (a missing
stamp column, a CSV that was never pulled).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

REPO = Path(__file__).resolve().parents[1]
T0 = 1_794_190_351_560_399_616
TAXELS, FRAMES, ROWS_PER_FRAME = 6, 30, 10


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sync = _load("check_tactile_sync", REPO / "crisp_gym" / "scripts" / "check_tactile_sync.py")
inspect = _load("inspect_dataset", REPO / "crisp_gym" / "scripts" / "inspect_dataset.py")
metrics = _load("code_metrics", REPO / "scripts" / "code_metrics.py")


def make_dataset(root, *, stamp_dtype="int64", handheld=False, shuffle_csv=False, episodes=2):
    """Two episodes; each frame stores the CSV row at an exact stamp, 10 rows apart."""
    rng = np.random.default_rng(0)
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "tactile_raw").mkdir()
    rows, features = [], {}
    for ep in range(episodes):
        m = FRAMES * ROWS_PER_FRAME
        per_side = {}
        for side in ("left", "right"):
            stamps = T0 + ep * 10**11 + np.cumsum(rng.integers(4_000_000, 6_000_000, m))
            values = rng.integers(0, 50, (m, TAXELS)).astype(float)
            csv = pd.DataFrame({"episode_index": ep, "source_stamp_ns": stamps, "frame_id": side})
            for j in range(TAXELS):
                csv[f"taxel_{j}"] = values[:, j]
            if shuffle_csv:
                csv = csv.sample(frac=1.0, random_state=1)
            csv.to_csv(root / "tactile_raw" / f"episode_{ep:06d}_{side}.csv", index=False)
            pick = np.arange(FRAMES) * ROWS_PER_FRAME + 5
            per_side[side] = (stamps[pick], values[pick])
        for i in range(FRAMES):
            pose = np.r_[0.4 + 0.002 * i, 0.0, 0.3, 1, 0, 0, 0, 1, 0]
            row = {
                "episode_index": ep,
                "frame_index": i,
                "observation.state.cartesian": pose.astype(np.float32),
                "extra.target_cartesian": np.zeros(9, np.float32)
                if handheld
                else (pose + [0.01, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32),
                "extra.joints": np.zeros(7, np.float32)
                if handheld
                else np.full(7, 0.01 * i, np.float32),
            }
            for side, (stamps, values) in per_side.items():
                key = f"extra.tactile.{side}"
                row[key] = values[i].astype(np.float32)
                row[key + ".source_stamp_ns"] = np.array([stamps[i]]).astype(stamp_dtype)
                row[key + ".sync_valid"] = np.array([True])
                row[key + ".sync_offset_ms"] = np.array([2.0], np.float32)
                row[key + ".sync_error_ms"] = np.array([1.0], np.float32)
            rows.append(row)
    for side in ("left", "right"):
        key = f"extra.tactile.{side}"
        features[key] = {"dtype": "float32", "shape": [TAXELS]}
        features[key + ".source_stamp_ns"] = {"dtype": stamp_dtype, "shape": [1]}
        features[key + ".sync_valid"] = {"dtype": "bool", "shape": [1]}
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    info = {
        "fps": 15,
        "total_episodes": episodes,
        "total_frames": episodes * FRAMES,
        "codebase_version": "v3.0",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    return root


def run(mod, argv, capsys):
    rc = mod.main([str(a) for a in argv])
    return rc, capsys.readouterr().out


# ── check_tactile_sync ───────────────────────────────────────────────────────


def test_int64_stamps_join_exactly(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds")
    rc, out = run(sync, [root], capsys)
    assert rc == 0
    assert out.count("exact 100%") == 2 and out.count("taxels equal at that row 100%") == 2
    assert out.count(f"csv rows per frame: median {ROWS_PER_FRAME}") == 2
    assert out.count(f"csv: {FRAMES * ROWS_PER_FRAME} rows") == 2
    assert "sync_valid 100%" in out and "sync_offset_ms median 2.00" in out
    assert "csv non-taxel columns: ['episode_index', 'source_stamp_ns', 'frame_id']" in out
    assert "command-measured position gap mm: median 10.00" in out  # the 10 mm offset
    assert "joints moving: 0.0" in out and "joints moving: 0.000" not in out


def test_float32_stamps_are_reported_as_unjoinable(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds", stamp_dtype="float32")
    _, out = run(sync, [root], capsys)
    assert "dataset stamp (float32) found in csv: exact 0%" in out
    assert "'.source_stamp_ns': 'float32'" in out


def test_handheld_signature(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds", handheld=True)
    _, out = run(sync, [root], capsys)
    assert "joints moving: 0.000 rad std" in out
    # the "gap" is just the distance of the pose from the origin: ~0.5 m here
    assert "command-measured position gap mm: median 52" in out


def test_episode_and_side_selection(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds")
    _, out = run(sync, [root, "--episode", "1", "--sides", "left"], capsys)
    assert out.startswith(f"episode 1: {FRAMES} frames")
    assert "--- left" in out and "--- right" not in out
    with pytest.raises(ValueError, match="episode 7 has no rows"):
        sync.main([str(root), "--episode", "7"])


def test_unsorted_csv_rows_still_join(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds", shuffle_csv=True)
    _, out = run(sync, [root], capsys)
    assert out.count("exact 100%") == 2
    assert out.count(f"csv rows per frame: median {ROWS_PER_FRAME}") == 2


def test_missing_stamp_column_and_unpulled_csv(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds")
    data = next((root / "data").rglob("*.parquet"))
    df = pd.read_parquet(data)
    df.drop(columns=[c for c in df.columns if c.endswith(".source_stamp_ns")]).to_parquet(data)
    csv = root / "tactile_raw" / "episode_000000_right.csv"
    csv.unlink()
    csv.with_name(csv.name + ".dvc").write_text("outs: []\n")
    rc, out = run(sync, [root], capsys)
    assert rc == 0
    assert "dataset has no extra.tactile.left.source_stamp_ns" in out
    assert "only its .dvc pointer: pull it" in out


# ── inspect_dataset ──────────────────────────────────────────────────────────


def test_inspect_dataset_layout(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds")
    (root / "tactile_raw" / "edit_report.json").write_text(json.dumps({"operation": "merge"}) * 50)
    rc, out = run(inspect, [root, "--report-chars", "40"], capsys)
    assert rc == 0
    assert out.splitlines()[0] == f"fps 15 | episodes 2 | frames {2 * FRAMES} | codebase v3.0"
    import re

    assert re.search(rf"^  extra\.tactile\.left +\[{TAXELS}\]  float32$", out, re.M)
    assert "4 CSV file(s) for 2 episode(s), sides ['left', 'right']" in out
    fields = 3 + TAXELS
    assert (
        f"episode_000000_left.csv: {fields} fields | episode_index, source_stamp_ns, frame_id"
        in out
    )
    assert f"| {FRAMES * ROWS_PER_FRAME} data rows" in out  # header not counted
    assert "edit_report.json (first 40 of" in out


def test_inspect_dataset_reports_missing_pieces(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds")
    csv = root / "tactile_raw" / "episode_000001_right.csv"
    csv.unlink()
    csv.with_name(csv.name + ".dvc").write_text("outs: []\n")
    _, out = run(inspect, [root, "--episode", "1"], capsys)
    assert "1 DVC pointer(s), 1 not pulled" in out
    assert "episode_000001_right.csv: only its DVC pointer (not pulled)" in out
    _, out = run(inspect, [root, "--episode", "9"], capsys)
    assert "episode_000009_left.csv: missing" in out
    assert inspect.main([str(tmp_path / "nothing")]) == 1


# ── code_metrics ─────────────────────────────────────────────────────────────


def test_code_metrics_orders_functions_by_length(tmp_path, capsys):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        "def short():\n    return 1\n\n\ndef long():\n" + "    x = 1\n" * 9 + "    return x\n"
    )
    rc, out = run(metrics, [tmp_path / "pkg", "--top", "2"], capsys)
    assert rc == 0
    first, second = out.splitlines()[:2]
    assert first.split()[:2] == ["11", "long"] and second.split()[:2] == ["2", "short"]
    assert "2 functions in 1 files; over 80 lines: 0, over 150 lines: 0" in out
    rc, out = run(metrics, [tmp_path / "pkg", "--files"], capsys)
    assert "1 files, 15 lines" in out
    assert metrics.main([str(tmp_path / "empty")]) == 1
