"""add_environment_state.py on synthetic datasets, in both on-disk layouts.

The tool copies a dataset and joins chosen `extra.*` columns into one new
column, `observation.environment_state`, which a lerobot diffusion or ACT
policy reads beside `observation.state`. Checked here:

  1. the new column is the chosen columns joined in the order given (a nested
     taxel grid is flattened row by row);
  2. every other column, the videos and the INPUT dataset are untouched;
  3. meta/info.json gains the feature (shape, names) right after
     observation.state;
  4. statistics for the new key appear in all three places lerobot keeps them
     (meta/stats.json, v3.0 meta/episodes/*.parquet, v2.x episodes_stats.jsonl),
     with the stat names of observation.state, and match numpy;
  5. meta/environment_state.json says which columns went where;
  6. it refuses what would train a model on nothing or on garbage: a missing
     column, a column that never changes (the handheld placeholders), NaN, an
     image column, a dataset that already has the key, an existing output.

Runs under pytest or directly:  python tests/test_add_environment_state.py
Needs numpy, pandas, pyarrow. No lerobot, no ROS.
"""

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "crisp_gym" / "scripts"
KEY = "observation.environment_state"
STATS = ("min", "max", "mean", "std", "count", "q01", "q99")


def _load_by_path(name, path):
    """Import the file at `path` as module `name`."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # registered first: dataclasses and pickle look a module up by its name
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _tool():
    # the tool imports its sibling migrate_euler_delta_to_rot6d the way it does
    # when run as a script: from its own directory
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return _load_by_path("add_environment_state", SCRIPTS / "add_environment_state.py")


def _features():
    vec = lambda n, names=None: {"dtype": "float32", "shape": [n], "names": names}  # noqa: E731
    return {
        "observation.state.cartesian": vec(9),
        "observation.state.gripper": vec(1, ["gripper"]),
        "observation.state": vec(10, [f"s{i}" for i in range(10)]),
        "observation.images.cam": {"dtype": "video", "shape": [4, 4, 3], "names": ["h", "w", "c"]},
        "action": vec(10, [f"a{i}" for i in range(10)]),
        "extra.ext_torque": vec(7, [f"ext_torque_{i}" for i in range(7)]),
        "extra.tactile.left": {"dtype": "float32", "shape": [2, 3], "names": None},
        "extra.target_joints": vec(7),  # a placeholder that never changes
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
    }


def _stat_block(dim, n):
    block = {s: [0.0] * dim for s in STATS if s != "count"}
    block["count"] = [n]
    return block


def _make_dataset(root: Path, layout: str, lengths=(5, 8, 3), seed=0, nan_at=None):
    """A small dataset. Returns the frames as a DataFrame (the ground truth)."""
    rng = np.random.default_rng(seed)
    (root / "meta").mkdir(parents=True)
    video = root / "videos" / "observation.images.cam" / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"FAKE VIDEO BYTES " * 7)

    rows, index = [], 0
    for ep, n in enumerate(lengths):
        for fi in range(n):
            torque = rng.normal(size=7).astype(np.float32)
            if nan_at == (ep, fi):
                torque[2] = np.nan
            rows.append(
                {
                    "observation.state.cartesian": rng.normal(size=9).astype(np.float32),
                    "observation.state.gripper": rng.uniform(size=1).astype(np.float32),
                    "observation.state": rng.normal(size=10).astype(np.float32),
                    "action": rng.normal(size=10).astype(np.float32),
                    "extra.ext_torque": torque,
                    # a 2 x 3 taxel grid stored nested, as a recorder may do
                    "extra.tactile.left": [list(r) for r in rng.uniform(size=(2, 3))],
                    "extra.target_joints": np.zeros(7, np.float32),
                    "episode_index": ep,
                    "frame_index": fi,
                    "index": index,
                }
            )
            index += 1
    frames = pd.DataFrame(rows)

    total = len(frames)
    json.dump(
        {"codebase_version": "v3.0" if layout == "v3" else "v2.1", "features": _features()},
        open(root / "meta" / "info.json", "w"),
    )
    json.dump(
        {
            "observation.state": _stat_block(10, total),
            "action": _stat_block(10, total),
            "extra.ext_torque": _stat_block(7, total),
            "observation.images.cam": {"mean": [[[0.5]]]},
        },
        open(root / "meta" / "stats.json", "w"),
    )

    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    if layout == "v2":
        for ep in range(len(lengths)):
            frames[frames["episode_index"] == ep].to_parquet(
                data / f"episode_{ep:06d}.parquet", index=False
            )
        with open(root / "meta" / "episodes_stats.jsonl", "w") as f:
            for ep, n in enumerate(lengths):
                stats = {"observation.state": _stat_block(10, n), "action": _stat_block(10, n)}
                f.write(json.dumps({"episode_index": ep, "stats": stats}) + "\n")
    else:
        frames.to_parquet(data / "file-000.parquet", index=False)
        ep_rows = []
        for ep, n in enumerate(lengths):
            row = {"episode_index": ep, "length": n}
            for key, dim in (("observation.state", 10), ("action", 10)):
                for stat in STATS:
                    row[f"stats/{key}/{stat}"] = (
                        np.array([n]) if stat == "count" else np.zeros(dim, np.float32)
                    )
            ep_rows.append(row)
        ep_dir = root / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True)
        pd.DataFrame(ep_rows).to_parquet(ep_dir / "file-000.parquet", index=False)
    return frames


def _read_frames(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    return (
        pd.concat([pd.read_parquet(f) for f in files]).sort_values("index").reset_index(drop=True)
    )


def _flat(cell) -> np.ndarray:
    arr = np.asarray(cell)
    if arr.dtype == object:
        return np.concatenate([_flat(c) for c in arr])
    return arr.astype(np.float32).reshape(-1)


def _digest(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _argv(src, out, columns, **flags):
    argv = ["--input", str(src), "--output", str(out), "--columns", *columns]
    argv += ["--log-level", "CRITICAL"]
    return argv + ["--" + flag.replace("_", "-") for flag, on in flags.items() if on]


def _run(tool, src, out, columns, **flags):
    """As from the command line: returns the exit code."""
    return tool.main(_argv(src, out, columns, **flags))


def _expected(frames: pd.DataFrame, columns) -> np.ndarray:
    return np.stack(
        [np.concatenate([_flat(row[c]) for c in columns]) for _, row in frames.iterrows()]
    )


def _check(layout, columns):
    tool = _tool()
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        truth = _make_dataset(src, layout)
        before = _digest(src)
        assert _run(tool, src, out, columns) == 0
        assert _digest(src) == before, "the input dataset must not change"

        expected = _expected(truth, columns)
        frames = _read_frames(out)

        # 1. the new column
        got = np.stack([np.asarray(v, np.float32) for v in frames[KEY]])
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, expected)

        # 2. everything else as it was
        assert list(frames.columns) == [*truth.columns, KEY]
        for col in truth.columns:
            for a, b in zip(truth[col], frames[col]):
                np.testing.assert_array_equal(_flat(a), _flat(b))
        for rel, digest in before.items():
            if rel.startswith("videos/"):
                assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == digest

        # 3. the feature
        features = json.load(open(out / "meta" / "info.json"))["features"]
        feature = features[KEY]
        assert feature["dtype"] == "float32" and feature["shape"] == [expected.shape[1]]
        assert len(feature["names"]) == expected.shape[1]
        assert len(set(feature["names"])) == expected.shape[1], "names must be unique"
        order = list(features)
        assert order.index(KEY) == order.index("observation.state") + 1
        assert {k: v for k, v in features.items() if k != KEY} == _features()

        # 4. statistics, in every place, with the stat names of observation.state
        stats = json.load(open(out / "meta" / "stats.json"))
        assert set(stats[KEY]) == set(STATS)
        np.testing.assert_allclose(stats[KEY]["min"], expected.min(0), rtol=1e-6)
        np.testing.assert_allclose(stats[KEY]["max"], expected.max(0), rtol=1e-6)
        np.testing.assert_allclose(stats[KEY]["mean"], expected.mean(0), rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(stats[KEY]["std"], expected.std(0), rtol=1e-4)
        np.testing.assert_allclose(
            stats[KEY]["q99"], np.quantile(expected.astype(np.float64), 0.99, axis=0), rtol=1e-5
        )
        assert stats[KEY]["count"] == [len(expected)]
        assert stats["action"] == _stat_block(10, len(expected)), "other keys untouched"

        episodes = truth["episode_index"].to_numpy()
        if layout == "v3":
            ep = pd.read_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
            for stat in STATS:
                assert f"stats/{KEY}/{stat}" in ep.columns
            for _, row in ep.iterrows():
                part = expected[episodes == row["episode_index"]]
                np.testing.assert_allclose(row[f"stats/{KEY}/max"], part.max(0), rtol=1e-6)
                np.testing.assert_allclose(
                    row[f"stats/{KEY}/mean"], part.mean(0), rtol=1e-5, atol=1e-7
                )
                assert np.asarray(row[f"stats/{KEY}/count"]).tolist() == [len(part)]
                # same form as the key it was modelled on
                assert np.asarray(row[f"stats/{KEY}/max"]).dtype == np.float32
                assert len(row["stats/action/max"]) == 10
        else:
            lines = [json.loads(x) for x in open(out / "meta" / "episodes_stats.jsonl")]
            assert len(lines) == 3
            for rec in lines:
                part = expected[episodes == rec["episode_index"]]
                assert set(rec["stats"][KEY]) == set(STATS)
                np.testing.assert_allclose(rec["stats"][KEY]["min"], part.min(0), rtol=1e-6)
                assert rec["stats"][KEY]["count"] == [len(part)]

        # 5. the note that says what the column is made of
        note = json.load(open(out / "meta" / "environment_state.json"))
        assert note["key"] == KEY and note["dim"] == expected.shape[1]
        assert [s["column"] for s in note["sources"]] == list(columns)
        start = 0
        for source in note["sources"]:
            width = int(np.prod(_features()[source["column"]]["shape"]))
            assert source["slice"] == [start, start + width]
            start += width


def test_one_column_v3():
    _check("v3", ["extra.ext_torque"])


def test_one_column_v2():
    _check("v2", ["extra.ext_torque"])


def test_two_columns_in_the_order_given_v3():
    _check("v3", ["extra.ext_torque", "extra.tactile.left"])


def test_the_order_given_is_the_order_stored_v2():
    _check("v2", ["extra.tactile.left", "extra.ext_torque"])


def test_a_nested_grid_is_flattened_row_by_row():
    tool = _tool()
    df = pd.DataFrame({"extra.tactile.left": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]})
    joined = tool.join_columns(df, ["extra.tactile.left"])
    np.testing.assert_array_equal(joined, [[1, 2, 3, 4, 5, 6]])


def _refused(columns, match, prepare=None, **flags):
    tool = _tool()
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        _make_dataset(src, "v3", **(prepare or {}))
        before = _digest(src)
        args = tool.build_parser().parse_args(_argv(src, out, columns, **flags))
        try:
            tool.run(args)
        except tool.Refused as refusal:
            assert match in str(refusal), str(refusal)
        else:
            raise AssertionError(f"expected a refusal mentioning {match!r}")
        # from the command line the same request is one error line and exit code 2
        assert _run(tool, src, out, columns, **flags) == 2
        assert not out.exists(), "a refused run must write nothing"
        assert _digest(src) == before


def test_refuses_a_column_that_is_not_there():
    _refused(["extra.no_such_column"], "extra.no_such_column")


def test_refuses_a_column_that_never_changes():
    """The handheld datasets carry zero placeholders for the robot-only columns."""
    _refused(["extra.ext_torque", "extra.target_joints"], "never changes")


def test_a_constant_column_can_be_allowed_on_purpose():
    tool = _tool()
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        _make_dataset(src, "v3")
        assert _run(tool, src, out, ["extra.target_joints"], allow_constant=True) == 0
        assert _read_frames(out)[KEY].map(len).eq(7).all()


def test_refuses_nan():
    _refused(["extra.ext_torque"], "NaN", prepare={"nan_at": (1, 2)})


def test_refuses_an_image_column():
    _refused(["observation.images.cam"], "observation.images.cam")


def test_refuses_the_same_column_twice():
    _refused(["extra.ext_torque", "extra.ext_torque"], "twice")


def test_refuses_a_dataset_that_already_has_the_key():
    tool = _tool()
    with tempfile.TemporaryDirectory() as tmp:
        src, once, twice = (Path(tmp) / n for n in ("src", "once", "twice"))
        _make_dataset(src, "v3")
        assert _run(tool, src, once, ["extra.ext_torque"]) == 0
        args = tool.build_parser().parse_args(_argv(once, twice, ["extra.tactile.left"]))
        try:
            tool.run(args)
        except tool.Refused as refusal:
            assert "already has" in str(refusal)
        else:
            raise AssertionError("expected a refusal")
        assert not twice.exists()


def test_refuses_an_existing_output_folder():
    tool = _tool()
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        _make_dataset(src, "v3")
        out.mkdir()
        assert _run(tool, src, out, ["extra.ext_torque"]) == 2
        assert list(out.iterdir()) == []


def test_dry_run_writes_nothing():
    tool = _tool()
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        _make_dataset(src, "v2")
        before = _digest(src)
        assert _run(tool, src, out, ["extra.ext_torque"], dry_run=True) == 0
        assert not out.exists() and _digest(src) == before


def test_the_result_has_no_schema_drift():
    """The repository's own checker: parquet columns and info.json must agree."""
    tool = _tool()
    drift = _load_by_path("check_schema_drift", SCRIPTS / "check_schema_drift.py")
    for layout in ("v2", "v3"):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "src", Path(tmp) / "out"
            _make_dataset(src, layout)
            assert _run(tool, src, out, ["extra.ext_torque", "extra.tactile.left"]) == 0
            orphan, missing = drift.report(out)
            assert orphan == set() and missing == set()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                import traceback

                print(f"FAIL  {name}: {exc}")
                traceback.print_exc()
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
    raise SystemExit(1 if failures else 0)
