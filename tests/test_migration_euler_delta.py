"""Round-trip tests for migrate_euler_delta_to_rot6d.py on synthetic datasets.

Builds legacy-schema datasets (Euler pose + delta-command action) in BOTH
on-disk layouts and runs the real migrate():
  - v2.x: data/chunk-*/episode_XXXXXX.parquet (one episode per file)
  - v3.0: data/chunk-*/file-*.parquet (multi-episode) + meta/episodes stats parquet

Asserts:
  1. cartesian Euler(6) -> rot6d(9) matches ground-truth rotations exactly.
  2. observation.state is rebuilt in sub-feature order (6+1+... -> 9+1+...).
  3. action[t] = absolute pose at t+1 with gripper from t+1; the LAST frame of
     each episode repeats its own pose; the lookahead NEVER crosses an episode
     boundary inside a shared v3.0 parquet.
  4. info.json feature shapes/names and stats (aggregate + v3.0 per-episode
     stats columns incl. quantiles) are updated; untouched keys stay untouched.
  5. video files are byte-identical (no re-encode).
  6. State-composition guard aborts on a scrambled observation.state.

Runs under pytest or directly:  python tests/test_migration_euler_delta.py
Requires numpy, scipy, pandas, pyarrow (the [postprocess] extra). No lerobot.
"""

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]


def _load_mig():
    spec = importlib.util.spec_from_file_location(
        "mig", REPO / "crisp_gym" / "scripts" / "migrate_euler_delta_to_rot6d.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _decode_rot6d(v9):
    a1, a2 = np.asarray(v9[3:6], float), np.asarray(v9[6:9], float)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    return np.stack([b1, b2, np.cross(b1, b2)])


def _features(state_names):
    return {
        "observation.state.cartesian": {"dtype": "float32", "shape": [6],
                                        "names": ["x", "y", "z", "roll", "pitch", "yaw"]},
        "observation.state.gripper": {"dtype": "float32", "shape": [1], "names": ["gripper"]},
        "observation.state": {"dtype": "float32", "shape": [7], "names": state_names},
        "observation.images.env_cam": {"dtype": "video", "shape": [64, 64, 3],
                                       "names": ["h", "w", "c"]},
        "action": {"dtype": "float32", "shape": [7],
                   "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
    }


def _make_dataset(root: Path, episodes, layout: str, scramble_state: bool = False):
    """episodes: list of (poses_R, poses_p, grips) per episode."""
    (root / "meta").mkdir(parents=True)
    vdir = root / "videos" / "observation.images.env_cam" / "chunk-000"
    vdir.mkdir(parents=True)
    (vdir / "file-000.mp4").write_bytes(b"FAKEVIDEO_BYTES")

    state_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    json.dump({"features": _features(state_names), "codebase_version": "v3.0"},
              open(root / "meta" / "info.json", "w"))
    json.dump({
        "observation.state.cartesian": {"mean": [0] * 6, "std": [1] * 6, "min": [0] * 6,
                                        "max": [1] * 6, "count": [1], "q50": [0] * 6},
        "observation.state": {"mean": [0] * 7, "std": [1] * 7, "min": [0] * 7,
                              "max": [1] * 7, "count": [1]},
        "action": {"mean": [0] * 7, "std": [1] * 7, "min": [0] * 7,
                   "max": [1] * 7, "count": [1]},
        "observation.images.env_cam": {"mean": [[[0.5]]]},
    }, open(root / "meta" / "stats.json", "w"))

    rows, idx = [], 0
    for ep, (Rs, ps, gs) in enumerate(episodes):
        for fi in range(len(Rs)):
            cart = np.concatenate([ps[fi], Rs[fi].as_euler("xyz")]).astype(np.float32)
            grip = np.array([gs[fi]], np.float32)
            state = np.concatenate([cart, grip]).astype(np.float32)
            if scramble_state:
                state = np.concatenate([grip, cart]).astype(np.float32)  # wrong order
            rows.append({
                "observation.state.cartesian": cart,
                "observation.state.gripper": grip,
                "observation.state": state,
                "action": np.random.randn(7).astype(np.float32),  # legacy delta (discarded)
                "episode_index": ep, "frame_index": fi, "index": idx,
            })
            idx += 1

    ddir = root / "data" / "chunk-000"
    ddir.mkdir(parents=True)
    if layout == "v2":
        df = pd.DataFrame(rows)
        for ep in sorted({r["episode_index"] for r in rows}):
            df[df["episode_index"] == ep].to_parquet(
                ddir / f"episode_{ep:06d}.parquet", index=False)
    else:  # v3: all episodes in ONE file + per-episode stats parquet
        pd.DataFrame(rows).to_parquet(ddir / "file-000.parquet", index=False)
        edir = root / "meta" / "episodes" / "chunk-000"
        edir.mkdir(parents=True)
        ep_rows = []
        for ep, (Rs, _, _) in enumerate(episodes):
            r = {"episode_index": ep, "length": len(Rs)}
            for k, D in [("observation.state.cartesian", 6), ("observation.state", 7),
                         ("action", 7), ("observation.state.gripper", 1)]:
                for st in ("min", "max", "mean", "std"):
                    r[f"stats/{k}/{st}"] = np.zeros(D, np.float32)
                r[f"stats/{k}/count"] = np.array([len(Rs)])
                r[f"stats/{k}/q50"] = np.zeros(D, np.float32)
            ep_rows.append(r)
        pd.DataFrame(ep_rows).to_parquet(edir / "file-000.parquet", index=False)


def _rand_episode(rng, n):
    Rs = [Rotation.random(random_state=int(rng.integers(1 << 30)))]
    ps = [rng.normal(size=3)]
    for _ in range(n - 1):
        Rs.append(Rotation.from_rotvec(rng.normal(scale=0.05, size=3)) * Rs[-1])
        ps.append(ps[-1] + rng.normal(scale=0.01, size=3))
    gs = rng.uniform(0, 1, size=n)
    return Rs, ps, gs


def _run_migration(mig, src: Path, dst: Path, dry_run=False):
    args = types.SimpleNamespace(input=str(src), output=str(dst),
                                 no_action_gripper=False, dry_run=dry_run,
                                 log_level="ERROR")
    return mig.migrate(args)


def _check_migrated(out: Path, episodes):
    frames = pd.concat(
        [pd.read_parquet(f) for f in sorted((out / "data").rglob("*.parquet"))]
    ).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    info = json.load(open(out / "meta" / "info.json"))
    assert info["features"]["observation.state.cartesian"]["shape"] == [9]
    assert info["features"]["observation.state"]["shape"] == [10]
    assert info["features"]["action"]["shape"] == [10]
    assert info["features"]["action"]["names"][-1] == "gripper"

    row = 0
    for Rs, ps, gs in episodes:
        n = len(Rs)
        for fi in range(n):
            r = frames.iloc[row + fi]
            cart9 = np.asarray(r["observation.state.cartesian"], float)
            # 1. rotation ground truth
            np.testing.assert_allclose(_decode_rot6d(cart9), Rs[fi].as_matrix(), atol=1e-6)
            np.testing.assert_allclose(cart9[:3], ps[fi], atol=1e-6)
            # 2. state = cartesian(9) + gripper(1), in order
            state = np.asarray(r["observation.state"], float)
            np.testing.assert_allclose(state[:9], cart9, atol=1e-7)
            np.testing.assert_allclose(state[9], gs[fi], atol=1e-6)
            # 3. action = pose at t+1 (last frame repeats), gripper from t+1
            nxt = min(fi + 1, n - 1)
            act = np.asarray(r["action"], float)
            np.testing.assert_allclose(_decode_rot6d(act), Rs[nxt].as_matrix(), atol=1e-6)
            np.testing.assert_allclose(act[:3], ps[nxt], atol=1e-6)
            np.testing.assert_allclose(act[9], gs[nxt], atol=1e-6)
        row += n

    # 4. stats updated (aggregate incl. quantiles)
    stats = json.load(open(out / "meta" / "stats.json"))
    assert len(stats["observation.state.cartesian"]["mean"]) == 9
    assert len(stats["observation.state.cartesian"]["q50"]) == 9
    assert len(stats["observation.state"]["mean"]) == 10
    assert len(stats["action"]["mean"]) == 10
    assert "observation.images.env_cam" in stats  # untouched key kept

    # 5. video byte-identical
    v = out / "videos" / "observation.images.env_cam" / "chunk-000" / "file-000.mp4"
    assert v.read_bytes() == b"FAKEVIDEO_BYTES"


def test_migration_v2_layout():
    mig = _load_mig()
    rng = np.random.default_rng(0)
    episodes = [_rand_episode(rng, 5), _rand_episode(rng, 3)]
    root = Path(tempfile.mkdtemp()) / "ds"
    _make_dataset(root, episodes, layout="v2")
    out = root.parent / "ds_rot6d"
    assert _run_migration(mig, root, out) == 0
    _check_migrated(out, episodes)


def test_migration_v3_layout_and_episode_boundary():
    mig = _load_mig()
    rng = np.random.default_rng(1)
    episodes = [_rand_episode(rng, 4), _rand_episode(rng, 6), _rand_episode(rng, 2)]
    root = Path(tempfile.mkdtemp()) / "ds"
    _make_dataset(root, episodes, layout="v3")
    out = root.parent / "ds_rot6d"
    assert _run_migration(mig, root, out) == 0
    _check_migrated(out, episodes)  # includes boundary check via last-frame repeat

    # v3.0 per-episode stats columns updated to new dims (incl. quantiles),
    # untouched keys (gripper) left alone
    ep = pd.read_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert len(ep["stats/observation.state.cartesian/mean"].iloc[0]) == 9
    assert len(ep["stats/observation.state.cartesian/q50"].iloc[0]) == 9
    assert len(ep["stats/observation.state/mean"].iloc[0]) == 10
    assert len(ep["stats/action/mean"].iloc[0]) == 10
    assert len(ep["stats/observation.state.gripper/mean"].iloc[0]) == 1
    assert int(np.asarray(ep["stats/observation.state.cartesian/count"].iloc[0])[0]) == 4


def test_dry_run_writes_nothing():
    mig = _load_mig()
    rng = np.random.default_rng(2)
    root = Path(tempfile.mkdtemp()) / "ds"
    _make_dataset(root, [_rand_episode(rng, 3)], layout="v3")
    out = root.parent / "ds_out"
    assert _run_migration(mig, root, out, dry_run=True) == 0
    assert not out.exists()


def test_state_composition_guard():
    mig = _load_mig()
    rng = np.random.default_rng(3)
    root = Path(tempfile.mkdtemp()) / "ds"
    _make_dataset(root, [_rand_episode(rng, 3)], layout="v2", scramble_state=True)
    out = root.parent / "ds_out"
    try:
        _run_migration(mig, root, out)
        raise AssertionError("scrambled observation.state was not rejected")
    except ValueError as e:
        assert "ordered concat" in str(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} migration tests passed.")
