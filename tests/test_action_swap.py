"""Test the target-cartesian action-swap training wrapper (no lerobot/torch needed).

train_action_from_target_cartesian.py replaces the ARM dims of `action` with
extra.target_cartesian for training. The risk it guards against is the FACTR/JIC
dataset, where target_cartesian is constant per episode — training on it is
worthless. These tests pin: the swap keeps the gripper and replaces the arm with
the windowed target; windowing is injected into delta_indices; the constant-
column guard refuses; a window-shape mismatch is caught; and stats are recomputed.

torch is stubbed (from_numpy = identity), so this runs anywhere numpy is present.

Run:  python tests/test_action_swap.py   (or via pytest)
"""

import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load():
    if "torch" not in sys.modules:
        t = types.ModuleType("torch")

        class _T(np.ndarray):
            pass

        t.Tensor = _T
        t.from_numpy = lambda a: a
        t.tensor = lambda a, dtype=None: np.asarray(a, dtype=np.float32)
        t.float32 = np.float32
        sys.modules["torch"] = t
    return SourceFileLoader(
        "tsw", str(REPO / "crisp_gym" / "scripts" / "train_action_from_target_cartesian.py")
    ).load_module()


class _Meta:
    def __init__(self, feats):
        self.features = feats
        self.stats = {}
        self.fps = 15


class _HF:
    def __init__(self, cols):
        self._c = cols

    def __getitem__(self, k):
        return self._c[k]


class _FakeDS:
    """Minimal LeRobotDataset stand-in: 2 episodes, an action horizon of H."""

    def __init__(self, n=8, h=4, tc_constant=False):
        self.meta = _Meta({
            "action": {"shape": [10], "names": list("abcdefghij")},
            "extra.target_cartesian": {"shape": [9]},
        })
        self.delta_indices = {"action": list(range(h))}
        self.delta_timestamps = {"action": [i / 15 for i in range(h)]}
        self.n, self.h = n, h
        ep = np.repeat([0, 1], n // 2)
        if tc_constant:
            tc = np.repeat(np.array([[0.5] * 9, [0.7] * 9]), n // 2, axis=0)
        else:
            tc = np.linspace(0, 1, n)[:, None] * np.ones((1, 9))
        self._tc = tc
        self.hf_dataset = _HF({
            "extra.target_cartesian": tc.tolist(),
            "episode_index": ep.tolist(),
        })

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        h = self.h
        act = np.zeros((h, 10), dtype=np.float32)
        act[:, :9] = -1.0        # sentinel: proves the arm was replaced
        act[:, 9] = 0.3          # gripper: must survive
        tc = self._tc[idx:idx + h]
        if tc.shape[0] < h:
            tc = np.vstack([tc, np.repeat(tc[-1:], h - tc.shape[0], 0)])
        item = {"action": act}
        if "extra.target_cartesian" in self.delta_indices:
            item["extra.target_cartesian"] = tc.astype(np.float32)
        return item


def test_swap_replaces_arm_keeps_gripper():
    m = _load()
    ds = _FakeDS(tc_constant=False)
    w = m.TargetCartesianActionDataset(ds)
    a = np.asarray(w[0]["action"])
    assert a.shape == (4, 10)
    assert np.allclose(a[:, 9], 0.3), "gripper (dim 9) must be preserved"
    assert np.allclose(a[:, :9], ds._tc[0:4]), "arm dims must equal windowed target_cartesian"


def test_windowing_injected_into_delta_indices():
    m = _load()
    ds = _FakeDS()
    m.TargetCartesianActionDataset(ds)
    assert ds.delta_indices["extra.target_cartesian"] == ds.delta_indices["action"]


def test_constant_target_is_refused():
    m = _load()
    try:
        m.TargetCartesianActionDataset(_FakeDS(tc_constant=True))
        raise AssertionError("constant target_cartesian (FACTR/JIC) not rejected")
    except ValueError as e:
        assert "CONSTANT" in str(e)


def test_window_shape_mismatch_is_caught():
    m = _load()

    class _BadDS(_FakeDS):
        def __getitem__(self, idx):
            it = _FakeDS.__getitem__(self, idx)
            it["extra.target_cartesian"] = np.zeros((9,), np.float32)  # not windowed
            return it

    bad = _BadDS()
    w = m.TargetCartesianActionDataset(bad)
    try:
        w.convert_item(bad[0])
        raise AssertionError("mismatched window shapes not caught")
    except ValueError as e:
        assert "disagree" in str(e)


class _LazyDS(_FakeDS):
    """delta_indices is EMPTY until the first __getitem__ populates it.

    Reproduces the lerobot version seen in the field: `action` is windowed at
    query time via delta_indices, but that dict is empty when make_dataset
    returns, so the wrapper's eager injection at __init__ finds no 'action' to
    mirror. The target is windowed ONLY if its key is present in delta_indices
    at query time — exactly the behaviour the __getitem__ re-query must handle.
    """

    def __init__(self, n=8, h=4):
        super().__init__(n=n, h=h, tc_constant=False)
        self.delta_indices = {}          # empty at construction (lazy)
        self.delta_timestamps = {}

    def __getitem__(self, idx):
        self.delta_indices.setdefault("action", list(range(self.h)))  # lazily built
        h = self.h
        act = np.zeros((h, 10), dtype=np.float32)
        act[:, :9] = -1.0
        act[:, 9] = 0.3
        item = {"action": act}
        if "extra.target_cartesian" in self.delta_indices:   # windowed only if present
            tc = self._tc[idx:idx + h]
            if tc.shape[0] < h:
                tc = np.vstack([tc, np.repeat(tc[-1:], h - tc.shape[0], 0)])
            item["extra.target_cartesian"] = tc.astype(np.float32)
        else:
            item["extra.target_cartesian"] = self._tc[idx].astype(np.float32)  # (9,) single
        return item


def test_lazy_delta_indices_requery_aligns_window():
    m = _load()
    ds = _LazyDS()
    w = m.TargetCartesianActionDataset(ds)   # eager inject finds nothing (empty)
    assert "extra.target_cartesian" not in ds.delta_indices
    a = np.asarray(w[0]["action"])           # first query: mismatch -> inject -> re-query
    assert a.shape == (4, 10), "target must be windowed to the action horizon after re-query"
    assert np.allclose(a[:, 9], 0.3), "gripper preserved"
    assert np.allclose(a[:, :9], ds._tc[0:4]), "arm == windowed target after re-query"
    assert ds.delta_indices["extra.target_cartesian"] == ds.delta_indices["action"]


def test_getattr_guarded_against_unpickle_recursion():
    """Under a 'spawn' DataLoader the wrapper is pickled; during unpickling
    __dict__ is empty, so __getattr__ must raise AttributeError, not recurse
    into self._dataset forever (the RecursionError seen on lerobot 0.6.1)."""
    m = _load()
    obj = object.__new__(m.TargetCartesianActionDataset)  # __init__ NOT run: no _dataset
    try:
        obj.some_missing_attr
        raise AssertionError("expected AttributeError")
    except RecursionError:
        raise AssertionError("__getattr__ recurses when _dataset is unset")
    except AttributeError:
        pass


def test_stats_recomputed_for_action():
    m = _load()
    w = m.TargetCartesianActionDataset(_FakeDS(tc_constant=False))
    m.recompute_action_stats(w, num_samples=8)
    assert "action" in w.meta.stats
    assert np.asarray(w.meta.stats["action"]["mean"]).shape == (10,)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} action-swap tests passed.")
