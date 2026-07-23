"""Unit tests for the relative-pose math (HANDOFF §3 invariants).

Covers, on synthetic SE(3) data (no robot, no dataset needed):
  1. rot6d <-> matrix round-trip is exact on random rotations.
  2. Current obs frame relative to itself -> identity pose.
  3. Round-trip T_base @ T_rel == T_abs, compared as MATRICES.
  4. World-frame invariance: left-multiplying ALL absolute poses by one random
     rigid transform leaves the relative outputs unchanged (the property that
     makes handheld->robot transfer work).
  5. Gripper dim passes through the conversion bit-exact.
  6. RelativePoseDataset.convert_item: identity obs, relative action, wrt_start
     shape (n_obs, 6) and rotation-only.
  7. RemotePolicy chunk composition uses the OBS-TIME base: all steps of a
     chunk reproduce ground truth even while the robot moves (regression for
     the execution-time-base bug).

Runs under pytest or directly:  python tests/test_pose_math.py
torch is stubbed if unavailable (the math under test is pure numpy).
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # remote_policy lazily imports crisp_gym.util.rot6d
    sys.path.insert(0, str(REPO))

# ── torch stub (lerobot_relative_pose imports torch; its math is numpy) ──────
try:
    import torch  # noqa: F401
except ImportError:
    torch = types.ModuleType("torch")

    class _Tensor:
        def __init__(self, a):
            self.a = np.asarray(a)

        def numpy(self):
            return self.a

        @property
        def ndim(self):
            return self.a.ndim

        @property
        def shape(self):
            return self.a.shape

    torch.Tensor = _Tensor
    torch.from_numpy = lambda a: _Tensor(a)
    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.Dataset = object
    sys.modules["torch"] = torch
    sys.modules["torch.utils"] = torch.utils
    sys.modules["torch.utils.data"] = torch.utils.data


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lrp = _load("lrp", REPO / "crisp_gym" / "scripts" / "lerobot_relative_pose.py")

_rng = np.random.default_rng(42)


def _rand_R() -> Rotation:
    return Rotation.random(random_state=int(_rng.integers(1 << 30)))


def _rand_pose9d() -> np.ndarray:
    R = _rand_R()
    return np.concatenate([_rng.normal(size=3), R.as_matrix()[:2, :].flatten()])


def _rand_T() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rand_R().as_matrix()
    T[:3, 3] = _rng.normal(size=3)
    return T


# ── 1. rot6d round-trip ───────────────────────────────────────────────────────

def test_rot6d_matrix_roundtrip():
    for _ in range(100):
        R = _rand_R().as_matrix()
        d6 = lrp.mat_to_rot6d(R)
        assert d6.shape == (6,)
        np.testing.assert_allclose(lrp.rot6d_to_mat(d6), R, atol=1e-12)
        # noisy 6d still decodes to a valid rotation (Zhou et al. continuity)
        Rn = lrp.rot6d_to_mat(d6 + _rng.normal(scale=0.1, size=6))
        np.testing.assert_allclose(Rn @ Rn.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(Rn), 1.0)


def test_pose9d_mat_roundtrip():
    for _ in range(100):
        p = _rand_pose9d()
        np.testing.assert_allclose(lrp.mat_to_pose9d(lrp.pose9d_to_mat(p)), p, atol=1e-12)


# ── 2-4. relative-pose invariants ─────────────────────────────────────────────

def test_relative_to_self_is_identity():
    identity = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64)
    for _ in range(50):
        p = _rand_pose9d()
        np.testing.assert_allclose(lrp.make_relative(p, p[None])[0], identity, atol=1e-12)


def test_roundtrip_as_matrices():
    for _ in range(50):
        base, target = _rand_pose9d(), _rand_pose9d()
        rel = lrp.make_relative(base, target[None])[0]
        T_recon = lrp.pose9d_to_mat(base) @ lrp.pose9d_to_mat(rel)
        # compare as MATRICES, never as raw 9d vectors (HANDOFF §3.3)
        np.testing.assert_allclose(T_recon, lrp.pose9d_to_mat(target), atol=1e-10)


def test_world_frame_invariance():
    for _ in range(50):
        base, target = _rand_pose9d(), _rand_pose9d()
        rel = lrp.make_relative(base, target[None])[0]
        G = _rand_T()  # one arbitrary global frame change applied to EVERYTHING
        base_g = lrp.mat_to_pose9d(G @ lrp.pose9d_to_mat(base))
        target_g = lrp.mat_to_pose9d(G @ lrp.pose9d_to_mat(target))
        rel_g = lrp.make_relative(base_g, target_g[None])[0]
        np.testing.assert_allclose(rel_g, rel, atol=1e-9)


# ── 5-6. convert_item (dataset wrapper semantics) ─────────────────────────────

def _fake_item(n_obs=2, horizon=4, grip=0.7):
    obs = np.stack([_rand_pose9d() for _ in range(n_obs)]).astype(np.float32)
    act = np.stack(
        [np.concatenate([_rand_pose9d(), [grip]]) for _ in range(horizon)]
    ).astype(np.float32)
    import torch as _t

    return {
        "observation.state.cartesian": _t.from_numpy(obs),
        "action": _t.from_numpy(act),
    }, obs, act


def test_convert_item_semantics():
    ds = lrp.RelativePoseDataset.__new__(lrp.RelativePoseDataset)
    ds._start_pose_noise_scale = 0.0
    ds._episode_start_pose = {}

    item, obs_abs, act_abs = _fake_item()
    start_pose = obs_abs[0][:9].astype(np.float64)
    out = ds.convert_item(dict(item), start_pose=start_pose)

    def npy(v):
        return v.numpy() if hasattr(v, "numpy") else np.asarray(v)

    obs_rel = npy(out["observation.state.cartesian"])
    act_rel = npy(out["action"])
    base = obs_abs[-1][:9].astype(np.float64)  # base = LAST obs timestep

    # current frame -> identity
    identity = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    np.testing.assert_allclose(obs_rel[-1], identity, atol=1e-6)
    # every action recomposes to its absolute pose (matrix compare)
    for k in range(act_abs.shape[0]):
        T = lrp.pose9d_to_mat(base) @ lrp.pose9d_to_mat(act_rel[k][:9].astype(np.float64))
        np.testing.assert_allclose(
            T, lrp.pose9d_to_mat(act_abs[k][:9].astype(np.float64)), atol=1e-5
        )
        # gripper dim bit-exact
        assert act_rel[k][9] == act_abs[k][9]
    # wrt_start: shape (n_obs, 6), decodes to valid rotations (rotation-only)
    wrt = npy(out[lrp.WRT_START_KEY])
    assert wrt.shape == (obs_abs.shape[0], 6)
    for row in wrt:
        R = lrp.rot6d_to_mat(row.astype(np.float64))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)


# ── 7. RemotePolicy obs-time chunk base (regression) ──────────────────────────

def test_remote_policy_chunk_base_is_obs_time():
    # stub the package import chain used by remote_policy
    pkg = types.ModuleType("crisp_gym")
    pol = types.ModuleType("crisp_gym.policy")
    polpol = types.ModuleType("crisp_gym.policy.policy")
    polpol.Policy = object
    polpol.Action = object
    polpol.Observation = dict
    polpol.register_policy = lambda name: (lambda cls: cls)
    saved = {k: sys.modules.get(k) for k in
             ("crisp_gym", "crisp_gym.policy", "crisp_gym.policy.policy")}
    sys.modules.update(
        {"crisp_gym": pkg, "crisp_gym.policy": pol, "crisp_gym.policy.policy": polpol}
    )
    try:
        rp = _load("rp_test", REPO / "crisp_gym" / "policy" / "remote_policy.py")
    finally:
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)

    # smooth ground-truth trajectory; obs at k=0
    d = np.array([0.004, 0.002, 0.001])
    W = Rotation.from_rotvec([0, 0, 0.02])
    T = [np.eye(4)]
    for _ in range(9):
        Tk = np.eye(4)
        Tk[:3, :3] = (W * Rotation.from_matrix(T[-1][:3, :3])).as_matrix()
        Tk[:3, 3] = T[-1][:3, 3] + d
        T.append(Tk)
    T = np.stack(T)
    T0inv = np.linalg.inv(T[0])
    chunk = []
    for k in range(8):
        Tr = T0inv @ T[k + 1]
        chunk.append(np.concatenate([Tr[:3, 3], Tr[:3, :3][:2, :].flatten(), [0.5]]))

    class _FakeRobot:
        k = 0

        @property
        def end_effector_pose(self):
            o = types.SimpleNamespace()
            o.position = T[self.k][:3, 3].copy()
            o.orientation = Rotation.from_matrix(T[self.k][:3, :3])
            return o

    env = types.SimpleNamespace(
        robot=_FakeRobot(),
        config=types.SimpleNamespace(orientation_representation="quaternion"),
    )
    policy = rp.RemotePolicy.__new__(rp.RemotePolicy)
    policy.env = env
    policy.relative_actions = True

    base = policy._current_pose_mat()  # snapshot at obs time (k=0)
    for k in range(8):
        env.robot.k = k  # robot HAS MOVED by execution time
        cmd = policy._compose_relative(np.asarray(chunk[k]), base)
        np.testing.assert_allclose(cmd[:3], T[k + 1][:3, 3], atol=1e-9)
        R_cmd = Rotation.from_quat(cmd[3:7]).as_matrix()
        np.testing.assert_allclose(R_cmd, T[k + 1][:3, :3], atol=1e-9)
        assert np.isclose(cmd[7], 0.5)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} pose-math tests passed.")
