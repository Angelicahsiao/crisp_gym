"""Tests for the local relative-pose deployment path (RelativeLerobotPolicy).

Pure-numpy checks of the client-side math against the TRAINING-side reference
(scripts/lerobot_relative_pose.py, loaded with the same torch stub as
tests/test_pose_math.py):

  1. compose_relative_pose inverts the training conversion: for every step of
     a synthetic trajectory, T_base(obs) @ T_rel(train) == T_abs — as MATRICES.
  2. Obs-time base invariant: composing a whole chunk against the single
     obs-time base reproduces ground truth even while the robot "moves".
  3. Gripper unit conversions round-trip and match the recording semantics
     (clip(width_m / reference_width, 0, 1)).
  4. build_obs_frame produces the training observation.state layout
     [cartesian9_abs, gripper_ref1] and rejects non-rot6d cartesian obs.

Runs under pytest or directly:  python tests/test_relative_deploy.py
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # helpers lazily import crisp_gym.util.rot6d
    sys.path.insert(0, str(REPO))

# ── torch stub (training script imports torch; its math is numpy) ────────────
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


def _load_policy_module():
    """Load relative_lerobot_policy with the crisp_gym package chain stubbed."""
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
        return _load(
            "rlp_test", REPO / "crisp_gym" / "policy" / "relative_lerobot_policy.py"
        )
    finally:
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)


rlp = _load_policy_module()
lrp = _load("lrp_ref", REPO / "crisp_gym" / "scripts" / "lerobot_relative_pose.py")


def _random_traj(n: int, seed: int) -> np.ndarray:
    """(n, 4, 4) smooth random SE(3) trajectory."""
    rng = np.random.default_rng(seed)
    T = [np.eye(4)]
    T[0][:3, :3] = Rotation.random(random_state=int(seed)).as_matrix()
    T[0][:3, 3] = rng.uniform(-0.5, 0.5, 3)
    for _ in range(n - 1):
        step = np.eye(4)
        step[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.03, 3)).as_matrix()
        step[:3, 3] = rng.normal(0, 0.01, 3)
        T.append(T[-1] @ step)
    return np.stack(T)


def _pose9(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[..., :3, 3], T[..., :2, :3].reshape(*T.shape[:-2], 6)],
                          axis=-1)


# ── 1. deploy composition inverts the training conversion ────────────────────

def test_compose_inverts_training_relative():
    for seed in range(5):
        T = _random_traj(10, seed)
        base9 = _pose9(T[0])
        rel = lrp.make_relative(base9, _pose9(T))  # training-side conversion
        for k in range(len(T)):
            T_cmd = rlp.compose_relative_pose(rel[k], T[0])
            np.testing.assert_allclose(T_cmd, T[k], atol=1e-9)


# ── 2. whole chunk composes against the ONE obs-time base ────────────────────

def test_chunk_uses_obs_time_base():
    T = _random_traj(9, seed=42)
    base = T[0]  # obs-time pose; the "robot" then moves through T[1..]
    rel_chunk = lrp.make_relative(_pose9(base), _pose9(T[1:]))
    for k, rel in enumerate(rel_chunk):
        # correct: obs-time base reproduces ground truth
        np.testing.assert_allclose(
            rlp.compose_relative_pose(rel, base), T[k + 1], atol=1e-9
        )
    # wrong: composing step k against the live pose T[k] does NOT reproduce
    # ground truth (this is the compounding bug the invariant prevents)
    live_err = np.linalg.norm(
        rlp.compose_relative_pose(rel_chunk[3], T[3])[:3, 3] - T[4][:3, 3]
    )
    assert live_err > 1e-4, "live-base composition should be measurably wrong"


# ── 3. gripper unit conversions ───────────────────────────────────────────────

def test_gripper_scaling_matches_recording():
    ref, dev_max = 0.09, 0.140  # Robotiq 2F-140
    for g_dev in [0.0, 0.2, 0.5, 0.643, 1.0]:
        g_ref = rlp.gripper_device_to_ref(g_dev, ref, dev_max)
        # recording semantics: clip(width_m / reference_width, 0, 1)
        assert abs(g_ref - np.clip(g_dev * dev_max / ref, 0, 1)) < 1e-9
    # round-trip holds wherever width_m <= reference_width (unclipped zone)
    for g_ref in [0.0, 0.3, 0.7, 1.0]:
        g_dev = rlp.gripper_ref_to_device(g_ref, ref, dev_max)
        assert abs(rlp.gripper_device_to_ref(g_dev, ref, dev_max) - g_ref) < 1e-9
        assert 0.0 <= g_dev <= 1.0
    # model outputs beyond [0,1] are clipped before scaling
    assert rlp.gripper_ref_to_device(1.7, ref, dev_max) == rlp.gripper_ref_to_device(
        1.0, ref, dev_max
    )


# ── 4. observation frame layout ───────────────────────────────────────────────

def test_build_obs_frame_layout_and_guards():
    ref, dev_max = 0.09, 0.140
    cart = _pose9(_random_traj(1, seed=7)[0]).astype(np.float32)
    obs = {
        "observation.state.cartesian": cart,
        "observation.state.gripper": np.array([0.5], dtype=np.float32),
        "observation.images.primary": np.zeros((224, 224, 3), np.uint8),
        "observation.images.wrist": np.zeros((224, 224, 3), np.uint8),
    }
    frame = rlp.build_obs_frame(obs, ref, dev_max,
                                image_keys=["observation.images.primary"])
    # training layout: [cartesian9 ABSOLUTE, gripper_ref1]
    assert frame["observation.state"].shape == (10,)
    np.testing.assert_allclose(frame["observation.state"][:9], cart, atol=0)
    expected_g = np.clip(0.5 * dev_max / ref, 0, 1)
    assert abs(frame["observation.state"][9] - expected_g) < 1e-6
    # only the requested image key is forwarded
    assert "observation.images.primary" in frame
    assert "observation.images.wrist" not in frame

    # non-rot6d cartesian obs (e.g. euler 6D) must be rejected loudly
    bad = dict(obs)
    bad["observation.state.cartesian"] = np.zeros(6, np.float32)
    try:
        rlp.build_obs_frame(bad, ref, dev_max)
        raise AssertionError("6D cartesian obs not rejected")
    except ValueError as e:
        assert "rotation_6d" in str(e)


# ── 5. training wrapper converts observation.state (the model input) ─────────

def test_training_converts_observation_state():
    T = _random_traj(2, seed=11)  # n_obs_steps = 2 window
    cart = _pose9(T).astype(np.float32)                      # (2, 9) absolute
    state = np.concatenate([cart, [[0.4], [0.5]]], axis=-1)  # (2, 10) + gripper
    ds = lrp.RelativePoseDataset(None, start_pose_noise_scale=0.0)
    item = ds.convert_item({
        "observation.state.cartesian": cart.copy(),
        "observation.state": state.copy(),
    })
    out = np.asarray(item["observation.state"].numpy(), dtype=np.float64)
    # current frame -> identity pose, gripper untouched
    np.testing.assert_allclose(out[-1, :9], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(out[:, 9], [0.4, 0.5], atol=1e-7)
    # pose dims match the sub-key conversion exactly
    np.testing.assert_allclose(
        out[:, :9], np.asarray(item["observation.state.cartesian"].numpy()),
        atol=1e-6,
    )
    # round-trip: T_base @ T_rel recovers the absolute window (as matrices)
    for k in range(2):
        np.testing.assert_allclose(
            rlp.compose_relative_pose(out[k, :9], T[-1]), T[k], atol=1e-5
        )


# ── 6. deploy-side window conversion matches the training wrapper ────────────

def test_worker_window_conversion_matches_training():
    T = _random_traj(2, seed=23)
    cart = _pose9(T).astype(np.float32)
    frames = [
        {
            "observation.state.cartesian": cart[k],
            "observation.state": np.concatenate([cart[k], [0.3 + 0.1 * k]]).astype(
                np.float32
            ),
            "observation.images.primary": np.zeros((4, 4, 3), np.uint8),
        }
        for k in range(2)
    ]
    converted = rlp.convert_window_state_to_relative(frames)

    # training reference on the same window
    ds = lrp.RelativePoseDataset(None, start_pose_noise_scale=0.0)
    ref = ds.convert_item({
        "observation.state.cartesian": cart.copy(),
        "observation.state": np.stack(
            [f["observation.state"] for f in frames]
        ).copy(),
    })
    for k in range(2):
        np.testing.assert_allclose(
            converted[k]["observation.state"],
            np.asarray(ref["observation.state"].numpy())[k],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            converted[k]["observation.state.cartesian"],
            np.asarray(ref["observation.state.cartesian"].numpy())[k],
            atol=1e-6,
        )
    # images pass through untouched; inputs not mutated
    assert converted[0]["observation.images.primary"] is frames[0][
        "observation.images.primary"
    ]
    np.testing.assert_allclose(frames[1]["observation.state"][:9], cart[1], atol=0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} relative-deploy tests passed.")
