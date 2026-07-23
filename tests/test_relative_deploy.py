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

def test_training_converts_observation_state_10d():
    """Old relative generation (append_wrt_start_to_state=False): 10-D state."""
    T = _random_traj(2, seed=11)  # n_obs_steps = 2 window
    cart = _pose9(T).astype(np.float32)                      # (2, 9) absolute
    state = np.concatenate([cart, [[0.4], [0.5]]], axis=-1)  # (2, 10) + gripper
    ds = lrp.RelativePoseDataset(
        None, start_pose_noise_scale=0.0, append_wrt_start_to_state=False
    )
    item = ds.convert_item({
        "observation.state.cartesian": cart.copy(),
        "observation.state": state.copy(),
    })
    out = np.asarray(item["observation.state"].numpy(), dtype=np.float64)
    assert out.shape == (2, 10)
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


def test_training_state_is_16d_umi_parity():
    """New generation (default): [rel_pose9, gripper1, rot_wrt_start6]."""
    T = _random_traj(2, seed=13)
    cart = _pose9(T).astype(np.float32)
    state = np.concatenate([cart, [[0.4], [0.5]]], axis=-1)
    start_pose = _pose9(T[0]).astype(np.float64)  # episode start = frame 0
    ds = lrp.RelativePoseDataset(None, start_pose_noise_scale=0.0)  # append=True
    item = ds.convert_item(
        {
            "observation.state.cartesian": cart.copy(),
            "observation.state": state.copy(),
        },
        start_pose=start_pose,
    )
    out = np.asarray(item["observation.state"].numpy(), dtype=np.float64)
    assert out.shape == (2, 16)
    # dims 0-9 unchanged semantics: rel pose + gripper
    np.testing.assert_allclose(out[-1, :9], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(out[:, 9], [0.4, 0.5], atol=1e-7)
    # appended dims == the WRT_START_KEY tensor (rotation-only, wrt start)
    wrt = np.asarray(item[lrp.WRT_START_KEY].numpy(), dtype=np.float64)
    np.testing.assert_allclose(out[:, 10:16], wrt, atol=1e-7)
    # frame 0 IS the (un-noised) start -> its wrt rotation is identity rot6d
    np.testing.assert_allclose(out[0, 10:16], [1, 0, 0, 0, 1, 0], atol=1e-6)
    # wrt rows are valid rotations
    for row in out[:, 10:16]:
        R = lrp.rot6d_to_mat(row)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    # convert_item without a start pose must fail loudly (dim mismatch trap)
    try:
        ds.convert_item({"observation.state": state.copy()})
        raise AssertionError("missing start pose not rejected")
    except ValueError as e:
        assert "start pose" in str(e)


# ── 6. deploy-side window conversion matches the training wrapper ────────────

def _client_frames(cart):
    return [
        {
            "observation.state.cartesian": cart[k],
            "observation.state": np.concatenate([cart[k], [0.3 + 0.1 * k]]).astype(
                np.float32
            ),
            "observation.images.primary": np.zeros((4, 4, 3), np.uint8),
        }
        for k in range(len(cart))
    ]


def test_worker_window_conversion_matches_training_10d():
    T = _random_traj(2, seed=23)
    cart = _pose9(T).astype(np.float32)
    frames = _client_frames(cart)
    converted = rlp.convert_window_state_to_relative(frames)

    # training reference on the same window (old relative generation)
    ds = lrp.RelativePoseDataset(
        None, start_pose_noise_scale=0.0, append_wrt_start_to_state=False
    )
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


def test_worker_16d_pipeline_matches_training():
    """Full deploy order (wrt-start FIRST from absolute, then relative
    conversion) == training convert_item with noise off."""
    T = _random_traj(2, seed=29)
    cart = _pose9(T).astype(np.float32)
    frames = _client_frames(cart)
    start_pose = np.asarray(frames[0]["observation.state"][:9], np.float64)

    with_wrt = rlp.append_wrt_start_to_window(frames, start_pose)
    converted = rlp.convert_window_state_to_relative(with_wrt)

    ds = lrp.RelativePoseDataset(None, start_pose_noise_scale=0.0)
    ref = ds.convert_item(
        {
            "observation.state.cartesian": cart.copy(),
            "observation.state": np.stack(
                [f["observation.state"] for f in frames]
            ).copy(),
        },
        start_pose=start_pose.copy(),
    )
    ref_state = np.asarray(ref["observation.state"].numpy())
    assert ref_state.shape == (2, 16)
    for k in range(2):
        assert converted[k]["observation.state"].shape == (16,)
        np.testing.assert_allclose(
            converted[k]["observation.state"], ref_state[k], atol=1e-6
        )
    # frame 0 == episode start -> wrt dims identity rot6d
    np.testing.assert_allclose(
        converted[0]["observation.state"][10:16], [1, 0, 0, 0, 1, 0], atol=1e-6
    )
    # inputs not mutated by either step
    np.testing.assert_allclose(frames[1]["observation.state"][:9], cart[1], atol=0)
    assert frames[0]["observation.state"].shape == (10,)


# ── 7. promoted-state frame build (legacy migrated layout) ───────────────────

def test_build_obs_frame_promoted_state_with_euler_target():
    """Legacy migrated layout: state = [cart9, grip1, joints7, target6-euler]
    = 23-D, built from a rot6d env whose target obs is 9-D."""
    ref = dev = 0.085  # migrated data: identity gripper conversion
    T = _random_traj(1, seed=31)[0]
    cart = _pose9(T).astype(np.float32)
    Tt = _random_traj(1, seed=37)[0]
    target9 = _pose9(Tt).astype(np.float32)
    joints = np.linspace(-1, 1, 7).astype(np.float32)
    obs = {
        "observation.state.cartesian": cart,
        "observation.state.gripper": np.array([0.6], np.float32),
        "observation.state.joints": joints,
        "observation.state.target": target9,
        "observation.images.primary": np.zeros((4, 4, 3), np.uint8),
    }
    frame = rlp.build_obs_frame(obs, ref, dev, target_to_euler=True)
    s = frame["observation.state"]
    assert s.shape == (23,)
    np.testing.assert_allclose(s[:9], cart, atol=0)          # cartesian first
    assert abs(s[9] - 0.6) < 1e-6                            # identity gripper
    np.testing.assert_allclose(s[10:17], joints, atol=0)     # joints untouched
    # target: 9-D rot6d -> 6-D euler, matching scipy on the same rotation
    np.testing.assert_allclose(s[17:20], target9[:3], atol=1e-6)
    expected_euler = Rotation.from_matrix(Tt[:3, :3]).as_euler("xyz")
    np.testing.assert_allclose(s[20:23], expected_euler, atol=1e-5)
    # without the flag the target stays 9-D -> 26-D state
    frame26 = rlp.build_obs_frame(obs, ref, dev, target_to_euler=False)
    assert frame26["observation.state"].shape == (26,)


# ── 7b. pose_repr stamp: machine-readable state composition ──────────────────

def test_stamp_state_components():
    import json
    import tempfile

    features = {
        "observation.state.cartesian": {"shape": (9,)},
        "observation.state.gripper": {"shape": (1,)},
        "observation.state.joints": {"shape": (7,)},
        "observation.state": {"shape": (23,)},  # 17 disk + 6 widened wrt
        "observation.images.primary": {"shape": (224, 224, 3)},
    }
    dataset = types.SimpleNamespace(
        meta=types.SimpleNamespace(features=features, fps=15), _append_wrt=True
    )
    with tempfile.TemporaryDirectory() as tmp:
        cfg = types.SimpleNamespace(
            output_dir=tmp,
            policy=types.SimpleNamespace(n_obs_steps=2, horizon=16, n_action_steps=8),
        )
        lrp.stamp_pose_repr(cfg, dataset)
        stamp = json.loads((Path(tmp) / "pose_repr.json").read_text())

    comps = stamp["observation"]["state_components"]
    assert [c.get("key", c.get("generated")) for c in comps] == [
        "observation.state.cartesian", "observation.state.gripper",
        "observation.state.joints", "rot_wrt_start",
    ]
    assert [c["dims"] for c in comps] == [9, 1, 7, 6]
    assert comps[0]["transform"] == "relative_to_last_obs_frame"
    assert comps[1]["transform"] == "none" and comps[2]["transform"] == "none"
    assert comps[3]["transform"] == "wrt_episode_start_rotation"
    assert sum(c["dims"] for c in comps) == 23  # matches the widened state dim

    # dims mismatch (sub-features can't explain the concat) -> components omitted
    features_bad = dict(features)
    features_bad["observation.state"] = {"shape": (29,)}
    dataset_bad = types.SimpleNamespace(
        meta=types.SimpleNamespace(features=features_bad, fps=15), _append_wrt=True
    )
    with tempfile.TemporaryDirectory() as tmp:
        cfg = types.SimpleNamespace(output_dir=tmp, policy=None)
        lrp.stamp_pose_repr(cfg, dataset_bad)
        stamp = json.loads((Path(tmp) / "pose_repr.json").read_text())
    assert stamp["observation"]["state_components"] is None


# ── 8. END-TO-END client loop: obs -> frame -> window -> chunk -> env.step ───

def test_client_loop_end_to_end():
    """Drive the REAL RelativeLerobotPolicy data_fn with a scripted env and a
    ground-truth 'server': every commanded pose must reproduce the ground-truth
    trajectory (as matrices), across TWO chunks, with the obs-time base held
    per chunk; gripper and state layout checked along the way."""
    GT = _random_traj(20, seed=41)  # ground-truth TCP trajectory
    ref = dev = 0.085

    class _FakeEnv:
        def __init__(self):
            self.k = 0                       # current pose index (perfect tracking)
            self.commands = []               # (t_index, action) received by step
            self.config = types.SimpleNamespace(
                orientation_representation="rotation_6d",
                use_relative_actions=False,
            )
            self.robot = types.SimpleNamespace()
            self._sync_robot()

        def _sync_robot(self):
            T = GT[self.k]
            self.robot.end_effector_pose = types.SimpleNamespace(
                position=T[:3, 3].copy(),
                orientation=Rotation.from_matrix(T[:3, :3]),
            )

        def get_obs(self):
            return {
                "observation.state.cartesian": _pose9(GT[self.k]).astype(np.float32),
                "observation.state.gripper": np.array([0.6], np.float32),
                "observation.state.joints": np.zeros(7, np.float32),
                "observation.state.target": _pose9(GT[self.k]).astype(np.float32),
                "observation.images.primary": np.zeros((4, 4, 3), np.uint8),
            }

        def step(self, action, block=False):
            assert action.shape == (10,), f"action dim {action.shape} != 10"
            self.commands.append((self.k, np.asarray(action, np.float64).copy()))
            self.k += 1                      # perfect tracking of the command
            self._sync_robot()

    class _FakeConn:
        """Emulates a CORRECT inference worker: returns the ground-truth
        relative chunk wrt the LAST frame of the received window."""

        def __init__(self, env):
            self.env = env
            self.windows = []

        def send(self, msg):
            kind, window = msg
            assert kind == "infer"
            self.windows.append(window)
            base9 = np.asarray(window[-1]["observation.state"][:9], np.float64)
            # sanity: the window's newest frame is the env's CURRENT abs pose
            np.testing.assert_allclose(base9, _pose9(GT[self.env.k]), atol=1e-6)
            T_base_inv = np.linalg.inv(_mat(base9))  # T_base from the sent obs
            chunk = []
            for j in range(8):
                T_rel = T_base_inv @ GT[self.env.k + 1 + j]
                chunk.append(np.concatenate(
                    [_pose9(T_rel), [0.7]]).astype(np.float32))
            self._chunk = np.stack(chunk)

        def recv(self):
            return self._chunk

    def _mat(p9):
        T = np.eye(4)
        a1, a2 = p9[3:6], p9[6:9]
        b1 = a1 / np.linalg.norm(a1)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / np.linalg.norm(b2)
        T[:3, :3] = np.stack([b1, b2, np.cross(b1, b2)])
        T[:3, 3] = p9[:3]
        return T

    from collections import deque

    env = _FakeEnv()
    pol = object.__new__(rlp.RelativeLerobotPolicy)
    pol.env = env
    pol.reference_width = ref
    pol.device_max_width = dev
    pol.target_to_euler = False
    pol.compose_mode = "coupled"
    pol.invert_gripper = False
    pol._log_actions = False
    pol._log_actions_n = 0
    pol._action_log_left = 0
    pol.meta = {"n_obs_steps": 2, "n_action_steps": 8, "state_input":
                "relative_wrt_start", "state_dim": 29 + 3,  # 26 disk (target 9D) + 6
                "image_keys": ["observation.images.primary"]}
    pol.n_obs_steps = 2
    pol.n_action_steps = 8
    pol._history = deque(maxlen=2)
    pol._state_dim_checked = False
    pol._chunk = None
    pol._chunk_idx = 0
    pol._chunk_base = None
    pol._tick_times = deque(maxlen=30)
    pol._last_infer_ms = 0.0
    pol._rate_log_every = 15
    pol._tick_count = 0
    pol.parent_conn = _FakeConn(env)

    fn = pol.make_data_fn()
    for _ in range(16):                      # two full chunks
        obs, act = fn()
        assert act is not None

    # every commanded pose reproduces ground truth — compared as MATRICES
    assert len(env.commands) == 16
    for k, action in env.commands:
        T_cmd = _mat(np.asarray(action[:9]))
        np.testing.assert_allclose(T_cmd, GT[k + 1], atol=1e-6)
        assert abs(action[9] - 0.7) < 1e-6   # identity gripper scaling
    # exactly two inferences (chunking works), window capped at n_obs_steps
    assert len(pol.parent_conn.windows) == 2
    assert len(pol.parent_conn.windows[1]) == 2
    # frames sent contain the full promoted state (26-D here: target kept 9-D)
    assert pol.parent_conn.windows[0][0]["observation.state"].shape == (26,)

    # reset clears client state
    pol.parent_conn.send_raw = None
    pol.parent_conn.send = lambda msg: None  # ignore the reset message
    pol.reset()
    assert pol._chunk is None and len(pol._history) == 0 and pol._chunk_base is None


# ── 9. decoupled composition = UMI 'rel' backward (base-frame position) ───────

def test_decoupled_composition_matches_umi_rel():
    """compose_mode='decoupled' must equal UMI 'rel' eval transform:
    pos = rel_pos + base_pos (base frame), rot = rel_rot @ base_rot."""
    T_base = _random_traj(1, seed=51)[0]
    Trel = _random_traj(1, seed=52)[0]
    action = np.concatenate([_pose9(Trel), [0.5]]).astype(np.float64)

    pol = object.__new__(rlp.RelativeLerobotPolicy)
    pol._chunk_base = T_base
    pol.reference_width = pol.device_max_width = 0.085
    pol.invert_gripper = False
    pol._log_actions = False
    pol._log_actions_n = 0
    pol._action_log_left = 0

    class _Cfg:
        orientation_representation = "rotation_6d"
    pol.env = types.SimpleNamespace(config=_Cfg())

    # coupled: position rotated by base -> EE frame
    pol.compose_mode = "coupled"
    out_c = pol._to_env_action(action)
    exp_c = rlp.compose_relative_pose(action[:9], T_base)[:3, 3]
    np.testing.assert_allclose(out_c[:3], exp_c, atol=1e-9)

    # decoupled: position added in base frame (UMI 'rel')
    pol.compose_mode = "decoupled"
    out_d = pol._to_env_action(action)
    np.testing.assert_allclose(out_d[:3], action[:3] + T_base[:3, 3], atol=1e-9)
    # the two modes give DIFFERENT commands (unless base rotation is identity)
    assert np.linalg.norm(out_c[:3] - out_d[:3]) > 1e-3

    # invert_gripper flips the commanded gripper (1 - a); identity ref==dev
    pol.compose_mode = "coupled"
    a2 = action.copy()
    a2[-1] = 0.2
    pol.invert_gripper = False
    assert abs(pol._to_env_action(a2)[-1] - 0.2) < 1e-6
    pol.invert_gripper = True
    assert abs(pol._to_env_action(a2)[-1] - 0.8) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} relative-deploy tests passed.")
