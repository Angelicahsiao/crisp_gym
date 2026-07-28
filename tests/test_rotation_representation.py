"""Round-trip math tests for the action/observation rotation representations.

Each representation crisp_gym can command or observe is an encode/decode pair
that must round-trip exactly on valid SO(3). The encode side lives in crisp_py
(``Pose.to_array``, ROS-coupled so not imported here) and the decode side in
crisp_gym (``ManipulatorBaseEnv.action_to_rotation``); they agree only by
CONVENTION, so this file pins that convention numerically and checks the
independent implementations against one another.

  EULER        encode: Rotation.as_euler("xyz")   (crisp_py Pose.to_pos_euler_array)
               decode: Rotation.from_euler("xyz")  (action_to_rotation EULER branch)
  ROTATION_6D  encode: first two ROWS of R, row-major, flattened  -- HANDOFF §1.1
               (crisp_py Pose.to_pos_rotation_6d_array == crisp_gym.util.rot6d.mat_to_rot6d)
               decode: Gram-Schmidt
               (crisp_gym.util.rot6d.rot6d_to_mat == action_to_rotation inline == script copy)

There are THREE copies of the rot6d Gram-Schmidt decode in the tree
(HANDOFF §1.1 warns they must stay in sync):
  1. crisp_gym/util/rot6d.py            -- the canonical module
  2. envs/manipulator_env.py            -- inline in action_to_rotation()
  3. scripts/lerobot_relative_pose.py   -- import-free copy for the GPU PC
This file asserts all three produce the same rotation.

Pure numpy + scipy; no ROS, no robot, no torch required. Run under pytest or
directly:  python tests/test_rotation_representation.py
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # allow `python tests/test_rotation_representation.py`
    sys.path.insert(0, str(REPO))

import crisp_gym.util.rot6d as rot6d  # noqa: E402  (needs REPO on sys.path first)

_rng = np.random.default_rng(0)


def _rand_R() -> Rotation:
    return Rotation.random(random_state=int(_rng.integers(1 << 30)))


def _assert_same_rotation(a: Rotation, b: Rotation, atol: float = 1e-9) -> None:
    """Compare as matrices (avoids quaternion double-cover sign ambiguity)."""
    np.testing.assert_allclose(a.as_matrix(), b.as_matrix(), atol=atol)


# ── encode side: ROS-free replicas of crisp_py Pose.to_array orientation parts ──

def _encode_euler(R: Rotation) -> np.ndarray:
    return R.as_euler("xyz", degrees=False)


def _encode_rot6d(R: Rotation) -> np.ndarray:
    # Pose.to_pos_rotation_6d_array: first two rows of the matrix, flattened.
    return R.as_matrix()[:2, :].flatten()


# ── decode side: replicas of crisp_gym action_to_rotation branches ─────────────

def _decode_euler(v: np.ndarray) -> Rotation:
    return Rotation.from_euler("xyz", v)


def _decode_rot6d_inline(v: np.ndarray) -> Rotation:
    """Verbatim copy of the action_to_rotation() ROTATION_6D branch."""
    a1, a2 = np.asarray(v[:3], float), np.asarray(v[3:6], float)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    return Rotation.from_matrix(np.stack([b1, b2, np.cross(b1, b2)]))


def _load_script_copy():
    """Load scripts/lerobot_relative_pose.py (its rot6d math is import-free but
    the module top imports torch — stub it if unavailable, like test_pose_math)."""
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            torch = types.ModuleType("torch")
            torch.Tensor = object
            torch.from_numpy = lambda a: a
            torch.utils = types.ModuleType("torch.utils")
            torch.utils.data = types.ModuleType("torch.utils.data")
            torch.utils.data.Dataset = object
            sys.modules.update(
                {
                    "torch": torch,
                    "torch.utils": torch.utils,
                    "torch.utils.data": torch.utils.data,
                }
            )
    path = REPO / "crisp_gym" / "scripts" / "lerobot_relative_pose.py"
    spec = importlib.util.spec_from_file_location("lrp_repr_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 1. EULER round-trip ────────────────────────────────────────────────────────

def test_euler_roundtrip():
    for _ in range(200):
        R = _rand_R()
        v = _encode_euler(R)
        assert v.shape == (3,)
        _assert_same_rotation(_decode_euler(v), R)


# ── 2. ROTATION_6D round-trip (canonical module + inline env copy) ─────────────

def test_rot6d_roundtrip_canonical():
    for _ in range(200):
        R = _rand_R()
        d6 = _encode_rot6d(R)
        assert d6.shape == (6,)
        np.testing.assert_allclose(rot6d.rot6d_to_mat(d6), R.as_matrix(), atol=1e-12)


def test_rot6d_roundtrip_inline_action_path():
    # Mirrors what action_to_rotation actually does with a rot6d action.
    for _ in range(200):
        R = _rand_R()
        _assert_same_rotation(_decode_rot6d_inline(_encode_rot6d(R)), R)


# ── 3. Convention guard: first two ROWS, row-major (HANDOFF §1.1) ──────────────

def test_rot6d_is_first_two_rows_not_columns():
    # A deliberately asymmetric rotation so rows != columns.
    R = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()
    d6 = rot6d.mat_to_rot6d(R)
    np.testing.assert_allclose(d6, R[:2, :].flatten(), atol=1e-15)  # rows, row-major
    # Must NOT be the first two columns (the classic wrong convention).
    assert not np.allclose(d6, R[:, :2].T.flatten())


# ── 4. Full pose-array round-trip in the [pos, *rot] action layout ────────────

def test_pose_array_roundtrip_euler():
    for _ in range(100):
        pos, R = _rng.normal(size=3), _rand_R()
        arr = np.concatenate([pos, _encode_euler(R)])  # 6D: [x,y,z, r,p,y]
        assert arr.shape == (6,)
        np.testing.assert_allclose(arr[:3], pos, atol=1e-15)
        _assert_same_rotation(_decode_euler(arr[3:]), R)


def test_pose_array_roundtrip_rot6d():
    for _ in range(100):
        pos, R = _rng.normal(size=3), _rand_R()
        arr = np.concatenate([pos, _encode_rot6d(R)])  # 9D: [x,y,z, rot6d...]
        assert arr.shape == (9,)
        np.testing.assert_allclose(rot6d.pose9d_to_mat(arr)[:3, 3], pos, atol=1e-15)
        np.testing.assert_allclose(rot6d.pose9d_to_mat(arr)[:3, :3], R.as_matrix(), atol=1e-12)


# ── 5. The three rot6d implementations must agree (anti-drift regression) ─────

def test_rot6d_implementations_agree():
    lrp = _load_script_copy()
    for _ in range(100):
        R = _rand_R().as_matrix()
        d6 = R[:2, :].flatten()
        # encode agrees
        np.testing.assert_allclose(rot6d.mat_to_rot6d(R), lrp.mat_to_rot6d(R), atol=1e-15)
        # decode agrees across canonical, script copy, and the inline env copy
        canon = rot6d.rot6d_to_mat(d6)
        script = lrp.rot6d_to_mat(d6)
        inline = _decode_rot6d_inline(d6).as_matrix()
        np.testing.assert_allclose(canon, script, atol=1e-12)
        np.testing.assert_allclose(canon, inline, atol=1e-12)


# ── 6. Decoded rot6d is always a valid rotation, even from noisy input ────────

def test_rot6d_decode_is_valid_SO3():
    for _ in range(200):
        d6 = _encode_rot6d(_rand_R()) + _rng.normal(scale=0.1, size=6)
        M = rot6d.rot6d_to_mat(d6)
        np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-9)  # orthonormal
        assert np.isclose(np.linalg.det(M), 1.0)  # right-handed (proper rotation)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} rotation-representation tests passed.")
