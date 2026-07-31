"""Cross-package consistency: crisp_py ``Pose.to_array`` <-> crisp_gym rot6d.

crisp_py (the ROS2 client library) encodes observation / target poses via
``Pose.to_array(representation)`` in ``crisp_py/utils/geometry.py``. crisp_gym
decodes and consumes them via ``crisp_gym.util.rot6d``. The two packages agree
only by CONVENTION (HANDOFF §1.1), so this test pins that they actually do:

  - crisp_py ``to_array(EULER)``       == ``[pos, as_euler("xyz")]`` and round-trips
  - crisp_py ``to_array(ROTATION_6D)`` uses the first two ROWS of the matrix,
    equals ``crisp_gym.util.rot6d.mat_to_rot6d``, and ``rot6d_to_mat`` decodes
    it back to the original rotation — the encode(crisp_py)/decode(crisp_gym)
    seam that the whole record -> train -> deploy pipeline relies on.

``crisp_py/utils/geometry.py`` hard-imports ROS message types at module top
(and the crisp_py package ``__init__`` needs an installed distribution + rclpy),
so the module is loaded DIRECTLY FROM SOURCE with the ROS message modules
stubbed. The math under test is pure numpy + scipy; no ROS, no robot.

Skips cleanly if the sibling crisp_py checkout is not present (``../crisp_py``).
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]  # crisp_gym repo root
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import crisp_gym.util.rot6d as rot6d  # noqa: E402  (needs REPO on sys.path first)


def _stub_ros_modules() -> None:
    """Ensure the ROS message modules geometry.py imports exist.

    Prefer the real modules (pixi/humble env); only install lightweight stubs
    when they are absent (bare / CI env). geometry.py uses these names only in
    annotations and the from_ros_msg/to_ros_msg helpers, none of which the math
    under test touches — empty classes are enough.
    """
    needed = {
        "builtin_interfaces.msg": {"Time": ()},
        "geometry_msgs.msg": {"PoseStamped": (), "TwistStamped": ()},
        "tf2_ros": {"TransformStamped": ()},
    }
    for name, attrs in needed.items():
        try:
            __import__(name)
        except Exception:
            # Create the full parent chain (e.g. builtin_interfaces -> .msg).
            parts = name.split(".")
            for i in range(len(parts)):
                sub = ".".join(parts[: i + 1])
                if sub not in sys.modules:
                    mod = types.ModuleType(sub)
                    sys.modules[sub] = mod
                    if i > 0:
                        setattr(sys.modules[".".join(parts[:i])], parts[i], mod)
        mod = sys.modules[name]
        for cls_name in attrs:
            if not hasattr(mod, cls_name):
                setattr(mod, cls_name, type(cls_name, (), {}))


def _load_crisp_py_geometry():
    geom_path = REPO.parent / "crisp_py" / "crisp_py" / "utils" / "geometry.py"
    if not geom_path.exists():
        pytest.skip(f"crisp_py checkout not found at {geom_path}")
    _stub_ros_modules()
    spec = importlib.util.spec_from_file_location("crisp_py_geometry_under_test", geom_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_geom = _load_crisp_py_geometry()
Pose = _geom.Pose
OrientationRepresentation = _geom.OrientationRepresentation

_rng = np.random.default_rng(0)


def _rand_R() -> Rotation:
    return Rotation.random(random_state=int(_rng.integers(1 << 30)))


def _rand_pos() -> np.ndarray:
    return _rng.normal(size=3)


# ── crisp_py Pose.to_array: layout + round-trip ───────────────────────────────

def test_to_array_default_is_euler():
    for _ in range(50):
        pose = Pose(_rand_pos(), _rand_R())
        np.testing.assert_allclose(
            pose.to_array(), pose.to_array(OrientationRepresentation.EULER)
        )


def test_to_array_euler_layout_and_roundtrip():
    for _ in range(200):
        p, R = _rand_pos(), _rand_R()
        arr = Pose(p, R).to_array(OrientationRepresentation.EULER)
        assert arr.shape == (6,)
        np.testing.assert_allclose(arr[:3], p, atol=1e-12)
        np.testing.assert_allclose(arr[3:], R.as_euler("xyz"), atol=1e-12)
        np.testing.assert_allclose(
            Rotation.from_euler("xyz", arr[3:]).as_matrix(), R.as_matrix(), atol=1e-9
        )


def test_to_array_rot6d_is_first_two_rows():
    for _ in range(200):
        p, R = _rand_pos(), _rand_R()
        arr = Pose(p, R).to_array(OrientationRepresentation.ROTATION_6D)
        assert arr.shape == (9,)
        np.testing.assert_allclose(arr[:3], p, atol=1e-12)
        # first two ROWS of the matrix, flattened row-major (HANDOFF §1.1)
        np.testing.assert_allclose(arr[3:], R.as_matrix()[:2, :].flatten(), atol=1e-12)


# ── consistency: crisp_py encode <-> crisp_gym encode/decode ──────────────────

def test_crisp_py_rot6d_encode_matches_crisp_gym():
    # crisp_py Pose.to_array(ROTATION_6D)[3:] must equal crisp_gym's encoder.
    for _ in range(200):
        R = _rand_R()
        arr = Pose(_rand_pos(), R).to_array(OrientationRepresentation.ROTATION_6D)
        np.testing.assert_allclose(arr[3:], rot6d.mat_to_rot6d(R.as_matrix()), atol=1e-15)


def test_crisp_gym_decodes_crisp_py_rot6d():
    # The cross-repo seam: crisp_gym rot6d_to_mat / pose9d_to_mat decode what
    # crisp_py encoded, recovering the original rotation and position.
    for _ in range(200):
        p, R = _rand_pos(), _rand_R()
        arr = Pose(p, R).to_array(OrientationRepresentation.ROTATION_6D)
        np.testing.assert_allclose(rot6d.rot6d_to_mat(arr[3:9]), R.as_matrix(), atol=1e-9)
        T = rot6d.pose9d_to_mat(arr)  # 9D -> 4x4 homogeneous
        np.testing.assert_allclose(T[:3, :3], R.as_matrix(), atol=1e-9)
        np.testing.assert_allclose(T[:3, 3], p, atol=1e-12)


def test_rot6d_rows_not_columns_regression():
    # Guard against a rows<->columns swap between the two packages.
    R = Rotation.from_euler("xyz", [0.1, 0.2, 0.3])
    arr = Pose(_rand_pos(), R).to_array(OrientationRepresentation.ROTATION_6D)
    M = R.as_matrix()
    np.testing.assert_allclose(arr[3:], M[:2, :].flatten())  # rows (correct)
    assert not np.allclose(arr[3:], M[:, :2].T.flatten())  # not columns


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} crisp_py<->crisp_gym consistency tests passed.")
