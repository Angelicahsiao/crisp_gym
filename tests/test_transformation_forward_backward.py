"""Forward / backward homogeneous-transformation tests for the CIC action path.

The relative-action deploy composes the CIC command as an SE(3) operation:

    forward :  T_cmd = T_base @ T_rel
    backward:  T_rel = (T_base)^-1 @ T_target

with the ANALYTICAL rigid-body inverse (the formula this test pins):

    H_B^A = (H_A^B)^-1 = [[ R^T , -R^T . t ],
                          [ 0_1x3,     1    ]]

The rot6d -> matrix path (`crisp_gym.util.rot6d`) obeys this exactly. The
EULER representation does NOT: negating euler angles is not the SE(3) inverse
and adding euler angles is not SE(3) composition — rotations do not commute,
and the translation term (-R^T t on the way back, R t on the way forward) is
dropped. So an euler-arithmetic CIC command is wrong for any non-trivial
rotation. These tests pin both facts.

Pure numpy + scipy; no ROS, no robot.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import crisp_gym.util.rot6d as rot6d  # noqa: E402  (needs REPO on sys.path first)

_rng = np.random.default_rng(0)


def _rand_R() -> Rotation:
    return Rotation.random(random_state=int(_rng.integers(1 << 30)))


def _mat(R: Rotation, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.as_matrix()
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def _rand_T() -> np.ndarray:
    return _mat(_rand_R(), _rng.normal(size=3))


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Analytical rigid-body inverse: H_B^A = [[R^T, -R^T t], [0, 1]]."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ── A. forward / backward on the SE(3) (rot6d/matrix) path — the ACCURATE one ──

def test_pose9d_to_mat_is_valid_se3():
    for _ in range(200):
        pose9 = np.concatenate([_rng.normal(size=3), _rand_R().as_matrix()[:2, :].flatten()])
        T = rot6d.pose9d_to_mat(pose9)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)  # orthonormal
        assert np.isclose(np.linalg.det(R), 1.0)  # proper rotation
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-15)  # homogeneous row


def test_analytical_inverse_equals_numeric_inverse():
    # The formula H_B^A = [[R^T, -R^T t], [0, 1]] must equal np.linalg.inv.
    for _ in range(200):
        T = _rand_T()
        np.testing.assert_allclose(se3_inverse(T), np.linalg.inv(T), atol=1e-9)
        np.testing.assert_allclose(T @ se3_inverse(T), np.eye(4), atol=1e-9)
        np.testing.assert_allclose(se3_inverse(T) @ T, np.eye(4), atol=1e-9)


def test_forward_then_backward_is_identity():
    for _ in range(200):
        T = _rand_T()
        np.testing.assert_allclose(se3_inverse(T) @ T, np.eye(4), atol=1e-9)
        # via rot6d encode/decode of the composed identity
        pose9 = rot6d.mat_to_pose9d(se3_inverse(T) @ T)
        identity9 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64)
        np.testing.assert_allclose(pose9, identity9, atol=1e-9)


def test_backward_then_forward_recovers_target():
    # T_rel = inv(T_base) @ T_target (backward), then T_base @ T_rel == T_target.
    for _ in range(200):
        T_base, T_target = _rand_T(), _rand_T()
        T_rel = se3_inverse(T_base) @ T_target
        np.testing.assert_allclose(T_base @ T_rel, T_target, atol=1e-9)
        # and the whole thing survives a rot6d encode/decode of T_rel
        T_rel_r = rot6d.pose9d_to_mat(rot6d.mat_to_pose9d(T_rel))
        np.testing.assert_allclose(T_base @ T_rel_r, T_target, atol=1e-9)


# ── B. the EULER representation does NOT obey the SE(3) formulas ───────────────

def test_euler_negation_is_not_se3_inverse():
    # A general pose whose rotation is far from identity.
    R = Rotation.from_euler("xyz", [0.7, -0.4, 1.1])
    t = np.array([0.3, -0.2, 0.5])
    T = _mat(R, t)

    # Accurate inverse (rot6d/matrix path) works:
    np.testing.assert_allclose(T @ se3_inverse(T), np.eye(4), atol=1e-12)

    # "Euler inverse" done the naive way: negate euler angles, negate position.
    R_neg = Rotation.from_euler("xyz", -R.as_euler("xyz"))
    T_euler_inv = _mat(R_neg, -t)
    # It is neither a left nor a right inverse of T.
    assert not np.allclose(T @ T_euler_inv, np.eye(4), atol=1e-6)
    assert not np.allclose(T_euler_inv @ T, np.eye(4), atol=1e-6)
    # Even the rotation part alone is wrong (euler negation != R^T for xyz).
    assert not np.allclose(R_neg.as_matrix(), R.as_matrix().T, atol=1e-6)


def test_euler_addition_is_not_se3_composition():
    R1 = Rotation.from_euler("xyz", [0.5, 0.3, -0.7])
    R2 = Rotation.from_euler("xyz", [-0.2, 0.9, 0.4])
    # Adding euler angles is not composing the rotations.
    R_add = Rotation.from_euler("xyz", R1.as_euler("xyz") + R2.as_euler("xyz"))
    assert not np.allclose(R_add.as_matrix(), (R1 * R2).as_matrix(), atol=1e-3)


def test_euler_arithmetic_gives_wrong_cic_command():
    # The concrete failure: composing the CIC command T_cmd = T_base @ T_rel.
    T_base = _mat(Rotation.from_euler("xyz", [0.2, -0.5, 0.8]), [0.1, 0.2, 0.3])
    T_rel = _mat(Rotation.from_euler("xyz", [0.3, 0.4, -0.6]), [0.05, -0.1, 0.2])
    T_cmd_true = T_base @ T_rel

    # rot6d/matrix path reproduces the true command exactly.
    rel9 = rot6d.mat_to_pose9d(T_rel)
    T_cmd_rot6d = T_base @ rot6d.pose9d_to_mat(rel9)
    np.testing.assert_allclose(T_cmd_rot6d, T_cmd_true, atol=1e-9)

    # euler-arithmetic path (add angles, add positions) is wrong on BOTH the
    # rotation (angles don't compose additively) and the translation (the
    # R_base @ t_rel term is dropped).
    base_e = Rotation.from_matrix(T_base[:3, :3]).as_euler("xyz")
    rel_e = Rotation.from_matrix(T_rel[:3, :3]).as_euler("xyz")
    R_cmd_euler = Rotation.from_euler("xyz", base_e + rel_e).as_matrix()
    pos_cmd_euler = T_base[:3, 3] + T_rel[:3, 3]
    assert not np.allclose(R_cmd_euler, T_cmd_true[:3, :3], atol=1e-3)
    assert not np.allclose(pos_cmd_euler, T_cmd_true[:3, 3], atol=1e-3)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} forward/backward transformation tests passed.")
