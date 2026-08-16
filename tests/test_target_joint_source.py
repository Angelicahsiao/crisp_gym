"""Test the robot.target_joint record source + its wiring in umi_robot_full.

robot.target_pose records the COMMANDED TCP pose, but crisp_py writes that field
only from set_target()/move_to(). A joint-driven env (FACTR leader -> JIC) calls
set_target_joint(), which touches _target_joint alone — so target_pose there is
just the post-home measured pose, frozen. robot.target_joint is the joint-space
mirror that carries the real command on that path.

umi_robot_full_record.yaml records BOTH, deliberately: one config serves the
Cartesian and the joint rig, and whichever column does not match the control
mode is dead weight rather than a correctness problem.

Run:  python tests/test_target_joint_source.py   (or via pytest)
"""

import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load_record_config():
    return SourceFileLoader(
        "rc_target_joint", str(REPO / "crisp_gym" / "record" / "record_config.py")
    ).load_module()


class _FakeEnv:
    def __init__(self, target_joint):
        self.robot = types.SimpleNamespace(target_joint=np.asarray(target_joint))


def test_target_joint_source_returns_the_command():
    rc = _load_record_config()
    q = [0.42, -0.01, -0.33, -2.69, 0.09, 4.25, -0.01]
    out = rc.SOURCE_REGISTRY["robot.target_joint"](_FakeEnv(q))
    np.testing.assert_allclose(out, q, atol=1e-6)
    assert out.dtype == np.float32
    assert out.shape == (7,)


def test_target_joint_source_is_float32_from_a_float64_target():
    """crisp_py keeps targets in float64; the dataset column is float32."""
    rc = _load_record_config()
    env = _FakeEnv(np.zeros(7, dtype=np.float64))
    assert rc.SOURCE_REGISTRY["robot.target_joint"](env).dtype == np.float32


def test_target_joint_needs_an_explicit_shape():
    """Unlike the pose sources, the dim cannot be derived — the YAML must say."""
    rc = _load_record_config()
    o = rc.ObsFieldConfig(key="extra.target_joints", source="robot.target_joint")
    try:
        o.resolved_shape()
        raise AssertionError("missing shape not rejected")
    except ValueError as e:
        assert "robot.target_joint" in str(e)


def test_umi_robot_full_records_both_commanded_targets():
    rc = _load_record_config()
    cfg = rc.RecordConfig.from_yaml(
        str(REPO / "crisp_gym" / "config" / "recording" / "umi_robot_full_record.yaml")
    )
    cfg.validate()

    by_key = {o.key: o for o in cfg.observations}
    tj = by_key["extra.target_joints"]
    assert tj.source == "robot.target_joint"
    assert tuple(tj.resolved_shape()) == (7,)          # FR3
    assert tj.include_in_state is False
    # The Cartesian target stays alongside it — one config, both rigs.
    assert by_key["extra.target_cartesian"].source == "robot.target_pose"

    feats = cfg.to_features(use_video=False)
    assert feats["extra.target_joints"]["shape"] == (7,)
    assert feats["extra.target_joints"]["names"] == [f"target_joints_{i}" for i in range(7)]
    # Stored, but never a policy input: neither target may leak into the
    # concatenated state, or the policy learns to copy the command.
    state_names = "".join(feats["observation.state"]["names"])
    assert "target" not in state_names


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} target-joint source tests passed.")
