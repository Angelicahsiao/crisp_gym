"""Test the gripper.target_width_normalized record source + its wiring.

observation.state.gripper records the MEASURED opening, which stalls at an
object's width however hard the gripper is told to close. The commanded opening
is the only place the "squeeze" survives, exactly as robot.target_pose is the
only place a contact force survives on the arm.

The source must share units with gripper.width_normalized (so the two subtract
into a residual), must not invent a command where none exists (handheld), and
must never leak into observation.state.

Run:  python tests/test_gripper_target_source.py   (or via pytest)
"""

import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE = "gripper.target_width_normalized"


def _load_record_config():
    return SourceFileLoader(
        "rc_gripper_target", str(REPO / "crisp_gym" / "record" / "record_config.py")
    ).load_module()


class _FakeGripper:
    """Mirrors crisp_py Gripper: `target` raises TypeError until a command is set
    (it normalizes `_target`, which starts as None)."""

    def __init__(self, value, target=None):
        self.value = value
        self._target = target

    @property
    def target(self):
        return float(np.clip((self._target - 0.0) / 1.0, 0.0, 1.0))


def _env(value, target=None):
    return types.SimpleNamespace(gripper=_FakeGripper(value, target))


PARAMS = dict(reference_width=0.09, device_max_width=0.085)


def test_command_and_measurement_share_units():
    """Same params -> same scale, so (target - measured) is a residual."""
    rc = _load_record_config()
    env = _env(value=0.5, target=0.5)
    cmd = rc.SOURCE_REGISTRY[SOURCE](env, **PARAMS)
    meas = rc.SOURCE_REGISTRY["gripper.width_normalized"](env, **PARAMS)
    np.testing.assert_allclose(cmd, meas, atol=1e-7)
    np.testing.assert_allclose(cmd, [0.5 * 0.085 / 0.09], atol=1e-7)
    assert cmd.dtype == np.float32 and cmd.shape == (1,)


def test_squeezing_shows_up_as_a_negative_residual():
    """Fingers blocked at 0.4 by an object, command says close fully."""
    rc = _load_record_config()
    env = _env(value=0.4, target=0.0)
    cmd = rc.SOURCE_REGISTRY[SOURCE](env, **PARAMS)
    meas = rc.SOURCE_REGISTRY["gripper.width_normalized"](env, **PARAMS)
    assert float(cmd[0] - meas[0]) < -0.3


def test_before_the_first_command_it_reports_the_measured_value():
    rc = _load_record_config()
    env = _env(value=0.7, target=None)
    np.testing.assert_allclose(
        rc.SOURCE_REGISTRY[SOURCE](env, **PARAMS), [0.7 * 0.085 / 0.09], atol=1e-7
    )


def test_raw_device_value_without_reference_scaling():
    rc = _load_record_config()
    np.testing.assert_allclose(rc.SOURCE_REGISTRY[SOURCE](_env(0.9, 0.25)), [0.25], atol=1e-7)


def test_handheld_env_is_refused_not_faked():
    """UmiHandheldEnv has no command; copying the measured width would be a
    column that looks like one."""
    rc = _load_record_config()
    handheld = types.SimpleNamespace(
        _gripper_normalized=0.5, config=types.SimpleNamespace(max_gripper_width=0.09)
    )
    for env in (handheld, types.SimpleNamespace(gripper=None)):
        try:
            rc.SOURCE_REGISTRY[SOURCE](env, **PARAMS)
            raise AssertionError("env without a gripper command not refused")
        except ValueError as e:
            assert "no gripper command" in str(e)


def test_reference_scaling_needs_the_device_width():
    rc = _load_record_config()
    try:
        rc.SOURCE_REGISTRY[SOURCE](_env(0.5, 0.5), reference_width=0.09)
        raise AssertionError("missing device_max_width not rejected")
    except ValueError as e:
        assert "device_max_width" in str(e)


def test_shape_and_name_need_no_yaml_help():
    rc = _load_record_config()
    o = rc.ObsFieldConfig(key="extra.gripper_target", source=SOURCE)
    assert tuple(o.resolved_shape()) == (1,)
    assert o.names() == ["gripper_target"]


def test_umi_robot_full_records_the_gripper_command():
    rc = _load_record_config()
    cfg = rc.RecordConfig.from_yaml(
        str(REPO / "crisp_gym" / "config" / "recording" / "umi_robot_full_record.yaml")
    )
    cfg.validate()
    by_key = {o.key: o for o in cfg.observations}
    cmd, meas = by_key["extra.gripper_target"], by_key["observation.state.gripper"]
    assert cmd.source == SOURCE
    assert cmd.include_in_state is False
    # Different scaling would turn the residual into a unit error.
    for k in ("reference_width", "device_max_width"):
        assert cmd.params[k] == meas.params[k], k

    feats = cfg.to_features(use_video=False)
    assert feats["extra.gripper_target"]["shape"] == (1,)
    # Stored, never a policy input.
    assert "target" not in "".join(feats["observation.state"]["names"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} gripper-target source tests passed.")
