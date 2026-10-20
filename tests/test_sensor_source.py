"""Test the generic sensor.value record source + the ext-effort example config.

The external-effort estimator now lives robot-side (crisp_controllers_robot_demos
external_effort_node) and publishes a topic; crisp_gym records it as any other
sensor via `source: sensor.value`. No Pinocchio dependency here.

The source was called `robot.sensor` until it was renamed: it reads env.sensors,
a sibling of env.robot, so the robot.* prefix was wrong for anything not bolted
to the arm (tactile skin on the gripper). The old name still loads — these tests
pin that, and pin that it normalizes, which is what keeps datasets recorded
before and after the rename mixable.

Run:  python tests/test_sensor_source.py   (or via pytest)
"""

import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load_record_config():
    return SourceFileLoader(
        "rc_sensor", str(REPO / "crisp_gym" / "record" / "record_config.py")
    ).load_module()


class _FakeSensor:
    def __init__(self, name, value):
        self.config = types.SimpleNamespace(name=name)
        self.value = np.asarray(value, dtype=np.float32)


class _FakeEnv:
    def __init__(self, sensors):
        self.sensors = sensors


def test_sensor_value_source_returns_named_sensor_value():
    rc = _load_record_config()
    env = _FakeEnv([_FakeSensor("ft_sensor", [1, 2, 3, 4, 5, 6]),
                    _FakeSensor("ext_effort", [0.1, -0.2, 0.3, 0, 0, 0])])
    src = rc.SOURCE_REGISTRY["sensor.value"]
    out = src(env, name="ext_effort")
    np.testing.assert_allclose(out, [0.1, -0.2, 0.3, 0, 0, 0], atol=1e-6)
    assert out.dtype == np.float32
    # F/T wrench works through the same source
    np.testing.assert_allclose(src(env, name="ft_sensor"), [1, 2, 3, 4, 5, 6])


def test_sensor_value_source_missing_name_raises():
    rc = _load_record_config()
    env = _FakeEnv([_FakeSensor("ft_sensor", [0] * 6)])
    try:
        rc.SOURCE_REGISTRY["sensor.value"](env, name="nope")
        raise AssertionError("missing sensor not rejected")
    except KeyError as e:
        assert "nope" in str(e) and "ft_sensor" in str(e)


def test_ext_effort_example_config_validates():
    rc = _load_record_config()
    cfg = rc.RecordConfig.from_yaml(
        str(REPO / "crisp_gym" / "config" / "recording" / "umi_robot_ext_effort_record.yaml")
    )
    cfg.validate()
    ee = [o for o in cfg.observations if o.key == "extra.external_effort"][0]
    assert ee.source == "sensor.value" and ee.params.get("name") == "ext_effort"
    assert ee.include_in_state is False and tuple(ee.resolved_shape()) == (6,)
    feats = cfg.to_features(use_video=False)
    # stored but NOT a policy input
    assert "extra.external_effort" in feats
    assert "external_effort" not in "".join(feats["observation.state"]["names"])


# ── the rename ────────────────────────────────────────────────────────────────

def test_deprecated_name_is_not_executable():
    """Aliases stay out of the registry so the unknown-source error's
    "Available: [...]" list names only sources you should be writing."""
    rc = _load_record_config()
    assert "sensor.value" in rc.SOURCE_REGISTRY
    assert "robot.sensor" not in rc.SOURCE_REGISTRY
    assert rc.DEPRECATED_SOURCES["robot.sensor"] == "sensor.value"


def test_deprecated_name_loads_and_normalizes():
    rc = _load_record_config()
    rc._source_deprecation_warned.clear()
    o = rc.ObsFieldConfig(key="extra.ext_torque", source="robot.sensor",
                          shape=[7], include_in_state=False)
    assert o.source == "sensor.value", "the old name must still load"
    assert rc._source_deprecation_warned == {"robot.sensor"}, "warned once, by old name"
    # the latch is keyed by the old name, so a second config does not re-warn
    rc.ObsFieldConfig(key="extra.other", source="robot.sensor", shape=[1],
                      include_in_state=False)
    assert rc._source_deprecation_warned == {"robot.sensor"}


def test_validate_accepts_the_deprecated_name():
    """from_yaml -> ObsFieldConfig normalizes before validate() sees it, so an
    unmigrated config on a robot still loads instead of dying on load."""
    rc = _load_record_config()
    cfg = rc.RecordConfig(
        observations=[rc.ObsFieldConfig(key="extra.skin", source="robot.sensor",
                                        shape=[15], include_in_state=False)],
        action=rc.ActionConfig(),
    )
    cfg.validate()


def test_old_and_new_spelling_stamp_the_same_contract():
    """`source` is compared by contracts_compatible for in-state observations,
    so without normalization a renamed source would split one dataset family
    into two incompatible halves."""
    rc = _load_record_config()

    def _meta(spelling):
        return rc.RecordConfig(
            observations=[
                rc.ObsFieldConfig(key="observation.state.cartesian",
                                  source="robot.tcp_pose",
                                  params={"representation": "rotation_6d"}),
                rc.ObsFieldConfig(key="observation.state.skin",
                                  source=spelling, shape=[15]),
            ],
            action=rc.ActionConfig(),
        ).to_metadata()

    old, new = _meta("robot.sensor"), _meta("sensor.value")
    assert old == new
    assert [o["source"] for o in old["observations"]][1] == "sensor.value"
    assert rc.RecordConfig.contracts_compatible(old, new)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} sensor-source tests passed.")
