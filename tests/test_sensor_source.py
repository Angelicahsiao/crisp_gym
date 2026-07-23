"""Test the generic robot.sensor record source + the ext-effort example config.

The external-effort estimator now lives robot-side (crisp_controllers_robot_demos
external_effort_node) and publishes a topic; crisp_gym records it as any other
sensor via `source: robot.sensor`. No Pinocchio dependency here.

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


def test_robot_sensor_source_returns_named_sensor_value():
    rc = _load_record_config()
    env = _FakeEnv([_FakeSensor("ft_sensor", [1, 2, 3, 4, 5, 6]),
                    _FakeSensor("ext_effort", [0.1, -0.2, 0.3, 0, 0, 0])])
    src = rc.SOURCE_REGISTRY["robot.sensor"]
    out = src(env, name="ext_effort")
    np.testing.assert_allclose(out, [0.1, -0.2, 0.3, 0, 0, 0], atol=1e-6)
    assert out.dtype == np.float32
    # F/T wrench works through the same source
    np.testing.assert_allclose(src(env, name="ft_sensor"), [1, 2, 3, 4, 5, 6])


def test_robot_sensor_source_missing_name_raises():
    rc = _load_record_config()
    env = _FakeEnv([_FakeSensor("ft_sensor", [0] * 6)])
    try:
        rc.SOURCE_REGISTRY["robot.sensor"](env, name="nope")
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
    assert ee.source == "robot.sensor" and ee.params.get("name") == "ext_effort"
    assert ee.include_in_state is False and tuple(ee.resolved_shape()) == (6,)
    feats = cfg.to_features(use_video=False)
    # stored but NOT a policy input
    assert "extra.external_effort" in feats
    assert "external_effort" not in "".join(feats["observation.state"]["names"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} sensor-source tests passed.")
