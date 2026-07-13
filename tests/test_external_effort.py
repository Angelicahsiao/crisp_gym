"""Tests for ExternalEffortEstimator + the robot.external_effort record source.

Pinocchio is stubbed (not installed in CI) so the ESTIMATOR MATH — gravity
subtraction, calibration fit, crisp_py<->pinocchio joint index mapping — is
verified against a controllable fake model. The source provider is checked for
lazy import, caching, calibration loading, and correct wiring to the env.

Run:  python tests/test_external_effort.py   (or via pytest)
"""

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# external_effort imports rclpy + std_msgs at module top (for the URDF fetch
# helper); stub them so the estimator math is testable without ROS.
if "rclpy" not in sys.modules:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        _stub("rclpy")
        _stub("rclpy.node", Node=object)
        _stub("rclpy.qos", DurabilityPolicy=types.SimpleNamespace(TRANSIENT_LOCAL=1),
              QoSProfile=lambda **k: object())
        _stub("std_msgs")
        _stub("std_msgs.msg", String=type("String", (), {}))


# ── fake pinocchio: a 1:1 "arm" whose gravity term is a fixed linear map ──────

def _install_fake_pinocchio(joint_names, gravity_fn):
    """gravity_fn(q_full) -> tau_full over ALL model joints (here == arm)."""
    pin = types.ModuleType("pinocchio")

    class _Joint:
        def __init__(self, idx_q):
            self.idx_q = idx_q

    class _Model:
        def __init__(self):
            self.names = ["universe"] + list(joint_names)
            self.nq = len(joint_names)
            # joint id: 0=universe, 1..n = arm joints
            self._name_to_id = {n: i + 1 for i, n in enumerate(joint_names)}
            self.joints = [_Joint(-1)] + [_Joint(i) for i in range(len(joint_names))]

        def getJointId(self, name):
            return self._name_to_id.get(name, -1)

        def existJointName(self, name):
            return name in self._name_to_id

        def createData(self):
            return object()

    _MODEL_SINGLETON = _Model()
    pin.buildModelFromXML = lambda urdf: _MODEL_SINGLETON
    pin.buildReducedModel = lambda model, lock_ids, q: _MODEL_SINGLETON  # already arm-only
    pin.neutral = lambda model: np.zeros(model.nq)
    pin.computeGeneralizedGravity = lambda model, data, q: gravity_fn(q)
    sys.modules["pinocchio"] = pin
    return _MODEL_SINGLETON


def _load_estimator_module():
    # fresh import each call so the fake pinocchio is picked up
    sys.modules.pop("crisp_gym.util.external_effort", None)
    spec = importlib.util.spec_from_file_location(
        "crisp_gym.util.external_effort",
        REPO / "crisp_gym" / "util" / "external_effort.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Register so a later `from crisp_gym.util.external_effort import ...`
    # (in the source provider) uses THIS fake-pinocchio-backed instance.
    sys.modules["crisp_gym.util.external_effort"] = mod
    return mod


JOINTS = [f"j{i}" for i in range(6)]
# ground-truth gravity: tau_g = G @ q  (diagonal-ish so per-joint fits are exact)
_G = np.diag([2.0, 1.5, 1.0, 0.5, 0.3, 0.1])


def test_gravity_subtraction():
    _install_fake_pinocchio(JOINTS, lambda q: _G @ q)
    mod = _load_estimator_module()
    est = mod.ExternalEffortEstimator(urdf="<xml/>", joint_names=JOINTS)

    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    contact = np.array([0, 0, 3.0, 0, 0, 0])  # 3 Nm external on joint 2
    tau_measured = _G @ q + contact
    np.testing.assert_allclose(est.gravity_effort(q), _G @ q, atol=1e-9)
    np.testing.assert_allclose(est.external_effort(q, tau_measured), contact, atol=1e-9)


def test_calibration_fit_recovers_scale_offset():
    _install_fake_pinocchio(JOINTS, lambda q: _G @ q)
    mod = _load_estimator_module()
    est = mod.ExternalEffortEstimator(urdf="<xml/>", joint_names=JOINTS)

    rng = np.random.default_rng(0)
    true_scale = np.array([1.1, 0.9, 1.05, 0.95, 1.2, 0.8])
    true_offset = np.array([0.2, -0.1, 0.05, 0.3, -0.2, 0.0])
    qs = rng.uniform(-1, 1, size=(200, 6))
    # contact-free: tau = scale*g(q) + offset (no external contact)
    taus = np.stack([true_scale * (_G @ q) + true_offset for q in qs])
    scale, offset = est.fit_calibration(qs, taus)
    np.testing.assert_allclose(scale, true_scale, atol=1e-6)
    np.testing.assert_allclose(offset, true_offset, atol=1e-6)
    # with calibration applied, a contact-free sample reads ~0 external
    q = qs[0]
    np.testing.assert_allclose(est.external_effort(q, taus[0]), 0.0, atol=1e-6)


def test_missing_joint_raises():
    _install_fake_pinocchio(JOINTS, lambda q: _G @ q)
    mod = _load_estimator_module()
    try:
        mod.ExternalEffortEstimator(urdf="<xml/>", joint_names=JOINTS + ["nonexistent"])
        raise AssertionError("missing joint not rejected")
    except ValueError as e:
        assert "nonexistent" in str(e)


# ── source provider (record_config) ───────────────────────────────────────────

def _load_record_config():
    # record_config imports numpy/yaml only at module top — safe without ROS
    from importlib.machinery import SourceFileLoader

    sys.modules.pop("rc_ext", None)
    return SourceFileLoader(
        "rc_ext", str(REPO / "crisp_gym" / "record" / "record_config.py")
    ).load_module()


class _FakeRobot:
    def __init__(self):
        self.node = object()
        self.config = types.SimpleNamespace(joint_names=JOINTS)
        self.joint_values = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        self.current_joint_effort = _G @ self.joint_values + np.array([0, 0, 3.0, 0, 0, 0])


class _FakeEnv:
    def __init__(self):
        self.robot = _FakeRobot()


def test_source_provider_lazy_and_cached(monkeypatch=None):
    _install_fake_pinocchio(JOINTS, lambda q: _G @ q)
    mod = _load_estimator_module()  # register the fake-backed module in sys.modules
    mod.fetch_robot_description = lambda node, **k: "<xml/>"  # skip ROS URDF fetch
    rc = _load_record_config()

    env = _FakeEnv()
    src = rc.SOURCE_REGISTRY["robot.external_effort"]
    out = src(env)
    np.testing.assert_allclose(out, [0, 0, 3.0, 0, 0, 0], atol=1e-6)
    assert out.dtype == np.float32
    # cached on the env: second call reuses the same estimator instance
    est1 = env._external_effort_estimator
    src(env)
    assert env._external_effort_estimator is est1


def test_source_provider_reads_calibration_file():
    _install_fake_pinocchio(JOINTS, lambda q: _G @ q)
    mod = _load_estimator_module()
    mod.fetch_robot_description = lambda node, **k: "<xml/>"
    # find_config lives in crisp_gym.config.path which imports crisp_py at top;
    # stub it so the provider falls back to the literal calibration path.
    _stub("crisp_gym.config.path", find_config=lambda name: None)
    rc = _load_record_config()

    scale = [1.0] * 6
    offset = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # pretend joint-2 offset
    cal = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"scale": scale, "offset": offset}, cal)
    cal.close()

    env = _FakeEnv()
    out = rc.SOURCE_REGISTRY["robot.external_effort"](env, calibration=cal.name)
    # external = tau - (1*g + offset) = contact - offset
    expected = np.array([0, 0, 3.0, 0, 0, 0]) - np.array(offset)
    np.testing.assert_allclose(out, expected, atol=1e-6)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} external-effort tests passed.")
