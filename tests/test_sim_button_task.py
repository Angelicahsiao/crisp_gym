"""The press-button simulation against statics, and its columns against the robot's layout.

The fingertips sit on a spring of stiffness K (crisp_gym/sim/button_task.py).
At rest the spring force K x (measured - command), the true contact force and
the summed taxels must agree; the button must click at its activation force
and not before; and the columns must follow the recording contract
(action[t] = measured pose of frame t+1, rot6d rows, state = pose + gripper +
button).

Needs the `mujoco` package (in tests/requirements.txt); skipped without it.
No ROS, no display, no lerobot.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from crisp_gym.sim import button_task as bt  # noqa: E402

STIFFNESSES = (200.0, 400.0, 800.0)


def _params(seed=0, **overrides):
    params = bt.sample_params(np.random.default_rng(seed))
    return bt.EpisodeParams(**{**params.__dict__, **overrides})


def _press_and_settle(cfg, params, force):
    """Move the command onto the button until the spring pushes with `force`, then wait."""
    spring = bt.button_spring(params)
    preload = bt.PRELOAD_FRACTION * params.activation_force
    depth = max(0.0, (force - preload) / spring["stiffness"])
    target_z = spring["protrusion"] - depth - force / cfg.stiffness
    sim = bt._Sim(cfg, params)
    bx, by = params.button_xy
    start = np.array([bx, by, spring["protrusion"] + 0.005])
    for z in np.linspace(start[2], target_z, 3000):
        sim.step(np.array([bx, by, z]))
    for _ in range(3000):
        sim.step(np.array([bx, by, target_z]))
    return sim, target_z


def test_the_button_spring_clicks_at_its_activation_force():
    params = _params(activation_force=10.0, travel=0.003)
    s = bt.button_spring(params)
    assert s["click_depth"] == pytest.approx(bt.CLICK_FRACTION * 0.003)
    assert s["stiffness"] * s["springref"] == pytest.approx(bt.PRELOAD_FRACTION * 10.0)
    assert s["stiffness"] * (s["springref"] + s["click_depth"]) == pytest.approx(10.0)
    assert s["protrusion"] > 0.003  # the cap stays above the panel at full travel


@pytest.mark.parametrize("stiffness", STIFFNESSES)
def test_below_the_activation_force_no_click_above_it_a_click(stiffness):
    cfg = bt.ButtonTaskConfig(stiffness=stiffness)
    params = _params(activation_force=8.0, travel=0.003)
    sim, _ = _press_and_settle(cfg, params, 0.9 * 8.0)
    assert sim.depth() < bt.button_spring(params)["click_depth"]
    sim, _ = _press_and_settle(cfg, params, 1.1 * 8.0)
    assert sim.depth() >= bt.button_spring(params)["click_depth"]


@pytest.mark.parametrize("stiffness", STIFFNESSES)
def test_at_rest_spring_force_contact_force_and_taxels_agree(stiffness):
    cfg = bt.ButtonTaskConfig(stiffness=stiffness)
    params = _params(activation_force=8.0, travel=0.003)
    sim, target_z = _press_and_settle(cfg, params, 6.0)
    spring_force = stiffness * (sim.tcp()[2] - target_z)
    sensed = sim.sensed()
    taxels, contact = sensed[: 2 * cfg.nx * cfg.ny], sensed[-3:]
    assert spring_force == pytest.approx(6.0, rel=0.02)
    assert contact[2] == pytest.approx(spring_force, rel=0.02)  # pushes the fingertips UP
    assert taxels.sum() == pytest.approx(contact[2], rel=0.02)
    # the fingertips stand on the cap, however far below it the command is
    cap_top = bt.button_spring(params)["protrusion"] - sim.depth()
    assert sim.tcp()[2] == pytest.approx(cap_top, abs=2e-4)


def test_a_centred_button_is_felt_by_both_fingertips():
    cfg = bt.ButtonTaskConfig()
    sim, _ = _press_and_settle(cfg, _params(), 6.0)
    left, right = sim.taxels("left"), sim.taxels("right")
    assert left.sum() > 1.0 and right.sum() > 1.0
    assert (left > 0).sum() < left.size  # only the taxels over the cap


@pytest.mark.parametrize("stiffness", STIFFNESSES)
@pytest.mark.parametrize("seed", (0, 1))
def test_a_demonstration_clicks_and_the_command_leads_by_force_over_k(stiffness, seed):
    cfg = bt.ButtonTaskConfig(stiffness=stiffness)
    episode = bt.run_episode(cfg, _params(seed))
    assert episode.clicked
    c, p = episode.columns, episode.params
    click = int(np.flatnonzero(c["extra.button_pressed"][:, 0] > 0)[0])
    gap = c["observation.state.cartesian"][click, 2] - c["extra.target_cartesian"][click, 2]
    # the measured pose stays at the button; only the command carries the push
    assert stiffness * gap == pytest.approx(p.activation_force, rel=0.15)


def test_sliding_pushes_with_the_commanded_force_and_feels_friction():
    cfg = bt.ButtonTaskConfig()
    episode = bt.run_episode(cfg, _params(1))
    c = episode.columns
    sliding = [i for i, ph in enumerate(episode.phases[:-1]) if ph == "slide"][3:]
    assert len(sliding) >= 5
    normal = c["extra.contact_force"][sliding, 2]
    lateral = np.linalg.norm(c["extra.contact_force"][sliding, :2], axis=1)
    taxels = c["extra.tactile.left"][sliding].sum(1) + c["extra.tactile.right"][sliding].sum(1)
    assert np.median(normal) == pytest.approx(episode.params.slide_force, rel=0.1)
    assert np.median(lateral) == pytest.approx(cfg.friction * np.median(normal), rel=0.15)
    np.testing.assert_allclose(taxels, normal, rtol=0.05, atol=0.05)
    # the fingertips are on the panel, not on the button
    assert np.abs(c["observation.state.cartesian"][sliding, 2]).max() < 2e-4
    assert c["extra.button_depth"][sliding].max() == 0.0


def test_in_free_motion_nothing_is_felt():
    episode = bt.run_episode(bt.ButtonTaskConfig(), _params(2))
    c = episode.columns
    free = [i for i, ph in enumerate(episode.phases[:-1]) if ph == "approach"][1:]
    assert c["extra.contact_force"][free].max() == 0.0
    assert c["extra.tactile.left"][free].max() == 0.0


def test_columns_follow_the_recording_contract():
    cfg = bt.ButtonTaskConfig()
    c = bt.run_episode(cfg, _params(3)).columns
    n = len(c["action"])
    taxels = cfg.nx * cfg.ny
    widths = {
        "observation.state.cartesian": 9,
        "observation.state.gripper": 1,
        "observation.state.button_in_tcp": 3,
        "observation.state": 13,
        "observation.environment_state": 3,
        "action": 10,
        "extra.target_cartesian": 9,
        "extra.tactile.left": taxels,
        "extra.tactile.right": taxels,
        "extra.contact_force": 3,
        "extra.button_depth": 1,
        "extra.button_pressed": 1,
    }
    assert set(c) == set(widths)
    for key, width in widths.items():
        assert c[key].shape == (n, width), key
        assert c[key].dtype == np.float32, key
    pose = c["observation.state.cartesian"]
    # lookahead 1: the action is the next frame's measured pose
    np.testing.assert_array_equal(c["action"][:-1, :9], pose[1:])
    np.testing.assert_array_equal(c["action"][:, 9], cfg.gripper)
    np.testing.assert_array_equal(
        c["observation.state"],
        np.concatenate(
            [pose, c["observation.state.gripper"], c["observation.state.button_in_tcp"]], 1
        ),
    )
    np.testing.assert_array_equal(
        c["observation.environment_state"], c["observation.state.button_in_tcp"]
    )
    # rot6d = the first two ROWS; the TCP's z axis points down into the panel
    rows = pose[:, 3:9].reshape(n, 2, 3)
    np.testing.assert_allclose(rows, np.broadcast_to(bt.TCP_ROT[:2], (n, 2, 3)))
    assert bt.TCP_ROT[:, 2] @ np.array([0.0, 0.0, 1.0]) == -1.0
    assert np.linalg.det(bt.TCP_ROT) == pytest.approx(1.0)
    # at the start the button is below the TCP: positive along the TCP's z
    assert c["observation.state.button_in_tcp"][0, 2] > 0.02


def test_the_same_parameters_give_the_same_episode():
    cfg = bt.ButtonTaskConfig()
    a = bt.run_episode(cfg, _params(4)).columns
    b = bt.run_episode(cfg, _params(4)).columns
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
