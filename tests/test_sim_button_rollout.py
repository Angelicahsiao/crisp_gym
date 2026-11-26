"""The rollout harness of the press-button simulation (crisp_gym/sim/button_rollout.py).

The scoring is checked on the scripted demonstrator, which must click and
slide on every setup. The closed loop is checked with FAKE policy workers that
speak the deploy worker's pipe protocol (meta first, then reset / infer): one
that pushes past the button must click it, and one that commands the button's
top, as a measured-pose label would, must not. The real worker with a real
checkpoint needs lerobot and torch; RUN_SHEETS.md sheet 6 runs it.

Needs `mujoco` and scipy (tests/requirements.txt); skipped without mujoco.
"""

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

from crisp_gym.sim import button_rollout as br  # noqa: E402
from crisp_gym.sim import button_task as bt  # noqa: E402

IDENTITY_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _setup(i, seed=br.SETUP_SEED):
    return bt.sample_params(np.random.default_rng([seed, i]))


def _fake_worker(push_m, seen=None, fail=None):
    """A policy that goes above the button, then commands `push_m` past its top."""

    def worker(conn, path, overrides, state_input="auto"):
        if fail == "start":
            conn.send(("error", "cannot load"))
            return
        conn.send({"n_obs_steps": 2, "n_action_steps": 12, "state_input": state_input})
        while True:
            msg = conn.recv()
            if msg is None:
                return
            kind, window = msg
            if seen is not None:
                seen.append((kind, window))
            if kind != "infer":
                continue
            if fail == "infer":
                conn.send(("error", "boom"))
                continue
            button = np.asarray(window[-1]["observation.state.button_in_tcp"], dtype=np.float64)
            target = button.copy()  # in the TCP frame, whose z points down
            if np.linalg.norm(button[:2]) > 0.002:
                target[2] -= 0.01  # first above the cap
            else:
                target[2] += push_m
            row = np.concatenate([target, IDENTITY_ROT6D, [0.0]])
            conn.send(np.tile(row, (16, 1)))

    return worker


@pytest.mark.parametrize("stiffness", (200.0, 400.0, 800.0))
def test_the_scripted_demonstrator_clicks_and_slides_on_every_setup(stiffness):
    cfg = bt.ButtonTaskConfig(stiffness=stiffness)
    for i in range(4):
        params = _setup(i)
        s = br.score(br.demonstrator_record(cfg, params), params)
        assert s.clicked and s.slid and not s.artefact, (i, s)
        # it holds after the click: just enough force, no slamming
        assert s.after_click_ratio == pytest.approx(1.0, abs=0.15)
        assert s.slide_m >= br.SLIDE_MIN_M


def test_a_policy_that_pushes_past_the_button_clicks_it():
    cfg = bt.ButtonTaskConfig(stiffness=400.0)
    client = br.PolicyClient("unused", worker=_fake_worker(push_m=0.06))
    try:
        params = _setup(0)
        s = br.score(br.rollout(cfg, params, client, 6.0, 0.03), params)
    finally:
        client.close()
    assert s.clicked and not s.artefact, s
    # 60 mm past the cap at 400 N/m pushes far harder than this button needs
    assert s.after_click_ratio > 1.5


def test_a_policy_that_commands_the_buttons_top_never_clicks():
    # what replaying measured poses does: the spring gets no gap, so no push
    cfg = bt.ButtonTaskConfig(stiffness=400.0)
    client = br.PolicyClient("unused", worker=_fake_worker(push_m=0.0))
    try:
        params = _setup(0)
        s = br.score(br.rollout(cfg, params, client, 6.0, 0.03), params)
    finally:
        client.close()
    assert not s.clicked and not s.slid, s
    assert s.peak_force_n < 0.5 * params.activation_force


def test_the_worker_gets_the_deploy_frames_and_one_reset_per_rollout():
    seen = []
    cfg = bt.ButtonTaskConfig()
    client = br.PolicyClient("unused", worker=_fake_worker(push_m=0.0, seen=seen))
    try:
        for i in range(2):
            br.rollout(cfg, _setup(i), client, 2.0, 0.03)
    finally:
        client.close()
    kinds = [k for k, _ in seen]
    assert kinds.count("reset") == 2 and kinds[0] == "reset"
    # 30 frames, a new chunk every 12: frames 0, 12 and 24 of each rollout
    assert kinds.count("infer") == 6
    first = next(w for k, w in seen if k == "infer")
    assert len(first) == 1  # the window grows to n_obs_steps from the reset on
    windows = [w for k, w in seen if k == "infer"]
    assert all(len(w) == 2 for w in windows[1:3])
    frame = windows[1][-1]
    assert frame["task"] == bt.TASK
    assert frame["observation.state"].shape == (13,)
    np.testing.assert_allclose(
        frame["observation.state"],
        np.concatenate(
            [
                frame["observation.state.cartesian"],
                frame["observation.state.gripper"],
                frame["observation.state.button_in_tcp"],
            ]
        ),
    )
    np.testing.assert_allclose(frame["observation.state.cartesian"][3:], bt.TCP_ROT6D)


def test_worker_errors_are_raised_not_swallowed():
    with pytest.raises(RuntimeError, match="did not start"):
        br.PolicyClient("unused", worker=_fake_worker(0.0, fail="start"))
    client = br.PolicyClient("unused", worker=_fake_worker(0.0, fail="infer"))
    try:
        with pytest.raises(RuntimeError, match="inference failed: boom"):
            br.rollout(bt.ButtonTaskConfig(), _setup(0), client, 1.0, 0.03)
    finally:
        client.close()


def test_score_marks_artefacts_and_a_missing_click():
    params = _setup(0)
    n = 10
    record = {"tcp": np.zeros((n, 3)), "force": np.zeros((n, 3)), "depth": np.zeros(n)}
    s = br.score(record, params)
    assert not s.clicked and not s.slid and s.time_to_click_s is None and not s.artefact
    record["tcp"][5, 2] = -0.003  # sunk 3 mm into the panel
    assert br.score(record, params).artefact


def test_the_stiffness_and_label_come_from_the_checkpoint(tmp_path):
    root = tmp_path / "datasets" / "sim_button_k800" / "lerobot"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "sim_config.json").write_text(json.dumps({"stiffness": 800.0}))
    ckpt = (
        tmp_path / "runs" / "sim_dit_k800_command" / "checkpoints" / "012570" / "pretrained_model"
    )
    ckpt.mkdir(parents=True)
    (ckpt / "train_config.json").write_text(json.dumps({"dataset": {"root": str(root)}}))
    assert br.stiffness_of(ckpt) == 800.0
    assert br.label_of(ckpt) == "sim_dit_k800_command@012570"
    assert br.stiffness_of(tmp_path) is None
