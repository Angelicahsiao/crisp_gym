"""Simulated press-button datasets look like robot recordings to the tools that read them.

The feature schema and meta/record_config.json come from the recorder's own
RecordConfig, so they are compared with the Franka recording contract
(config/recording/umi_robot_full_record.yaml). The round trip through lerobot
needs the real lerobot and is marked `lerobot`.

Needs `mujoco` (tests/requirements.txt); skipped without it. No ROS.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from crisp_gym.record.record_config import RecordConfig  # noqa: E402
from crisp_gym.sim import button_dataset as bd  # noqa: E402
from crisp_gym.sim import button_task as bt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FRANKA_CONTRACT = REPO / "crisp_gym" / "config" / "recording" / "umi_robot_full_record.yaml"


def test_the_schema_has_the_robot_names_for_pose_gripper_command_and_action():
    sim = bd.dataset_features(bt.ButtonTaskConfig())
    real = RecordConfig.from_yaml(FRANKA_CONTRACT).to_features(joint_count=7, use_video=False)
    for key in ("observation.state.cartesian", "observation.state.gripper", "action"):
        assert sim[key] == real[key], key
    assert sim["extra.target_cartesian"] == real["extra.target_cartesian"]
    # the robot's 10-D state, then the extra input after the gripper (D-10)
    assert sim["observation.state"]["names"] == real["observation.state"]["names"] + [
        f"button_in_tcp_{i}" for i in range(3)
    ]


def test_the_contract_has_the_franka_action_but_is_never_mixed_with_robot_data():
    sim = bd.record_contract(bt.ButtonTaskConfig()).to_metadata()
    real = RecordConfig.from_yaml(FRANKA_CONTRACT).to_metadata()
    for field in RecordConfig.CONTRACT_FIELDS:  # rate and the action's meaning
        assert sim[field] == real[field], field
    # what lerobot_relative_pose.read_action_source turns into kind "measured"
    assert sim["action"]["definition"] == "next_tcp_pose"
    assert sim["action"]["lookahead"] == 1
    assert sim["robot_type"] == bd.ROBOT_TYPE
    # the policy inputs differ (the button's position instead of a camera), so
    # the check postprocess_align_datasets.py relies on refuses to mix them
    assert not RecordConfig.contracts_compatible(sim, real)


def test_every_column_of_an_episode_has_a_feature_of_its_shape():
    cfg = bt.ButtonTaskConfig()
    features = bd.dataset_features(cfg)
    columns = bt.run_episode(cfg, bt.sample_params(bd.episode_rng(0, 0))).columns
    assert set(columns) == set(features)
    for key, column in columns.items():
        assert column.shape[1:] == tuple(features[key]["shape"]), key
        assert len(features[key]["names"]) == column.shape[1], key


def test_preview_prints_a_click():
    assert bd.main(["--preview", "--stiffness", "400"]) == 0


@pytest.mark.lerobot
def test_written_dataset_reads_back_with_its_contract(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "sim_button_k400" / "lerobot"
    cfg = bt.ButtonTaskConfig(stiffness=400.0)
    result = bd.write_dataset(root, "sim_button_k400", cfg, 2, seed=7, log=lambda _: None)
    assert result["episodes"] == 2

    dataset = LeRobotDataset("sim_button_k400", root=root)
    assert dataset.num_episodes == 2 and dataset.fps == 15
    assert dataset.meta.robot_type == bd.ROBOT_TYPE
    record = json.loads((root / "meta" / "record_config.json").read_text())
    assert record["action"]["definition"] == "next_tcp_pose"
    made = json.loads((root / "meta" / "sim_config.json").read_text())
    assert made["stiffness"] == 400.0 and len(made["episodes"]) == 2

    # the first episode on disk is the one (seed, index 0) simulates
    first = made["episodes"][0]["index"]
    columns = bt.run_episode(cfg, bt.sample_params(bd.episode_rng(7, first))).columns
    frame = dataset[3]
    for key in ("observation.state", "action", "extra.target_cartesian", "extra.tactile.left"):
        np.testing.assert_allclose(frame[key].numpy(), columns[key][3], rtol=1e-6, err_msg=key)

    with pytest.raises(FileExistsError):
        bd.write_dataset(root, "again", cfg, 1, seed=7)
