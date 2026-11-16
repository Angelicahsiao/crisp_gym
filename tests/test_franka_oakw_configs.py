"""The single-camera Franka configs agree with each other.

A recording/deployment mismatch in camera keys or size does not raise; it
silently degrades the policy. These YAML-only checks pin what the files'
comments promise: the record config's camera exists in both recording envs at
the stored size (so no per-frame resize), the torque sensor the record config
reads exists in both recording envs, and the deploy envs see the same camera,
pin the 10-D state, and carry the per-step target limit.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CONFIG = Path(__file__).resolve().parents[1] / "crisp_gym" / "config"
RECORDING_ENVS = ("dric_franka_oakw_umi", "factr_dric_franka_oakw")
DEPLOY_ENVS = ("dric_franka_oakw_deploy_umi", "dric_franka_oakw_deploy_groot")


def _env(name):
    return yaml.safe_load((CONFIG / "envs" / f"{name}.yaml").read_text())


def _record():
    return yaml.safe_load((CONFIG / "recording" / "umi_robot_full_record.yaml").read_text())


def _record_cameras():
    return {
        o["camera"]: o["shape"]
        for o in _record()["observations"]
        if o.get("source") == "camera.image"
    }


@pytest.mark.parametrize("name", RECORDING_ENVS)
def test_recording_env_provides_what_the_record_config_reads(name):
    env = _env(name)
    cams = {c["camera_name"]: c["resolution"] for c in env["camera_configs"]}
    for camera, shape in _record_cameras().items():
        assert camera in cams, f"{name} has no camera {camera!r}"
        assert list(cams[camera]) == list(shape[:2]), "a size mismatch resizes every frame"
    sensors = {s["name"] for s in env.get("sensor_configs") or []}
    wanted = {o["name"] for o in _record()["observations"] if o.get("source") == "robot.sensor"}
    assert wanted <= sensors, f"{name} lacks sensors {wanted - sensors}: KeyError on frame 1"
    assert env["robot_config"]["has_effort_feedback"] is True  # extra.joint_efforts


@pytest.mark.parametrize("name", DEPLOY_ENVS)
def test_deploy_env_matches_the_recorded_data(name):
    env = _env(name)
    assert [c["camera_name"] for c in env["camera_configs"]] == list(_record_cameras())
    assert env["observations_to_include_to_state"] == [
        "observation.state.cartesian",
        "observation.state.gripper",
    ]
    assert env["use_relative_actions"] is False
    assert env["robot_config"]["has_effort_feedback"] is False  # would add 7 state dims
    assert env["max_target_step_m"] == 0.03 and env["max_target_step_deg"] == 5.0


def test_one_camera_topic_for_the_whole_family():
    topics = {
        (c["camera_color_image_topic"], c["camera_frame"])
        for name in RECORDING_ENVS + DEPLOY_ENVS
        for c in _env(name)["camera_configs"]
    }
    assert topics == {("/wrist/oakw_cam/rgb/image_raw", "wrist_link")}
