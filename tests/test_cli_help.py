"""Every GPU-side script still starts: `python <script> --help` exits 0.

An import error in one of these scripts would otherwise show up only on the
GPU machine. Each script runs in its own process, the way the owner runs it:
by file path, with no `crisp_gym` install assumed.

Not covered, on purpose:
  * the ROS entry points (deploy_policy, the record scripts,
    leader_follower_teleop, check_config): they import rclpy at the top;
  * the training launchers (lerobot_relative_pose, train_absolute_next_pose,
    train_action_from_target_cartesian) and check_relative_pose: they have no
    argument parser of their own and hand `--help` on to lerobot;
  * check_available_policies: it takes no arguments.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "crisp_gym" / "scripts"

# Start with nothing more than tests/requirements.txt installed.
ROS_FREE = [
    SCRIPTS / "add_environment_state.py",
    SCRIPTS / "audit_command_residual.py",
    SCRIPTS / "inspect_dataset.py",
    SCRIPTS / "check_tactile_sync.py",
    SCRIPTS / "check_schema_drift.py",
    SCRIPTS / "fix_dataset_task.py",
    SCRIPTS / "swap_action_offline.py",
    SCRIPTS / "migrate_euler_delta_to_rot6d.py",
    SCRIPTS / "postprocess_align_datasets.py",
    SCRIPTS / "viz_encoder_attention.py",
    SCRIPTS / "groot_preflight.py",
    SCRIPTS / "deploy_preflight.py",
    REPO / "scripts" / "code_metrics.py",
    REPO / "scripts" / "policy_server.py",
    # import mujoco only when they simulate (and lerobot only when writing),
    # so --help works without them
    REPO / "crisp_gym" / "sim" / "spring_pad.py",
    REPO / "crisp_gym" / "sim" / "button_dataset.py",
    REPO / "crisp_gym" / "sim" / "button_rollout.py",
]

# Import torch or lerobot at the top, so they start only where those are.
NEEDS_MORE = [
    pytest.param(SCRIPTS / "check_policy_openloop.py", marks=pytest.mark.lerobot),
    pytest.param(SCRIPTS / "reencode_dataset_videos.py", marks=pytest.mark.lerobot),
]


def _help(script: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("script", ROS_FREE + NEEDS_MORE, ids=lambda p: p.name)
def test_script_answers_help(script):
    assert script.is_file(), f"{script} is gone: update the list in this file"
    result = _help(script)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage" in result.stdout.lower()


def test_every_script_is_listed_or_excluded_on_purpose():
    """A new script must be added above, or to the exclusions with a reason."""
    excluded = {
        # ROS entry points
        "deploy_policy.py",
        "record_lerobot_format_leader_follower.py",
        "record_umi_handheld.py",
        "leader_follower_teleop.py",
        "check_config.py",
        # no argument parser of their own
        "lerobot_relative_pose.py",
        "train_absolute_next_pose.py",
        "train_action_from_target_cartesian.py",
        "check_relative_pose.py",
        "check_available_policies.py",
    }
    listed = {p.name for p in ROS_FREE} | {p.values[0].name for p in NEEDS_MORE}
    on_disk = {p.name for p in SCRIPTS.glob("*.py")}
    assert on_disk - listed - excluded == set()
    elsewhere = {
        "code_metrics.py",
        "policy_server.py",
        "spring_pad.py",
        "button_dataset.py",
        "button_rollout.py",
    }
    assert (listed | excluded) - on_disk - elsewhere == set()
