"""scripts/train_multi_task_dit_server.sh: the command it builds, and its gates.

The script runs in its documented layout (a temp WORKDIR with a `crisp_gym`
symlink to this repo beside it), with a FAKE `python` first on PATH that
records its argv and exits. That captures the exact argument list handed to the
training launcher and to the preflight, after all of bash's quoting, without
lerobot, torch or a GPU.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "train_multi_task_dit_server.sh"

# lerobot 0.6.1's default image transforms (transforms.py), minus "affine".
DEFAULT_TFS_NO_AFFINE = {
    "brightness": ("ColorJitter", {"brightness": [0.8, 1.2]}),
    "contrast": ("ColorJitter", {"contrast": [0.8, 1.2]}),
    "saturation": ("ColorJitter", {"saturation": [0.5, 1.5]}),
    "hue": ("ColorJitter", {"hue": [-0.05, 0.05]}),
    "sharpness": ("SharpnessJitter", {"sharpness": [0.5, 1.5]}),
}


@pytest.fixture
def layout(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(SCRIPT, work)
    (work / "crisp_gym").symlink_to(REPO, target_is_directory=True)

    ds = tmp_path / "datasets" / "toy"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text("{}")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "python"
    fake.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import json, os, sys
        with open(os.environ["FAKE_PY_LOG"], "a") as f:
            f.write(json.dumps(sys.argv[1:]) + "\\n")
        if sys.argv[1:] and sys.argv[1].endswith(("lerobot_relative_pose.py", "train_absolute_next_pose.py")):
            sys.exit(int(os.environ.get("FAKE_TRAIN_EXIT", "0")))
    """))
    fake.chmod(0o755)
    return {"work": work, "ds": ds, "bin": fake_bin, "log": tmp_path / "py_calls.jsonl", "tmp": tmp_path}


def _run(layout, env=None, args=()):
    full_env = dict(os.environ)
    full_env.update({
        "PATH": f"{layout['bin']}{os.pathsep}{os.environ['PATH']}",
        "FAKE_PY_LOG": str(layout["log"]),
        "DATASET": str(layout["ds"]),
        "OUT": str(layout["tmp"] / "runs" / "run1"),
        "SKIP_HUB_CHECK": "1",
    })
    full_env.update(env or {})
    proc = subprocess.run(
        ["bash", str(layout["work"] / SCRIPT.name), *args],
        capture_output=True, text=True, cwd=layout["tmp"], env=full_env, timeout=60,
    )
    calls = []
    if layout["log"].exists():
        calls = [json.loads(line) for line in layout["log"].read_text().splitlines()]
    return proc, calls


def _train_argv(calls):
    train = [c for c in calls if c and c[0].endswith(("lerobot_relative_pose.py", "train_absolute_next_pose.py"))]
    assert len(train) == 1, calls
    return train[0]


def _flag(argv, name):
    """Value of --name=value in argv, or None."""
    hits = [a.split("=", 1)[1] for a in argv if a.startswith(f"{name}=")]
    assert len(hits) <= 1, f"{name} passed {len(hits)} times"
    return hits[0] if hits else None


# ── the default command ───────────────────────────────────────────────────────

def test_default_is_relative_16d_with_the_pinned_settings(layout):
    proc, calls = _run(layout)
    assert proc.returncode == 0, proc.stderr
    argv = _train_argv(calls)

    assert argv[0] == str(layout["work"] / "crisp_gym" / "crisp_gym" / "scripts" / "lerobot_relative_pose.py")
    assert argv[1] == "--wrt-start"
    assert "--epochs" not in argv
    expected = {
        "--policy.type": "multi_task_dit",
        "--policy.device": "cuda",
        "--policy.push_to_hub": "false",
        "--policy.horizon": "16",
        "--policy.n_action_steps": "8",
        "--policy.objective": "diffusion",
        "--policy.image_resize_shape": "[236,236]",
        "--policy.image_crop_shape": "[224,224]",
        "--policy.noise_scheduler_type": "DDIM",
        "--policy.num_inference_steps": "10",
        "--dataset.image_transforms.enable": "true",
        "--dataset.root": str(layout["ds"]),
        "--batch_size": "64",
        "--steps": "100000",
        "--save_freq": "20000",
        "--num_workers": "8",
        "--wandb.enable": "false",
    }
    for flag, value in expected.items():
        assert _flag(argv, flag) == value, flag
    # Not set unless asked for.
    assert _flag(argv, "--policy.scheduler_warmup_steps") is None
    assert _flag(argv, "--policy.num_integration_steps") is None


def test_default_augmentation_is_lerobots_set_without_affine(layout):
    proc, calls = _run(layout)
    assert proc.returncode == 0, proc.stderr
    tfs = json.loads(_flag(_train_argv(calls), "--dataset.image_transforms.tfs"))
    assert set(tfs) == set(DEFAULT_TFS_NO_AFFINE)          # complete list, no affine
    for name, (typ, kwargs) in DEFAULT_TFS_NO_AFFINE.items():
        assert tfs[name] == {"weight": 1.0, "type": typ, "kwargs": kwargs}, name


def test_preflight_runs_without_gripper_exclusion(layout):
    proc, calls = _run(layout, {"N_ACTION_STEPS": "6"})
    assert proc.returncode == 0, proc.stderr
    pre = [c for c in calls if c and c[0].endswith("groot_preflight.py")]
    assert pre == [[
        str(layout["work"] / "crisp_gym" / "crisp_gym" / "scripts" / "groot_preflight.py"),
        str(layout["ds"]), "--exclude", "--chunk-size", "6",
    ]]


def test_preflight_failure_stops_before_training(layout):
    fake = layout["bin"] / "python"
    fake.write_text(fake.read_text() + "if sys.argv[1:] and sys.argv[1].endswith('groot_preflight.py'):\n    sys.exit(1)\n")
    proc, calls = _run(layout)
    assert proc.returncode == 1
    assert "Preflight FAILED" in proc.stderr
    assert not any(c and c[0].endswith("lerobot_relative_pose.py") for c in calls)


def test_skip_preflight(layout):
    proc, calls = _run(layout, {"SKIP_PREFLIGHT": "1"})
    assert proc.returncode == 0, proc.stderr
    assert not any(c and c[0].endswith("groot_preflight.py") for c in calls)


def test_hub_check_runs_unless_skipped(layout):
    proc, calls = _run(layout, {"SKIP_HUB_CHECK": "0"})
    assert proc.returncode == 0, proc.stderr
    assert ["-"] in calls                                  # the heredoc hub probe


# ── modes and knobs ───────────────────────────────────────────────────────────

def test_wrt_start_0_gives_the_10d_state(layout):
    argv = _train_argv(_run(layout, {"WRT_START": "0"})[1])
    assert argv[1] == "--no-wrt-start"


def test_absolute_mode_uses_the_absolute_launcher(layout):
    argv = _train_argv(_run(layout, {"ABSOLUTE": "1"})[1])
    assert argv[0].endswith("train_absolute_next_pose.py")
    assert "--wrt-start" not in argv and "--no-wrt-start" not in argv
    assert _flag(argv, "--policy.type") == "multi_task_dit"


def test_epochs_is_passed_to_the_launcher(layout):
    for env, flag_at in (({}, 2), ({"ABSOLUTE": "1"}, 1)):
        layout["log"].unlink(missing_ok=True)
        argv = _train_argv(_run(layout, {**env, "EPOCHS": "120"})[1])
        assert argv[flag_at:flag_at + 2] == ["--epochs", "120"]
        # Still passed: with EPOCHS they set the checkpoint count.
        assert _flag(argv, "--steps") == "100000" and _flag(argv, "--save_freq") == "20000"


def test_flow_matching_sets_integration_steps_not_ddim(layout):
    argv = _train_argv(_run(layout, {"OBJECTIVE": "flow_matching", "INFERENCE_STEPS": "12"})[1])
    assert _flag(argv, "--policy.objective") == "flow_matching"
    assert _flag(argv, "--policy.num_integration_steps") == "12"
    assert _flag(argv, "--policy.noise_scheduler_type") is None
    assert _flag(argv, "--policy.num_inference_steps") is None


def test_augment_all_and_off(layout):
    argv = _train_argv(_run(layout, {"AUGMENT": "all"})[1])
    assert _flag(argv, "--dataset.image_transforms.enable") == "true"
    assert _flag(argv, "--dataset.image_transforms.tfs") is None     # lerobot's full default set
    layout["log"].unlink()
    argv = _train_argv(_run(layout, {"AUGMENT": "0"})[1])
    assert not any(a.startswith("--dataset.image_transforms") for a in argv)


def test_warmup_resize_and_sizes_are_forwarded(layout):
    argv = _train_argv(_run(layout, {
        "WARMUP_STEPS": "500", "IMAGE_RESIZE": "256", "HORIZON": "24", "N_ACTION_STEPS": "12",
        "BATCH_SIZE": "32", "NUM_WORKERS": "4",
    })[1])
    assert _flag(argv, "--policy.scheduler_warmup_steps") == "500"
    assert _flag(argv, "--policy.image_resize_shape") == "[256,256]"
    assert _flag(argv, "--policy.image_crop_shape") == "[224,224]"    # CLIP's input size
    assert _flag(argv, "--policy.horizon") == "24"
    assert _flag(argv, "--policy.n_action_steps") == "12"
    assert _flag(argv, "--batch_size") == "32" and _flag(argv, "--num_workers") == "4"


def test_extra_arguments_are_appended_last(layout):
    argv = _train_argv(_run(layout, args=["--policy.num_layers=8", "--seed=7"])[1])
    assert argv[-2:] == ["--policy.num_layers=8", "--seed=7"]


# ── refusals: all before the preflight, the hub check or training ────────────

@pytest.mark.parametrize(
    ("env", "args", "message"),
    [
        ({"EPOCHS": "0"}, [], "EPOCHS must be a positive number"),
        ({"EPOCHS": "abc"}, [], "EPOCHS must be a positive number"),
        ({}, ["--epochs", "10"], "use EPOCHS=N"),
        ({}, ["--epochs=10"], "use EPOCHS=N"),
        ({"ABSOLUTE": "2"}, [], "ABSOLUTE must be 0 or 1"),
        ({"WRT_START": "yes"}, [], "WRT_START must be 0 or 1"),
        ({"OBJECTIVE": "ddpm"}, [], "OBJECTIVE must be diffusion or flow_matching"),
        ({"AUGMENT": "1"}, [], "AUGMENT must be noaffine, all or 0"),
        ({"STEPS": "abc"}, [], "STEPS must be a positive integer"),
        ({"INFERENCE_STEPS": "0"}, [], "INFERENCE_STEPS must be a positive integer"),
        ({"WARMUP_STEPS": "-1"}, [], "WARMUP_STEPS must be a non-negative integer"),
        ({"IMAGE_RESIZE": "200"}, [], "IMAGE_RESIZE must be >= 224"),
        ({"HORIZON": "8", "N_ACTION_STEPS": "12"}, [], "cannot exceed HORIZON"),
    ],
)
def test_bad_settings_exit_2_before_anything_runs(layout, env, args, message):
    proc, calls = _run(layout, {**env, "SKIP_HUB_CHECK": "0"}, args)
    assert proc.returncode == 2
    assert message in proc.stderr
    assert calls == []                                     # no preflight, no hub check, no training


def test_missing_dataset_exits_2(layout):
    proc, calls = _run(layout, {"DATASET": str(layout["tmp"] / "nope")})
    assert proc.returncode == 2
    assert "No meta/info.json" in proc.stderr


def test_nested_lerobot_dataset_dir_is_found(layout):
    nested = layout["tmp"] / "datasets" / "nested"
    (nested / "lerobot" / "meta").mkdir(parents=True)
    (nested / "lerobot" / "meta" / "info.json").write_text("{}")
    argv = _train_argv(_run(layout, {"DATASET": str(nested), "SKIP_PREFLIGHT": "1"})[1])
    assert _flag(argv, "--dataset.root") == str(nested / "lerobot")


def test_missing_launcher_exits_2(layout):
    (layout["work"] / "crisp_gym").unlink()                # no clone beside the script
    proc, calls = _run(layout)
    assert proc.returncode == 2
    assert "Training launcher missing" in proc.stderr
    assert calls == []


# ── dry run, logs and exit status ─────────────────────────────────────────────

def test_dry_run_launches_nothing(layout):
    proc, calls = _run(layout, {"DRY_RUN": "1", "SKIP_PREFLIGHT": "1"})
    assert proc.returncode == 0, proc.stderr
    assert "(dry run" in proc.stdout
    assert "multi_task_dit" in proc.stdout
    assert calls == []


def test_logs_are_kept_beside_a_run_that_never_wrote_a_checkpoint(layout):
    proc, _ = _run(layout, {"SKIP_PREFLIGHT": "1"})
    assert proc.returncode == 0, proc.stderr
    out = layout["tmp"] / "runs" / "run1"
    assert not out.exists()                                # the fake never creates it
    launch = Path(f"{out}.launch_command.txt").read_text()
    assert "lerobot_relative_pose.py" in launch and "multi_task_dit" in launch
    assert Path(f"{out}.log").exists()


def test_training_failure_status_is_propagated(layout):
    proc, _ = _run(layout, {"SKIP_PREFLIGHT": "1", "FAKE_TRAIN_EXIT": "3"})
    assert proc.returncode == 3
    assert "FAILED (exit 3)" in proc.stdout


def test_success_message_matches_the_mode(layout):
    proc, _ = _run(layout, {"SKIP_PREFLIGHT": "1"})
    assert 'state_input: "relative_wrt_start"' in proc.stdout
    assert "SAME task string" in proc.stdout
    layout["log"].unlink()
    proc, _ = _run(layout, {"SKIP_PREFLIGHT": "1", "ABSOLUTE": "1",
                            "OUT": str(layout["tmp"] / "runs" / "run2")})
    assert "absolute_lerobot_policy.yaml" in proc.stdout
