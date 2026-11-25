"""Image augmentation in the two GR00T launchers, and its parity with the DiT launcher.

scripts/train_groot.sh (--augment) and scripts/train_groot_server.sh (AUGMENT)
run with FAKE `python` / `python3` executables first on PATH that record their
argv and exit, so the exact lerobot argument list is checked after bash's
quoting, without lerobot, torch or a GPU. The transform list must stay
byte-identical to scripts/train_multi_task_dit_server.sh's.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FLAG_SCRIPT = REPO / "scripts" / "train_groot.sh"
SERVER_SCRIPT = REPO / "scripts" / "train_groot_server.sh"
DIT_SCRIPT = REPO / "scripts" / "train_multi_task_dit_server.sh"

# lerobot 0.6.1's default image transforms, minus "affine".
DEFAULT_TFS_NO_AFFINE = {
    "brightness": ("ColorJitter", {"brightness": [0.8, 1.2]}),
    "contrast": ("ColorJitter", {"contrast": [0.8, 1.2]}),
    "saturation": ("ColorJitter", {"saturation": [0.5, 1.5]}),
    "hue": ("ColorJitter", {"hue": [-0.05, 0.05]}),
    "sharpness": ("SharpnessJitter", {"sharpness": [0.5, 1.5]}),
}
TRAINERS = ("lerobot.scripts.lerobot_train", "lerobot_relative_pose.py", "epoch_steps.py")


@pytest.fixture
def env(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # The real interpreter in the shebang: `#!/usr/bin/env python3` would find
    # the fake python3 itself on PATH and call itself forever.
    fake = f"#!{sys.executable}\n" + textwrap.dedent("""\
        import json, os, sys
        with open(os.environ["FAKE_PY_LOG"], "a") as f:
            f.write(json.dumps(sys.argv[1:]) + "\\n")
    """)
    for name in ("python", "python3"):
        (fake_bin / name).write_text(fake)
        (fake_bin / name).chmod(0o755)

    ds = tmp_path / "datasets" / "toy"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text("{}")

    # the server script's documented layout: a crisp_gym clone beside it
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(SERVER_SCRIPT, work)
    (work / "crisp_gym").symlink_to(REPO, target_is_directory=True)
    return {"tmp": tmp_path, "bin": fake_bin, "ds": ds, "work": work, "log": tmp_path / "py.jsonl"}


def _run(env, script, settings=None, args=()):
    full = dict(os.environ)
    full.update(
        {
            "PATH": f"{env['bin']}{os.pathsep}{os.environ['PATH']}",
            "FAKE_PY_LOG": str(env["log"]),
            "SKIP_HUB_CHECK": "1",
            "SKIP_PREFLIGHT": "1",
            "DATASET": str(env["ds"]),
            "OUT": str(env["tmp"] / "runs" / "server_run"),
        }
    )
    full.update(settings or {})
    if script == "flag":
        argv = ["bash", str(FLAG_SCRIPT), "--dataset", str(env["ds"])]
        argv += ["--output", str(env["tmp"] / "runs" / "flag_run"), "--skip-preflight", *args]
    else:
        argv = ["bash", str(env["work"] / SERVER_SCRIPT.name), *args]
    proc = subprocess.run(
        argv, capture_output=True, text=True, cwd=env["tmp"], env=full, timeout=30
    )
    calls = []
    if env["log"].exists():
        calls = [json.loads(line) for line in env["log"].read_text().splitlines()]
    return proc, calls


def _train_argv(calls):
    train = [c for c in calls if any(t in " ".join(c[:2]) for t in TRAINERS)]
    assert len(train) == 1, calls
    return train[0]


def _flag(argv, name):
    hits = [a.split("=", 1)[1] for a in argv if a.startswith(f"{name}=")]
    assert len(hits) <= 1, f"{name} passed {len(hits)} times"
    return hits[0] if hits else None


def _augment(script, value):
    """The way each launcher is told the augmentation mode."""
    if script == "flag":
        return {}, ("--augment", value)
    return {"AUGMENT": value}, ()


MODES = {
    "default": ({}, ()),
    "se3": ({"SE3": "1"}, ("--se3",)),
    "epochs": ({"EPOCHS": "10"}, ("--epochs", "10")),
}


@pytest.mark.parametrize("script", ("flag", "server"))
@pytest.mark.parametrize("mode", MODES)
def test_default_is_lerobots_set_without_affine_in_every_mode(env, script, mode):
    settings, args = MODES[mode]
    proc, calls = _run(
        env, script, settings if script == "server" else {}, args if script == "flag" else ()
    )
    assert proc.returncode == 0, proc.stderr
    argv = _train_argv(calls)
    assert _flag(argv, "--dataset.image_transforms.enable") == "true"
    tfs = json.loads(_flag(argv, "--dataset.image_transforms.tfs"))
    assert set(tfs) == set(DEFAULT_TFS_NO_AFFINE)  # the complete list, no affine
    for name, (typ, kwargs) in DEFAULT_TFS_NO_AFFINE.items():
        assert tfs[name] == {"weight": 1.0, "type": typ, "kwargs": kwargs}, name


@pytest.mark.parametrize("script", ("flag", "server"))
def test_all_keeps_lerobots_full_set_and_0_turns_it_off(env, script):
    settings, args = _augment(script, "all")
    argv = _train_argv(_run(env, script, settings, args)[1])
    assert _flag(argv, "--dataset.image_transforms.enable") == "true"
    assert _flag(argv, "--dataset.image_transforms.tfs") is None  # lerobot's defaults, affine too

    env["log"].unlink()
    settings, args = _augment(script, "0")
    argv = _train_argv(_run(env, script, settings, args)[1])
    assert not any(a.startswith("--dataset.image_transforms") for a in argv)


@pytest.mark.parametrize("script", ("flag", "server"))
def test_an_unknown_mode_stops_before_anything_runs(env, script):
    settings, args = _augment(script, "1")
    proc, calls = _run(env, script, settings, args)
    assert proc.returncode == 2
    assert "must be noaffine, all or 0, got: 1" in proc.stderr
    assert calls == []


def test_the_flag_launcher_refuses_augment_without_a_value(env):
    proc, calls = _run(env, "flag", args=("--augment",))
    assert proc.returncode == 2
    assert "--augment needs a value" in proc.stderr
    assert calls == []


@pytest.mark.parametrize("script", ("flag", "server"))
def test_appended_arguments_come_after_the_augmentation_so_they_win(env, script):
    extra = "--dataset.image_transforms.max_num_transforms=2"
    args = ("--", extra) if script == "flag" else (extra,)
    argv = _train_argv(_run(env, script, args=args)[1])
    assert argv[-1] == extra
    assert argv.index("--dataset.image_transforms.enable=true") < argv.index(extra)


def test_the_transform_list_is_identical_in_all_three_launchers():
    def block(path):
        text = path.read_text()
        match = re.search(
            r"^if \[ \"\$AUGMENT\" != \"0\" \].*?\n    CMD\+=\(--dataset\.image_transforms\.tfs=\"\$IMAGE_TFS\"\)\nfi\n",
            text,
            re.S | re.M,
        )
        assert match, path
        return match.group(0)

    reference = block(DIT_SCRIPT)
    assert block(FLAG_SCRIPT) == reference
    assert block(SERVER_SCRIPT) == reference
