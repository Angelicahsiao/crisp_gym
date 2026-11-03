"""--epochs across the training launchers and the GR00T shell scripts.

Covers what tests/test_epoch_steps.py cannot: that the launchers find
epoch_steps.py WITHOUT crisp_gym being importable (an uninstalled repo clone, as
in scripts/train_groot_server.sh, or one script copied alone to a GPU PC), that
epoch_steps.py runs as plain lerobot-train, and that the shell scripts compose
the right command.

The launchers run in subprocesses against a FAKE `lerobot.scripts.lerobot_train`
(placed first on PYTHONPATH, so it shadows a real install too). Its main()
builds a config, calls the dataset factory through the module global — exactly
how lerobot's train() resolves it, so the launcher's patch is what runs — and
prints the resulting schedule. A second fake, a `crisp_gym` package whose
import raises, forces the by-path fallback even where crisp_gym IS installed.

Only train_absolute_next_pose.py is run end to end: the other two launchers
import numpy/torch at module level and wrap the dataset, which a fake cannot
satisfy. They carry a byte-identical copy of the same helper block, which
test_epoch_helper_block_is_identical_in_all_launchers enforces.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "crisp_gym" / "scripts"
UTIL = REPO / "crisp_gym" / "util"
EPOCH_STEPS = UTIL / "epoch_steps.py"
ABS_LAUNCHER = SCRIPTS / "train_absolute_next_pose.py"
LAUNCHERS = [
    SCRIPTS / "lerobot_relative_pose.py",
    SCRIPTS / "train_absolute_next_pose.py",
    SCRIPTS / "train_action_from_target_cartesian.py",
]
BLOCK_START = "# ── --epochs support"
BLOCK_END = "# ── end --epochs support"

# 9,000 samples, batch 64, 100k steps / 20k save_freq (5 checkpoints), and
# --epochs 120 -> target 16,875, save every 3,375, total 23,625.
EXPECTED_STEPS = 23_625
EXPECTED_SAVE = 3_375

_FAKE_TRAIN = '''
import json, os, sys
from types import SimpleNamespace


class _Dataset:
    def __len__(self):
        return 9000


{factory}


def main():
    leaked = [a for a in sys.argv[1:] if a.startswith("--epochs")]
    assert not leaked, "--epochs reached lerobot: %r" % (sys.argv,)
    cfg = SimpleNamespace(steps=100000, batch_size=64, save_freq=20000,
                          output_dir=os.environ["FAKE_OUTPUT_DIR"])
    {call}
    print("RESULT " + json.dumps({{
        "steps": cfg.steps, "save_freq": cfg.save_freq,
        "argv": sys.argv[1:], "sys_path": [os.path.abspath(p or os.curdir) for p in sys.path],
    }}))
'''

_LAYOUTS = {
    # lerobot >= 0.5
    "0.6": (
        "def make_train_eval_datasets(cfg):\n    return _Dataset(), None",
        "make_train_eval_datasets(cfg)",
    ),
    # lerobot 0.4.x
    "0.4": ("def make_dataset(cfg):\n    return _Dataset()", "make_dataset(cfg)"),
    # neither factory: the launcher has nowhere to hook
    "none": ("", "pass"),
}


def _fake_lerobot(root: Path, layout: str) -> Path:
    factory, call = _LAYOUTS[layout]
    pkg = root / f"fake_lerobot_{layout}"
    (pkg / "lerobot" / "scripts").mkdir(parents=True)
    (pkg / "lerobot" / "__init__.py").write_text("")
    (pkg / "lerobot" / "scripts" / "__init__.py").write_text("")
    (pkg / "lerobot" / "scripts" / "lerobot_train.py").write_text(
        _FAKE_TRAIN.format(factory=factory, call=call)
    )
    return pkg


@pytest.fixture
def fake06(tmp_path):
    return _fake_lerobot(tmp_path, "0.6")


@pytest.fixture
def no_crisp_gym(tmp_path):
    """A `crisp_gym` that cannot be imported, shadowing any real install."""
    d = tmp_path / "block_crisp_gym"
    (d / "crisp_gym").mkdir(parents=True)
    (d / "crisp_gym" / "__init__.py").write_text(
        'raise ImportError("crisp_gym deliberately unimportable in this test")\n'
    )
    return d


def _run(script: Path, args, pythonpath, tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in pythonpath)
    env["FAKE_OUTPUT_DIR"] = str(tmp_path / "out")
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True, cwd=tmp_path, env=env, timeout=120,
    )
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    return proc, result


# ── epoch_steps.py as plain lerobot-train ─────────────────────────────────────

@pytest.mark.parametrize("layout", ["0.6", "0.4"])
def test_epoch_steps_runs_as_plain_lerobot_train(tmp_path, layout):
    fake = _fake_lerobot(tmp_path, layout)
    proc, res = _run(EPOCH_STEPS, ["--epochs", "120", "--policy.type=groot"], [fake], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert res["steps"] == EXPECTED_STEPS
    assert res["save_freq"] == EXPECTED_SAVE
    # --epochs stripped, every other argument passed through untouched.
    assert res["argv"] == ["--policy.type=groot"]


def test_epoch_steps_without_epochs_leaves_the_schedule_alone(tmp_path, fake06):
    proc, res = _run(EPOCH_STEPS, ["--policy.type=groot"], [fake06], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert res["steps"] == 100_000
    assert res["save_freq"] == 20_000


def test_epoch_steps_refuses_epochs_when_there_is_no_factory_to_hook(tmp_path):
    fake = _fake_lerobot(tmp_path, "none")
    proc, res = _run(EPOCH_STEPS, ["--epochs", "120"], [fake], tmp_path)
    assert proc.returncode != 0
    assert res is None                       # never reached training
    assert "neither" in proc.stderr


def test_epoch_steps_drops_its_own_directory_from_sys_path(tmp_path, fake06):
    # Running the file puts crisp_gym/util/ on sys.path, where sibling modules
    # (prompt.py, rot6d.py, ...) could shadow something lerobot imports.
    proc, res = _run(EPOCH_STEPS, ["--epochs", "120"], [fake06], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert str(UTIL) not in res["sys_path"]


def test_epoch_steps_needs_no_crisp_gym(tmp_path, fake06, no_crisp_gym):
    proc, res = _run(EPOCH_STEPS, ["--epochs", "120"], [no_crisp_gym, fake06], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert res["steps"] == EXPECTED_STEPS


# ── launchers: finding epoch_steps.py without crisp_gym ───────────────────────

def test_launcher_uses_the_package_when_crisp_gym_is_importable(tmp_path, fake06):
    proc, res = _run(ABS_LAUNCHER, ["--epochs", "120"], [fake06, REPO], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert res["steps"] == EXPECTED_STEPS
    # The logger name shows WHICH copy ran: the package, not a by-path load.
    assert "crisp_gym.util.epoch_steps:[epochs]" in proc.stderr


def test_launcher_in_an_uninstalled_repo_clone(tmp_path, fake06, no_crisp_gym):
    """train_groot_server.sh's layout: the repo is cloned, never pip-installed."""
    proc, res = _run(ABS_LAUNCHER, ["--epochs", "120"], [no_crisp_gym, fake06], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert res["steps"] == EXPECTED_STEPS
    assert res["save_freq"] == EXPECTED_SAVE
    assert "crisp_gym_epoch_steps:[epochs]" in proc.stderr   # loaded by path


def test_launcher_copied_with_epoch_steps_beside_it(tmp_path, fake06, no_crisp_gym):
    """The GPU-PC layout: two files copied into one folder."""
    pair = tmp_path / "pair"
    pair.mkdir()
    shutil.copy(ABS_LAUNCHER, pair)
    shutil.copy(EPOCH_STEPS, pair)
    proc, res = _run(pair / ABS_LAUNCHER.name, ["--epochs", "120"], [no_crisp_gym, fake06], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert res["steps"] == EXPECTED_STEPS


def test_launcher_copied_alone_refuses_epochs(tmp_path, fake06, no_crisp_gym):
    solo = tmp_path / "solo"
    solo.mkdir()
    shutil.copy(ABS_LAUNCHER, solo)
    proc, res = _run(solo / ABS_LAUNCHER.name, ["--epochs", "120"], [no_crisp_gym, fake06], tmp_path)
    assert proc.returncode != 0
    assert res is None                       # never fell back to --steps
    assert "--epochs needs epoch_steps.py" in proc.stderr


def test_launcher_copied_alone_still_trains_without_epochs(tmp_path, fake06, no_crisp_gym):
    # The regression this fixes: before, a lone copy died on `import crisp_gym`
    # even when --epochs was never passed.
    solo = tmp_path / "solo"
    solo.mkdir()
    shutil.copy(ABS_LAUNCHER, solo)
    proc, res = _run(solo / ABS_LAUNCHER.name, ["--policy.type=diffusion"], [no_crisp_gym, fake06], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert res["steps"] == 100_000
    assert res["argv"] == ["--policy.type=diffusion"]
    assert "epoch_steps.py not found" in proc.stderr


# ── the duplicated helper ─────────────────────────────────────────────────────

def _split(path: Path):
    text = path.read_text()
    start = text.index(BLOCK_START)
    end = text.index(BLOCK_END)
    return text[start:end], text[:start] + text[end:]


def test_epoch_helper_block_is_identical_in_all_launchers():
    blocks = {p.name: _split(p)[0] for p in LAUNCHERS}
    reference = blocks[ABS_LAUNCHER.name]          # the copy tested end to end
    assert "def _setup_epochs" in reference
    for name, block in blocks.items():
        assert block == reference, f"{name}: --epochs helper drifted from {ABS_LAUNCHER.name}"


@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_launchers_import_crisp_gym_only_inside_the_helper(path):
    _, outside = _split(path)
    hard = [
        line for line in outside.splitlines()
        if not line.strip().startswith("#")
        and re.match(r"\s*(from|import)\s+crisp_gym\b", line)
    ]
    assert hard == [], f"{path.name}: hard crisp_gym import outside the helper: {hard}"


# ── the GR00T shell scripts (dry run: composed command only) ──────────────────

def _bash(args, env_extra, cwd):
    env = dict(os.environ)
    env.update({"SKIP_HUB_CHECK": "1"})
    env.update(env_extra)
    return subprocess.run(
        ["bash", *args], capture_output=True, text=True, cwd=cwd, env=env, timeout=120,
    )


def _dataset(tmp_path) -> Path:
    ds = tmp_path / "datasets" / "toy"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text("{}")
    return ds


def _groot(tmp_path, *extra):
    ds = _dataset(tmp_path)
    return _bash(
        [str(REPO / "scripts" / "train_groot.sh"), "--dataset", str(ds),
         "--output", str(tmp_path / "out" / "run"), "--skip-preflight", "--dry-run", *extra],
        {}, tmp_path,
    )


def test_train_groot_default_mode_is_unchanged_without_epochs(tmp_path):
    proc = _groot(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "-m lerobot.scripts.lerobot_train" in proc.stdout
    assert "--epochs" not in proc.stdout


def test_train_groot_default_mode_with_epochs_runs_epoch_steps(tmp_path):
    proc = _groot(tmp_path, "--epochs", "10")
    assert proc.returncode == 0, proc.stderr
    assert str(EPOCH_STEPS) in proc.stdout
    assert "-m lerobot.scripts.lerobot_train" not in proc.stdout
    assert "--epochs 10" in proc.stdout
    # Still passed: with --epochs they set the checkpoint count.
    assert "--steps=100000" in proc.stdout and "--save_freq=20000" in proc.stdout
    # GR00T's own relative-action flags are untouched in this mode.
    assert "--policy.use_relative_actions=true" in proc.stdout


def test_train_groot_se3_with_epochs_passes_it_to_the_relative_launcher(tmp_path):
    proc = _groot(tmp_path, "--se3", "--epochs", "10")
    assert proc.returncode == 0, proc.stderr
    assert "lerobot_relative_pose.py" in proc.stdout
    assert "epoch_steps.py" not in proc.stdout
    assert "--epochs 10" in proc.stdout


@pytest.mark.parametrize("bad", ["0", "abc", "-1", "1e3", "0.0"])
def test_train_groot_rejects_bad_epochs_before_doing_anything(tmp_path, bad):
    proc = _groot(tmp_path, "--epochs", bad)
    assert proc.returncode == 2
    assert "--epochs must be a positive number" in proc.stderr


def _server(tmp_path, env):
    """train_groot_server.sh in its documented layout: a crisp_gym clone beside it."""
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(REPO / "scripts" / "train_groot_server.sh", work)
    (work / "crisp_gym").symlink_to(REPO, target_is_directory=True)
    ds = _dataset(tmp_path)
    base = {"DATASET": str(ds), "SKIP_PREFLIGHT": "1", "DRY_RUN": "1"}
    base.update(env)
    return _bash([str(work / "train_groot_server.sh")], base, tmp_path), work


def test_server_default_mode_is_unchanged_without_epochs(tmp_path):
    proc, _ = _server(tmp_path, {})
    assert proc.returncode == 0, proc.stderr
    assert "-m lerobot.scripts.lerobot_train" in proc.stdout
    assert "--epochs" not in proc.stdout


def test_server_default_mode_with_epochs_runs_epoch_steps(tmp_path):
    proc, work = _server(tmp_path, {"EPOCHS": "10"})
    assert proc.returncode == 0, proc.stderr
    assert str(work / "crisp_gym" / "crisp_gym" / "util" / "epoch_steps.py") in proc.stdout
    assert "--epochs 10" in proc.stdout
    assert "--steps=100000" in proc.stdout and "--save_freq=20000" in proc.stdout


def test_server_se3_with_epochs_passes_it_to_the_relative_launcher(tmp_path):
    proc, _ = _server(tmp_path, {"SE3": "1", "EPOCHS": "10"})
    assert proc.returncode == 0, proc.stderr
    assert "lerobot_relative_pose.py" in proc.stdout
    assert "--epochs 10" in proc.stdout


@pytest.mark.parametrize("bad", ["0", "abc", "-1"])
def test_server_rejects_bad_epochs(tmp_path, bad):
    proc, _ = _server(tmp_path, {"EPOCHS": bad})
    assert proc.returncode == 2
    assert "EPOCHS must be a positive number" in proc.stderr
