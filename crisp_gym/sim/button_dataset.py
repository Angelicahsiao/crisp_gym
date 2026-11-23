r"""Simulated press-button demonstrations, previewed or written as a LeRobot dataset.

What it is: the command line of crisp_gym/sim/button_task.py. `--preview` runs
one demonstration and prints what it shows; with `--root` it writes a dataset
whose columns, names and `meta/record_config.json` follow a real Franka
recording (umi_robot_full_record.yaml), so the audit, `swap_action_offline.py`
and the training launchers run on it unchanged. One dataset holds one spring
stiffness; write one per stiffness.

Runs on any machine with mujoco; writing also needs lerobot (0.4.4 or newer),
so on the GPU PC. Imports numpy, mujoco (through button_task),
crisp_gym.record.record_config (numpy and yaml only) and, only when writing,
lerobot. Tests: tests/test_sim_button_dataset.py.

    python crisp_gym/sim/button_dataset.py --preview --stiffness 400
    python crisp_gym/sim/button_dataset.py --root /workspace/datasets/sim_button_k400/lerobot \\
        --stiffness 400 --episodes 200 --seed 0

What differs from robot data, on purpose:
  * No camera. The button's position, in the TCP frame, is in
    `observation.state` after the gripper (the agreed place for extra inputs,
    ROADMAP D-10) and again as `observation.environment_state`, because
    lerobot's diffusion refuses to train with neither an image nor an
    environment state.
  * No joints and no joint torques (`extra.joints`, `extra.ext_torque`): the
    audit's stiffness steps need a simulated FR3 and skip themselves here.
  * Columns the robot cannot have: `extra.contact_force` (the true force on the
    fingertips, world frame), `extra.button_depth`, `extra.button_pressed`.
  * `meta/sim_config.json` holds the stiffness, the seed and every episode's
    parameters, so a dataset can be regenerated exactly.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

if __package__ in (None, ""):  # run as a script: make `crisp_gym` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crisp_gym.record.record_config import ActionConfig, ObsFieldConfig, RecordConfig
from crisp_gym.sim import button_task as bt

ROBOT_TYPE = "sim_button_tcp"
CONTRACT_NAME = "sim_button_task"
# extra observations: key -> (source label, shape); the labels only document
# where a column comes from, they are not record sources
_SIM_EXTRAS = {
    "extra.tactile.left": ("sim.taxels_left", [72]),
    "extra.tactile.right": ("sim.taxels_right", [72]),
    "extra.contact_force": ("sim.contact_force", [3]),
    "extra.button_depth": ("sim.button_depth", [1]),
    "extra.button_pressed": ("sim.button_pressed", [1]),
}


def record_contract(cfg: bt.ButtonTaskConfig) -> RecordConfig:
    """The data contract of a simulated dataset, in the recorder's own terms."""
    taxels = cfg.nx * cfg.ny
    pose = {"representation": "rotation_6d"}
    observations = [
        ObsFieldConfig("observation.state.cartesian", "robot.tcp_pose", dict(pose)),
        ObsFieldConfig("observation.state.gripper", "gripper.width_normalized"),
        ObsFieldConfig("observation.state.button_in_tcp", "sim.button_in_tcp", shape=[3]),
        ObsFieldConfig("extra.target_cartesian", "robot.target_pose", dict(pose), None, False),
    ]
    for key, (source, shape) in _SIM_EXTRAS.items():
        shape = [taxels] if key.startswith("extra.tactile") else shape
        observations.append(ObsFieldConfig(key, source, shape=shape, include_in_state=False))
    action = ActionConfig(definition="next_tcp_pose", lookahead=1, include_gripper=True)
    return RecordConfig(
        observations=observations,
        action=action,
        rate_hz=bt.RATE_HZ,
        name=CONTRACT_NAME,
        robot_type=ROBOT_TYPE,
    )


def dataset_features(cfg: bt.ButtonTaskConfig) -> dict[str, dict]:
    """The LeRobot feature schema: the contract's, plus the environment state for diffusion."""
    features = record_contract(cfg).to_features(use_video=False)
    features["observation.environment_state"] = {
        "dtype": "float32",
        "shape": (3,),
        "names": [f"button_in_tcp_{i}" for i in range(3)],
    }
    return features


def episode_rng(seed: int, index: int) -> np.random.Generator:
    """The random stream of one episode: the same (seed, index) gives the same demonstration."""
    return np.random.default_rng([seed, index])


def sim_config(cfg: bt.ButtonTaskConfig, seed: int, kept: list, skipped: list) -> dict:
    """What meta/sim_config.json records about how the dataset was made."""
    return {
        "simulator": "crisp_gym/sim/button_task.py",
        "stiffness": cfg.stiffness,
        "config": asdict(cfg),
        "seed": seed,
        "rate_hz": bt.RATE_HZ,
        "physics_steps_per_frame": bt.STEPS_PER_FRAME,
        "contact_solref": bt.CONTACT_SOLREF,
        "contact_solimp": bt.CONTACT_SOLIMP,
        "task": bt.TASK,
        "episodes": kept,
        "skipped_without_click": skipped,
    }


def write_dataset(
    root: Path,
    repo_id: str,
    cfg: bt.ButtonTaskConfig,
    n_episodes: int,
    seed: int,
    log: Callable[[str], None] = print,
) -> dict:
    """Simulate `n_episodes` demonstrations that click the button and write them under `root`."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        raise FileExistsError(f"{root} exists; choose a new --root or delete it first")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(bt.RATE_HZ),
        features=dataset_features(cfg),
        root=root,
        robot_type=ROBOT_TYPE,
        use_videos=False,
    )
    # lerobot >= 0.5 takes the task as an argument, older versions as a frame key
    task_argument = "task" in inspect.signature(dataset.add_frame).parameters
    kept, skipped, index = [], [], 0
    while len(kept) < n_episodes:
        if index >= 2 * n_episodes + 10:
            raise RuntimeError(f"{len(skipped)} demonstrations did not click the button; stopping")
        params = bt.sample_params(episode_rng(seed, index))
        episode = bt.run_episode(cfg, params)
        record = {"index": index, **asdict(params)}
        index += 1
        if not episode.clicked:
            skipped.append(record)
            continue
        frames = len(episode.columns["action"])
        for t in range(frames):
            frame = {key: column[t] for key, column in episode.columns.items()}
            if task_argument:
                dataset.add_frame(frame, task=bt.TASK)
            else:
                dataset.add_frame({**frame, "task": bt.TASK})
        dataset.save_episode()
        kept.append({**record, "frames": frames})
        if len(kept) % 10 == 0 or len(kept) == n_episodes:
            log(f"{len(kept)}/{n_episodes} episodes written")
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    meta = root / "meta"
    (meta / "record_config.json").write_text(
        json.dumps(record_contract(cfg).to_metadata(), indent=4)
    )
    (meta / "sim_config.json").write_text(
        json.dumps(sim_config(cfg, seed, kept, skipped), indent=2)
    )
    return {
        "episodes": len(kept),
        "frames": sum(e["frames"] for e in kept),
        "skipped": len(skipped),
    }


def summary(cfg: bt.ButtonTaskConfig, episode: bt.Episode) -> str:
    """A few numbers that show what one demonstration did."""
    c, p = episode.columns, episode.params
    gap_mm = 1e3 * (c["observation.state.cartesian"][:, 2] - c["extra.target_cartesian"][:, 2])
    force = c["extra.contact_force"][:, 2]
    taxels = c["extra.tactile.left"].sum(axis=1) + c["extra.tactile.right"].sum(axis=1)
    peak = int(np.argmax(force))
    sliding = [i for i, phase in enumerate(episode.phases[:-1]) if phase == "slide"]
    lines = [
        f"spring {cfg.stiffness:g} N/m | button: {p.activation_force:.1f} N to click, "
        f"{1e3 * p.travel:.1f} mm travel | {len(gap_mm)} frames, {len(gap_mm) / bt.RATE_HZ:.1f} s",
        f"clicked: {'yes' if episode.clicked else 'NO'}",
        f"command below measured, largest: {gap_mm.max():.1f} mm "
        f"(clicking needs {1e3 * p.activation_force / cfg.stiffness:.1f} mm at this spring)",
        f"largest contact force {force[peak]:.2f} N; summed taxels in that frame {taxels[peak]:.2f} N",
    ]
    if sliding:
        lines.append(
            f"sliding: commanded push {p.slide_force:.2f} N, measured "
            f"{np.median(force[sliding]):.2f} N (median over {len(sliding)} frames)"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--preview", action="store_true", help="simulate one episode and print it")
    what.add_argument("--root", type=Path, help="dataset directory to create (must not exist)")
    p.add_argument("--repo-id", help="dataset name (default: from --root)")
    p.add_argument("--stiffness", type=float, default=400.0, help="spring, N/m (default 400)")
    p.add_argument("--episodes", type=int, default=200, help="episodes to write (default 200)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    return p


def main(argv: list[str] | None = None) -> int:
    """Preview one demonstration, or write a dataset."""
    args = build_parser().parse_args(argv)
    cfg = bt.ButtonTaskConfig(stiffness=args.stiffness)
    if args.preview:
        episode = bt.run_episode(cfg, bt.sample_params(episode_rng(args.seed, 0)))
        print(summary(cfg, episode))
        return 0 if episode.clicked else 1
    root = args.root
    repo_id = args.repo_id or (root.parent.name if root.name == "lerobot" else root.name)
    result = write_dataset(root, repo_id, cfg, args.episodes, args.seed)
    print(
        f"Done: {result['episodes']} episodes, {result['frames']} frames, "
        f"{result['skipped']} skipped without a click, at {args.stiffness:g} N/m -> {root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
