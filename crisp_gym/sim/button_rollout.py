r"""Run trained checkpoints in the press-button simulation and count what they achieve.

What it is: the closed-loop test of ROADMAP F7. Each checkpoint plays the same
new button setups (seed 2; the training data used seed 0 and the open-loop
check seed 1) for up to `--time-limit` seconds. The model side IS the deploy
code: `relative_lerobot_policy.inference_worker` loads the checkpoint, its
normalisation and its pose_repr.json, and converts every observation window
exactly as on the robot. Around it this file plays the robot: every 1/15 s it
builds the frame with the deploy `build_obs_frame`, asks for a chunk when the
last one is used up, composes each action against the pose at observation time
(`compose_relative_pose`), cuts it with the deploy step limit
(`limit_target_step`, 30 mm), and sends it to the simulated spring, moving the
command over the frame rather than jumping it (see spring_pad.py). The
gripper and the orientation stay fixed, as in the training data.

Reported per checkpoint, and for the scripted demonstrator on the same setups
as a reference: how often the button clicks; how often it clicks AND the
fingertips then slide at least 10 mm on the panel with at least 1 N; time to
the click; how hard it pushes after the click (the largest force in the next
second over the force that button needs: about 1 = just enough); the peak
force; and how many rollouts hit a simulation artefact (fingertips sunk
into the panel or the button pushed through its stop), which are listed, not
hidden.

Runs on the GPU PC (or any machine with lerobot, torch and mujoco; CPU works,
slower). Imports numpy and mujoco; crisp_gym's deploy helpers, scipy and
lerobot only when it runs. Tests: tests/test_sim_button_rollout.py.

    python crisp_gym/sim/button_rollout.py \
        --checkpoint outputs/train/sim_dit_k400_measured/checkpoints/<step>/pretrained_model \
        --checkpoint outputs/train/sim_dit_k400_command/checkpoints/<step>/pretrained_model \
        --out outputs/sim_rollouts_k400.json

Only observation.state, the task text and images reach the model: the deploy
path drops `observation.environment_state` (numpy_obs_to_torch), so this
serves `multi_task_dit` and GR00T, not diffusion trained on that input.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import deque
from dataclasses import asdict, dataclass
from multiprocessing import Pipe
from pathlib import Path
from typing import Callable

import numpy as np

if __package__ in (None, ""):  # run as a script: make `crisp_gym` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crisp_gym.sim import button_task as bt

SETUP_SEED = 2  # training data: seed 0; held-out open-loop data: seed 1
# A slide counts from this far along the panel. The scripted demonstrations
# slide 16-37 mm (50 setups at 200, 400 and 800 N/m): at low stiffness the
# fingertips lag up to ~20 mm behind a command that stops and lifts.
SLIDE_MIN_M = 0.01
SLIDE_MIN_N = 1.0  # ... pushing at least this hard
ON_PANEL_M = 0.0003  # fingertip faces this close to the panel's top: on it, not on the cap
# How hard a model pushes is read in the second AFTER the click. AT the click
# the force always equals what the button needs (that is what makes it click),
# and the recorded force is a frame average, which a fast press drags down.
AFTER_CLICK_FRAMES = int(bt.RATE_HZ)
# the recording contract's gripper scale (umi_robot_full_record.yaml); the
# simulated gripper reads 0, which is 0 in either unit
REFERENCE_WIDTH, DEVICE_MAX_WIDTH = 0.09, 0.085


@dataclass
class Score:
    """What one rollout achieved."""

    clicked: bool
    slid: bool  # clicked, then slid on the panel
    time_to_click_s: float | None
    after_click_ratio: float | None  # largest force in the second after the click / needed
    activation_force_n: float
    peak_force_n: float
    slide_m: float
    artefact: bool


def score(record: dict[str, np.ndarray], params: bt.EpisodeParams) -> Score:
    """Score per-frame arrays: tcp [N, 3], force [N, 3] (on the fingertips), depth [N]."""
    tcp, force, depth = record["tcp"], record["force"], record["depth"]
    click_depth = bt.button_spring(params)["click_depth"]
    pressed = np.flatnonzero(depth >= click_depth)
    click = int(pressed[0]) if len(pressed) else None
    slide = 0.0
    if click is not None:
        on_panel = (depth < 1e-4) & (tcp[:, 2] < ON_PANEL_M) & (force[:, 2] >= SLIDE_MIN_N)
        both = on_panel[click + 1 :] & on_panel[click:-1]
        steps = np.linalg.norm(np.diff(tcp[click:, :2], axis=0), axis=1)
        slide = float(steps[both].sum())
    artefact = bool((tcp[:, 2] < -0.002).any() or (depth > params.travel + 0.001).any())
    return Score(
        clicked=click is not None,
        slid=click is not None and slide >= SLIDE_MIN_M,
        time_to_click_s=None if click is None else click / bt.RATE_HZ,
        after_click_ratio=None
        if click is None
        else float(force[click : click + AFTER_CLICK_FRAMES, 2].max() / params.activation_force),
        activation_force_n=params.activation_force,
        peak_force_n=float(np.linalg.norm(force, axis=1).max()) if len(force) else 0.0,
        slide_m=slide,
        artefact=artefact,
    )


def demonstrator_record(cfg: bt.ButtonTaskConfig, params: bt.EpisodeParams) -> dict:
    """The scripted demonstrator on one setup, as the arrays `score` reads."""
    c = bt.run_episode(cfg, params).columns
    return {
        "tcp": c["observation.state.cartesian"][:, :3].astype(np.float64),
        "force": c["extra.contact_force"].astype(np.float64),
        "depth": c["extra.button_depth"][:, 0].astype(np.float64),
    }


class PolicyClient:
    """The deploy inference worker in a thread, spoken to over its own pipe protocol.

    `worker` defaults to `relative_lerobot_policy.inference_worker`; tests pass
    a fake with the same protocol: send a meta dict (or ("error", msg)) first,
    then answer ("reset", None) with nothing and ("infer", window) with a chunk.
    """

    def __init__(self, checkpoint: Path, worker: Callable | None = None) -> None:
        """Start the worker on `checkpoint` and wait for its meta (n_obs_steps, ...)."""
        if worker is None:
            from crisp_gym.policy.relative_lerobot_policy import inference_worker as worker
        self._conn, child = Pipe()
        self._thread = threading.Thread(
            target=worker, args=(child, str(checkpoint), {}, "auto"), daemon=True
        )
        self._thread.start()
        meta = self._receive(timeout_s=1800)  # first use downloads the text encoder
        if not isinstance(meta, dict):
            raise RuntimeError(f"the policy worker did not start: {meta}")
        self.meta = meta
        self.n_obs_steps = int(meta.get("n_obs_steps", 1))
        self.n_action_steps = int(meta.get("n_action_steps", 1))

    def _receive(self, timeout_s: float) -> object:
        if not self._conn.poll(timeout_s):
            raise TimeoutError(f"no answer from the policy worker in {timeout_s:.0f} s")
        return self._conn.recv()

    def reset(self) -> None:
        """Start a new episode: the worker forgets the episode-start pose."""
        self._conn.send(("reset", None))

    def infer(self, window: list[dict]) -> np.ndarray:
        """One action chunk for an observation window (oldest frame first)."""
        self._conn.send(("infer", window))
        chunk = self._receive(timeout_s=600)
        if isinstance(chunk, tuple) and chunk and chunk[0] == "error":
            raise RuntimeError(f"inference failed: {chunk[1]}")
        return np.asarray(chunk, dtype=np.float64)

    def close(self) -> None:
        """Stop the worker."""
        self._conn.send(None)
        self._thread.join(timeout=30)


def observation(sim: bt._Sim, cfg: bt.ButtonTaskConfig) -> dict:
    """The raw observation the deploy env would give: the dataset's observation.state.* keys."""
    tcp = sim.tcp()
    return {
        "observation.state.cartesian": np.concatenate([tcp, bt.TCP_ROT6D]),
        "observation.state.gripper": np.array([cfg.gripper]),
        "observation.state.button_in_tcp": bt.TCP_ROT.T @ (sim.cap_top() - tcp),
        "task": bt.TASK,
    }


def _step_frame(sim: bt._Sim, start: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Move the command from `start` to `target` over one frame; return the mean contact force."""
    total = np.zeros(3)
    for i in range(1, bt.STEPS_PER_FRAME + 1):
        sim.step(start + (target - start) * (i / bt.STEPS_PER_FRAME))
        total += sim.sensed()[-3:]
    return total / bt.STEPS_PER_FRAME


def _limit(previous: np.ndarray, target: np.ndarray, max_step_m: float | None) -> np.ndarray:
    """The deploy step limit on the commanded position (the orientation never changes here)."""
    from scipy.spatial.transform import Rotation

    from crisp_gym.util.safety import limit_target_step

    rot = Rotation.from_matrix(bt.TCP_ROT)
    position, _, _, _ = limit_target_step(previous, rot, target, rot, max_step_m=max_step_m)
    return np.asarray(position, dtype=np.float64)


def rollout(
    cfg: bt.ButtonTaskConfig,
    params: bt.EpisodeParams,
    client: PolicyClient,
    time_limit_s: float,
    max_step_m: float | None,
) -> dict:
    """One closed-loop rollout; returns the per-frame arrays `score` reads."""
    from crisp_gym.policy.relative_lerobot_policy import build_obs_frame, compose_relative_pose
    from crisp_gym.util.rot6d import pose9d_to_mat

    sim = bt._Sim(cfg, params)
    client.reset()
    history: deque = deque(maxlen=client.n_obs_steps)
    command = np.array(params.start, dtype=np.float64)
    force = sim.sensed()[-3:]
    chunk, index, base = None, 0, None
    record: dict[str, list] = {"tcp": [], "force": [], "depth": []}
    for _ in range(int(round(time_limit_s * bt.RATE_HZ))):
        raw = observation(sim, cfg)
        history.append(build_obs_frame(raw, REFERENCE_WIDTH, DEVICE_MAX_WIDTH))
        record["tcp"].append(sim.tcp())
        record["force"].append(force)
        record["depth"].append(sim.depth())
        if chunk is None or index >= min(client.n_action_steps, len(chunk)):
            # the base is the pose of THIS observation (HANDOFF §1.1)
            base = pose9d_to_mat(raw["observation.state.cartesian"])
            chunk, index = client.infer(list(history)), 0
        target = compose_relative_pose(chunk[index], base)[:3, 3]
        index += 1
        target = _limit(command, target, max_step_m)
        force = _step_frame(sim, command, target)
        command = target
    return {key: np.asarray(value, dtype=np.float64) for key, value in record.items()}


def stiffness_of(checkpoint: Path) -> float | None:
    """The spring stiffness of the simulated dataset a checkpoint was trained on, if recorded."""
    try:
        train = json.loads((checkpoint / "train_config.json").read_text())
        root = Path(train["dataset"]["root"])
        return float(json.loads((root / "meta" / "sim_config.json").read_text())["stiffness"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def label_of(checkpoint: Path) -> str:
    """`<run directory>@<step>` for .../<run>/checkpoints/<step>/pretrained_model."""
    parts = checkpoint.resolve().parts
    if "checkpoints" in parts:
        i = len(parts) - 1 - parts[::-1].index("checkpoints")
        if i + 1 < len(parts):
            return f"{parts[i - 1]}@{parts[i + 1]}"
    return checkpoint.name


def summary_row(name: str, scores: list[Score]) -> str:
    """One line of the results table."""
    n = len(scores)
    clicked = [s for s in scores if s.clicked]
    ratio = [s.after_click_ratio for s in clicked]
    times = [s.time_to_click_s for s in clicked]
    median = lambda v: f"{np.median(v):.2f}" if v else "  -  "  # noqa: E731
    return (
        f"{name:<34} {n:>4}  {100 * len(clicked) / n:5.0f}%  "
        f"{100 * sum(s.slid for s in scores) / n:5.0f}%  {median(times):>8} s  "
        f"{median(ratio):>9}  {max(s.peak_force_n for s in scores):7.1f} N  "
        f"{sum(s.artefact for s in scores):>4}"
    )


TABLE_HEADER = (
    f"{'':<34} {'runs':>4}  {'click':>6}  {'+slide':>6}  {'to click':>10}  "
    f"{'push/need':>9}  {'peak F':>9}  {'artef':>5}\n"
    f"{'':<34} {'':>4}  {'':>6}  {'':>6}  {'(median)':>10}  {'(median)':>9}  {'(max)':>9}"
)


def evaluate(args: argparse.Namespace, log: Callable[[str], None] = print) -> dict:
    """Score the demonstrator and every checkpoint on the same setups."""
    stiffness = args.stiffness
    for checkpoint in args.checkpoint:
        trained = stiffness_of(checkpoint)
        if stiffness is None:
            stiffness = trained
        elif trained is not None and trained != stiffness:
            log(
                f"WARNING: {label_of(checkpoint)} was trained at {trained:g} N/m, not {stiffness:g}"
            )
    if stiffness is None:
        raise SystemExit("no stiffness recorded with the checkpoint: pass --stiffness")
    cfg = bt.ButtonTaskConfig(stiffness=stiffness)
    setups = [bt.sample_params(np.random.default_rng([args.seed, i])) for i in range(args.rollouts)]
    results: dict[str, list[Score]] = {}
    if not args.no_reference:
        results["scripted demonstrator"] = [score(demonstrator_record(cfg, p), p) for p in setups]
    max_step = None if args.max_step_mm <= 0 else args.max_step_mm / 1e3
    for checkpoint in args.checkpoint:
        name = label_of(checkpoint)
        client = PolicyClient(checkpoint)
        log(
            f"{name}: arm action = {client.meta.get('action_source')}, state = {client.meta.get('state_input')}"
        )
        scores = []
        try:
            for i, params in enumerate(setups):
                scores.append(
                    score(rollout(cfg, params, client, args.time_limit, max_step), params)
                )
                if (i + 1) % 10 == 0 or i + 1 == len(setups):
                    log(f"  {name}: {i + 1}/{len(setups)} rollouts")
        finally:
            client.close()
        results[name] = scores
    log(f"\n{len(setups)} setups (seed {args.seed}), {stiffness:g} N/m, {args.time_limit:g} s each")
    log(TABLE_HEADER)
    for name, scores in results.items():
        log(summary_row(name, scores))
    return {
        "stiffness": stiffness,
        "seed": args.seed,
        "time_limit_s": args.time_limit,
        "max_step_mm": args.max_step_mm,
        "results": {name: [asdict(s) for s in scores] for name, scores in results.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="a pretrained_model directory; repeat to compare several",
    )
    p.add_argument(
        "--stiffness",
        type=float,
        help="spring, N/m (default: the one the checkpoint's dataset was made with)",
    )
    p.add_argument("--rollouts", type=int, default=50, help="setups per checkpoint (default 50)")
    p.add_argument("--seed", type=int, default=SETUP_SEED, help="setup seed (default 2)")
    p.add_argument(
        "--time-limit", type=float, default=20.0, help="seconds per rollout (default 20)"
    )
    p.add_argument(
        "--max-step-mm",
        type=float,
        default=30.0,
        help="deploy step limit per command, mm; 0 = none (default 30)",
    )
    p.add_argument("--no-reference", action="store_true", help="skip the scripted demonstrator row")
    p.add_argument("--out", type=Path, help="write every rollout's numbers as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    """Roll out every checkpoint and print the comparison."""
    args = build_parser().parse_args(argv)
    report = evaluate(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
