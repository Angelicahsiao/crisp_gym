"""Post-process a crisp_gym LeRobot dataset to add observation.state.umilike.

observation.state.umilike = concat(observation.state.cartesian, observation.state.gripper)

This UMI-style flat state vector (end-effector pose + gripper) is useful for training
policies that expect a compact proprioceptive input rather than the full state vector.

Datasets recorded with crisp_gym always store individual sub-keys
(observation.state.cartesian, observation.state.gripper, observation.state.joints, …)
alongside the flat observation.state, so this script reads those columns directly.

Usage
-----
    python add_umilike_state_observation.py \\
        --source-dataset <repo_id> \\
        [--output-dataset <repo_id>] \\
        [--output-dir <path>] \\
        [--push-to-hub]

The output dataset preserves every original feature unchanged and appends a new
``observation.state.umilike`` column.

Dependencies
------------
    pip install lerobot einops rich
    (no crisp_gym install required)
"""

import argparse
from inspect import signature
from pathlib import Path

import einops
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rich import print
from rich.progress import Progress

# ---------------------------------------------------------------------------
# Compatibility shim: lerobot >= v3.0 passes ``task`` as an explicit argument
# to add_frame; older versions expect it as a key inside the frame dict.
# ---------------------------------------------------------------------------
_ADD_FRAME_HAS_TASK = "task" in signature(LeRobotDataset.add_frame).parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_state_value(frame: dict, key: str) -> np.ndarray:
    """Return a state feature as a flat float32 array (handles scalar tensors)."""
    arr = np.asarray(frame[key], dtype=np.float32)
    return arr.reshape(1) if arr.ndim == 0 else arr.ravel()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add observation.state.umilike to a crisp_gym LeRobot dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-dataset",
        required=True,
        help="repo_id of the source LeRobot dataset (local or HF Hub).",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help=(
            "repo_id for the output dataset. "
            "Defaults to <source>_umilike (or replaces the last _vN suffix)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Local directory where the output dataset will be stored. "
            "Defaults to HF_LEROBOT_HOME / <output-dataset> when not set."
        ),
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the output dataset to Hugging Face Hub after conversion.",
    )
    args = parser.parse_args()

    source_id: str = args.source_dataset
    if args.output_dataset:
        output_id: str = args.output_dataset
    elif "_v" in source_id:
        output_id = source_id.rsplit("_v", 1)[0] + "_umilike"
    else:
        output_id = source_id + "_umilike"

    # ── Load source dataset ──────────────────────────────────────────────────
    print(f"\nLoading source dataset: [bold]{source_id}[/bold]")
    dataset = LeRobotDataset(repo_id=source_id)

    print(f"  Total episodes : {dataset.meta.total_episodes}")
    print(f"  FPS            : {dataset.fps}")
    print(f"  Features       : {list(dataset.features.keys())}")

    # ── Validate required sub-keys are present ───────────────────────────────
    for required in ("observation.state.cartesian", "observation.state.gripper"):
        if required not in dataset.features:
            raise ValueError(
                f"Dataset is missing '{required}'. "
                "Ensure it was recorded with crisp_gym, which stores individual "
                "observation.state sub-keys alongside the flat observation.state."
            )

    cart_info = dataset.features["observation.state.cartesian"]
    grip_info = dataset.features["observation.state.gripper"]

    cart_names: list[str] = list(cart_info["names"])
    grip_names: list[str] = list(grip_info["names"])
    umilike_names = cart_names + grip_names
    umilike_dim = int(cart_info["shape"][0]) + int(grip_info["shape"][0])

    print(f"\n[bold]observation.state.umilike[/bold] → {umilike_dim}D")
    print(f"  names : {umilike_names}")

    # ── Build output feature spec ────────────────────────────────────────────
    new_features = dict(dataset.features)
    new_features["observation.state.umilike"] = {
        "dtype": "float32",
        "shape": (umilike_dim,),
        "names": umilike_names,
    }

    dir_msg = str(args.output_dir) if args.output_dir else "HF_LEROBOT_HOME (default)"
    print(f"\nOutput dataset : [bold]{output_id}[/bold]")
    print(f"Output dir     : {dir_msg}")

    # ── Create output dataset ────────────────────────────────────────────────
    new_dataset = LeRobotDataset.create(
        repo_id=output_id,
        features=new_features,
        fps=dataset.fps,
        root=args.output_dir,
    )

    # ── Iterate and copy frames ──────────────────────────────────────────────
    with Progress() as progress:
        task_bar = progress.add_task(
            "Processing episodes…", total=dataset.meta.total_episodes
        )
        current_episode = 0

        for frame in dataset:
            episode_idx = int(frame["episode_index"])

            if episode_idx > current_episode:
                new_dataset.save_episode()
                progress.update(task_bar, advance=1)
                current_episode = episode_idx

            new_frame: dict = {}

            for key in dataset.features:
                if key == "action":
                    new_frame[key] = np.asarray(frame[key], dtype=np.float32)
                elif key.startswith("observation.images"):
                    # LeRobot stores images as (C, H, W); add_frame expects (H, W, C)
                    new_frame[key] = einops.rearrange(frame[key], "c h w -> h w c")
                elif key.startswith("observation.state"):
                    new_frame[key] = _copy_state_value(frame, key)
                else:
                    new_frame[key] = frame[key]

            cart = _copy_state_value(frame, "observation.state.cartesian")
            grip = _copy_state_value(frame, "observation.state.gripper")
            new_frame["observation.state.umilike"] = np.concatenate([cart, grip])

            task_label: str = frame["task"] if "task" in frame else ""

            if _ADD_FRAME_HAS_TASK:
                new_dataset.add_frame(new_frame, task=task_label)
            else:
                new_frame["task"] = task_label
                new_dataset.add_frame(new_frame)

        new_dataset.save_episode()
        progress.update(task_bar, advance=1)

    print(f"\n[green]Done.[/green] Output dataset: [bold]{output_id}[/bold]")
    print(f"Stored at: {new_dataset.root}")

    if args.push_to_hub:
        print("Pushing to Hugging Face Hub…")
        new_dataset.push_to_hub()
        print(f"[green]Pushed:[/green] {output_id}")
    else:
        print("[yellow]Tip:[/yellow] add --push-to-hub to upload the dataset.")


if __name__ == "__main__":
    main()
