"""Post-process a crisp_gym LeRobot dataset to add observation.state.umilike.

observation.state.umilike = concat(observation.state.cartesian, observation.state.gripper)

This UMI-style flat state vector (end-effector pose + gripper) is useful for training
policies that expect a compact proprioceptive input rather than the full state vector.

The script works on lerobot 0.4.4 datasets recorded with crisp_gym. It supports both:
- Datasets that have individual observation.state.cartesian / observation.state.gripper
  columns (the normal crisp_gym recording path).
- Older datasets that only expose a flat observation.state vector; in that case the
  cartesian and gripper slices are identified via the feature ``names`` metadata.

Usage
-----
    python add_umilike_state_observation.py \\
        --source-dataset <repo_id> \\
        [--output-dataset <repo_id>] \\
        [--push-to-hub]

The output dataset preserves every original feature unchanged and appends a new
``observation.state.umilike`` column.
"""

import argparse
from inspect import signature

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.utils import einops
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

def _cartesian_meta(features: dict) -> tuple[list[str], int]:
    """Return (names, dim) for the cartesian observation."""
    if "observation.state.cartesian" in features:
        info = features["observation.state.cartesian"]
        return list(info["names"]), int(info["shape"][0])
    # Fall back: infer from the flat observation.state names
    state_names = features["observation.state"]["names"]
    cart_names = [n for n in state_names if n in {"x", "y", "z", "roll", "pitch", "yaw"}]
    return cart_names, len(cart_names)


def _gripper_meta(features: dict) -> tuple[list[str], int]:
    """Return (names, dim) for the gripper observation."""
    if "observation.state.gripper" in features:
        info = features["observation.state.gripper"]
        return list(info["names"]), int(info["shape"][0])
    return ["gripper"], 1


def _flat_indices(features: dict) -> tuple[list[int], list[int]]:
    """Return the cartesian and gripper index positions within the flat observation.state."""
    state_names = features["observation.state"]["names"]
    cart_set = {"x", "y", "z", "roll", "pitch", "yaw"}
    cart_idx = [i for i, n in enumerate(state_names) if n in cart_set]
    grip_idx = [i for i, n in enumerate(state_names) if n == "gripper"]
    return cart_idx, grip_idx


def _build_umilike(
    frame: dict,
    features: dict,
    flat_cart_idx: list[int],
    flat_grip_idx: list[int],
) -> np.ndarray:
    """Compute observation.state.umilike for a single frame."""
    if "observation.state.cartesian" in frame and "observation.state.gripper" in frame:
        cart = np.asarray(frame["observation.state.cartesian"], dtype=np.float32).ravel()
        grip = np.asarray(frame["observation.state.gripper"], dtype=np.float32).ravel()
        return np.concatenate([cart, grip])

    # Fallback: slice from the flat state vector
    state = np.asarray(frame["observation.state"], dtype=np.float32).ravel()
    indices = flat_cart_idx + flat_grip_idx
    return state[indices].astype(np.float32)


def _copy_frame_value(frame: dict, key: str) -> np.ndarray:
    """Copy a scalar-or-array state value from a frame dict to a plain numpy array."""
    val = frame[key]
    arr = np.asarray(val, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.ravel()


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

    if "observation.state" not in dataset.features:
        raise ValueError("Dataset does not contain 'observation.state'.")

    # ── Resolve cartesian / gripper metadata ────────────────────────────────
    cart_names, cart_dim = _cartesian_meta(dataset.features)
    grip_names, grip_dim = _gripper_meta(dataset.features)
    flat_cart_idx, flat_grip_idx = _flat_indices(dataset.features)

    if not flat_cart_idx and "observation.state.cartesian" not in dataset.features:
        raise ValueError(
            "Cannot locate cartesian dimensions (x, y, z, roll, pitch, yaw) in the "
            "dataset. Make sure the dataset was recorded with crisp_gym and that "
            "observation.state.cartesian or named state entries exist."
        )
    if not flat_grip_idx and "observation.state.gripper" not in dataset.features:
        raise ValueError(
            "Cannot locate 'gripper' in the dataset. Make sure observation.state.gripper "
            "or a named 'gripper' entry in observation.state exists."
        )

    umilike_dim = cart_dim + grip_dim
    umilike_names = cart_names + grip_names

    print(f"\n[bold]observation.state.umilike[/bold] → {umilike_dim}D")
    print(f"  names : {umilike_names}")

    # ── Build output feature spec ────────────────────────────────────────────
    new_features = dict(dataset.features)
    new_features["observation.state.umilike"] = {
        "dtype": "float32",
        "shape": (umilike_dim,),
        "names": umilike_names,
    }

    print(f"\nOutput dataset: [bold]{output_id}[/bold]")

    # ── Create output dataset ────────────────────────────────────────────────
    new_dataset = LeRobotDataset.create(
        repo_id=output_id,
        features=new_features,
        fps=dataset.fps,
    )

    # ── Iterate and copy frames ──────────────────────────────────────────────
    with Progress() as progress:
        task_bar = progress.add_task(
            "Processing episodes…", total=dataset.meta.total_episodes
        )
        current_episode = 0

        for frame in dataset:
            episode_idx = int(frame["episode_index"])

            # Episode boundary: save the completed episode
            if episode_idx > current_episode:
                new_dataset.save_episode()
                progress.update(task_bar, advance=1)
                current_episode = episode_idx

            new_frame: dict = {}

            # Copy every original feature
            for key in dataset.features:
                if key == "action":
                    new_frame[key] = np.asarray(frame[key], dtype=np.float32)
                elif key.startswith("observation.images"):
                    # LeRobot stores images as (C, H, W); add_frame expects (H, W, C)
                    new_frame[key] = einops.rearrange(frame[key], "c h w -> h w c")
                elif key.startswith("observation.state"):
                    new_frame[key] = _copy_frame_value(frame, key)
                else:
                    new_frame[key] = frame[key]

            # Add the new umilike feature
            new_frame["observation.state.umilike"] = _build_umilike(
                frame, dataset.features, flat_cart_idx, flat_grip_idx
            )

            task_label: str = ""
            if isinstance(frame, dict) and "task" in frame:
                task_label = frame["task"]

            if _ADD_FRAME_HAS_TASK:
                new_dataset.add_frame(new_frame, task=task_label)
            else:
                new_frame["task"] = task_label
                new_dataset.add_frame(new_frame)

        # Save the last episode
        new_dataset.save_episode()
        progress.update(task_bar, advance=1)

    print(f"\n[green]Done.[/green] Output dataset: [bold]{output_id}[/bold]")

    if args.push_to_hub:
        print("Pushing to Hugging Face Hub…")
        new_dataset.push_to_hub()
        print(f"[green]Pushed:[/green] {output_id}")
    else:
        print("[yellow]Tip:[/yellow] add --push-to-hub to upload the dataset.")


if __name__ == "__main__":
    main()
