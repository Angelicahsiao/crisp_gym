"""Post-process a crisp_gym LeRobot dataset to add observation.state.umilike.

observation.state.umilike = concat(observation.state.cartesian, observation.state.gripper)

This UMI-style flat state vector (end-effector pose + gripper) is useful for training
policies that expect a compact proprioceptive input rather than the full state vector.

Datasets recorded with crisp_gym always store individual sub-keys
(observation.state.cartesian, observation.state.gripper, observation.state.joints, …)
alongside the flat observation.state, so this script reads those columns directly.

The script works by copying the source dataset directory and patching only the parquet
data files — video files are copied as-is without any re-encoding.

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
    pip install lerobot pandas einops rich
    (no crisp_gym install required)
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rich import print
from rich.progress import Progress

try:
    from lerobot.utils.constants import HF_LEROBOT_HOME
except ImportError:
    from lerobot.constants import HF_LEROBOT_HOME


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
    source_root: Path = dataset.root

    print(f"  Total episodes : {dataset.meta.total_episodes}")
    print(f"  FPS            : {dataset.fps}")
    print(f"  Features       : {list(dataset.features.keys())}")

    # ── Validate required sub-keys ───────────────────────────────────────────
    for required in ("observation.state.cartesian", "observation.state.gripper"):
        if required not in dataset.features:
            raise ValueError(
                f"Dataset is missing '{required}'. "
                "Ensure it was recorded with crisp_gym, which stores individual "
                "observation.state sub-keys alongside the flat observation.state."
            )

    cart_names: list[str] = list(dataset.features["observation.state.cartesian"]["names"])
    grip_names: list[str] = list(dataset.features["observation.state.gripper"]["names"])
    umilike_names = cart_names + grip_names
    umilike_dim = (
        int(dataset.features["observation.state.cartesian"]["shape"][0])
        + int(dataset.features["observation.state.gripper"]["shape"][0])
    )

    print(f"\n[bold]observation.state.umilike[/bold] → {umilike_dim}D")
    print(f"  names : {umilike_names}")

    # ── Resolve output path ──────────────────────────────────────────────────
    output_root: Path = args.output_dir if args.output_dir else HF_LEROBOT_HOME / output_id
    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_root}\n"
            "Remove it or choose a different --output-dir / --output-dataset."
        )

    print(f"\nOutput dataset : [bold]{output_id}[/bold]")
    print(f"Output dir     : {output_root}")

    # ── Step 1: copy the entire source dataset ───────────────────────────────
    # Videos are copied as-is; no re-encoding occurs.
    print("\nCopying dataset (videos are not re-encoded)…")
    shutil.copytree(source_root, output_root)

    # ── Step 2: patch every parquet file under data/ ─────────────────────────
    parquet_files = sorted((output_root / "data").glob("**/*.parquet"))
    print(f"Found {len(parquet_files)} parquet file(s) to patch.")

    all_umilike: list[np.ndarray] = []

    with Progress() as progress:
        task_bar = progress.add_task("Patching parquet files…", total=len(parquet_files))

        for parquet_path in parquet_files:
            df = pd.read_parquet(parquet_path)

            cart = np.stack(df["observation.state.cartesian"].values).astype(np.float32)
            grip = np.stack(df["observation.state.gripper"].values).astype(np.float32)
            if grip.ndim == 1:
                grip = grip.reshape(-1, 1)

            umilike = np.concatenate([cart, grip], axis=1)
            df["observation.state.umilike"] = list(umilike)
            all_umilike.append(umilike)

            df.to_parquet(parquet_path, index=False)
            progress.advance(task_bar)

    # ── Step 3: update meta/info.json ────────────────────────────────────────
    info_path = output_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["features"]["observation.state.umilike"] = {
        "dtype": "float32",
        "shape": [umilike_dim],
        "names": umilike_names,
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # ── Step 4: update stats.json if present ─────────────────────────────────
    stats_path = output_root / "meta" / "stats.json"
    if stats_path.exists():
        all_data = np.concatenate(all_umilike, axis=0)

        with open(stats_path) as f:
            stats = json.load(f)

        stats["observation.state.umilike"] = {
            "mean": all_data.mean(axis=0).tolist(),
            "std": all_data.std(axis=0).tolist(),
            "min": all_data.min(axis=0).tolist(),
            "max": all_data.max(axis=0).tolist(),
        }

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print("Updated meta/stats.json with umilike statistics.")
    else:
        print("[yellow]meta/stats.json not found — stats not updated.[/yellow]")

    print(f"\n[green]Done.[/green] Output dataset: [bold]{output_id}[/bold]")
    print(f"Stored at: {output_root}")

    if args.push_to_hub:
        print("Pushing to Hugging Face Hub…")
        output_dataset = LeRobotDataset(repo_id=output_id, root=output_root)
        output_dataset.push_to_hub()
        print(f"[green]Pushed:[/green] {output_id}")
    else:
        print("[yellow]Tip:[/yellow] add --push-to-hub to upload the dataset.")


if __name__ == "__main__":
    main()
