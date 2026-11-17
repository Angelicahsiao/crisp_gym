r"""Offline: copy a dataset and join chosen columns into `observation.environment_state`.

Why this exists. A lerobot diffusion or ACT policy reads exactly two
low-dimensional inputs: `observation.state` and `observation.environment_state`.
Any other column (`extra.ext_torque`, `extra.tactile.left`, and also a column
renamed to `observation.state.<name>` by `postprocess_align_datasets.py
--promote`) is loaded and normalised but never enters the network. To let the
model see joint torque, force or tactile, this script writes the chosen columns,
joined in the order given, into ONE new column with that exact name:

    observation.environment_state[t] = [ columns[0][t], columns[1][t], ... ]

`observation.state`, `action` and every other column stay as they are, so the
relative-pose training wrapper, the checkpoint stamps and the state-width
checks are unaffected.

WHICH POLICIES READ IT (checked in the lerobot source, 2026-11):
    diffusion, act                 yes
    multi_task_dit                 NO: it reads observation.state, the images and
                                   the task text only. This column would be ignored.
Check any other policy before training on the result:
    grep -n "OBS_ENV_STATE" <lerobot>/policies/<policy>/modeling_*.py

WHAT IT DOES. File surgery, like swap_action_offline.py: the dataset is copied
(videos byte for byte, never re-encoded), the new column is added to every data
parquet, the feature is added to meta/info.json, and its statistics are added
wherever the dataset keeps them (meta/stats.json; v3.0 meta/episodes/*.parquet;
v2.x meta/episodes_stats.jsonl) with the same stat names `observation.state`
has. meta/environment_state.json records which column went into which slice.
The input dataset is never modified.

WHAT IT REFUSES, before writing anything: a column that is not in the dataset,
an image column, a column that never changes (handheld datasets carry all-zero
placeholders for the robot-only columns: `--allow-constant` overrides), a value
that is NaN or infinite, and a dataset that already has the key.

NOT DONE YET: deployment. `relative_lerobot_policy` does not produce this key,
so a checkpoint trained on the result can be trained and checked open loop but
not run on the robot until the deploy side is written (GUIDANCE WP-10).

    python crisp_gym/scripts/add_environment_state.py \
        --input  ../../datasets/franka_electricbox/lerobot \
        --output ../../datasets/franka_electricbox_torque/lerobot \
        --columns extra.ext_torque

    python crisp_gym/scripts/add_environment_state.py --input IN --output OUT \
        --columns extra.tactile.left extra.tactile.right --dry-run

GPU-PC / preprocessing script: numpy + pandas + pyarrow. No ROS, no crisp_py,
no lerobot. Reuses the tested statistics code of migrate_euler_delta_to_rot6d.py
(same directory). Tests: tests/test_add_environment_state.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# sys.path[0] is this file's directory when it is run as a script, so the
# sibling module resolves without an install (as in swap_action_offline.py).
from migrate_euler_delta_to_rot6d import (
    _compute_stats,
    data_parquet_files,
    load_info,
)

logger = logging.getLogger(__name__)

ENV_STATE_KEY = "observation.environment_state"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
NOTE_FILE = "environment_state.json"
DEFAULT_STAT_NAMES = ("min", "max", "mean", "std", "count")
NOT_NUMERIC = ("video", "image", "string")


class Refused(ValueError):
    """The request would produce a column that is wrong or useless; nothing was written."""


def flatten_cell(cell: object) -> np.ndarray:
    """One row's value as a flat float32 vector; a nested grid is read row by row."""
    arr = np.asarray(cell)
    if arr.dtype == object:  # parquet returns a 2-D cell as an array of arrays
        return np.concatenate([flatten_cell(part) for part in arr])
    return arr.astype(np.float32).reshape(-1)


def join_columns(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """The chosen columns of every row, flattened and joined in the order given: [N, D]."""
    parts = [np.stack([flatten_cell(cell) for cell in df[column]]) for column in columns]
    return np.concatenate(parts, axis=1).astype(np.float32)


def plan_sources(features: dict, columns: list[str]) -> list[dict]:
    """Where each chosen column goes in the new vector. Raises `Refused` on a bad request."""
    if ENV_STATE_KEY in features:
        raise Refused(
            f"the dataset already has '{ENV_STATE_KEY}'. Start again from the dataset it "
            "was made from and name every column you want in one run."
        )
    if len(set(columns)) != len(columns):
        raise Refused(f"a column is named twice in {columns}")
    sources, start = [], 0
    for column in columns:
        if column not in features:
            candidates = sorted(k for k in features if k.startswith("extra."))
            raise Refused(f"the dataset has no column '{column}'. Its extra columns: {candidates}")
        dtype = str(features[column].get("dtype", ""))
        if dtype in NOT_NUMERIC:
            raise Refused(f"'{column}' is a {dtype} column, not numbers")
        dim = int(np.prod(features[column]["shape"]))
        stem = column.removeprefix("extra.").replace(".", "_")
        sources.append(
            {
                "column": column,
                "dim": dim,
                "slice": [start, start + dim],
                "names": [f"{stem}_{i}" for i in range(dim)],
            }
        )
        start += dim
    return sources


def read_joined(
    src: Path, sources: list[dict], allow_constant: bool
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    """Join the columns of every data file and check the result. Writes nothing.

    Returns the joined values per data file (keyed by its path relative to the
    dataset) and per episode.
    """
    columns = [s["column"] for s in sources]
    width = sources[-1]["slice"][1]
    per_file: dict[str, np.ndarray] = {}
    per_episode: dict[int, list[np.ndarray]] = {}
    for file in data_parquet_files(src):
        df = pd.read_parquet(file, columns=[*columns, "episode_index"])
        joined = join_columns(df, columns)
        if joined.shape[1] != width:
            raise Refused(
                f"{file.name}: the columns are {joined.shape[1]} numbers wide together, but "
                f"meta/info.json declares {width}. The dataset and its info.json disagree; "
                "run check_schema_drift.py."
            )
        bad = ~np.isfinite(joined)
        if bad.any():
            row, col = np.argwhere(bad)[0]
            source = next(s for s in sources if s["slice"][0] <= col < s["slice"][1])
            raise Refused(
                f"{int(bad.sum())} NaN or infinite value(s) in {file.name}; the first is in "
                f"'{source['column']}', row {int(row)}. A model cannot be trained on that."
            )
        per_file[str(file.relative_to(src))] = joined
        episodes = df["episode_index"].to_numpy()
        for episode in np.unique(episodes):
            per_episode.setdefault(int(episode), []).append(joined[episodes == episode])

    stacked = {episode: np.concatenate(parts) for episode, parts in per_episode.items()}
    everything = np.concatenate(list(stacked.values()))
    spread = everything.max(axis=0) - everything.min(axis=0)
    for source in sources:
        lo, hi = source["slice"]
        still = int((spread[lo:hi] == 0).sum())
        if still == hi - lo and not allow_constant:
            raise Refused(
                f"'{source['column']}' never changes: every frame holds "
                f"{np.round(everything[0, lo:hi], 6).tolist()}. That is what a placeholder "
                "looks like (a handheld dataset has no joint torque; a joint-driven one has "
                "no Cartesian target). The model would learn nothing from it. Pass "
                "--allow-constant if this is intended."
            )
        if still:
            logger.warning(
                "'%s': %d of %d values never change (dead taxels?).",
                source["column"],
                still,
                hi - lo,
            )
    return per_file, stacked


def _template_key(present: object) -> str | None:
    """The key whose stat names the new key copies: observation.state, else action."""
    return next((key for key in (STATE_KEY, ACTION_KEY) if key in present), None)


def add_stats_json(out: Path, values: np.ndarray) -> None:
    """Add the new key to meta/stats.json, with the stat names of observation.state."""
    path = out / "meta" / "stats.json"
    if not path.exists():
        logger.warning("no meta/stats.json: the aggregate statistics were not written.")
        return
    stats = json.loads(path.read_text())
    template = _template_key(stats)
    names = list(stats[template]) if template else list(DEFAULT_STAT_NAMES)
    computed = _compute_stats(values, names)
    block = {
        name: np.asarray(computed[name]).astype(np.float64).tolist()
        for name in names
        if computed.get(name) is not None
    }
    if "count" in block:
        block["count"] = [int(values.shape[0])]
    updated = {}
    for key, value in stats.items():
        updated[key] = value
        if key == template:
            updated[ENV_STATE_KEY] = block
    updated.setdefault(ENV_STATE_KEY, block)
    path.write_text(json.dumps(updated, indent=4))
    logger.info("  meta/stats.json: added %s (%s)", ENV_STATE_KEY, ", ".join(block))


def add_v3_episode_stats(out: Path, per_episode: dict[int, np.ndarray]) -> None:
    """Add `stats/<key>/<stat>` columns to the v3.0 meta/episodes parquet files."""
    for file in sorted((out / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(file)
        template = next(
            (k for k in (STATE_KEY, ACTION_KEY) if any(c.startswith(f"stats/{k}/") for c in df)),
            None,
        )
        if template is None:
            logger.warning("%s has no per-episode statistics to model on; skipped.", file.name)
            continue
        prefix = f"stats/{template}/"
        names = [c[len(prefix) :] for c in df.columns if c.startswith(prefix)]
        new_columns: dict[str, list] = {name: [] for name in names}
        for i, episode in enumerate(df["episode_index"].to_numpy()):
            values = per_episode.get(int(episode))
            if values is None:
                raise Refused(
                    f"meta/episodes lists episode {int(episode)}, which has no frames in data/."
                )
            computed = _compute_stats(values, names)
            for name in names:
                like = df[prefix + name].iloc[i]  # copy the form of the key modelled on
                like_is_array = isinstance(like, np.ndarray) and like.ndim >= 1
                if name == "count":
                    n = int(values.shape[0])
                    cell = np.full(like.shape, n, dtype=like.dtype) if like_is_array else n
                elif computed.get(name) is None:
                    cell = None
                else:
                    cell = np.asarray(computed[name]).reshape(-1)
                    cell = cell.astype(like.dtype if like_is_array else np.float32)
                new_columns[name].append(cell)
        for name, cells in new_columns.items():
            if any(cell is not None for cell in cells):
                df[f"stats/{ENV_STATE_KEY}/{name}"] = cells
        df.to_parquet(file, index=False)
        logger.info("  %s: added per-episode statistics", file.relative_to(out))


def add_v2_episode_stats(out: Path, per_episode: dict[int, np.ndarray]) -> None:
    """Add the new key to every line of the v2.x meta/episodes_stats.jsonl."""
    path = out / "meta" / "episodes_stats.jsonl"
    if not path.exists():
        return
    lines = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        values = per_episode.get(record.get("episode_index"))
        if values is not None and "stats" in record:
            template = _template_key(record["stats"])
            names = list(record["stats"][template]) if template else list(DEFAULT_STAT_NAMES)
            computed = _compute_stats(values, names)
            record["stats"][ENV_STATE_KEY] = {
                name: [int(values.shape[0])]
                if name == "count"
                else np.asarray(computed[name]).astype(np.float64).tolist()
                for name in names
                if computed.get(name) is not None
            }
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n")
    logger.info("  meta/episodes_stats.jsonl: added per-episode statistics")


def add_feature(out: Path, sources: list[dict]) -> None:
    """Declare the new column in meta/info.json, right after observation.state."""
    path = out / "meta" / "info.json"
    info = json.loads(path.read_text())
    feature = {
        "dtype": "float32",
        "shape": [sources[-1]["slice"][1]],
        "names": [name for source in sources for name in source["names"]],
    }
    features = {}
    for key, value in info["features"].items():
        features[key] = value
        if key == STATE_KEY:
            features[ENV_STATE_KEY] = feature
    features.setdefault(ENV_STATE_KEY, feature)
    info["features"] = features
    path.write_text(json.dumps(info, indent=4))


def run(args: argparse.Namespace) -> int:
    """Validate, then copy and extend. Raises `Refused`; returns 2 for a bad path."""
    src = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not (src / "meta" / "info.json").exists():
        logger.error("No meta/info.json under %s: is this a LeRobot dataset root?", src)
        return 2
    if out.exists() and not args.dry_run:
        logger.error("Output %s already exists: remove it first.", out)
        return 2

    sources = plan_sources(load_info(src)["features"], list(args.columns))
    per_file, per_episode = read_joined(src, sources, args.allow_constant)
    everything = np.concatenate([per_episode[e] for e in sorted(per_episode)])

    logger.info("%s will be %d numbers per frame:", ENV_STATE_KEY, everything.shape[1])
    for source in sources:
        lo, hi = source["slice"]
        logger.info(
            "  [%3d:%3d]  %-28s  min %.4g  max %.4g",
            lo,
            hi,
            source["column"],
            float(everything[:, lo:hi].min()),
            float(everything[:, lo:hi].max()),
        )
    if args.dry_run:
        logger.info("--dry-run: the request is valid. Nothing was written.")
        return 0

    logger.info("Copying %s -> %s (videos byte for byte)", src, out)
    shutil.copytree(src, out)
    for relative, joined in per_file.items():
        file = out / relative
        df = pd.read_parquet(file).reset_index(drop=True)
        df[ENV_STATE_KEY] = list(joined)
        df.to_parquet(file, index=False)

    add_feature(out, sources)
    add_stats_json(out, everything)
    add_v3_episode_stats(out, per_episode)
    add_v2_episode_stats(out, per_episode)
    (out / "meta" / NOTE_FILE).write_text(
        json.dumps(
            {
                "key": ENV_STATE_KEY,
                "dim": int(everything.shape[1]),
                "sources": [
                    {"column": s["column"], "dim": s["dim"], "slice": s["slice"]} for s in sources
                ],
                "derived_from": str(src),
                "note": "Joined copies of the listed columns, in this order. Read by "
                "lerobot's diffusion and ACT policies; NOT by multi_task_dit.",
            },
            indent=2,
        )
    )
    logger.info(
        "Done: %d frames, %d episodes. New dataset: %s", len(everything), len(per_episode), out
    )
    logger.info(
        "Read by lerobot's diffusion and ACT policies. NOT read by multi_task_dit. "
        "Deployment does not produce this key yet."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="source dataset root (has data/ and meta/)")
    p.add_argument("--output", required=True, help="destination dataset root (must not exist)")
    p.add_argument(
        "--columns",
        required=True,
        nargs="+",
        help="columns to join, in this order, e.g. extra.ext_torque extra.tactile.left",
    )
    p.add_argument(
        "--allow-constant",
        action="store_true",
        help="accept a column that never changes (refused by default: it is usually a placeholder)",
    )
    p.add_argument("--dry-run", action="store_true", help="validate and report; write nothing")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the tool; a refusal is reported in one line and returns 2."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    try:
        return run(args)
    except Refused as refusal:
        logger.error("Refused, nothing written: %s", refusal)
        return 2


if __name__ == "__main__":
    sys.exit(main())
