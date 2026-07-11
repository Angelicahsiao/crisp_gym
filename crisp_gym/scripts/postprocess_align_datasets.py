"""Align LeRobot datasets from different devices so they can be mixed in training.

Use case: robot episodes recorded with the UMI contract PLUS extra states
(umi_robot_full_record.yaml: joints, efforts, target pose, ...) must be
schema-identical to handheld episodes (which have none of those) before
LeRobot can concatenate them. This script:

1. Reads each dataset's stamped meta/record_config.json and verifies the CORE
   contract matches (action definition/lookahead/representation, rate_hz, and
   the include_in_state observation entries). A core mismatch is unfixable
   data — the script refuses (no silent corruption).
2. Computes the shared column set and strips non-shared columns from the
   Parquet files (extras like observation.state.joint_efforts are dropped in
   the aligned copy; originals are untouched).
3. Optional fixups (explicit flags only):
     --rescale-gripper OLD_REF NEW_REF   gripper recorded against a wrong
                                         reference width -> rescale obs+action
                                         gripper dims by OLD_REF/NEW_REF, clip [0,1]
     --promote extra.foo [extra.bar ...] rename extra.* columns to
                                         observation.state.* so LeRobot treats
                                         them as policy STATE inputs (with
                                         temporal windowing + stats). Use for
                                         ROBOT-ONLY training — promoted
                                         datasets are no longer mixable with
                                         handheld data lacking those columns.
4. Rewrites meta features (info.json) and record_config.json accordingly.

Output: <dataset>_aligned copies, ready to concatenate.

Usage:
    python postprocess_align_datasets.py \\
        --datasets ~/.cache/.../umi_handheld_demo ~/.cache/.../umi_ur7e_demo \\
        [--output-suffix _aligned] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bookkeeping columns LeRobot needs — never stripped. Note: exact-match names
# plus "next." prefix; "observation.state" is exact (sub-keys like
# observation.state.joints ARE strippable extras).
LEROBOT_INTERNAL_EXACT = {
    "timestamp", "frame_index", "episode_index", "index", "task", "task_index",
    "action", "observation.state",
}
LEROBOT_INTERNAL_PREFIXES = ("next.",)


def load_contract(dataset_dir: Path) -> dict:
    path = dataset_dir / "meta" / "record_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_dir} has no meta/record_config.json — it was not recorded "
            "with the config-driven recorder. Cannot verify mixability."
        )
    with open(path) as f:
        return json.load(f)


def core_contract(meta: dict) -> dict:
    """The part of the contract that must match for datasets to be mixable."""
    return {
        "rate_hz": meta.get("rate_hz"),
        "action": meta.get("action"),
        # device_max_width is device-specific (Robotiq vs handheld) — the
        # shared scale is reference_width, so exclude it like RecordConfig does.
        "state_observations": {
            o["key"]: {k: v for k, v in o.items()
                       if k not in ("include_in_state", "device_max_width")}
            for o in meta.get("observations", [])
            if o.get("include_in_state", True)
        },
    }


def episode_files(dataset_dir: Path) -> list[Path]:
    """All frame-data parquet files, layout-agnostic.

    v2.x: data/chunk-*/episode_XXXXXX.parquet (one episode per file).
    v3.0: data/chunk-*/file-XXX.parquet (many episodes per file). All the
    rewrites in this script (column drop/rename, per-row gripper rescale)
    operate per-row, so multi-episode files are handled transparently.
    """
    files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No frame parquet files under {dataset_dir}/data")
    return files


def dataset_columns(dataset_dir: Path) -> set[str]:
    return set(pd.read_parquet(episode_files(dataset_dir)[0]).columns)


def is_internal(col: str) -> bool:
    return col in LEROBOT_INTERNAL_EXACT or any(
        col.startswith(p) for p in LEROBOT_INTERNAL_PREFIXES
    )


def promoted_name(key: str) -> str:
    """extra.foo -> observation.state.foo"""
    if not key.startswith("extra."):
        raise ValueError(f"--promote expects 'extra.*' keys, got '{key}'")
    return "observation.state." + key[len("extra."):]


def align(
    dataset_dirs: list[Path],
    output_suffix: str,
    rescale_gripper: tuple[float, float] | None,
    promote: list[str],
    dry_run: bool,
) -> None:
    # ── 1. contract verification ─────────────────────────────────────────────
    contracts = {d: load_contract(d) for d in dataset_dirs}
    cores = {d: core_contract(m) for d, m in contracts.items()}
    ref_dir, ref_core = next(iter(cores.items()))
    for d, core in cores.items():
        if core != ref_core:
            for field in ref_core:
                if core[field] != ref_core[field]:
                    logger.error(
                        f"CORE CONTRACT MISMATCH on '{field}':\n"
                        f"  {ref_dir.name}: {ref_core[field]}\n"
                        f"  {d.name}: {core[field]}"
                    )
            raise SystemExit(
                "Datasets are not mixable: core contracts differ (see above). "
                "This cannot be fixed by post-processing — the recorded action "
                "semantics/rates/policy-inputs are different data."
            )
    logger.info("✓ Core contracts match across all datasets.")

    # ── 2. shared column set ─────────────────────────────────────────────────
    col_sets = {d: dataset_columns(d) for d in dataset_dirs}
    shared = set.intersection(*col_sets.values())

    if promote:
        # Promoted columns must exist everywhere (a policy input cannot be
        # missing in part of the training data) and must not be stripped.
        missing = {d.name: sorted(set(promote) - col_sets[d]) for d in dataset_dirs
                   if set(promote) - col_sets[d]}
        if missing:
            raise SystemExit(
                f"--promote columns missing in some datasets: {missing}. "
                "Promotion makes them policy inputs, so every dataset in the "
                "mix must contain them (robot-only training)."
            )
        shared |= set(promote)
    for d, cols in col_sets.items():
        extras = sorted(c for c in cols - shared if not is_internal(c))
        if extras:
            logger.info(f"{d.name}: stripping non-shared columns: {extras}")

    # ── 3. per-dataset rewrite ───────────────────────────────────────────────
    for d in dataset_dirs:
        out = d.parent / (d.name + output_suffix)
        drop = sorted(c for c in col_sets[d] - shared if not is_internal(c))

        if dry_run:
            logger.info(f"[DRY RUN] {d.name} -> {out.name}: drop {drop or 'nothing'}"
                        + (f", rescale gripper x{rescale_gripper[0]/rescale_gripper[1]:.4f}"
                           if rescale_gripper else ""))
            continue

        logger.info(f"Copying {d} -> {out}")
        shutil.copytree(d, out, dirs_exist_ok=True)

        rename_map = {k: promoted_name(k) for k in promote}

        for ep in episode_files(out):
            df = pd.read_parquet(ep)
            df = df.drop(columns=[c for c in drop if c in df.columns])
            if rename_map:
                df = df.rename(columns=rename_map)

            if rescale_gripper is not None:
                old_ref, new_ref = rescale_gripper
                factor = old_ref / new_ref
                if "observation.state.gripper" in df.columns:
                    df["observation.state.gripper"] = [
                        np.clip(np.asarray(v, np.float32) * factor, 0.0, 1.0)
                        for v in df["observation.state.gripper"]
                    ]
                if "action" in df.columns:
                    def _fix_action(v):
                        a = np.asarray(v, np.float32).copy()
                        a[-1] = np.clip(a[-1] * factor, 0.0, 1.0)
                        return a
                    df["action"] = [_fix_action(v) for v in df["action"]]
                # observation.state (concatenated) ends with the gripper dim
                if "observation.state" in df.columns:
                    def _fix_state(v):
                        s = np.asarray(v, np.float32).copy()
                        s[-1] = np.clip(s[-1] * factor, 0.0, 1.0)
                        return s
                    df["observation.state"] = [_fix_state(v) for v in df["observation.state"]]

            df.to_parquet(ep, index=False)

        # ── meta updates ──
        info_path = out / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            feats = info.get("features", {})
            for c in drop:
                feats.pop(c, None)
            for old, new in rename_map.items():
                if old in feats:
                    feats[new] = feats.pop(old)
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4)

        rc_path = out / "meta" / "record_config.json"
        meta = contracts[d]
        meta["observations"] = [
            o for o in meta.get("observations", []) if o["key"] in shared
        ]
        for o in meta["observations"]:
            if o["key"] in rename_map:
                o["key"] = rename_map[o["key"]]
                o["include_in_state"] = True   # it IS a policy input now
        if rename_map:
            meta["promoted"] = rename_map
        meta["aligned_from"] = str(d)
        if rescale_gripper is not None:
            meta["gripper_rescaled"] = {"old_ref": rescale_gripper[0],
                                        "new_ref": rescale_gripper[1]}
        with open(rc_path, "w") as f:
            json.dump(meta, f, indent=4)

        logger.info(f"  ✓ {out.name} written ({len(episode_files(out))} data files)")

    if not dry_run:
        logger.info("Done. Aligned datasets are schema-identical and mixable.")


def main():
    parser = argparse.ArgumentParser(
        description="Align LeRobot datasets from different devices for mixed training."
    )
    parser.add_argument("--datasets", type=Path, nargs="+", required=True,
                        help="Dataset root directories (each with data/ and meta/).")
    parser.add_argument("--output-suffix", type=str, default="_aligned",
                        help="Suffix for the aligned output copies (default: _aligned).")
    parser.add_argument("--rescale-gripper", type=float, nargs=2, default=None,
                        metavar=("OLD_REF", "NEW_REF"),
                        help="Rescale gripper dims recorded against OLD_REF meters "
                             "to NEW_REF meters (value * OLD/NEW, clipped [0,1]).")
    parser.add_argument("--promote", type=str, nargs="+", default=[],
                        metavar="EXTRA_KEY",
                        help="Rename extra.* columns to observation.state.* so "
                             "LeRobot uses them as policy inputs (robot-only "
                             "training; all datasets must contain them).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing anything.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    dirs = [d.expanduser().resolve() for d in args.datasets]
    for d in dirs:
        if not d.exists():
            raise FileNotFoundError(f"Dataset not found: {d}")

    for k in args.promote:
        promoted_name(k)  # validate prefix early, before any copying

    align(dirs, args.output_suffix,
          tuple(args.rescale_gripper) if args.rescale_gripper else None,
          args.promote, args.dry_run)


if __name__ == "__main__":
    main()
