#!/usr/bin/env python3
"""Diagnose (and optionally repair) parquet-vs-info.json column drift.

LeRobot casts every data parquet to the schema declared in meta/info.json.
If the parquet carries a column info.json does not declare, the cast fails
at TRAIN time with `CastError: ... because column names don't match` -- long
after every merge-time gate has passed, because none of those gates ever
look at the parquet columns.

Usage:
    python check_schema_drift.py <dataset_root> [<dataset_root> ...]
    python check_schema_drift.py <dataset_root> --repair

<dataset_root> is the dir holding meta/ and data/ (e.g. .../franka_electricbox/lerobot).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def data_files(root: Path) -> list[Path]:
    """Every frame-data parquet under data/, layout-agnostic."""
    return sorted((root / "data").rglob("*.parquet"))


def declared_features(root: Path) -> dict:
    """The `features` block of meta/info.json."""
    with open(root / "meta" / "info.json") as f:
        return json.load(f).get("features", {})


def report(root: Path) -> tuple[set[str], set[str]]:
    """Print the column/feature comparison; return (orphan columns, missing columns)."""
    feats = declared_features(root)
    files = data_files(root)
    if not files:
        raise SystemExit(f"{root}: no parquet under data/")

    cols = set(pd.read_parquet(files[0]).columns)
    # video features live as mp4 on disk, never as a parquet column
    declared = {k for k, v in feats.items() if v.get("dtype") != "video"}

    orphan_cols = cols - declared          # in parquet, not declared -> CastError
    missing_cols = declared - cols         # declared, absent from parquet -> also fatal

    print(f"\n=== {root}")
    print(f"  data files        : {len(files)}")
    print(f"  parquet columns   : {len(cols)}")
    print(f"  declared (non-vid): {len(declared)}")
    if orphan_cols:
        print("  ** ORPHAN COLUMNS (in parquet, NOT in info.json) -> this is the CastError:")
        for c in sorted(orphan_cols):
            print(f"       {c}")
    if missing_cols:
        print("  ** MISSING COLUMNS (in info.json, NOT in parquet):")
        for c in sorted(missing_cols):
            print(f"       {c}")
    if not orphan_cols and not missing_cols:
        print("  OK - parquet columns and info.json features agree.")
    return orphan_cols, missing_cols


def repair(root: Path, orphans: set[str]) -> None:
    """Drop orphan columns from data parquets AND from every stats location."""
    if not orphans:
        print(f"{root}: nothing to repair.")
        return

    for p in data_files(root):
        df = pd.read_parquet(p)
        drop = [c for c in orphans if c in df.columns]
        if drop:
            df.drop(columns=drop).to_parquet(p, index=False)
            print(f"  data   {p.name}: dropped {drop}")

    # meta/episodes/*.parquet carries flattened stats/<key>/<stat> columns
    for p in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(p)
        stale = [c for c in df.columns
                 if c.startswith("stats/") and c.split("/")[1] in orphans]
        if stale:
            df.drop(columns=stale).to_parquet(p, index=False)
            print(f"  epstat {p.name}: dropped {len(stale)} stats column(s)")

    # meta/stats.json is keyed directly by feature name
    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        removed = [k for k in list(stats) if k in orphans]
        for k in removed:
            stats.pop(k)
        if removed:
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4)
            print(f"  stats.json: dropped {removed}")



# ─────────────────────── video timeline integrity ────────────────────────────

def _probe_duration(path: Path) -> float | None:
    """Seconds, via PyAV then ffprobe. None when neither is available."""
    try:
        import av  # type: ignore
        with av.open(str(path)) as c:
            if c.duration:
                return float(c.duration) / 1_000_000.0
            st = c.streams.video[0]
            if st.duration and st.time_base:
                return float(st.duration * st.time_base)
    except ImportError:
        pass
    except Exception:
        return None
    import shutil as _sh
    import subprocess
    if _sh.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(path)],
                capture_output=True, text=True, check=True).stdout.strip()
            return float(out)
        except Exception:
            return None
    return None


def check_videos(root: Path) -> bool:
    """Does every episode's video reference resolve to a file that is long enough?

    v3.0 packs many episodes into one mp4; an episode is located by
    videos/<key>/{chunk_index,file_index} plus from/to_timestamp. A missing
    file, or a file shorter than to_timestamp, means that episode reads
    someone else's frames -- or fails outright.
    """
    feats = declared_features(root)
    video_keys = [k for k, v in feats.items() if v.get("dtype") == "video"]
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not video_keys or not ep_files:
        print("  (no video features or no meta/episodes parquet - skipping video check)")
        return True

    eps = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    template = info.get("video_path",
                        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

    print(f"  episodes in meta   : {len(eps)}"
          f"   (info.json total_episodes: {info.get('total_episodes')})")
    ok = len(eps) == info.get("total_episodes")
    if not ok:
        print("  ** episode count in meta/episodes does not match info.json")

    for key in video_keys:
        cols = [f"videos/{key}/{c}" for c in
                ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
        if any(c not in eps.columns for c in cols):
            print(f"  ** {key}: episodes parquet lacks {[c for c in cols if c not in eps.columns]}")
            ok = False
            continue

        on_disk = set((root).glob(f"videos/{key}/chunk-*/file-*.mp4"))
        referenced, missing, too_short = set(), [], []
        # longest to_timestamp per (chunk, file) is all we need to bound-check
        need = eps.groupby([cols[0], cols[1]])[cols[3]].max()
        for (chunk, file_idx), max_to in need.items():
            rel = template.format(video_key=key, chunk_index=int(chunk), file_index=int(file_idx))
            path = root / rel
            referenced.add(path)
            if not path.exists():
                missing.append(rel)
                continue
            dur = _probe_duration(path)
            if dur is not None and max_to > dur + 0.05:
                too_short.append(f"{rel}: episodes need {max_to:.2f}s, file is {dur:.2f}s")

        orphans = sorted(p.relative_to(root) for p in on_disk - referenced)
        print(f"  {key}: {len(on_disk)} file(s) on disk, {len(referenced)} referenced")
        if missing:
            ok = False
            print(f"  ** MISSING video files referenced by episodes ({len(missing)}) - DATA LOSS:")
            for m in missing[:10]:
                print(f"       {m}")
        if too_short:
            ok = False
            print(f"  ** TRUNCATED video files ({len(too_short)}):")
            for t in too_short[:10]:
                print(f"       {t}")
        if orphans:
            print(f"  -- unreferenced files on disk ({len(orphans)}), harmless but stale:")
            for o in orphans[:10]:
                print(f"       {o}")
        if _probe_duration.__doc__ and not missing and not too_short:
            pass
    return ok


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--videos", action="store_true",
                    help="Also check that every episode's video reference resolves "
                         "to a file that exists and is long enough.")
    ap.add_argument("--repair", action="store_true",
                    help="Drop orphan columns in place (parquet + stats). "
                         "Back the dataset up first.")
    args = ap.parse_args()

    any_bad = False
    for root in args.roots:
        orphans, missing = report(root)
        if args.videos and not check_videos(root):
            any_bad = True
        if missing:
            any_bad = True
            print(f"  !! {root}: MISSING columns cannot be repaired by dropping — "
                  "info.json declares data the parquet does not have.")
        if orphans:
            any_bad = True
            if args.repair:
                repair(root, orphans)
    sys.exit(1 if (any_bad and not args.repair) else 0)


if __name__ == "__main__":
    main()
