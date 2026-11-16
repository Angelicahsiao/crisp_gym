r"""Print the layout of a LeRobot dataset folder and the files recorded next to it.

What it shows, so a dataset can be described without opening it by hand:

  * `meta/info.json`: fps, episode and frame counts, and every feature with its
    shape and dtype;
  * the high-rate tactile CSVs (`<dataset>/tactile_raw/episode_<NNNNNN>_<side>.csv`):
    how many there are, how many are only DVC pointers, and for one episode the
    number of fields, the first and last column names and the row count;
  * the start of `tactile_raw/edit_report.json` (written when sessions are merged).

Standard library only: no numpy, no pandas, no ROS. Runs on any machine that
can see the dataset.

    python crisp_gym/scripts/inspect_dataset.py ../../datasets/umi_electricbox/lerobot
    python crisp_gym/scripts/inspect_dataset.py DATASET --episode 3 --report-chars 0

This is the one-liner from the tactile investigation, made a script: the row
count printed here excludes the header line (`wc -l` counted it).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

RAW_DIR_NAME = "tactile_raw"
CSV_NAME = re.compile(r"^episode_(\d{6})_(.+)\.csv$")


def describe_info(info: dict) -> list[str]:
    """Lines for meta/info.json: counts, then one line per feature."""
    lines = [
        f"fps {info.get('fps')} | episodes {info.get('total_episodes')} | "
        f"frames {info.get('total_frames')}"
        + (f" | codebase {info['codebase_version']}" if "codebase_version" in info else "")
    ]
    features = info.get("features") or {}
    lines.append(f"features ({len(features)}):")
    width = max((len(k) for k in features), default=0)
    for key, spec in features.items():
        spec = spec if isinstance(spec, dict) else {}
        lines.append(f"  {key:<{width}}  {spec.get('shape')}  {spec.get('dtype', '?')}")
    return lines


def describe_csv(path: Path) -> str:
    """One line: field count, first and last three columns, data rows."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    if len(header) > 6:
        cols = ", ".join(header[:3]) + " ... " + ", ".join(header[-2:])
    else:
        cols = ", ".join(header)
    return f"{path.name}: {len(header)} fields | {cols} | {rows} data rows"


def describe_raw_dir(raw_dir: Path, episode: int, report_chars: int) -> list[str]:
    """Lines for the tactile_raw folder."""
    if not raw_dir.is_dir():
        return [f"{raw_dir.name}/: not present"]
    csvs = sorted(p for p in raw_dir.iterdir() if CSV_NAME.match(p.name))
    pointers = sorted(p for p in raw_dir.iterdir() if p.name.endswith(".csv.dvc"))
    pulled = {p.name for p in csvs}
    not_pulled = [p for p in pointers if p.name[: -len(".dvc")] not in pulled]
    episodes = sorted({int(CSV_NAME.match(p.name).group(1)) for p in csvs})
    sides = sorted({CSV_NAME.match(p.name).group(2) for p in csvs})
    lines = [
        f"{raw_dir.name}/: {len(csvs)} CSV file(s) for {len(episodes)} episode(s), "
        f"sides {sides}; {len(pointers)} DVC pointer(s), {len(not_pulled)} not pulled"
    ]
    for side in sides or ["(none)"]:
        path = raw_dir / f"episode_{episode:06d}_{side}.csv"
        if path.is_file():
            lines.append("  " + describe_csv(path))
        elif path.with_name(path.name + ".dvc").is_file():
            lines.append(f"  {path.name}: only its DVC pointer (not pulled)")
        else:
            lines.append(f"  {path.name}: missing")
    report = raw_dir / "edit_report.json"
    if report.is_file() and report_chars > 0:
        text = report.read_text(errors="replace")
        cut = "" if len(text) <= report_chars else f" (first {report_chars} of {len(text)})"
        lines.append(f"{raw_dir.name}/edit_report.json{cut}:")
        lines.append(text[:report_chars])
    elif report.is_file():
        lines.append(f"{raw_dir.name}/edit_report.json: present")
    return lines


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dataset", type=Path, help="LeRobot dataset root (contains meta/info.json)")
    p.add_argument("--episode", type=int, default=0, help="episode whose CSVs are shown")
    p.add_argument("--raw-dir", default=RAW_DIR_NAME, help="folder of the high-rate CSVs")
    p.add_argument(
        "--report-chars",
        type=int,
        default=1200,
        help="characters of edit_report.json to print (0 = only say it is there)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Print the dataset layout. Returns 1 when meta/info.json is missing."""
    args = build_parser().parse_args(argv)
    info_path = args.dataset / "meta" / "info.json"
    if not info_path.is_file():
        print(f"{info_path} not found — not a LeRobot dataset root?", file=sys.stderr)
        return 1
    lines = describe_info(json.loads(info_path.read_text()))
    lines += describe_raw_dir(args.dataset / args.raw_dir, args.episode, args.report_chars)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
