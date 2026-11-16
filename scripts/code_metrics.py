#!/usr/bin/env python3
r"""Size of the code: the longest functions, and how many pass a length.

The numbers in CODE_REVIEW.md (finding A4) come from this. Standard library
only; reads the files, imports nothing from them.

    python scripts/code_metrics.py                       # crisp_gym/ and scripts/
    python scripts/code_metrics.py crisp_gym/policy --top 5
    python scripts/code_metrics.py --files                # longest files instead

A function's length is from its `def` line to its last line, so it includes its
docstring and any functions nested in it (nested ones are also listed on their
own).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

DEFAULT_PATHS = ("crisp_gym", "scripts")
THRESHOLDS = (80, 150)


def python_files(paths: list[Path]) -> list[Path]:
    """Every .py file under the given files or directories, without caches."""
    out: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            out.append(path)
        elif path.is_dir():
            out += [p for p in sorted(path.rglob("*.py")) if "__pycache__" not in p.parts]
    return out


def function_lengths(files: list[Path]) -> list[tuple[int, str, str]]:
    """(lines, "file:line", name) for every function and method."""
    rows = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as e:
            print(f"skipped {path}: {e}", file=sys.stderr)
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                rows.append((node.end_lineno - node.lineno + 1, f"{path}:{node.lineno}", node.name))
    return sorted(rows, key=lambda r: (-r[0], r[1]))


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("paths", nargs="*", type=Path, help=f"default: {' '.join(DEFAULT_PATHS)}")
    p.add_argument("--top", type=int, default=15, help="how many rows to list")
    p.add_argument("--files", action="store_true", help="list the longest files instead")
    return p


def main(argv: list[str] | None = None) -> int:
    """Print the table. Returns 1 when no Python file was found."""
    args = build_parser().parse_args(argv)
    files = python_files(args.paths or [Path(p) for p in DEFAULT_PATHS])
    if not files:
        print("no Python files found", file=sys.stderr)
        return 1
    if args.files:
        sizes = sorted(((len(p.read_text().splitlines()), str(p)) for p in files), reverse=True)
        for lines, path in sizes[: args.top]:
            print(f"{lines:6d}  {path}")
        print(f"{len(files)} files, {sum(n for n, _ in sizes)} lines")
        return 0
    rows = function_lengths(files)
    for lines, where, name in rows[: args.top]:
        print(f"{lines:6d}  {name:<32} {where}")
    over = ", ".join(f"over {t} lines: {sum(r[0] > t for r in rows)}" for t in THRESHOLDS)
    print(f"{len(rows)} functions in {len(files)} files; {over}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
