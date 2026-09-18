#!/usr/bin/env python3
"""Is a recorded LeRobot dataset fit to fine-tune GR00T N1.7?

WHY THIS EXISTS
    GR00T's relative-action path is the reason to prefer it for rot6d poses,
    and it is configured by two flags plus one property of the DATA. Each can
    be wrong without anything failing:

    1. relative_exclude_joints matches per-DIMENSION action names read from
       info.json. lerobot's _infer_n1_7_action_groups() opens with

           if not action_names or action_dim <= 0:
               return []

       so a dataset whose `action` feature carries no `names` silently gets NO
       action groups, and --policy.relative_exclude_joints='["gripper"]' does
       nothing at all -- the gripper is folded into the relative arm group and
       trained as a delta. crisp_gym does write names (record_config.py:464,
       ActionSpec.names appends "gripper"), but a hand-built, migrated or
       aggregated dataset may not.

    2. GR00T wants ABSOLUTE actions. GrootN17PackInputsStep caches the raw
       state and GrootN17ActionDecodeStep composes the prediction back through
       it, so the relative conversion is GR00T's job, done with SE(3)
       composition on xyz+rot6d. Feeding it a dataset that crisp_gym already
       converted to relative double-converts. Nothing errors -- the model just
       learns deltas of deltas.

    3. The pose block must be xyz(3) + rot6d(6). GR00T's relative_eef_to_absolute
       consumes the first 9 dims that way; euler or quaternion poses do not map.

    This script answers all three from the dataset alone. It reads parquet and
    info.json only -- no torch, no lerobot, no GPU -- so it runs on a laptop
    before you commit a training slot.

USAGE
    python groot_preflight.py <dataset_root>
    python groot_preflight.py <dataset_root> --exclude gripper --chunk-size 40
    python groot_preflight.py <dataset_root> --samples 5000

<dataset_root> holds meta/ and data/ (e.g. .../franka_electricbox/lerobot).

EXIT CODES
    0  fit to train, possibly with warnings
    1  at least one check FAILED
    2  could not read the dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BAR = "=" * 72
OK, WARN, FAIL = "PASS", "WARN", "FAIL"
_results: list[tuple[str, str, str]] = []


def report(check: str, status: str, message: str) -> None:
    """Record and print one check outcome."""
    _results.append((check, status, message))
    print(f"  [{status}] {check}: {message}")


def detail(text: str) -> None:
    """Print an indented explanatory line under a check."""
    print(f"         {text}")


def head(title: str) -> None:
    """Print a section banner."""
    print(f"\n{BAR}\n  {title}\n{BAR}")


# ── dataset loading ───────────────────────────────────────────────────────────


def load_info(root: Path) -> dict:
    """meta/info.json as a dict."""
    path = root / "meta" / "info.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — is {root} a LeRobot dataset root?")
    with open(path) as f:
        return json.load(f)


def data_files(root: Path) -> list[Path]:
    """Every frame-data parquet, layout-agnostic."""
    return sorted((root / "data").rglob("*.parquet"))


def read_frames(root: Path, columns: list[str], limit: int) -> pd.DataFrame:
    """Read up to `limit` frames of the requested columns, in file order."""
    frames, total = [], 0
    for path in data_files(root):
        available = set(pd.read_parquet(path, columns=None).columns) if total == 0 else None
        want = [c for c in columns if available is None or c in available]
        df = pd.read_parquet(path, columns=want)
        frames.append(df)
        total += len(df)
        if total >= limit:
            break
    if not frames:
        raise SystemExit(f"no parquet files under {root / 'data'}")
    return pd.concat(frames, ignore_index=True).head(limit)


def stack(series: pd.Series) -> np.ndarray:
    """A parquet list-column as a 2-D float array."""
    return np.stack([np.asarray(v, dtype=np.float64) for v in series])


# ── G1: per-dimension action names ────────────────────────────────────────────


def check_names(info: dict, exclude: list[str]) -> list[str] | None:
    """The gripper must be namable, or relative_exclude_joints silently no-ops."""
    action = info.get("features", {}).get("action", {})
    names = action.get("names")
    shape = action.get("shape") or []
    dim = int(shape[0]) if shape else 0

    if not names:
        report("G1 action names", FAIL,
               "info.json features.action has no `names`")
        detail("_infer_n1_7_action_groups() returns [] when names are missing, so")
        detail(f"--policy.relative_exclude_joints={exclude!r} would do NOTHING and the")
        detail("gripper would be trained as a relative delta alongside the pose.")
        detail("Re-record with crisp_gym (record_config.py writes names), or add them")
        detail("to info.json by hand: they are purely metadata, one string per dim.")
        return None

    if len(names) != dim:
        report("G1 action names", FAIL,
               f"{len(names)} names for a {dim}-dim action")
        return names

    matched = [n for n in names if any(t.lower() in str(n).lower() for t in exclude)]
    if not exclude:
        report("G1 action names", WARN, f"{len(names)} names present, no --exclude given")
        detail(f"names: {names}")
        detail("With relative_exclude_joints empty, EVERY dim is relative, gripper included.")
    elif not matched:
        report("G1 action names", FAIL,
               f"no action dim matches {exclude!r}")
        detail(f"names: {names}")
        detail("The exclude token is matched case-insensitively as a substring against")
        detail("each dim's name. Nothing matches, so the gripper stays relative.")
    elif len(matched) > 1:
        report("G1 action names", WARN,
               f"{len(matched)} dims match {exclude!r}: {matched}")
        detail("Each becomes its own absolute group; intended only if that is really true.")
    else:
        report("G1 action names", OK,
               f"{matched[0]!r} (dim {names.index(matched[0])}) will stay absolute")
        detail(f"names: {names}")
    return names


# ── G2: absolute vs already-relative actions ──────────────────────────────────


def check_absolute(root: Path, df: pd.DataFrame, names: list[str] | None) -> None:
    """GR00T does its own relative conversion; a pre-converted dataset double-converts."""
    for stamp in ("action_repr.json", "pose_repr.json"):
        path = root / stamp
        if path.exists():
            try:
                with open(path) as f:
                    detail(f"stamp {stamp}: {json.load(f)}")
            except (OSError, json.JSONDecodeError):
                pass

    if "action" not in df.columns:
        report("G2 absolute actions", FAIL, "no `action` column in the data parquet")
        return

    act = stack(df["action"])
    xyz = act[:, :3]
    radius = float(np.median(np.linalg.norm(xyz, axis=1)))

    # An absolute TCP pose sits somewhere in the robot's workspace, tens of cm
    # from the base. A per-step delta at 15 fps is millimetres. Two orders of
    # magnitude apart, so the median norm separates them cleanly.
    if radius < 0.02:
        report("G2 absolute actions", FAIL,
               f"median |action xyz| = {radius:.4f} m — these look RELATIVE")
        detail("GR00T composes its own relative actions from absolute ones. Training on")
        detail("an already-relative dataset makes it learn deltas of deltas.")
        detail("Point it at the raw recording, not a lerobot_relative_pose.py output.")
    else:
        report("G2 absolute actions", OK,
               f"median |action xyz| = {radius:.3f} m — absolute workspace poses")

    # rot6d identity is [1,0,0,0,1,0]; relative rotations cluster on it.
    if act.shape[1] >= 9:
        rot = act[:, 3:9]
        identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        near = float(np.mean(np.linalg.norm(rot - identity, axis=1) < 0.05))
        if near > 0.9:
            report("G2 rot6d spread", FAIL,
                   f"{near:.0%} of rotations are within 0.05 of identity")
            detail("A relative-converted dataset looks exactly like this.")
        else:
            report("G2 rot6d spread", OK,
                   f"{1 - near:.0%} of rotations are away from identity")

    # next_tcp_pose with lookahead 1 means action[t] IS state[t+1]. Checking it
    # directly is the strongest available evidence, and it also catches a
    # lookahead or off-by-one that no other check would see.
    if "observation.state" in df.columns and "episode_index" in df.columns:
        state = stack(df["observation.state"])
        ep = df["episode_index"].to_numpy()
        same = ep[:-1] == ep[1:]
        if same.any() and state.shape[1] >= 3:
            err = np.linalg.norm(act[:-1, :3][same] - state[1:, :3][same], axis=1)
            med = float(np.median(err))
            if med < 1e-4:
                report("G2 action == state[t+1]", OK,
                       f"median residual {med:.2e} m — next_tcp_pose, lookahead 1")
            elif med < 0.05:
                report("G2 action == state[t+1]", WARN,
                       f"median residual {med:.4f} m")
                detail("Close but not exact: a lookahead > 1, or a different action")
                detail("definition (e.g. commanded target rather than measured pose).")
            else:
                report("G2 action == state[t+1]", WARN,
                       f"median residual {med:.3f} m — action is not the next measured pose")
                detail("Fine for GR00T as long as actions are ABSOLUTE (G2 above);")
                detail("just means the action is not simply the next state.")


# ── G3: pose layout ───────────────────────────────────────────────────────────


def check_layout(info: dict, names: list[str] | None) -> None:
    """GR00T's relative_eef_to_absolute needs xyz(3) + rot6d(6) as the first 9 dims."""
    feats = info.get("features", {})
    a_shape = feats.get("action", {}).get("shape") or []
    s_shape = feats.get("observation.state", {}).get("shape") or []
    a_dim = int(a_shape[0]) if a_shape else 0
    s_dim = int(s_shape[0]) if s_shape else 0

    if names and len(names) >= 9:
        rot = [n for n in names[3:9] if "rot6d" in str(n).lower()]
        if len(rot) == 6 and [str(n).lower() for n in names[:3]] == ["x", "y", "z"]:
            report("G3 pose layout", OK,
                   f"xyz(3) + rot6d(6) + {a_dim - 9} extra = {a_dim} dims")
        else:
            report("G3 pose layout", FAIL,
                   f"first 9 action dims are not xyz+rot6d: {names[:9]}")
            detail("GR00T's relative_eef_to_absolute reads dims 0:3 as xyz and 3:9 as")
            detail("rot6d. Euler or quaternion poses do not map onto it.")
    else:
        report("G3 pose layout", WARN, f"cannot verify layout from names ({a_dim}-dim action)")

    report("G3 dims", OK if s_dim and a_dim else WARN,
           f"observation.state {s_dim}, action {a_dim}")
    if s_dim and a_dim and s_dim != a_dim:
        detail(f"state and action widths differ ({s_dim} vs {a_dim}). Fine in general,")
        detail("but GR00T's relative composition pairs action dims with STATE dims —")
        detail("confirm the pose block occupies the same indices in both.")


# ── G4/G5/G6: images, language, scale ─────────────────────────────────────────


def check_images(info: dict) -> None:
    """Report camera keys and resolution; the backbone resizes, so this is informational."""
    feats = info.get("features", {})
    cams = {k: v for k, v in feats.items()
            if k.startswith("observation.images.")
            and (v.get("dtype") in {"image", "video"})}
    if not cams:
        report("G4 cameras", FAIL, "no observation.images.* features")
        return
    report("G4 cameras", OK, f"{len(cams)}: {', '.join(sorted(k.split('.')[-1] for k in cams))}")
    for key in sorted(cams):
        shape = cams[key].get("shape")
        detail(f"{key.split('.')[-1]:<12} {shape}  dtype={cams[key].get('dtype')}")
    detail("GR00T's Qwen3-VL image processor resizes internally, so native")
    detail("resolution is not a constraint — but every camera view costs tokens")
    detail("and VRAM, and the deploy env must publish the SAME keys.")


def check_language(root: Path, info: dict) -> None:
    """GR00T is language-conditioned: the task string is a model input."""
    path = root / "meta" / "tasks.parquet"
    if not path.exists():
        report("G5 language", FAIL, "meta/tasks.parquet missing")
        return
    df = pd.read_parquet(path)
    tasks = list(df["task"]) if "task" in df.columns else list(df.index)
    n = info.get("total_tasks")
    status = OK if tasks and all(str(t).strip() for t in tasks) else FAIL
    report("G5 language", status, f"{len(tasks)} task string(s), info.json total_tasks={n}")
    for t in tasks[:5]:
        detail(f"{t!r}")
    if any(str(t).strip().lower() in {"finish the task.", "", "none"} for t in tasks):
        report("G5 task text", WARN, "a placeholder-looking task string is present")
        detail("The sentence is a model INPUT: the checkpoint learns this exact text")
        detail("and you must say it again at deploy. Fix with scripts/fix_dataset_task.py.")


def check_scale(info: dict, chunk: int) -> None:
    """Episode length must exceed the action chunk, and fps sets the latency budget."""
    eps = info.get("total_episodes") or 0
    frames = info.get("total_frames") or 0
    fps = info.get("fps") or 0
    report("G6 scale", OK if eps else WARN,
           f"{eps} episodes, {frames} frames, fps={fps}")
    if eps and frames:
        mean_len = frames / eps
        detail(f"mean episode length {mean_len:.0f} frames ({mean_len / fps:.1f} s)" if fps
               else f"mean episode length {mean_len:.0f} frames")
        if mean_len < chunk:
            report("G6 episode vs chunk", FAIL,
                   f"mean episode {mean_len:.0f} frames < chunk_size {chunk}")
            detail("Most episodes cannot supply a full action chunk; every sample would")
            detail("be padded. Lower --policy.chunk_size or record longer episodes.")
        elif mean_len < chunk * 3:
            report("G6 episode vs chunk", WARN,
                   f"mean episode {mean_len:.0f} frames is only {mean_len / chunk:.1f}x chunk_size")
    if fps:
        report("G6 chunk period", OK,
               f"{chunk} steps / {fps} fps = {1000 * chunk / fps:.0f} ms per chunk")
        detail("Inference must finish inside this, or enable async_inference.")


def check_nans(df: pd.DataFrame) -> None:
    """A NaN anywhere in state or action poisons the normalization statistics."""
    for col in ("observation.state", "action"):
        if col not in df.columns:
            continue
        arr = stack(df[col])
        bad = int(np.count_nonzero(~np.isfinite(arr)))
        report(f"G7 {col} finite", OK if bad == 0 else FAIL,
               "no NaN/Inf" if bad == 0 else f"{bad} non-finite values")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Dataset root holding meta/ and data/")
    ap.add_argument("--exclude", nargs="*", default=["gripper"],
                    help="relative_exclude_joints tokens to verify (default: gripper)")
    ap.add_argument("--chunk-size", type=int, default=40,
                    help="GR00T action chunk (default: 40, the N1.7 native horizon)")
    ap.add_argument("--samples", type=int, default=20000,
                    help="Max frames to read for the numeric checks (default: 20000)")
    args = ap.parse_args()

    try:
        info = load_info(args.root)
    except SystemExit as exc:
        print(exc)
        sys.exit(2)

    head(f"GR00T N1.7 dataset preflight — {args.root}")
    print(f"  robot_type={info.get('robot_type')}  codebase={info.get('codebase_version')}\n")

    names = check_names(info, args.exclude)
    check_layout(info, names)
    check_images(info)
    check_language(args.root, info)
    check_scale(info, args.chunk_size)

    df = read_frames(args.root, ["observation.state", "action", "episode_index"], args.samples)
    print(f"\n  (numeric checks on {len(df)} frames)")
    check_absolute(args.root, df, names)
    check_nans(df)

    head("verdict")
    fails = [c for c, s, _ in _results if s == FAIL]
    warns = [c for c, s, _ in _results if s == WARN]
    if fails:
        print(f"  NOT READY — {len(fails)} failed: {', '.join(fails)}")
    else:
        print("  READY to fine-tune GR00T N1.7 on this dataset.")
    if warns:
        print(f"  {len(warns)} warning(s): {', '.join(warns)}")
    print()
    print("  Remember the two flags — neither default is right for rot6d poses:")
    print("    --policy.use_relative_actions=true")
    print(f"    --policy.relative_exclude_joints='{json.dumps(args.exclude)}'")
    print(BAR)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
