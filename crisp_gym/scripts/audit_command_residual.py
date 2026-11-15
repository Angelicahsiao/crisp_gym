r"""Audit the COMMAND-minus-MEASURED residual of a Cartesian-teleop dataset.

Read-only. Answers, from data you already recorded, the questions that decide
whether a tactile "residual trajectory correction" model is worth building:

  A. How big is the residual  T_res = T_tcp^-1 @ T_cmd  in contact vs free
     motion?  (If the contact residual is a few mm, there is little to learn.)
  B. How much of the FREE-motion residual is plain tracking lag
     (residual ~ tau * velocity)?
  C. System identification: what stiffness/damping does the Cartesian impedance
     controller actually deliver?  Fits  F_ext = -K r + D v  per TCP axis, with
     F_ext rebuilt from the recorded external joint torques. Compare against the
     nominal gains — the nominal value is whatever the controller was launched
     with, which is NOT always what an env/control YAML in this repo says.
  D. Can the tactile signal predict the residual?  A linear, single-frame ridge
     probe scored on HELD-OUT EPISODES, next to a velocity-only baseline and
     (when available) an external-wrench reference. A linear probe is a LOWER
     bound on what a real model will reach.
  E. The same residual statistics for the gripper channel, when the dataset
     carries a gripper command column.
  F. How different are the two ACTION LABELS a policy can be trained on — the
     measured pose a few frames ahead (`next_tcp_pose`, what `action` stores)
     and the command? The gap between command[t] and pose[t + L] for L = 0..8
     frames, in free motion and in contact. A lookahead that closes the gap in
     free motion says how much of the difference is plain lag; what is left in
     contact at that lookahead is the part only the command carries.

The residual is expressed in the TCP (body) frame, like the relative-pose
convention in lerobot_relative_pose.py, so it is independent of the world frame
and comparable between the robot and a handheld device.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEPS A-E ARE ONLY MEANINGFUL FOR CARTESIAN-DRIVEN ROBOT DATA (pose -> CIC).
The script refuses, like train_action_from_target_cartesian.py does, when the
command is not a command:
  * HANDHELD data: no robot, so extra.target_cartesian / extra.joints are
    placeholders (all zeros, never moving);
  * joint-driven data (FACTR -> JIC): extra.target_cartesian is constant.
For those, --tactile-only reports the tactile signal alone (no command needed):
baseline, noise, dead taxels, contact share and, with the high-rate CSVs, the
rate and gaps of the stream. Run it on the handheld AND the robot dataset and
compare — a model trained on one only transfers if these agree.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Columns (each overridable; a bare name also matches a unique dotted suffix, so
`left_tactile` finds `extra.left_tactile` or `observation.state.left_tactile`):

  observation.state.cartesian   measured TCP pose, 9-D pos + rot6d   REQUIRED
  extra.target_cartesian        commanded TCP pose, 9-D              REQUIRED (A-E)
  tactile skins, ANY shape      extra.tactile.left / .right, or      optional
                                left_tactile / right_tactile
  <tactile>.sync_valid          per-frame flag; false frames are     optional
                                left out of steps C, D and the tactile report
  <tactile>.source_stamp_ns     stamp of the sample used, joins the  optional
                                frame to <dataset>/tactile_raw/*.csv
  extra.ext_torque              external joint torque, 7 (Franka)    optional (C)
  extra.joints                  joint positions, 7                   optional (C)
  observation.state.gripper     measured gripper                     optional (E)
  extra.gripper_target          commanded gripper                    optional (E)

Step C needs the arm kinematics to turn joint torques into a TCP wrench. Only
the Franka FR3/Panda arm is built in (DH parameters); the Jacobian is taken at
the RECORDED TCP position, so no tool/gripper model is needed. The script checks
that recorded TCP and DH flange stay a rigid distance apart and skips C if they
do not (wrong arm, wrong base frame, or joints/pose out of sync).

Runs on the GPU PC or anywhere with numpy + pandas + pyarrow (+ pyyaml for
--control-config). ROS-free, crisp-import-free, lerobot-free. Works on both the
v2.x (per-episode parquet) and v3.0 (file-*.parquet) layouts.

Usage:

    python audit_command_residual.py /path/to/dataset \
        --k-pos 400 --k-rot 30 \
        --out audit.json --dump-npz audit_frames.npz

    # gains from a YAML instead (crisp_gym control YAML with flat "task.k_pos_x"
    # keys, or a ros2_control controllers.yaml):
    python audit_command_residual.py /path/to/dataset \
        --control-config .../config/fr3/controllers.yaml \
        --controller cartesian_impedance_controller

    # handheld (or any) dataset: the tactile signal alone
    python audit_command_residual.py /path/to/dataset --tactile-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

POSE_KEY = "observation.state.cartesian"
TARGET_KEY = "extra.target_cartesian"
# Tried in order, per sensor, when --tactile-keys is not given. The first is
# the layout written by the handheld recorder (extra.tactile.left + sync columns).
TACTILE_KEY_CANDIDATES = (("tactile.left", "left_tactile"), ("tactile.right", "right_tactile"))
SYNC_VALID_SUFFIX = ".sync_valid"
SOURCE_STAMP_SUFFIX = ".source_stamp_ns"
RAW_DIR_NAME = "tactile_raw"
EXT_TORQUE_KEY = "extra.ext_torque"
JOINTS_KEY = "extra.joints"
GRIPPER_KEY = "observation.state.gripper"
GRIPPER_TARGET_KEY = "extra.gripper_target"

AXES = ("x", "y", "z")


# ── rot6d / SE(3) (local copy BY DESIGN — this file stays crisp-import-free; ──
# ── the convention is pinned by tests/test_pose_math.py)                     ──

def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """[..., 6] rot6d (first two ROWS of R) -> [..., 3, 3] via Gram-Schmidt."""
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def pose9d_to_mat(pose: np.ndarray) -> np.ndarray:
    """[..., 9] (pos + rot6d) -> [..., 4, 4]."""
    pose = np.asarray(pose, dtype=np.float64)
    T = np.tile(np.eye(4), (*pose.shape[:-1], 1, 1))
    T[..., :3, :3] = rot6d_to_mat(pose[..., 3:9])
    T[..., :3, 3] = pose[..., :3]
    return T


def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """[..., 3, 3] -> [..., 3] rotation vector (axis * angle, rad).

    Accurate for the small rotations this script deals with (residuals and
    per-step motion). Near 180 deg the axis is taken from the diagonal instead
    of the vanishing skew part.
    """
    R = np.asarray(R, dtype=np.float64)
    skew = np.stack(
        (R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]),
        axis=-1,
    )
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    sin = np.linalg.norm(skew, axis=-1) / 2.0
    # atan2, not arccos: arccos loses half its digits near 0, which is exactly
    # where the residuals and per-step rotations live.
    angle = np.arctan2(sin, cos)
    # angle / (2 sin(angle)) -> 1/2 as angle -> 0
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(sin > 1e-8, angle / (2.0 * sin), 0.5)
    out = skew * scale[..., None]

    near_pi = (angle > np.pi - 1e-4) & (sin <= 1e-8)
    if np.any(near_pi):
        diag = np.stack((R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]), axis=-1)
        axis = np.sqrt(np.clip((diag + 1.0) / 2.0, 0.0, None))
        out = np.where(near_pi[..., None], axis * angle[..., None], out)
    return out


def tcp_frame_residual(T_tcp: np.ndarray, T_cmd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T_res = T_tcp^-1 @ T_cmd as (translation [N,3] m, rotvec [N,3] rad).

    Body-frame, hence unchanged by any rigid change of the world frame — the
    property that lets a model trained on robot data label handheld data.
    """
    R = T_tcp[..., :3, :3]
    Rt = np.swapaxes(R, -1, -2)
    pos = np.einsum("...ij,...j->...i", Rt, T_cmd[..., :3, 3] - T_tcp[..., :3, 3])
    rot = mat_to_rotvec(Rt @ T_cmd[..., :3, :3])
    return pos, rot


def body_velocity(T: np.ndarray, episode: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """TCP twist in the TCP frame by central differences WITHIN each episode.

    Returns (linear [N,3] m/s, angular [N,3] rad/s). Episode ends use a
    one-sided difference; a single-frame episode gets zeros.
    """
    n = len(T)
    lin = np.zeros((n, 3))
    ang = np.zeros((n, 3))
    for ep in np.unique(episode):
        idx = np.flatnonzero(episode == ep)
        if len(idx) < 2:
            continue
        prev = np.concatenate(([idx[0]], idx[:-1]))
        nxt = np.concatenate((idx[1:], [idx[-1]]))
        span = (nxt - prev).astype(np.float64) * dt  # dt at the ends, 2 dt inside
        Rt = np.swapaxes(T[idx, :3, :3], -1, -2)
        dp = T[nxt, :3, 3] - T[prev, :3, 3]
        lin[idx] = np.einsum("nij,nj->ni", Rt, dp) / span[:, None]
        # rotation prev -> next, expressed in the frame at `idx` (first order)
        dR = np.swapaxes(T[prev, :3, :3], -1, -2) @ T[nxt, :3, :3]
        ang[idx] = mat_to_rotvec(dR) / span[:, None]
    return lin, ang


# ── Franka FR3 / Panda kinematics (modified DH, Franka's published table) ────
# row i: (a_{i-1}, d_i, alpha_{i-1}); the last row is the fixed flange offset.
_FR3_DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
_FR3_FLANGE_D = 0.107


def _dh(a: float, d: float, alpha: float, theta: np.ndarray) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.zeros((*theta.shape, 4, 4))
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 3] = ct, -st, a
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2], T[..., 1, 3] = st * ca, ct * ca, -sa, -d * sa
    T[..., 2, 0], T[..., 2, 1], T[..., 2, 2], T[..., 2, 3] = st * sa, ct * sa, ca, d * ca
    T[..., 3, 3] = 1.0
    return T


def fr3_joint_frames(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint frames [N,7,4,4] and the flange [N,4,4] in the base frame, for q [N,7]."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 7:
        raise ValueError(f"FR3 kinematics need 7 joints, got {q.shape[-1]}")
    frames = []
    T = np.tile(np.eye(4), (len(q), 1, 1))
    for i, (a, d, alpha) in enumerate(_FR3_DH):
        T = T @ _dh(a, d, alpha, q[:, i])
        frames.append(T)
    flange = T.copy()
    flange[:, :3, 3] += T[:, :3, 2] * _FR3_FLANGE_D
    return np.stack(frames, axis=1), flange


def fr3_jacobian_at(q: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Geometric Jacobian [N,6,7] (linear rows first) of a point on the last link.

    The point is given in the base frame. Using the RECORDED TCP position as
    the point means no tool/gripper model is required.
    """
    frames, _ = fr3_joint_frames(q)
    z = frames[:, :, :3, 2]            # [N,7,3] joint axes
    o = frames[:, :, :3, 3]            # [N,7,3] joint origins
    lin = np.cross(z, point[:, None, :] - o)
    return np.concatenate((np.swapaxes(lin, 1, 2), np.swapaxes(z, 1, 2)), axis=1)


def tcp_wrench_from_joint_torque(
    q: np.ndarray, tau: np.ndarray, T_tcp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares external wrench AT the TCP, in the TCP frame.

    Solves J^T w = tau per frame (7 equations, 6 unknowns). Returns
    (wrench [N,6] = force N then moment Nm, sigma_min [N] of the Jacobian) —
    frames with a small sigma_min are near a singularity and their wrench is
    noise-amplified; the caller drops them.
    """
    J = fr3_jacobian_at(q, T_tcp[:, :3, 3])
    JT = np.swapaxes(J, 1, 2)                                   # [N,7,6]
    w_base = np.einsum("nij,nj->ni", np.linalg.pinv(JT), np.asarray(tau, dtype=np.float64))
    sigma_min = np.linalg.svd(J, compute_uv=False)[:, -1]
    Rt = np.swapaxes(T_tcp[:, :3, :3], -1, -2)
    w_tcp = np.concatenate(
        (np.einsum("nij,nj->ni", Rt, w_base[:, :3]), np.einsum("nij,nj->ni", Rt, w_base[:, 3:])),
        axis=1,
    )
    return w_tcp, sigma_min


def tool_offset_check(q: np.ndarray, T_tcp: np.ndarray) -> dict:
    """Check that the recorded TCP is rigidly attached to the DH flange."""
    _, flange = fr3_joint_frames(q)
    rel = np.linalg.inv(flange) @ T_tcp
    dist = np.linalg.norm(rel[:, :3, 3], axis=-1)
    return {
        "tool_offset_mean_m": float(np.mean(dist)),
        "tool_offset_std_m": float(np.std(rel[:, :3, 3], axis=0).max()),
    }


# ── small statistics helpers ─────────────────────────────────────────────────

def _percentiles(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0}
    p50, p90, p99 = np.percentile(x, [50, 90, 99])
    return {"n": int(len(x)), "p50": float(p50), "p90": float(p90), "p99": float(p99),
            "max": float(np.max(x))}


def fit_linear(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Ordinary least squares with an intercept. Returns (coef incl. bias, R2)."""
    A = np.column_stack((X, np.ones(len(X))))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else float("nan")
    return coef, r2


def ridge_cv(
    X: np.ndarray, Y: np.ndarray, groups: np.ndarray, eval_mask: np.ndarray,
    lam: float = 1.0, folds: int = 5,
) -> dict | None:
    """Ridge regression scored on HELD-OUT GROUPS (episodes).

    Frames of one episode are strongly correlated, so a random frame split
    would report memorization. Features are standardized on the training fold.
    R2 is out-of-sample: 1 - SSE / SST with SST around the TRAINING mean, per
    target column, pooled over folds and restricted to `eval_mask` frames.
    Returns None when there are fewer than 2 groups.
    """
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return None
    folds = min(folds, len(uniq))
    fold_of = {g: i % folds for i, g in enumerate(uniq)}
    fold = np.array([fold_of[g] for g in groups])

    sse = np.zeros(Y.shape[1])
    sst = np.zeros(Y.shape[1])
    n_eval = 0
    for f in range(folds):
        tr, te = fold != f, (fold == f) & eval_mask
        if not te.any() or not tr.any():
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ymu = Y[tr].mean(0)
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(X.shape[1]), Xtr.T @ (Y[tr] - ymu))
        pred = Xte @ W + ymu
        sse += np.sum((Y[te] - pred) ** 2, axis=0)
        sst += np.sum((Y[te] - ymu) ** 2, axis=0)
        n_eval += int(te.sum())
    if n_eval == 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(sst > 0, 1.0 - sse / sst, np.nan)
    return {"n_eval": n_eval, "r2": r2.tolist(), "rmse": np.sqrt(sse / n_eval).tolist()}


# ── the analysis (pure arrays in, plain dict out — unit-tested) ──────────────

def lookahead_gap(
    T_tcp: np.ndarray, T_cmd: np.ndarray, episode: np.ndarray, contact: np.ndarray,
    fps: float, max_lookahead: int = 8,
) -> list[dict]:
    """Gap between command[t] and the measured pose L frames later, per L.

    `pose[t + L]` is the measured-pose action label with lookahead L; the
    command is the other label. Pairs never cross an episode boundary. The
    contact/free split is taken at the command's frame t.
    """
    episode = np.asarray(episode)
    n = len(episode)
    rows = []
    for L in range(max_lookahead + 1):
        s = np.arange(n - L)
        s = s[episode[s] == episode[s + L]]
        if len(s) == 0:
            break
        pos, rot = tcp_frame_residual(T_tcp[s + L], T_cmd[s])
        pos_mm = np.linalg.norm(pos, axis=1) * 1e3
        rot_deg = np.degrees(np.linalg.norm(rot, axis=1))
        c = np.asarray(contact, bool)[s]
        rows.append({
            "lookahead": L, "ms": 1e3 * L / float(fps),
            "pos_mm": {"free": _percentiles(pos_mm[~c]), "contact": _percentiles(pos_mm[c])},
            "rot_deg": {"free": _percentiles(rot_deg[~c]), "contact": _percentiles(rot_deg[c])},
        })
    return rows


def detect_action_lookahead(
    action: np.ndarray, pose9: np.ndarray, episode: np.ndarray, max_lookahead: int = 8,
    tol_m: float = 1e-4,
) -> int | None:
    """The lookahead L with action[t] == pose[t + L], read from the data.

    None when the `action` column is not a measured pose at any tested
    lookahead (a swapped/commanded action, a delta action, another layout).
    """
    action = np.asarray(action, dtype=np.float64)
    pose9 = np.asarray(pose9, dtype=np.float64)
    episode = np.asarray(episode)
    if action.ndim != 2 or action.shape[1] < 3:
        return None
    best = None
    for L in range(1, max_lookahead + 1):
        s = np.arange(len(episode) - L)
        s = s[episode[s] == episode[s + L]]
        if len(s) == 0:
            break
        err = float(np.median(np.linalg.norm(action[s, :3] - pose9[s + L, :3], axis=1)))
        if err < tol_m and (best is None or err < best[1]):
            best = (L, err)
    return None if best is None else best[0]


class NoCommandError(ValueError):
    """The dataset carries no usable Cartesian command (handheld or joint-driven)."""


def check_command(
    episode: np.ndarray,
    pose9: np.ndarray,
    target9: np.ndarray,
    joints: np.ndarray | None,
    notes: list[str],
) -> None:
    """Refuse a command column that is not a command. Runs BEFORE any pose math.

    Two cases look alike (the command never moves while the TCP does) and mean
    different things, so they are told apart:
      * handheld: no robot at all — the command is a placeholder (all zeros,
        which is not even a valid rot6d) and the joints never move either;
      * joint-driven robot (FACTR -> JIC): the joints move, the Cartesian
        command is simply not what drives the arm.
    """
    episode = np.asarray(episode)
    pose9 = np.asarray(pose9, dtype=np.float64)
    target9 = np.asarray(target9, dtype=np.float64)
    eps = np.unique(episode)

    # all-zero rot6d = a placeholder; pose9d_to_mat would turn it into NaN
    placeholder = np.linalg.norm(target9[:, 3:9], axis=1) < 1e-9
    frozen = []
    for ep in eps:
        m = episode == ep
        moved = np.ptp(pose9[m, :3], axis=0).max()
        if np.ptp(target9[m, :3], axis=0).max() < 1e-6 and moved > 1e-3:
            frozen.append(int(ep))

    if placeholder.all() or len(frozen) == len(eps):
        joints_still = joints is None or all(
            np.ptp(np.asarray(joints, dtype=np.float64)[episode == ep], axis=0).max() < 1e-6
            for ep in eps
        )
        hint = " Use --tactile-only to report the tactile signal of this dataset."
        if placeholder.all() or (joints is not None and joints_still):
            raise NoCommandError(
                "no command was recorded: the command column is a placeholder ("
                + ("all zeros" if placeholder.all() else "constant")
                + ") and "
                + ("there is no joints column" if joints is None else "the joints never move")
                + ". This is HANDHELD data (no robot). Steps A-E need Cartesian-driven "
                "robot teleop data." + hint
            )
        raise NoCommandError(
            "the commanded pose is CONSTANT within every episode while the arm moves: "
            "this is a joint-driven dataset (FACTR -> JIC), where target_cartesian is "
            "not a command. Steps A-E only apply to Cartesian-driven data." + hint
        )
    if placeholder.any():
        bad = sorted({int(e) for e in episode[placeholder]})
        raise NoCommandError(
            f"{len(bad)} of {len(eps)} episodes carry a placeholder command (all-zero "
            f"rotation), e.g. {bad[:10]}: robot and handheld episodes are mixed. Pass "
            "--episodes with the robot episodes only."
        )
    if frozen:
        notes.append(f"{len(frozen)} episode(s) have a frozen command and were kept: {frozen[:10]}")


def remove_baseline(raw: np.ndarray, episode: np.ndarray, baseline_frames: int,
                    valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract each episode's own zero: the mean of its first frames.

    Returns (baseline-removed [N, T], per-episode baselines [E, T], a bool [N]
    marking the frames the baselines were taken from). With `valid`, the first
    `baseline_frames` VALID frames of each episode are used.
    """
    raw = np.asarray(raw, dtype=np.float64)
    out = np.empty_like(raw)
    is_base = np.zeros(len(raw), dtype=bool)
    baselines = []
    for ep in np.unique(episode):
        idx = np.flatnonzero(episode == ep)
        pool = idx if valid is None or not valid[idx].any() else idx[valid[idx]]
        first = pool[:min(baseline_frames, len(pool))]
        base = raw[first].mean(axis=0)
        out[idx] = raw[idx] - base
        is_base[first] = True
        baselines.append(base)
    return out, np.asarray(baselines), is_base


def row_width_hint(feat: np.ndarray, min_frames: int = 50) -> dict | None:
    """Guess the grid width of a skin stored ROW BY ROW, from the data alone.

    Neighbouring taxels are pressed together. In a row-major array of width W
    the taxel directly below taxel i is i + W, so the mean correlation between
    taxels i and i + k peaks at k = W (next to k = 1, the neighbour in the same
    row). Only widths that divide the taxel count are scored. This is a hint to
    confirm by pressing one corner, not a measurement.
    """
    feat = np.asarray(feat, dtype=np.float64)
    n, taxels = feat.shape
    if n < min_frames or (feat.std(axis=0) > 0).sum() < 4:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(feat, rowvar=False)
    scores = {}
    for width in range(2, taxels // 2 + 1):
        if taxels % width:
            continue
        pair = np.diagonal(corr, offset=width)
        pair = pair[np.isfinite(pair)]
        if len(pair):
            scores[width] = float(pair.mean())
    if not scores:
        return None
    best = max(scores, key=scores.get)
    others = [v for k, v in scores.items() if k != best]
    # a guess is only offered when one width clearly stands out
    confident = scores[best] > 0.3 and (not others or scores[best] - max(others) > 0.05)
    same_row = np.diagonal(corr, offset=1)
    same_row = same_row[np.isfinite(same_row)]
    return {"width": int(best), "rows": int(taxels // best), "confident": bool(confident),
            "correlation_by_width": {str(k): round(v, 3) for k, v in scores.items()},
            "correlation_next_index": float(same_row.mean()) if len(same_row) else float("nan")}


def tactile_report(
    episode: np.ndarray,
    tactile: dict[str, np.ndarray],
    *,
    baseline_frames: int = 10,
    tactile_threshold: float | None = None,
    valid: dict[str, np.ndarray] | None = None,
) -> dict:
    """The tactile signal on its own, PER SENSOR. Needs no command and no robot.

    Meant to be run on two datasets (handheld, robot) and compared: taxel count,
    raw level, noise, dead taxels, how often and how hard the skin is in
    contact. All values are in the sensor's own units.
    """
    episode = np.asarray(episode)
    n = len(episode)
    eps = np.unique(episode)
    out: dict = {}
    for name, arr in tactile.items():
        raw = np.asarray(arr, dtype=np.float64).reshape(n, -1)
        ok = np.ones(n, dtype=bool) if not valid or name not in valid else np.asarray(valid[name], bool)
        feat, baselines, is_base = remove_baseline(raw, episode, baseline_frames, ok)
        mag = np.linalg.norm(feat, axis=1)
        base_mag = mag[is_base]
        thr = float(base_mag.mean() + 6.0 * base_mag.std()) if tactile_threshold is None \
            else float(tactile_threshold)
        contact = (mag > thr) & ok
        taxel_noise = feat[is_base].std(axis=0)
        per_ep = np.array([contact[(episode == ep) & ok].mean() if ((episode == ep) & ok).any()
                           else 0.0 for ep in eps])
        # a taxel counts as active above 6 sigma of its own rest noise, and never
        # below an even share of the contact threshold (quantized sensors read an
        # exact 0 noise at rest)
        taxel_thr = np.maximum(6.0 * taxel_noise, thr / np.sqrt(raw.shape[1]))
        active = (np.abs(feat[contact]) > taxel_thr).sum(axis=1)
        entry = {
            "taxels": int(raw.shape[1]),
            "frames": int(n),
            "invalid_frames": int((~ok).sum()),
            "dead_taxels": np.flatnonzero(np.ptp(raw[ok], axis=0) == 0).tolist() if ok.any() else [],
            "raw": {"min": float(raw[ok].min()), "median": float(np.median(raw[ok])),
                    "max": float(raw[ok].max())} if ok.any() else {},
            "baseline_drift": float(np.median(baselines.std(axis=0))) if len(baselines) > 1 else 0.0,
            "taxel_noise": float(np.median(taxel_noise)),
            "noise_floor": float(base_mag.mean()),
            "threshold": thr,
            "contact_fraction": float(contact[ok].mean()) if ok.any() else 0.0,
            "contact_fraction_per_episode": {
                "min": float(per_ep.min()), "median": float(np.median(per_ep)),
                "max": float(per_ep.max())},
            "episodes_without_contact": int((per_ep == 0).sum()),
            "contact_magnitude": _percentiles(mag[contact]),
            "peak_taxel_in_contact": _percentiles(np.abs(feat[contact]).max(axis=1))
            if contact.any() else _percentiles(np.empty(0)),
            "active_taxels_in_contact": _percentiles(active.astype(np.float64)),
        }
        hint = row_width_hint(feat[contact])
        if hint:
            entry["row_width_hint"] = hint
        out[name] = entry
    return out


def analyze(
    *,
    episode: np.ndarray,
    pose9: np.ndarray,
    target9: np.ndarray,
    fps: float,
    tactile: dict[str, np.ndarray] | None = None,
    joints: np.ndarray | None = None,
    ext_torque: np.ndarray | None = None,
    gripper: np.ndarray | None = None,
    gripper_target: np.ndarray | None = None,
    k_pos: float | None = None,
    k_rot: float | None = None,
    baseline_frames: int = 10,
    contact_from: str = "auto",
    tactile_threshold: float | None = None,
    force_threshold: float = 3.0,
    min_sigma: float = 0.03,
    ridge_lambda: float = 1.0,
    folds: int = 5,
    max_tool_offset_std: float = 0.005,
    tactile_valid: np.ndarray | None = None,
    action: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """Run every step. Rows must be sorted by (episode, frame).

    Returns (report dict, per-frame arrays dict for --dump-npz).
    """
    episode = np.asarray(episode)
    n = len(episode)
    dt = 1.0 / float(fps)
    tactile = tactile or {}
    report: dict = {"frames": n, "episodes": int(len(np.unique(episode))), "fps": float(fps)}
    notes: list[str] = []

    check_command(episode, pose9, target9, joints, notes)
    T_tcp = pose9d_to_mat(pose9)
    T_cmd = pose9d_to_mat(target9)

    res_pos, res_rot = tcp_frame_residual(T_tcp, T_cmd)
    vel, omega = body_velocity(T_tcp, episode, dt)
    frames = {"episode": episode, "res_pos": res_pos, "res_rot": res_rot,
              "vel": vel, "omega": omega}

    # ── tactile: per-episode baseline removal, magnitude ────────────────────
    tac_feat = None
    tac_mag = None
    if tactile:
        cols, shapes = [], {}
        for name, arr in tactile.items():
            arr = np.asarray(arr, dtype=np.float64).reshape(n, -1)
            shapes[name] = int(arr.shape[1])
            cols.append(arr)
        raw = np.concatenate(cols, axis=1)
        tac_feat, _, is_base = remove_baseline(raw, episode, baseline_frames, tactile_valid)
        tac_mag = np.linalg.norm(tac_feat, axis=1)
        base_mag = tac_mag[is_base]
        auto_thr = float(base_mag.mean() + 6.0 * base_mag.std())
        report["tactile"] = {
            "taxels": shapes,
            "baseline_frames": baseline_frames,
            "noise_floor": float(base_mag.mean()),
            "auto_threshold": auto_thr,
        }
        frames["tactile_mag"] = tac_mag
        notes.append(
            f"tactile baseline = mean of the first {baseline_frames} frames of each episode; "
            "an episode that STARTS in contact hides that contact."
        )

    # ── external wrench at the TCP (FR3 only) ───────────────────────────────
    wrench = None
    good = np.ones(n, dtype=bool)
    if joints is not None and ext_torque is not None:
        joints = np.asarray(joints, dtype=np.float64)
        ext_torque = np.asarray(ext_torque, dtype=np.float64)
        if joints.shape[1] != 7 or ext_torque.shape[1] != 7:
            notes.append(
                f"system identification skipped: built-in kinematics are FR3/Panda (7 joints), "
                f"got joints {joints.shape[1]} / ext_torque {ext_torque.shape[1]}."
            )
        else:
            chk = tool_offset_check(joints, T_tcp)
            report["kinematics_check"] = chk
            if chk["tool_offset_std_m"] > max_tool_offset_std:
                notes.append(
                    "system identification skipped: recorded TCP is not rigid w.r.t. the FR3 "
                    f"flange (std {chk['tool_offset_std_m'] * 1e3:.1f} mm) — wrong arm, a base "
                    "frame that is not the arm base, or joints/pose out of sync."
                )
            else:
                wrench, sigma = tcp_wrench_from_joint_torque(joints, ext_torque, T_tcp)
                good = sigma > min_sigma
                frames["wrench_tcp"] = wrench
                frames["jacobian_sigma_min"] = sigma
                report["kinematics_check"]["near_singular_frames"] = int((~good).sum())
    elif joints is not None or ext_torque is not None:
        notes.append("system identification skipped: needs BOTH joints and ext_torque.")
    if tactile and tactile_valid is not None and not np.all(tactile_valid):
        bad = int((~np.asarray(tactile_valid, bool)).sum())
        good = good & np.asarray(tactile_valid, bool)
        notes.append(f"{bad} frame(s) with tactile sync_valid = false were left out of "
                     "steps C and D (they still count in A, B and the contact share).")

    # ── contact flag ────────────────────────────────────────────────────────
    source = contact_from
    if source == "auto":
        source = "tactile" if tac_mag is not None else ("wrench" if wrench is not None else "none")
    if source == "tactile":
        if tac_mag is None:
            raise ValueError("--contact-from tactile but no tactile column was found")
        thr = report["tactile"]["auto_threshold"] if tactile_threshold is None else tactile_threshold
        contact = tac_mag > thr
        report["contact"] = {"source": "tactile", "threshold": float(thr)}
    elif source == "wrench":
        if wrench is None:
            raise ValueError("--contact-from wrench but no TCP wrench could be computed")
        contact = (np.linalg.norm(wrench[:, :3], axis=1) > force_threshold) & good
        report["contact"] = {"source": "wrench", "threshold_N": float(force_threshold)}
    else:
        contact = np.zeros(n, dtype=bool)
        report["contact"] = {"source": "none"}
        notes.append("no tactile and no wrench: contact/free split unavailable.")
    report["contact"]["fraction"] = float(contact.mean())
    frames["contact"] = contact
    free = ~contact

    # Does the skin SEE the contacts the arm feels? A skin only covers the
    # finger pads; a load path that bypasses them (gripper body, outer finger)
    # shows up in the joint torques and not in the tactile signal.
    if tac_mag is not None and wrench is not None:
        thr = report["tactile"]["auto_threshold"] if tactile_threshold is None else tactile_threshold
        by_skin = (tac_mag > thr)[good]
        by_wrench = (np.linalg.norm(wrench[:, :3], axis=1) > force_threshold)[good]
        report["contact"]["skin_vs_wrench"] = {
            "force_threshold_N": float(force_threshold),
            "both": float(np.mean(by_skin & by_wrench)),
            "wrench_only": float(np.mean(~by_skin & by_wrench)),
            "skin_only": float(np.mean(by_skin & ~by_wrench)),
        }

    # ── A. residual magnitude ───────────────────────────────────────────────
    pos_mm = np.linalg.norm(res_pos, axis=1) * 1e3
    rot_deg = np.degrees(np.linalg.norm(res_rot, axis=1))
    report["residual"] = {
        "pos_mm": {"all": _percentiles(pos_mm), "free": _percentiles(pos_mm[free]),
                   "contact": _percentiles(pos_mm[contact])},
        "rot_deg": {"all": _percentiles(rot_deg), "free": _percentiles(rot_deg[free]),
                    "contact": _percentiles(rot_deg[contact])},
    }
    if k_pos:
        report["residual"]["nominal_force_N"] = {
            "free": _percentiles(pos_mm[free] * 1e-3 * k_pos),
            "contact": _percentiles(pos_mm[contact] * 1e-3 * k_pos),
        }

    # ── B. tracking lag in free motion: residual_i ~ tau_i * v_i ────────────
    lag = {}
    if free.sum() > 20:
        for i, ax in enumerate(AXES):
            coef, r2 = fit_linear(vel[free, i:i + 1], res_pos[free, i])
            lag[ax] = {"tau_ms": float(coef[0] * 1e3), "r2": float(r2),
                       "unexplained_rms_mm": float(
                           np.std(res_pos[free, i] - coef[0] * vel[free, i] - coef[1]) * 1e3)}
    report["lag_free_motion"] = lag

    # ── C. system identification: F_ext,i = -K_i r_i + D_i v_i + b_i ───────
    if wrench is not None:
        sysid: dict = {"frames_used": int(good.sum()), "translation": {}, "rotation": {}}
        for group, r, v, w, nominal in (
            ("translation", res_pos, vel, wrench[:, :3], k_pos),
            ("rotation", res_rot, omega, wrench[:, 3:], k_rot),
        ):
            for i, ax in enumerate(AXES):
                X = np.column_stack((r[good, i], v[good, i]))
                coef, r2 = fit_linear(X, w[good, i])
                entry = {"stiffness": float(-coef[0]), "damping": float(coef[1]),
                         "bias": float(coef[2]), "r2": float(r2)}
                if nominal:
                    entry["stiffness_vs_nominal"] = float(-coef[0] / nominal)
                sysid[group][ax] = entry
        ks = [sysid["translation"][ax]["stiffness"] for ax in AXES]
        if all(k < 0 for k in ks):
            notes.append(
                "identified stiffness is NEGATIVE on every axis: the torque column uses the "
                "opposite sign convention (torque exerted BY the robot). Read |stiffness|."
            )
        report["system_identification"] = sysid
        notes.append(
            "stiffness is identified from steady contact and is reliable at 15 Hz; the damping "
            "term is coarse at this rate and absorbs unmodelled inertia and friction."
        )

    # ── D. can tactile predict the residual? (held-out episodes) ────────────
    target = np.column_stack((res_pos, res_rot))
    twist = np.column_stack((vel, omega))
    feature_sets: dict[str, np.ndarray] = {"velocity": twist}
    if tac_feat is not None:
        feature_sets["tactile"] = tac_feat
        feature_sets["tactile+velocity"] = np.column_stack((tac_feat, twist))
    if wrench is not None:
        feature_sets["wrench+velocity"] = np.column_stack((wrench, twist))
    probe: dict = {}
    eval_sets = {"contact": contact & good, "all": good}
    for fname, X in feature_sets.items():
        probe[fname] = {}
        for ename, mask in eval_sets.items():
            if mask.sum() < 20:
                continue
            keep = good  # never train on near-singular wrench frames
            out = ridge_cv(X[keep], target[keep], episode[keep], mask[keep],
                           lam=ridge_lambda, folds=folds)
            if out is not None:
                probe[fname][ename] = {
                    "n_eval": out["n_eval"],
                    "r2_pos": out["r2"][:3], "r2_rot": out["r2"][3:],
                    "rmse_pos_mm": [v * 1e3 for v in out["rmse"][:3]],
                    "rmse_rot_deg": [float(np.degrees(v)) for v in out["rmse"][3:]],
                }
    report["residual_probe"] = probe
    if report["episodes"] < 2:
        notes.append("residual probe skipped: needs at least 2 episodes to hold one out.")

    # ── E. gripper channel ──────────────────────────────────────────────────
    if gripper is not None and gripper_target is not None:
        g = np.asarray(gripper, dtype=np.float64).reshape(n, -1)[:, 0]
        gt = np.asarray(gripper_target, dtype=np.float64).reshape(n, -1)[:, 0]
        gres = gt - g                       # < 0 : commanded more closed than measured
        entry = {"abs": {"free": _percentiles(np.abs(gres[free])),
                         "contact": _percentiles(np.abs(gres[contact]))},
                 "squeezing_fraction_in_contact":
                     float(np.mean(gres[contact] < -0.02)) if contact.any() else float("nan")}
        if tac_feat is not None:
            out = ridge_cv(tac_feat, gres[:, None], episode, np.ones(n, dtype=bool),
                           lam=ridge_lambda, folds=folds)
            if out is not None:
                entry["tactile_probe_r2"] = out["r2"][0]
        report["gripper_residual"] = entry
        frames["gripper_residual"] = gres

    # ── F. command vs the measured-pose action label ────────────────────────
    rows = lookahead_gap(T_tcp, T_cmd, episode, contact, fps)
    labels: dict = {"by_lookahead": rows}
    scored = [r for r in rows if r["pos_mm"]["free"].get("n")]
    if scored:
        best = min(scored, key=lambda r: r["pos_mm"]["free"]["p50"])
        labels["best_free_lookahead"] = best["lookahead"]
    if action is not None:
        labels["action_lookahead"] = detect_action_lookahead(action, pose9, episode)
        if labels["action_lookahead"] is None:
            notes.append(
                "the `action` column is not the measured pose at any lookahead up to 8 frames "
                "(already swapped to the command, a delta action, or another layout): step F "
                "cannot mark which row is the recorded action.")
    report["action_labels"] = labels

    report["notes"] = notes
    return report, frames


# ── dataset / config I/O ─────────────────────────────────────────────────────

def resolve_key(columns: list[str], key: str, *, required: bool) -> str | None:
    """Exact column name, else a UNIQUE dotted-suffix match."""
    if key in columns:
        return key
    hits = [c for c in columns if c.endswith("." + key)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise KeyError(f"'{key}' is ambiguous, matches {hits}; pass the full column name.")
    if required:
        raise KeyError(f"required column '{key}' not found. Columns: {sorted(columns)}")
    return None


def _to_array(series: "pd.Series") -> np.ndarray:
    """A parquet list/nested-list column -> [N, prod(shape)] float64."""
    return np.asarray(
        [np.asarray(v.tolist() if hasattr(v, "tolist") else v, dtype=np.float64).reshape(-1)
         for v in series.to_numpy()]
    )


def _data_files(root: Path) -> list[Path]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'} — not a LeRobot dataset?")
    return files


def dataset_columns(root: Path) -> list[str]:
    """Column names of the data parquets (read from the first file's schema)."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(_data_files(root)[0]).schema_arrow.names


def default_tactile_keys(columns: list[str]) -> list[str]:
    """The tactile columns present, per sensor, from TACTILE_KEY_CANDIDATES."""
    found = []
    for candidates in TACTILE_KEY_CANDIDATES:
        for key in candidates:
            col = resolve_key(columns, key, required=False)
            if col:
                found.append(col)
                break
    return found


def _first_scalar(series: "pd.Series", dtype: type) -> np.ndarray:
    """A scalar or length-1 list column -> [N] of `dtype`, without a float detour.

    An int64 nanosecond stamp does not survive float64 exactly.
    """
    return np.asarray([np.asarray(v).reshape(-1)[0] for v in series.to_numpy()], dtype=dtype)


def load_dataset(
    root: Path, wanted: dict[str, tuple[str, bool]], episodes: list[int] | None
) -> tuple[dict, dict, float, dict]:
    """Read only the needed low-dim columns from every data parquet.

    wanted: {role: (key, required)}. Returns (arrays by role, resolved column
    names, fps, the declared features from meta/info.json). Roles starting with
    `stamp:` are returned as int64 [N], roles starting with `valid:` as bool [N].
    """
    import pandas as pd

    files = _data_files(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    columns = dataset_columns(root)
    resolved = {role: resolve_key(columns, key, required=req) for role, (key, req) in wanted.items()}
    read_cols = ["episode_index", "frame_index"] + [c for c in resolved.values() if c]
    df = pd.concat([pd.read_parquet(f, columns=read_cols) for f in files], ignore_index=True)
    if episodes:
        df = df[df["episode_index"].isin(episodes)]
        if df.empty:
            raise ValueError(f"none of the requested episodes {episodes} exist in the dataset")
    df = df.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)

    arrays = {"episode": df["episode_index"].to_numpy()}
    for role, col in resolved.items():
        if not col:
            continue
        if role.startswith("stamp:"):
            arrays[role] = _first_scalar(df[col], np.int64)
        elif role.startswith("valid:"):
            arrays[role] = _first_scalar(df[col], bool)
        else:
            arrays[role] = _to_array(df[col])
    return arrays, resolved, float(info["fps"]), info.get("features", {})


def _tactile_raw_module():  # noqa: ANN202 - a module object
    """Load crisp_gym/util/tactile_raw.py by PATH.

    This script keeps working without importing the crisp_gym package (GPU PC,
    no ROS).
    """
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "util" / "tactile_raw.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (needed to read the tactile CSVs)")
    spec = importlib.util.spec_from_file_location("tactile_raw", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_stream_report(
    raw_dir: Path,
    episode: np.ndarray,
    tactile: dict[str, np.ndarray],
    stamps: dict[str, np.ndarray],
) -> dict:
    """Report the high-rate CSVs per sensor.

    Rate, gaps, and whether every dataset frame is found in its CSV by stamp
    with the same taxel values.
    """
    tr = _tactile_raw_module()
    episode = np.asarray(episode)
    out: dict = {}
    for name, values in tactile.items():
        side = tr.side_of(name)
        dts, missing, steps = [], [], []
        frames = matched = equal = 0
        n_eps = 0
        for ep in np.unique(episode):
            path = tr.csv_path(raw_dir, int(ep), side)
            if not path.exists():
                missing.append(int(ep))
                continue
            n_eps += 1
            csv_stamps, csv_taxels = tr.load_csv(path)
            dts.append(np.diff(csv_stamps) / 1e6)
            if name not in stamps:
                continue
            m = episode == ep
            row = tr.match_frames(csv_stamps, stamps[name][m])
            hit = row >= 0
            frames += int(m.sum())
            matched += int(hit.sum())
            frame_values = np.asarray(values)[m].reshape(int(m.sum()), -1)
            if hit.any() and csv_taxels.shape[1] == frame_values.shape[1]:
                equal += int(np.isclose(csv_taxels[row[hit]], frame_values[hit], atol=1e-3)
                             .all(axis=1).sum())
            if hit.sum() > 1:
                steps.append(np.diff(row[hit]))
        entry: dict = {"side": side, "episodes_with_csv": n_eps, "episodes_missing_csv": len(missing)}
        if missing:
            entry["missing_examples"] = missing[:10]
        if dts:
            entry.update(tr.interval_stats(np.concatenate(dts)))
        if frames:
            entry["frames_joined_by_stamp"] = matched / frames
            entry["taxels_equal_at_joined_row"] = equal / matched if matched else 0.0
            if steps:
                entry["rows_per_frame_median"] = float(np.median(np.concatenate(steps)))
        out[name] = entry
    return out


def load_gains(path: Path, controller: str) -> tuple[float | None, float | None]:
    """Read nominal (k_pos, k_rot) from a YAML.

    Accepts a crisp_gym control YAML (flat 'task.k_pos_x' keys) or a
    ros2_control controllers.yaml (nested under <controller>/ros__parameters).
    Returns the mean over x/y/z and warns when the axes differ.
    """
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}

    def find(node: object, name: str) -> object:
        if isinstance(node, dict):
            if name in node:
                return node[name]
            for v in node.values():
                hit = find(v, name)
                if hit is not None:
                    return hit
        return None

    flat = {k: v for k, v in data.items() if isinstance(k, str) and k.startswith("task.")}
    if flat:
        task = {k.split(".", 1)[1]: v for k, v in flat.items()}
    else:
        ctrl = find(data, controller)
        if ctrl is None:
            raise KeyError(f"controller '{controller}' not found in {path}")
        task = find(ctrl, "task") or {}

    def mean(prefix: str) -> float | None:
        vals = [float(task[f"{prefix}_{ax}"]) for ax in AXES if f"{prefix}_{ax}" in task]
        if not vals:
            return None
        if max(vals) - min(vals) > 1e-9:
            logger.warning("%s differs per axis %s; using the mean for nominal comparisons",
                           prefix, vals)
        return float(np.mean(vals))

    return mean("k_pos"), mean("k_rot")


ENV_GAINS_KEY = "cartesian_control_param_config"


def config_search_paths(extra: list[Path] | None = None) -> list[Path]:
    """Directories a config name like `control/x.yaml` is looked up in.

    The SAME order as crisp_gym.config.path.CRISP_CONFIG_PATHS (`find_config`
    loads the first match), rebuilt here so this script needs neither crisp_gym
    nor crisp_py imported: crisp_py's config dir, then crisp_gym's; with
    CRISP_CONFIG_PATH set, its directories first, then crisp_gym's, then
    crisp_py's. `extra` directories (--config-path) go in front of everything.
    """
    import importlib.util

    gym = Path(__file__).resolve().parents[1] / "config"
    py = None
    try:
        spec = importlib.util.find_spec("crisp_py")       # locates, does not import
        if spec and spec.submodule_search_locations:
            py = Path(list(spec.submodule_search_locations)[0]) / "config"
    except (ImportError, ValueError):
        pass
    if py is None:                                        # a checkout next to crisp_gym
        py = Path(__file__).resolve().parents[3] / "crisp_py" / "crisp_py" / "config"
    user = [Path(part.strip()) for part in os.environ.get("CRISP_CONFIG_PATH", "").split(":")
            if part.strip()]
    paths = list(extra or []) + (user + [gym, py] if user else [py, gym])
    return [path for path in paths if path.is_dir()]


def control_file_from_env_config(
    env_yaml: Path, search: list[Path] | None = None
) -> tuple[Path | None, str]:
    """The control YAML an env config pushes to the Cartesian controller.

    Returns (path or None, a sentence saying where the gains come from). None
    means the env does not set the gains: the controller then keeps whatever
    the robot bring-up started it with, and no file in this repo knows that.
    """
    import yaml

    search = config_search_paths() if search is None else search
    data = yaml.safe_load(Path(env_yaml).read_text()) or {}
    name = Path(env_yaml).name
    if ENV_GAINS_KEY not in data:
        return None, (
            f"{name} has no '{ENV_GAINS_KEY}' key, so the env CLASS default applies (Franka: "
            "control/default_cartesian_impedance.yaml). Pass that file with --control-config."
        )
    value = data[ENV_GAINS_KEY]
    if value is None:
        return None, (
            f"{name} sets {ENV_GAINS_KEY}: null — the controller keeps the gains of the robot "
            "bring-up. Pass the bring-up's controllers.yaml with --control-config, or "
            "--k-pos/--k-rot."
        )
    matches = [d / str(value) for d in search if (d / str(value)).is_file()]
    if not matches and Path(str(value)).is_file():
        matches = [Path(str(value))]
    if not matches:
        return None, (
            f"{name} names '{value}', which is in none of {[str(d) for d in search]}. "
            "Add its directory with --config-path, or pass the file with --control-config."
        )
    note = (f"nominal gains read from {matches[0]} (named by {name}). The copy on the PC that "
            "RECORDED the data is the one that ran; check it is the same file.")
    if len(matches) > 1:
        note += (" SHADOWED: the same name also exists in "
                 + ", ".join(str(m.parent.parent) for m in matches[1:])
                 + " and was ignored, as find_config does.")
    return matches[0], note


# ── report ───────────────────────────────────────────────────────────────────

def _fmt_p(d: dict, unit: str) -> str:
    if not d.get("n"):
        return "n=0"
    return (f"n={d['n']:<7d} median {d['p50']:7.2f}  p90 {d['p90']:7.2f}  "
            f"p99 {d['p99']:7.2f}  max {d['max']:7.2f} {unit}")


def format_report(r: dict) -> str:
    """Render the report dict as the text printed to the terminal."""
    out = []
    add = out.append
    add(f"Dataset: {r['episodes']} episodes, {r['frames']} frames at {r['fps']:g} Hz")
    if "columns" in r:
        for role, col in r["columns"].items():
            add(f"  {role:<16s} <- {col if col else '(not found)'}")
    g = r.get("nominal_gains") or {}
    if g.get("k_pos") is not None or g.get("k_rot") is not None:
        add(f"Nominal gains: k_pos {g.get('k_pos')} N/m, k_rot {g.get('k_rot')} Nm/rad"
            f"  ({g.get('source', 'given')})")
    c = r["contact"]
    add(f"Contact flag: {c['source']}, {100 * c.get('fraction', 0):.1f}% of frames in contact")
    if "tactile" in r:
        t = r["tactile"]
        add(f"  tactile taxels {t['taxels']}, noise floor {t['noise_floor']:.4g}, "
            f"auto threshold {t['auto_threshold']:.4g}")
    if "skin_vs_wrench" in c:
        s = c["skin_vs_wrench"]
        add(f"  skin vs joint-torque contact (>{s['force_threshold_N']:g} N): "
            f"both {100 * s['both']:.1f}%, torque only {100 * s['wrench_only']:.1f}% "
            f"(contact the skin does not see), skin only {100 * s['skin_only']:.1f}%")

    add("\nA. Residual  T_tcp^-1 @ T_cmd  (TCP frame)")
    for label, key, unit in (("translation", "pos_mm", "mm"), ("rotation", "rot_deg", "deg")):
        for split in ("free", "contact"):
            add(f"  {label:<11s} {split:<8s} {_fmt_p(r['residual'][key][split], unit)}")
    if "nominal_force_N" in r["residual"]:
        for split in ("free", "contact"):
            add(f"  k_pos*|res|  {split:<8s} {_fmt_p(r['residual']['nominal_force_N'][split], 'N')}")

    add("\nB. Tracking lag in free motion (residual ~ tau * velocity)")
    if not r["lag_free_motion"]:
        add("  (too few free-motion frames)")
    for ax, v in r["lag_free_motion"].items():
        add(f"  {ax}: tau {v['tau_ms']:7.1f} ms   R2 {v['r2']:5.2f}   "
            f"unexplained {v['unexplained_rms_mm']:.2f} mm rms")

    add("\nC. System identification  F_ext = -K r + D v  (TCP frame)")
    s = r.get("system_identification")
    if not s:
        add("  (skipped — see notes)")
    else:
        add(f"  frames used: {s['frames_used']}")
        for group, ku, du in (("translation", "N/m", "Ns/m"), ("rotation", "Nm/rad", "Nms/rad")):
            for ax, v in s[group].items():
                vs = f"  ({100 * v['stiffness_vs_nominal']:.0f}% of nominal)" \
                    if "stiffness_vs_nominal" in v else ""
                add(f"  {group:<11s} {ax}: K {v['stiffness']:9.1f} {ku:<6s} D {v['damping']:8.1f} "
                    f"{du:<7s} R2 {v['r2']:5.2f}{vs}")

    add("\nD. Predicting the residual on HELD-OUT episodes (linear ridge, single frame)")
    if not r["residual_probe"]:
        add("  (skipped — see notes)")
    for fname, per in r["residual_probe"].items():
        for ename, v in per.items():
            add(f"  {fname:<17s} on {ename:<8s} R2 pos {np.round(v['r2_pos'], 2).tolist()} "
                f"rot {np.round(v['r2_rot'], 2).tolist()}  "
                f"rmse {np.round(v['rmse_pos_mm'], 2).tolist()} mm")

    g = r.get("gripper_residual")
    if g:
        add("\nE. Gripper  (command - measured, normalized units)")
        for split in ("free", "contact"):
            add(f"  |residual| {split:<8s} {_fmt_p(g['abs'][split], '')}")
        add(f"  squeezing (command >2% more closed) in {100 * g['squeezing_fraction_in_contact']:.0f}%"
            " of contact frames")
        if "tactile_probe_r2" in g:
            add(f"  tactile -> gripper residual, held-out R2 {g['tactile_probe_r2']:.2f}")

    f = r.get("action_labels")
    if f and f["by_lookahead"]:
        add("\nF. Command vs measured-pose action label  (gap between command[t] and pose[t+L])")
        rec, best = f.get("action_lookahead"), f.get("best_free_lookahead")
        for row in f["by_lookahead"]:
            fr, co = row["pos_mm"]["free"], row["pos_mm"]["contact"]
            cell = lambda d: f"{d['p50']:6.2f} / {d['p90']:6.2f}" if d.get("n") else "     -          "  # noqa: E731
            tags = []
            if row["lookahead"] == rec:
                tags.append("<- the recorded `action`: what the two training targets differ by")
            if row["lookahead"] == best:
                tags.append("<- smallest in free motion")
            add(f"  L={row['lookahead']} ({row['ms']:4.0f} ms)  free {cell(fr)} mm   contact "
                f"{cell(co)} mm   (median / p90)  " + "  ".join(tags))
        if best is not None:
            b = next(row for row in f["by_lookahead"] if row["lookahead"] == best)
            if b["pos_mm"]["contact"].get("n"):
                add(f"  best lookahead L={best}: {b['pos_mm']['free']['p50']:.1f} mm remain in free "
                    f"motion, {b['pos_mm']['contact']['p50']:.1f} mm in contact (median). The contact "
                    "excess is what a measured-pose label does not carry.")

    if r.get("notes"):
        add("\nNotes")
        for note in r["notes"]:
            add(f"  - {note}")
    return "\n".join(out)


def format_tactile_report(r: dict) -> str:
    """Render the --tactile-only report."""
    out = []
    add = out.append
    add(f"Dataset: {r['episodes']} episodes, {r['frames']} frames at {r['fps']:g} Hz")
    add(f"Tactile signal per sensor, in sensor units. Zero = mean of the first "
        f"{r['baseline_frames']} valid frames of each episode.")
    if not r["tactile"]:
        add("  (no tactile column found — pass --tactile-keys)")
    for name, t in r["tactile"].items():
        dead = t["dead_taxels"]
        add(f"\n  {name}: {t['taxels']} taxels, {t['frames']} frames "
            f"({t['invalid_frames']} with sync_valid false), dead taxels: "
            f"{dead if dead else 'none'}")
        if t["raw"]:
            add(f"    raw level       min {t['raw']['min']:.4g}  median {t['raw']['median']:.4g}  "
                f"max {t['raw']['max']:.4g}")
        add(f"    noise           per taxel {t['taxel_noise']:.4g} | norm at rest "
            f"{t['noise_floor']:.4g} -> contact threshold {t['threshold']:.4g}")
        add(f"    baseline drift  {t['baseline_drift']:.4g} between episodes (median taxel)")
        pe = t["contact_fraction_per_episode"]
        add(f"    in contact      {100 * t['contact_fraction']:.1f}% of frames | per episode "
            f"min {100 * pe['min']:.0f}% median {100 * pe['median']:.0f}% max {100 * pe['max']:.0f}%"
            f" | {t['episodes_without_contact']} episode(s) never in contact")
        add(f"    contact norm    {_fmt_p(t['contact_magnitude'], '')}")
        add(f"    strongest taxel {_fmt_p(t['peak_taxel_in_contact'], '')}")
        add(f"    active taxels   {_fmt_p(t['active_taxels_in_contact'], '')}")
        h = t.get("row_width_hint")
        if h and not h["confident"]:
            add("    layout hint     no clear grid structure in the contact frames (too little or "
                "too uniform contact)")
        elif h:
            ranked = sorted(h["correlation_by_width"].items(), key=lambda kv: -kv[1])[:3]
            add(f"    layout hint     if stored row by row: {h['rows']} rows x {h['width']} "
                f"(neighbour correlation by width: "
                + ", ".join(f"{k}: {v:.2f}" for k, v in ranked)
                + f"; next index {h['correlation_next_index']:.2f}) — confirm by pressing a corner")
        s = r.get("raw_stream", {}).get(name)
        if s:
            add(f"    high-rate CSV   '{s['side']}': {s['episodes_with_csv']} episode(s) found, "
                f"{s['episodes_missing_csv']} missing")
            if "dt_ms_median" in s:
                add(f"                    mean {s['mean_rate_hz']:.0f} Hz | interval median "
                    f"{s['dt_ms_median']:.2f} ms  p99 {s['dt_ms_p99']:.2f}  max {s['dt_ms_max']:.2f}"
                    f" | {100 * s['gap_time_fraction']:.0f}% of the time inside gaps (>2x median)")
            if "frames_joined_by_stamp" in s:
                add(f"                    frames joined by stamp {100 * s['frames_joined_by_stamp']:.1f}%"
                    f" | taxels equal at the joined row {100 * s['taxels_equal_at_joined_row']:.1f}%"
                    f" | rows per frame, median {s.get('rows_per_frame_median', float('nan')):.0f}")
    if r.get("notes"):
        add("\nNotes")
        for note in r["notes"]:
            add(f"  - {note}")
    return "\n".join(out)


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("dataset", type=Path, help="LeRobot dataset root (contains data/ and meta/)")
    p.add_argument("--episodes", type=int, nargs="*", help="only these episode indices")
    p.add_argument("--pose-key", default=POSE_KEY)
    p.add_argument("--target-key", default=TARGET_KEY)
    p.add_argument("--tactile-keys", nargs="*", default=None,
                   help="tactile columns, any shape (default: extra.tactile.left/right, "
                        "else left_tactile/right_tactile, whichever the dataset has)")
    p.add_argument("--tactile-only", action="store_true",
                   help="report the tactile signal alone; needs no command, so it also runs on "
                        "handheld and joint-driven data")
    p.add_argument("--tactile-raw-dir", type=Path, default=None,
                   help="directory with the high-rate CSVs episode_<NNNNNN>_<side>.csv "
                        "(default: <dataset>/tactile_raw when it exists)")
    p.add_argument("--tactile-shape", type=int, nargs="+", default=None,
                   help="expected per-sensor shape, e.g. 8 10 — only CHECKED against the data; "
                        "the script itself adapts to whatever shape is stored")
    p.add_argument("--ext-torque-key", default=EXT_TORQUE_KEY)
    p.add_argument("--joints-key", default=JOINTS_KEY)
    p.add_argument("--gripper-key", default=GRIPPER_KEY)
    p.add_argument("--gripper-target-key", default=GRIPPER_TARGET_KEY)
    p.add_argument("--action-key", default="action",
                   help="the recorded action column; only used to mark, in step F, which "
                        "lookahead it was recorded with")
    p.add_argument("--k-pos", type=float, help="nominal translational stiffness, N/m")
    p.add_argument("--k-rot", type=float, help="nominal rotational stiffness, Nm/rad")
    p.add_argument("--control-config", type=Path,
                   help="YAML to read nominal gains from (crisp_gym control YAML or "
                        "ros2_control controllers.yaml); --k-pos/--k-rot override it")
    p.add_argument("--controller", default="cartesian_impedance_controller",
                   help="controller block to read in a controllers.yaml")
    p.add_argument("--env-config", type=Path,
                   help="crisp_gym env YAML the data was recorded with: the gains are read from "
                        "the control file its cartesian_control_param_config names, found the "
                        "way the env finds it. --control-config and --k-pos/--k-rot override it")
    p.add_argument("--config-path", type=Path, action="append", default=None,
                   help="extra config directory searched FIRST for that control file (repeatable); "
                        "CRISP_CONFIG_PATH is honoured too")
    p.add_argument("--baseline-frames", type=int, default=10,
                   help="frames at the start of each episode averaged as the tactile zero")
    p.add_argument("--contact-from", choices=("auto", "tactile", "wrench", "none"), default="auto")
    p.add_argument("--tactile-threshold", type=float,
                   help="contact threshold on the baseline-removed tactile norm (default: auto)")
    p.add_argument("--force-threshold", type=float, default=3.0,
                   help="contact threshold on the TCP force norm, N (--contact-from wrench)")
    p.add_argument("--min-sigma", type=float, default=0.03,
                   help="drop frames whose Jacobian smallest singular value is below this")
    p.add_argument("--ridge-lambda", type=float, default=1.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", type=Path, help="write the full report as JSON")
    p.add_argument("--dump-npz", type=Path,
                   help="write per-frame arrays (residual, velocity, contact, wrench) for plotting")
    return p


def main(argv: list[str] | None = None) -> int:
    """Load the dataset, run the audit, print and optionally save the report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    wanted: dict[str, tuple[str, bool]] = {}
    if not args.tactile_only:
        wanted = {
            "pose9": (args.pose_key, True),
            "target9": (args.target_key, True),
            "joints": (args.joints_key, False),
            "ext_torque": (args.ext_torque_key, False),
            "gripper": (args.gripper_key, False),
            "gripper_target": (args.gripper_target_key, False),
            "action": (args.action_key, False),
        }
    columns = dataset_columns(args.dataset)
    keys = default_tactile_keys(columns) if args.tactile_keys is None else args.tactile_keys
    for key in keys:
        wanted[f"tactile:{key}"] = (key, False)
        col = resolve_key(columns, key, required=False)
        for role, suffix in (("valid", SYNC_VALID_SUFFIX), ("stamp", SOURCE_STAMP_SUFFIX)):
            if col and col + suffix in columns:
                wanted[f"{role}:{key}"] = (col + suffix, False)
    arrays, resolved, fps, features = load_dataset(args.dataset, wanted, args.episodes)

    def by_role(prefix: str) -> dict[str, np.ndarray]:
        return {role.split(":", 1)[1]: arr for role, arr in arrays.items()
                if role.startswith(prefix + ":")}

    tactile, valid, stamps = by_role("tactile"), by_role("valid"), by_role("stamp")
    raw_dir = args.tactile_raw_dir
    if raw_dir is None and (args.dataset / RAW_DIR_NAME).is_dir():
        raw_dir = args.dataset / RAW_DIR_NAME
    if args.tactile_shape:
        want = int(np.prod(args.tactile_shape))
        for name, arr in tactile.items():
            if arr.shape[1] != want:
                raise ValueError(
                    f"--tactile-shape {args.tactile_shape} = {want} values, but '{name}' "
                    f"stores {arr.shape[1]} (declared shape "
                    f"{features.get(resolved['tactile:' + name], {}).get('shape')})."
                )

    k_pos, k_rot = args.k_pos, args.k_rot
    gain_notes: list[str] = []
    source = "--k-pos/--k-rot" if (k_pos is not None or k_rot is not None) else None
    control_config = args.control_config
    if args.env_config:
        found, note = control_file_from_env_config(
            args.env_config, config_search_paths(args.config_path))
        if control_config is None:
            control_config = found
            gain_notes.append(note)
    if control_config and (k_pos is None or k_rot is None):
        cfg_pos, cfg_rot = load_gains(control_config, args.controller)
        k_pos = k_pos if k_pos is not None else cfg_pos
        k_rot = k_rot if k_rot is not None else cfg_rot
        source = f"{source} + {control_config}" if source else str(control_config)

    if args.tactile_only:
        episode = arrays["episode"]
        report = {
            "frames": int(len(episode)), "episodes": int(len(np.unique(episode))), "fps": fps,
            "baseline_frames": args.baseline_frames, "columns": resolved, "notes": [],
            "tactile": tactile_report(
                episode, tactile, baseline_frames=args.baseline_frames,
                tactile_threshold=args.tactile_threshold, valid=valid),
        }
        if tactile and raw_dir is not None:
            report["raw_stream"] = raw_stream_report(raw_dir, episode, tactile, stamps)
        elif tactile:
            report["notes"].append(
                f"no high-rate CSVs: {args.dataset / RAW_DIR_NAME} does not exist "
                "(pass --tactile-raw-dir).")
        report["notes"].append(
            "an episode that STARTS in contact hides that contact (its zero is taken there).")
        print(format_tactile_report(report))
        if args.out:
            args.out.write_text(json.dumps(report, indent=2))
            print(f"\nwrote {args.out}")
        return 0

    all_valid = None
    if valid:
        all_valid = np.all(np.column_stack(list(valid.values())), axis=1)
    report, frames = analyze(
        episode=arrays["episode"], pose9=arrays["pose9"], target9=arrays["target9"], fps=fps,
        tactile=tactile, joints=arrays.get("joints"), ext_torque=arrays.get("ext_torque"),
        gripper=arrays.get("gripper"), gripper_target=arrays.get("gripper_target"),
        k_pos=k_pos, k_rot=k_rot, baseline_frames=args.baseline_frames,
        contact_from=args.contact_from, tactile_threshold=args.tactile_threshold,
        force_threshold=args.force_threshold, min_sigma=args.min_sigma,
        ridge_lambda=args.ridge_lambda, folds=args.folds, tactile_valid=all_valid,
        action=arrays.get("action"),
    )
    report["columns"] = resolved
    report["nominal_gains"] = {"k_pos": k_pos, "k_rot": k_rot, "source": source}
    report["notes"] = gain_notes + report["notes"]

    print(format_report(report))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    if args.dump_npz:
        np.savez_compressed(args.dump_npz, **frames)
        print(f"wrote {args.dump_npz}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NoCommandError as e:      # an expected outcome, not a crash: no traceback
        sys.exit(f"audit_command_residual: {e}")
