"""Tests for scripts/audit_command_residual.py on SYNTHETIC data with known answers.

The audit is only worth running if its numbers can be trusted, so every step is
pinned against a simulated Cartesian-impedance episode whose stiffness, damping
and tactile response are chosen here:

    F_ext = -K r + D v          (what the controller does, quasi-statically)
    r     = T_tcp^-1 @ T_cmd    (the residual the script reports)

A wrong sign, a wrong frame (world instead of TCP), a wrong rot6d convention or
a wrong DH table each break at least one assertion below.

Numpy-only for the core; the end-to-end test also needs pandas + pyarrow and is
skipped without them.

Run:  python tests/test_audit_command_residual.py   (or via pytest)
"""

import json
import tempfile
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

K_POS, K_ROT = 400.0, 30.0
D_POS, D_ROT = 40.0, 1.5
FPS = 15.0


def _load():
    return SourceFileLoader(
        "audit_command_residual",
        str(REPO / "crisp_gym" / "scripts" / "audit_command_residual.py"),
    ).load_module()


def _exp(rv):
    """Rotation vector -> matrix (Rodrigues); test-side inverse of mat_to_rotvec."""
    rv = np.asarray(rv, dtype=np.float64)
    th = np.linalg.norm(rv, axis=-1)[..., None, None]
    K = np.zeros((*rv.shape[:-1], 3, 3))
    K[..., 0, 1], K[..., 0, 2] = -rv[..., 2], rv[..., 1]
    K[..., 1, 0], K[..., 1, 2] = rv[..., 2], -rv[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -rv[..., 1], rv[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(th > 1e-9, np.sin(th) / th, 1.0)
        b = np.where(th > 1e-9, (1 - np.cos(th)) / th ** 2, 0.5)
    return np.eye(3) + a * K + b * (K @ K)


def _pose9(T):
    return np.concatenate([T[..., :3, 3], T[..., :2, :3].reshape(*T.shape[:-2], 6)], axis=-1)


def _tool():
    T = np.eye(4)
    T[:3, :3] = _exp(np.array([0.0, 0.0, 0.3]))
    T[:3, 3] = [0.01, 0.0, 0.18]
    return T


def _smooth(rng, n, dims, scale, fps=FPS):
    """Sum of slow sinusoids — smooth, non-periodic over an episode."""
    t = np.arange(n) / fps
    out = np.zeros((n, dims))
    for _ in range(3):
        f = rng.uniform(0.05, 0.35, dims)
        ph = rng.uniform(0, 2 * np.pi, dims)
        out += np.sin(2 * np.pi * f * t[:, None] + ph) * rng.uniform(0.3, 1.0, dims)
    return out * scale / 3.0


def synth(mod, *, episodes=6, n=160, seed=0, tactile_shape=(8, 10), torque_noise=0.01):
    """Simulated teleop episodes. Returns kwargs for mod.analyze plus the truth."""
    rng = np.random.default_rng(seed)
    q0 = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8])
    taxels = int(np.prod(tactile_shape))
    A_left = rng.normal(0, 1.0, (6, taxels))
    A_right = rng.normal(0, 1.0, (6, taxels))
    out = {k: [] for k in ("episode", "pose9", "target9", "joints", "ext_torque", "left",
                           "right", "gripper", "gripper_target", "res_pos", "res_rot", "wrench")}
    tool = _tool()
    for ep in range(episodes):
        q = q0 + _smooth(rng, n, 7, 0.25)
        _, flange = mod.fr3_joint_frames(q)
        T = flange @ tool
        ep_idx = np.full(n, ep)
        v, w = mod.body_velocity(T, ep_idx, 1.0 / FPS)

        # contact wrench in the TCP frame: zero for the first 40 frames
        window = np.clip((np.arange(n) - 40) / 15.0, 0.0, 1.0)[:, None]
        F = np.concatenate([_smooth(rng, n, 3, 12.0), _smooth(rng, n, 3, 0.8)], axis=1) * window

        r_pos = (-F[:, :3] + D_POS * v) / K_POS
        r_rot = (-F[:, 3:] + D_ROT * w) / K_ROT
        Tres = np.tile(np.eye(4), (n, 1, 1))
        Tres[:, :3, :3] = _exp(r_rot)
        Tres[:, :3, 3] = r_pos
        Tcmd = T @ Tres

        R = T[:, :3, :3]
        w_base = np.concatenate([np.einsum("nij,nj->ni", R, F[:, :3]),
                                 np.einsum("nij,nj->ni", R, F[:, 3:])], axis=1)
        J = mod.fr3_jacobian_at(q, T[:, :3, 3])
        tau = np.einsum("nji,nj->ni", J, w_base) + rng.normal(0, torque_noise, (n, 7))

        grasp = window[:, 0]
        g = np.full(n, 0.6)
        out["episode"].append(ep_idx)
        out["pose9"].append(_pose9(T))
        out["target9"].append(_pose9(Tcmd))
        out["joints"].append(q)
        out["ext_torque"].append(tau)
        out["left"].append(F @ A_left + rng.normal(0, 0.02, (n, taxels)) + rng.normal(0, 5.0, taxels))
        out["right"].append(F @ A_right + rng.normal(0, 0.02, (n, taxels)) + rng.normal(0, 5.0, taxels))
        out["gripper"].append(g)
        out["gripper_target"].append(g - 0.15 * grasp)
        out["res_pos"].append(r_pos)
        out["res_rot"].append(r_rot)
        out["wrench"].append(F)
    d = {k: np.concatenate(v) for k, v in out.items()}
    kwargs = dict(
        episode=d["episode"], pose9=d["pose9"], target9=d["target9"], fps=FPS,
        tactile={"left_tactile": d["left"], "right_tactile": d["right"]},
        joints=d["joints"], ext_torque=d["ext_torque"],
        gripper=d["gripper"], gripper_target=d["gripper_target"],
        k_pos=K_POS, k_rot=K_ROT,
    )
    return kwargs, d


# ── geometry ─────────────────────────────────────────────────────────────────

def test_rotvec_roundtrip():
    mod = _load()
    rng = np.random.default_rng(1)
    rv = rng.normal(0, 0.7, (500, 3))
    rv[:50] *= 1e-6                      # the small-angle branch
    np.testing.assert_allclose(mod.mat_to_rotvec(_exp(rv)), rv, atol=1e-9)


def test_rot6d_is_the_first_two_ROWS():
    """HANDOFF §1.1 — columns instead of rows silently transposes every rotation."""
    mod = _load()
    R = _exp(np.array([0.3, -0.5, 0.9]))
    np.testing.assert_allclose(mod.rot6d_to_mat(R[:2, :].reshape(6)), R, atol=1e-12)


def test_residual_is_recovered_and_world_frame_invariant():
    mod = _load()
    kwargs, truth = synth(mod, episodes=2)
    T, Tc = mod.pose9d_to_mat(kwargs["pose9"]), mod.pose9d_to_mat(kwargs["target9"])
    pos, rot = mod.tcp_frame_residual(T, Tc)
    np.testing.assert_allclose(pos, truth["res_pos"], atol=1e-9)
    np.testing.assert_allclose(rot, truth["res_rot"], atol=1e-9)

    # Re-express everything in another world frame (handheld: OptiTrack world).
    W = np.eye(4)
    W[:3, :3] = _exp(np.array([0.4, 1.1, -0.7]))
    W[:3, 3] = [1.0, -2.0, 0.5]
    pos2, rot2 = mod.tcp_frame_residual(W @ T, W @ Tc)
    np.testing.assert_allclose(pos2, pos, atol=1e-9)
    np.testing.assert_allclose(rot2, rot, atol=1e-9)


def test_fr3_jacobian_matches_finite_differences():
    """Pins the DH table AND the Jacobian against each other."""
    mod = _load()
    rng = np.random.default_rng(2)
    q = np.array([0.1, -0.5, 0.2, -2.0, 0.3, 1.6, 0.7]) + rng.normal(0, 0.2, (5, 7))
    tool = _tool()

    def tcp(qq):
        return mod.fr3_joint_frames(qq)[1] @ tool

    T0 = tcp(q)
    J = mod.fr3_jacobian_at(q, T0[:, :3, 3])
    eps = 1e-6
    for j in range(7):
        dq = np.zeros(7)
        dq[j] = eps
        T1 = tcp(q + dq)
        lin = (T1[:, :3, 3] - T0[:, :3, 3]) / eps
        ang_body = mod.mat_to_rotvec(np.swapaxes(T0[:, :3, :3], -1, -2) @ T1[:, :3, :3]) / eps
        ang = np.einsum("nij,nj->ni", T0[:, :3, :3], ang_body)
        np.testing.assert_allclose(J[:, :3, j], lin, atol=1e-5)
        np.testing.assert_allclose(J[:, 3:, j], ang, atol=1e-5)


def test_fr3_flange_at_the_zero_pose():
    """Franka's published zero-configuration flange: x 0.088, z 0.333+0.316+0.384-0.107."""
    mod = _load()
    _, flange = mod.fr3_joint_frames(np.zeros((1, 7)))
    np.testing.assert_allclose(flange[0, :3, 3], [0.088, 0.0, 0.926], atol=1e-9)


def test_fr3_home_pose_matches_frankas_published_value():
    """Second, independent anchor for the DH table: at the standard "ready" pose
    the Franka Hand TCP (0.1034 m beyond the flange) sits at (0.3069, 0, 0.4869),
    pointing straight down."""
    mod = _load()
    q = np.array([[0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]])
    _, flange = mod.fr3_joint_frames(q)
    np.testing.assert_allclose(flange[0, :3, 2], [0.0, 0.0, -1.0], atol=1e-9)
    tcp = flange[0, :3, 3] + 0.1034 * flange[0, :3, 2]
    np.testing.assert_allclose(tcp, [0.3069, 0.0, 0.4869], atol=5e-4)


def test_wrench_is_recovered_from_joint_torques():
    mod = _load()
    kwargs, truth = synth(mod, episodes=1, torque_noise=0.0)
    T = mod.pose9d_to_mat(kwargs["pose9"])
    w, sigma = mod.tcp_wrench_from_joint_torque(kwargs["joints"], kwargs["ext_torque"], T)
    assert sigma.min() > 0.03
    np.testing.assert_allclose(w, truth["wrench"], atol=1e-8)


# ── the analysis ─────────────────────────────────────────────────────────────

def test_system_identification_recovers_the_gains():
    mod = _load()
    kwargs, _ = synth(mod)
    report, _ = mod.analyze(**kwargs)
    sysid = report["system_identification"]
    for ax in "xyz":
        t, r = sysid["translation"][ax], sysid["rotation"][ax]
        assert abs(t["stiffness"] - K_POS) / K_POS < 0.02, t
        assert abs(r["stiffness"] - K_ROT) / K_ROT < 0.02, r
        assert abs(t["damping"] - D_POS) / D_POS < 0.10, t
        assert abs(t["stiffness_vs_nominal"] - 1.0) < 0.02
        assert t["r2"] > 0.98 and r["r2"] > 0.95
    assert report["kinematics_check"]["tool_offset_std_m"] < 1e-9


def test_flipped_torque_sign_is_reported_not_hidden():
    mod = _load()
    kwargs, _ = synth(mod)
    kwargs["ext_torque"] = -kwargs["ext_torque"]
    report, _ = mod.analyze(**kwargs)
    assert report["system_identification"]["translation"]["x"]["stiffness"] < 0
    assert any("NEGATIVE" in n for n in report["notes"])


def test_lag_time_constant_in_free_motion():
    mod = _load()
    kwargs, _ = synth(mod)
    report, _ = mod.analyze(**kwargs)
    for ax in "xyz":
        tau = report["lag_free_motion"][ax]["tau_ms"]
        assert abs(tau - 1e3 * D_POS / K_POS) < 15.0, (ax, tau)


def test_contact_residual_is_larger_than_free():
    mod = _load()
    kwargs, _ = synth(mod)
    report, _ = mod.analyze(**kwargs)
    res = report["residual"]["pos_mm"]
    assert 0.2 < report["contact"]["fraction"] < 0.9
    assert res["contact"]["p50"] > 2.0 * res["free"]["p50"]
    assert "nominal_force_N" in report["residual"]


def test_tactile_probe_beats_velocity_on_held_out_episodes():
    mod = _load()
    kwargs, _ = synth(mod)
    report, _ = mod.analyze(**kwargs)
    probe = report["residual_probe"]
    tv = probe["tactile+velocity"]["contact"]
    vel = probe["velocity"]["contact"]
    assert min(tv["r2_pos"]) > 0.95 and min(tv["r2_rot"]) > 0.90, tv
    assert max(vel["r2_pos"]) < 0.5, vel
    assert min(probe["wrench+velocity"]["contact"]["r2_pos"]) > 0.95


def test_probe_scores_on_held_out_GROUPS_not_random_frames():
    """A feature that is just the episode id must NOT look predictive."""
    mod = _load()
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(8), 50)
    offsets = rng.normal(0, 1.0, 8)
    y = (offsets[groups] + rng.normal(0, 0.01, 400))[:, None]
    X = np.eye(8)[groups]                       # one-hot episode id
    out = mod.ridge_cv(X, y, groups, np.ones(400, dtype=bool), lam=1e-3, folds=4)
    assert out["r2"][0] < 0.1, out


def test_gripper_residual_block():
    mod = _load()
    kwargs, _ = synth(mod)
    report, frames = mod.analyze(**kwargs)
    g = report["gripper_residual"]
    assert g["squeezing_fraction_in_contact"] > 0.8
    assert g["abs"]["free"]["p50"] < 0.02
    assert frames["gripper_residual"].min() < -0.1


def test_tactile_shape_is_free():
    """Any number of taxels, flat or nested — the user's 8x10 is not baked in."""
    mod = _load()
    for shape in ((8, 10), (4, 4), (15,)):
        kwargs, _ = synth(mod, episodes=3, tactile_shape=shape)
        report, _ = mod.analyze(**kwargs)
        want = int(np.prod(shape))
        assert report["tactile"]["taxels"] == {"left_tactile": want, "right_tactile": want}


def test_runs_without_optional_columns():
    mod = _load()
    kwargs, _ = synth(mod, episodes=3)
    for drop in ("tactile", "joints", "ext_torque", "gripper", "gripper_target"):
        kwargs[drop] = None
    kwargs["k_pos"] = kwargs["k_rot"] = None
    report, _ = mod.analyze(**kwargs)
    assert report["contact"]["source"] == "none"
    assert "system_identification" not in report
    assert "velocity" in report["residual_probe"]
    mod.format_report(report)


def test_frozen_command_is_refused():
    """FACTR/JIC data: target_cartesian is a constant, not a command."""
    mod = _load()
    kwargs, _ = synth(mod, episodes=2)
    for ep in (0, 1):
        m = kwargs["episode"] == ep
        kwargs["target9"][m] = kwargs["target9"][m][0]
    try:
        mod.analyze(**kwargs)
        raise AssertionError("frozen command not refused")
    except ValueError as e:
        assert "joint-driven" in str(e)


def test_kinematics_mismatch_skips_sysid_instead_of_lying():
    mod = _load()
    kwargs, _ = synth(mod, episodes=3)
    kwargs["joints"] = kwargs["joints"][::-1].copy()      # joints out of sync with pose
    report, _ = mod.analyze(**kwargs)
    assert "system_identification" not in report
    assert any("not rigid" in n for n in report["notes"])


def test_skin_vs_wrench_flags_contact_the_skin_misses():
    mod = _load()
    kwargs, _ = synth(mod, episodes=4)
    report, _ = mod.analyze(**kwargs)
    assert report["contact"]["skin_vs_wrench"]["wrench_only"] < 0.02

    # A skin that sees nothing: every joint-torque contact becomes "wrench only".
    rng = np.random.default_rng(5)
    n = len(kwargs["episode"])
    kwargs["tactile"] = {"left_tactile": rng.normal(0, 0.02, (n, 80))}
    report, _ = mod.analyze(**kwargs, contact_from="wrench")
    assert report["contact"]["skin_vs_wrench"]["wrench_only"] > 0.2


# ── I/O ──────────────────────────────────────────────────────────────────────

def test_resolve_key_suffix_and_ambiguity():
    mod = _load()
    cols = ["extra.left_tactile", "observation.state.cartesian", "a.x", "b.x"]
    assert mod.resolve_key(cols, "left_tactile", required=False) == "extra.left_tactile"
    assert mod.resolve_key(cols, "observation.state.cartesian", required=True)
    assert mod.resolve_key(cols, "right_tactile", required=False) is None
    for key, exc in (("x", "ambiguous"), ("extra.target_cartesian", "not found")):
        try:
            mod.resolve_key(cols, key, required=True)
            raise AssertionError(f"{key} not rejected")
        except KeyError as e:
            assert exc in str(e)


def test_load_gains_from_both_yaml_layouts():
    mod = _load()
    with tempfile.TemporaryDirectory() as d:
        flat = Path(d) / "control.yaml"
        flat.write_text(
            "task.k_pos_x: 600.0\ntask.k_pos_y: 600.0\ntask.k_pos_z: 600.0\n"
            "task.k_rot_x: 30.0\ntask.k_rot_y: 30.0\ntask.k_rot_z: 30.0\n"
        )
        assert mod.load_gains(flat, "unused") == (600.0, 30.0)

        nested = Path(d) / "controllers.yaml"
        nested.write_text(
            "/**:\n"
            "  gravity_compensation:\n    ros__parameters:\n      task:\n"
            "        k_pos_x: 0.0\n        k_pos_y: 0.0\n        k_pos_z: 0.0\n"
            "  cartesian_impedance_controller:\n    ros__parameters:\n      task:\n"
            "        k_pos_x: 400.0\n        k_pos_y: 400.0\n        k_pos_z: 400.0\n"
            "        k_rot_x: 30.0\n        k_rot_y: 30.0\n        k_rot_z: 30.0\n"
        )
        assert mod.load_gains(nested, "cartesian_impedance_controller") == (400.0, 30.0)
        assert mod.load_gains(nested, "gravity_compensation") == (0.0, None)


def test_the_shipped_fr3_controllers_yaml_layout_parses():
    """The bring-up file the dric Cartesian env actually takes its gains from."""
    path = (REPO.parent / "crisp_controllers_demos" / "crisp_controllers_robot_demos"
            / "config" / "fr3" / "controllers.yaml")
    if not path.exists():
        print("skip: crisp_controllers_demos not checked out next to crisp_gym")
        return
    k_pos, k_rot = _load().load_gains(path, "cartesian_impedance_controller")
    assert k_pos and k_pos > 0 and k_rot and k_rot > 0


def test_end_to_end_on_a_parquet_dataset():
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError:
        print("skip: pandas/pyarrow not installed")
        return
    mod = _load()
    kwargs, d = synth(mod, episodes=4)
    n = len(d["episode"])
    frame = np.concatenate([np.arange((d["episode"] == e).sum()) for e in np.unique(d["episode"])])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "meta").mkdir()
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(json.dumps({
            "fps": FPS, "features": {"extra.left_tactile": {"shape": [8, 10]}}}))
        df = pd.DataFrame({
            "episode_index": d["episode"], "frame_index": frame,
            "observation.state.cartesian": list(d["pose9"].astype(np.float32)),
            "extra.target_cartesian": list(d["target9"].astype(np.float32)),
            "extra.joints": list(d["joints"].astype(np.float32)),
            "extra.ext_torque": list(d["ext_torque"].astype(np.float32)),
            # nested 8x10, the way a 2-D feature is stored
            "extra.left_tactile": [x.reshape(8, 10).tolist() for x in d["left"].astype(np.float32)],
            "extra.right_tactile": list(d["right"].astype(np.float32)),
            "observation.state.gripper": list(d["gripper"].astype(np.float32)[:, None]),
            "extra.gripper_target": list(d["gripper_target"].astype(np.float32)[:, None]),
        })
        # two files, rows shuffled: the loader must restore (episode, frame) order
        df = df.sample(frac=1.0, random_state=0)
        df.iloc[: n // 2].to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
        df.iloc[n // 2:].to_parquet(root / "data" / "chunk-000" / "file-001.parquet")

        out, npz = root / "audit.json", root / "frames.npz"
        rc = mod.main([str(root), "--k-pos", str(K_POS), "--k-rot", str(K_ROT),
                       "--tactile-shape", "8", "10", "--out", str(out), "--dump-npz", str(npz)])
        assert rc == 0
        report = json.loads(out.read_text())
        assert report["columns"]["tactile:left_tactile"] == "extra.left_tactile"
        assert report["tactile"]["taxels"] == {"left_tactile": 80, "right_tactile": 80}
        k = report["system_identification"]["translation"]["z"]["stiffness"]
        assert abs(k - K_POS) / K_POS < 0.03, k          # float32 storage
        assert np.load(npz)["res_pos"].shape == (n, 3)

        try:
            mod.main([str(root), "--tactile-shape", "4", "4"])
            raise AssertionError("wrong --tactile-shape not rejected")
        except ValueError as e:
            assert "80" in str(e)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("ALL OK")
