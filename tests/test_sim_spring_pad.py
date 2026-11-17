"""The spring-and-skin simulation against what statics says it must show.

A pad on a spring is pressed onto a fixed peg (crisp_gym/sim/spring_pad.py).
Once it is at rest the spring force and the contact force must be equal, the
pad must stand at the peg's surface, and the lit taxels must lie under the peg.

Needs the `mujoco` package (in tests/requirements.txt); skipped without it.
No ROS, no display.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from crisp_gym.sim.spring_pad import (  # noqa: E402
    PadConfig,
    build_model,
    format_table,
    main,
    model_xml,
    press,
    read_taxels,
    taxel_centre,
)


def _centroid(cfg, normal):
    """Force-weighted mean position of the contact, in metres from the pad centre."""
    rows, cols = np.indices(normal.shape)
    total = normal.sum()
    return taxel_centre(cfg, (rows * normal).sum() / total, (cols * normal).sum() / total)


def test_three_values_per_taxel():
    for nx, ny in ((6, 12), (4, 5)):
        cfg = PadConfig(nx=nx, ny=ny)
        model = build_model(cfg)
        assert int(model.sensor_dim[0]) == 3 * nx * ny
        result = press(cfg, [0.01], settle_s=1.0)[0]
        assert result.normal.shape == (ny, nx)
        assert result.shear.shape == (2, ny, nx)


def test_short_of_the_peg_nothing_is_felt_and_the_pad_follows_the_command():
    result = press(PadConfig(), [-0.010])[0]
    assert result.lit_taxels == 0
    assert result.taxel_force == 0.0
    assert abs(result.measured_depth - (-0.010)) < 5e-5
    assert abs(result.spring_force) < 0.02


def test_in_contact_the_spring_force_is_the_taxel_force():
    cfg = PadConfig()
    for result in press(cfg, [0.002, 0.012, 0.042]):
        expected = cfg.stiffness * result.command_depth
        # the pad stops at the surface (the contact gives way by a fraction of a mm) ...
        assert 0.0 <= result.measured_depth < 5e-4
        # ... so the whole commanded depth is spring gap, and the skin feels that force
        assert result.spring_force == pytest.approx(expected, rel=0.02)
        assert result.taxel_force == pytest.approx(result.spring_force, rel=0.01, abs=1e-3)
        assert result.lit_taxels >= 2


def test_a_measured_pose_label_carries_no_force():
    """The same measured depth for a 2 mm and a 42 mm command: only the command differs."""
    shallow, deep = press(PadConfig(), [0.002, 0.042])
    assert abs(deep.measured_depth - shallow.measured_depth) < 5e-4  # commands differ by 40 mm
    assert deep.taxel_force > 15 * shallow.taxel_force


def test_force_scales_with_the_stiffness():
    soft = press(PadConfig(stiffness=200.0), [0.02])[0]
    stiff = press(PadConfig(stiffness=800.0), [0.02])[0]
    assert stiff.taxel_force / soft.taxel_force == pytest.approx(4.0, rel=0.03)


def test_a_stiff_spring_does_not_push_the_pad_through_the_peg():
    """Holds because the command MOVES to its depth; see the next test."""
    for stiffness in (800.0, 1600.0, 5000.0):
        cfg = PadConfig(stiffness=stiffness)
        early = press(cfg, [0.02], settle_s=1.0)[0]
        late = press(cfg, [0.02], settle_s=5.0)[0]
        assert not late.broke_through
        assert abs(late.measured_depth - early.measured_depth) < 1e-5  # at rest, not creeping
        assert late.taxel_force == pytest.approx(late.spring_force, rel=0.01)
        # the contact gives way a little (more on older MuJoCo), never by much
        assert late.spring_force > 0.9 * stiffness * 0.02


def test_a_command_jump_at_high_stiffness_is_reported_not_trusted():
    """After a jump the pad hits the peg fast, sinks in and stays there with a
    fraction of the force. The result is flagged, and the table warns."""
    cfg = PadConfig(stiffness=1600.0, command_speed=1e9)
    result = press(cfg, [0.02])[0]
    assert result.broke_through
    assert result.taxel_force < 0.5 * cfg.stiffness * 0.02
    assert "WARNING" in format_table(cfg, [result])
    assert "WARNING" not in format_table(PadConfig(), press(PadConfig(), [0.02]))


def test_the_lit_taxels_lie_under_the_peg():
    """Pins the orientation of the grid: rows run along y, columns along x, unmirrored."""
    for peg_xy in ((0.004, 0.008), (-0.005, -0.011), (0.006, -0.002)):
        cfg = PadConfig(peg_xy=peg_xy)
        result = press(cfg, [0.02])[0]
        x, y = _centroid(cfg, result.normal)
        pitch_x, pitch_y = 2 * cfg.half_x / cfg.nx, 2 * cfg.half_y / cfg.ny
        assert abs(x - peg_xy[0]) < pitch_x, (peg_xy, x, y)
        assert abs(y - peg_xy[1]) < pitch_y, (peg_xy, x, y)
        # nothing lights up further than the peg's radius plus one taxel away
        rows, cols = np.nonzero(result.normal > 1e-6)
        for row, col in zip(rows, cols):
            tx, ty = taxel_centre(cfg, row, col)
            assert abs(tx - peg_xy[0]) < cfg.peg_radius + pitch_x
            assert abs(ty - peg_xy[1]) < cfg.peg_radius + pitch_y


def test_no_force_is_lost_at_the_edge_of_the_pad():
    """A peg half off the pad: every contact still falls into a taxel."""
    cfg = PadConfig(peg_xy=(0.0105, 0.018))
    result = press(cfg, [0.02])[0]
    assert result.taxel_force == pytest.approx(result.spring_force, rel=0.01)


def test_a_site_looking_away_from_the_pad_reads_zero():
    """The trap the model's comment warns about: contact, force, and a silent skin."""
    cfg = PadConfig()
    xml = model_xml(cfg)
    assert xml.count('zaxis="0 0 1"') == 1
    model = mujoco.MjModel.from_xml_string(xml.replace('zaxis="0 0 1"', 'zaxis="0 0 -1"'))
    data = mujoco.MjData(model)
    data.ctrl[0] = -(cfg.clearance + 0.02)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    assert data.ncon > 0
    assert np.abs(read_taxels(model, data, cfg)).sum() == 0.0


def test_the_table_and_the_command_line(capsys):
    cfg = PadConfig()
    table = format_table(cfg, press(cfg, [-0.01, 0.012]))
    assert "72 taxels" in table and "400 N/m" in table
    assert "-0.00" not in table
    assert (
        main(["--depths-mm", "12", "--taxels", "4", "8", "--stiffness", "200", "--show-grid"]) == 0
    )
    out = capsys.readouterr().out
    assert "32 taxels" in out and "200 N/m" in out
    assert "normal force per taxel" in out
