r"""A taxel pad on a spring, pressed onto a peg: the contact question in its smallest form.

The real arm is driven by an impedance controller, which acts like a spring
between the commanded pose and the measured pose. In contact the measured pose
stops at the surface while the command can go on; the spring turns that gap
into force. This script builds exactly that in MuJoCo, with a tactile skin:

  * a pad of `nx` x `ny` taxels that slides along one axis;
  * a position servo of `stiffness` N/m between its command and its position;
  * a fixed peg in the way;
  * MuJoCo's `touch_grid` sensor reading the force on every taxel.

For a list of commanded depths it prints the measured depth, the spring force
`stiffness * (command - measured)` and the force summed over the taxels. In
steady contact the two forces are equal, and the measured depth stays at the
surface however deep the command goes: a label made of measured poses carries
no force, the command does.

    python crisp_gym/sim/spring_pad.py
    python crisp_gym/sim/spring_pad.py --stiffness 200 --depths-mm 5 10 20 --show-grid
    python crisp_gym/sim/spring_pad.py --taxels 4 8 --peg-xy-mm -5 10

Depth is measured along the pressing direction from the peg's surface:
negative = short of it, positive = commanded past it.

Runs on any machine: `pip install mujoco numpy`, CPU, no display. No ROS, no
crisp_py, no torch. Tests: tests/test_sim_spring_pad.py. Background and the
experiments this is the first step of: RESEARCH_FORCE_TACTILE.md.

What it does not model: a soft skin that spreads load, sensor noise and drift,
friction in the arm, and the 15 Hz / 50 Hz / 1 kHz timing of the real stack.

One artefact to know: the command must MOVE to its depth (`command_speed`), not
jump. After a jump a stiff spring (from about 800 N/m) drives the pad into the
peg fast enough to break through it, and it stays sunk at a fraction of the
force. The table says so when that happens.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    import mujoco

PAD_THICKNESS = 0.004  # m
# A pad that went further than this share of the commanded depth into the peg
# has broken through it (see `press`). A sound contact gives way by a few
# percent at most, more on older MuJoCo releases, whose contacts are softer.
BROKEN_THROUGH = 0.25
PEG_TOP = 0.06  # m, height of the peg's upper face in the world
# The sensor bins contacts by ANGLE as seen from its site. From far away the
# angles are small and equal angles are equal distances, so every taxel-sized
# collision box falls into its own bin.
SITE_HEIGHT = 1.0  # m above the pad


@dataclass(frozen=True)
class PadConfig:
    """Geometry and spring of the pad, and where the peg stands.

    `peg_xy` is the peg's centre measured from the pad's centre, in metres:
    x across the pad (`nx` taxels), y along it (`ny` taxels).
    """

    nx: int = 6
    ny: int = 12
    half_x: float = 0.011
    half_y: float = 0.019
    stiffness: float = 400.0  # N/m, the Franka bring-up's translational gain
    damping: float | None = None  # N s/m; None = critical, 2 * sqrt(stiffness * mass)
    mass: float = 0.5  # kg
    command_speed: float = 0.05  # m/s, how fast the command moves to a new depth
    clearance: float = 0.02  # m between pad and peg at the start
    peg_xy: tuple[float, float] = (0.004, 0.008)
    peg_radius: float = 0.004
    timestep: float = 0.001  # s


@dataclass(frozen=True)
class PressResult:
    """What one commanded depth produced once the pad came to rest."""

    command_depth: float  # m past the peg's surface that was commanded
    measured_depth: float  # m past the peg's surface that was reached
    spring_force: float  # N, stiffness * (command - measured)
    normal: np.ndarray  # [ny, nx] N, force on each taxel along the pressing direction
    shear: np.ndarray  # [2, ny, nx] N, the two tangential components as MuJoCo reports them

    @property
    def taxel_force(self) -> float:
        """Normal force summed over all taxels, in N."""
        return float(self.normal.sum())

    @property
    def lit_taxels(self) -> int:
        """Number of taxels that carry any normal force."""
        return int((np.abs(self.normal) > 1e-6).sum())

    @property
    def broke_through(self) -> bool:
        """True when the pad sank into the peg: a simulation artefact, not a result."""
        return self.measured_depth > max(1e-4, BROKEN_THROUGH * self.command_depth)


def model_xml(cfg: PadConfig) -> str:
    """The MuJoCo model as XML text.

    The pad is built from one collision box per taxel: a single rigid box would
    touch the peg at one point and light one taxel only.
    """
    sx, sy = cfg.half_x / cfg.nx, cfg.half_y / cfg.ny
    boxes = "\n      ".join(
        f'<geom type="box" pos="{-cfg.half_x + sx * (2 * i + 1):.6f} '
        f'{-cfg.half_y + sy * (2 * j + 1):.6f} 0" '
        f'size="{sx:.6f} {sy:.6f} {PAD_THICKNESS / 2:.6f}"/>'
        for j in range(cfg.ny)
        for i in range(cfg.nx)
    )
    # touch_grid takes the HALF angle of its field of view, in degrees
    fov_x = math.degrees(math.atan(cfg.half_x / SITE_HEIGHT))
    fov_y = math.degrees(math.atan(cfg.half_y / SITE_HEIGHT))
    pad_z = PEG_TOP + cfg.clearance + PAD_THICKNESS / 2
    damping = cfg.damping if cfg.damping is not None else 2 * math.sqrt(cfg.stiffness * cfg.mass)
    peg_x, peg_y = cfg.peg_xy
    return f"""
<mujoco model="spring_pad">
  <extension><plugin plugin="mujoco.sensor.touch_grid"/></extension>
  <option timestep="{cfg.timestep}" gravity="0 0 0"/>
  <worldbody>
    <body name="pad" pos="0 0 {pad_z:.6f}">
      <joint name="pad_z" type="slide" axis="0 0 1" damping="{damping:.6f}"/>
      <inertial pos="0 0 0" mass="{cfg.mass}" diaginertia="1e-4 1e-4 1e-4"/>
      {boxes}
      <!-- The sensor looks along its site's -z axis, like a camera. The pad
           faces down, so the site sits above it with +z up. Turned the other
           way round every taxel reads zero and nothing warns about it. -->
      <site name="skin" pos="0 0 {SITE_HEIGHT}" zaxis="0 0 1"/>
    </body>
    <geom name="peg" type="cylinder" pos="{peg_x:.6f} {peg_y:.6f} {PEG_TOP / 2:.6f}"
          size="{cfg.peg_radius:.6f} {PEG_TOP / 2:.6f}"/>
  </worldbody>
  <actuator>
    <position name="servo" joint="pad_z" kp="{cfg.stiffness}" ctrlrange="-1 1"/>
  </actuator>
  <sensor>
    <plugin name="skin" plugin="mujoco.sensor.touch_grid" objtype="site" objname="skin">
      <config key="size" value="{cfg.nx} {cfg.ny}"/>
      <config key="fov" value="{fov_x:.6f} {fov_y:.6f}"/>
      <config key="gamma" value="0"/>
      <config key="nchannel" value="3"/>
    </plugin>
  </sensor>
</mujoco>
"""


def _mujoco() -> ModuleType:
    """Import MuJoCo only when a simulation runs, so `--help` works without it."""
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("this simulation needs MuJoCo: pip install mujoco") from exc
    return mujoco


def build_model(cfg: PadConfig) -> mujoco.MjModel:
    """Compile the model. Needs the `mujoco` package (its pip wheel has the sensor plugin)."""
    return _mujoco().MjModel.from_xml_string(model_xml(cfg))


def read_taxels(model: mujoco.MjModel, data: mujoco.MjData, cfg: PadConfig) -> np.ndarray:
    """The skin's reading as [3, ny, nx]: normal force first, then the two shear components."""
    return np.array(data.sensordata[: int(model.sensor_dim[0])]).reshape(3, cfg.ny, cfg.nx)


def press(
    cfg: PadConfig, depths: list[float] | tuple[float, ...], settle_s: float = 2.0
) -> list[PressResult]:
    """Move the command to each depth in turn, wait for the pad to rest, and read it.

    The command travels at `cfg.command_speed`, as a hand or a policy would move
    it, rather than jumping. After a jump a stiff spring makes the pad hit the
    peg fast enough to sink millimetres into it. MuJoCo separates overlapping
    shapes along the shortest way out, which for a taxel-sized box that deep in
    the peg is sideways: measured at 1600 N/m, all 15 contact normals then
    point sideways (35 of 36 point along the pressing direction in a normal
    press), so the contact no longer pushes back and the pad stays sunk,
    13.6 mm deep. Thicker boxes do not help. Checked to hold from 50 to
    5000 N/m at the default speed.

    Args:
        cfg: the pad, its spring and the peg.
        depths: commanded depths in metres past the peg's surface, in the order
            they are applied (the pad is not reset in between).
        settle_s: simulated seconds to wait once the command has arrived.
    """
    mujoco = _mujoco()
    model = build_model(cfg)
    data = mujoco.MjData(model)
    settle_steps = int(round(settle_s / cfg.timestep))
    max_step = cfg.command_speed * cfg.timestep
    results = []
    for depth in depths:
        # joint position 0 is the start, `clearance` above the surface
        target = -(cfg.clearance + depth)
        while abs(target - data.ctrl[0]) > max_step:
            data.ctrl[0] += math.copysign(max_step, target - data.ctrl[0])
            mujoco.mj_step(model, data)
        data.ctrl[0] = target
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        measured = -(float(data.qpos[0]) + cfg.clearance)
        taxels = read_taxels(model, data, cfg)
        results.append(
            PressResult(
                command_depth=float(depth),
                measured_depth=measured,
                spring_force=cfg.stiffness * (float(depth) - measured),
                normal=taxels[0],
                shear=taxels[1:],
            )
        )
    return results


def taxel_centre(cfg: PadConfig, row: float, col: float) -> tuple[float, float]:
    """Position (x, y) on the pad, in metres from its centre, of a taxel index.

    Fractional indices are allowed, so a force-weighted mean row and column can
    be turned into a position.
    """
    x = -cfg.half_x + (col + 0.5) * 2 * cfg.half_x / cfg.nx
    y = -cfg.half_y + (row + 0.5) * 2 * cfg.half_y / cfg.ny
    return x, y


def format_table(cfg: PadConfig, results: list[PressResult]) -> str:
    """The results as the text table `main` prints."""
    lines = [
        f"pad {cfg.nx} x {cfg.ny} = {cfg.nx * cfg.ny} taxels, 3 values each | "
        f"spring {cfg.stiffness:g} N/m",
        "command(mm)  measured(mm)  gap(mm)  spring force(N)  taxel force(N)  lit taxels",
    ]
    for r in results:
        # rounded first, and + 0.0, so a residual of -0.0001 does not print as "-0.00"
        gap_mm = round((r.command_depth - r.measured_depth) * 1e3, 2) + 0.0
        spring = round(r.spring_force, 2) + 0.0
        lines.append(
            f"{r.command_depth * 1e3:11.2f}  {r.measured_depth * 1e3:12.2f}  "
            f"{gap_mm:7.2f}  {spring:15.2f}  {r.taxel_force:14.2f}  {r.lit_taxels:10d}"
        )
    if any(r.broke_through for r in results):
        lines.append(
            "WARNING: the pad sank into the peg (it went more than "
            f"{BROKEN_THROUGH:.0%} of the commanded depth in). That is a simulation artefact "
            "of a command that moved too fast for this stiffness; lower command_speed. "
            "Do not use these rows."
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stiffness", type=float, default=400.0, help="spring, N/m (default 400)")
    p.add_argument(
        "--taxels",
        type=int,
        nargs=2,
        default=[6, 12],
        metavar=("NX", "NY"),
        help="taxels across and along the pad (default 6 12)",
    )
    p.add_argument(
        "--depths-mm",
        type=float,
        nargs="+",
        default=[-10.0, 2.0, 12.0, 22.0, 42.0],
        help="commanded depths past the peg's surface, mm (default -10 2 12 22 42)",
    )
    p.add_argument(
        "--peg-xy-mm",
        type=float,
        nargs=2,
        default=[4.0, 8.0],
        metavar=("X", "Y"),
        help="peg centre from the pad centre, mm (default 4 8)",
    )
    p.add_argument(
        "--show-grid", action="store_true", help="print the taxel forces of the last depth"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Press at the requested depths and print what the spring and the skin report."""
    args = build_parser().parse_args(argv)
    cfg = PadConfig(
        nx=args.taxels[0],
        ny=args.taxels[1],
        stiffness=args.stiffness,
        peg_xy=(args.peg_xy_mm[0] / 1e3, args.peg_xy_mm[1] / 1e3),
    )
    results = press(cfg, [d / 1e3 for d in args.depths_mm])
    print(format_table(cfg, results))
    if args.show_grid:
        with np.printoptions(precision=1, suppress=True, linewidth=160):
            print("normal force per taxel, N (rows along the pad, columns across):")
            print(results[-1].normal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
