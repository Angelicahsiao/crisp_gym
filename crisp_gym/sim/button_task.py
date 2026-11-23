"""Two fingertips on a spring press a button, then slide on the panel: a press-button task in MuJoCo.

What it is: the closed gripper's two fingertips, each with a 6 x 12 taxel skin,
held by a 3-D spring of stiffness K the way the Cartesian impedance controller
holds the real TCP, above a panel with a spring-loaded button. A scripted
demonstrator moves the command as teleoperation would: approach, press until
the button clicks, hold, lift clear, come down on the panel beside the button,
slide along it away from the button, lift.
`run_episode` returns the frames at 15 Hz under the column names of a real
Franka recording (config/recording/umi_robot_full_record.yaml), plus a few
columns the robot cannot measure (the true contact force, the button's state).

Runs on any machine: CPU, no display. Imports numpy and mujoco only (no ROS,
crisp_py, torch or lerobot). Tests: tests/test_sim_button_task.py. The command
line, which previews an episode or writes a LeRobot dataset, is
crisp_gym/sim/button_dataset.py.

Conventions, chosen to match the robot data:
  * World: the panel's top face at z = 0, z up, metres.
  * TCP: the centre of the two fingertip faces. Its z axis points DOWN into the
    panel and its y axis is the jaw axis, as fr3_hand_tcp when pressing
    straight down. The springs act on position only, so the orientation never
    changes and the rot6d part of every pose is the constant `TCP_ROT6D`.
  * Gripper: closed and position-controlled, so its column is a constant.
  * Tactile: the normal force on each taxel, 72 per fingertip, 12 rows of 6,
    flattened row by row.
  * `action[t]` is the measured pose of the next frame (lookahead 1), and the
    last simulated frame only serves as the previous frame's action, as the
    recorder does.

Left out: rotation, a soft skin, sensor noise, the arm's own friction and the
15 Hz / 1 kHz timing of the real stack. The demonstrator knows when the button
clicks, as a person feels and hears it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

RATE_HZ = 15.0
STEPS_PER_FRAME = 64  # physics steps per recorded frame: a 1.04 ms step
SITE_HEIGHT = 1.0  # m above the fingertip faces; see spring_pad.py for why
# Contacts stiffer than MuJoCo's default (0.02 s, 0.9-0.95). Its softness is
# scaled by how hard a body is pushed, so the light button cap, pushed by its
# own spring, sank millimetres into the fingertips when they lifted off and
# knocked them down. The time constant stays above two physics steps.
CONTACT_SOLREF = "0.004 1"
CONTACT_SOLIMP = "0.95 0.99 0.001"
CLICK_FRACTION = 0.75  # the button clicks at this share of its travel
PRELOAD_FRACTION = 0.6  # share of the click force the spring already pushes with at rest
PROTRUSION_MARGIN = 0.0005  # m the cap still stands above the panel at full travel
CLEARANCE = 0.003  # m above the cap while moving beside the button
# The skins' field of view is this much wider than the fingertips: a contact on a
# fingertip's very edge lies exactly on the boundary and would otherwise be lost.
FOV_MARGIN = 1.001
REST_S = 0.3  # s of stillness at the end of an episode
MAX_EPISODE_S = 40.0
TASK = "press the button"

# Orientation of the TCP in the world: columns are the TCP axes. x_tcp = world y,
# y_tcp (jaw axis) = world x, z_tcp = world -z (pointing down into the panel).
TCP_ROT = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
TCP_ROT6D = TCP_ROT[:2].reshape(-1)  # the first two ROWS, the repository's rot6d


@dataclass(frozen=True)
class ButtonTaskConfig:
    """The fingertips, their spring and the panel: what stays fixed within a dataset."""

    stiffness: float = 400.0  # N/m in x, y and z, the Franka bring-up's translational gain
    mass: float = 2.0  # kg moved by the spring
    damping_ratio: float = 1.0  # 1 = critical
    friction: float = 0.5  # sliding friction between fingertips and panel or button
    nx: int = 6  # taxels per fingertip along the jaw axis (world x)
    ny: int = 12  # taxels per fingertip across it (world y)
    pad_half_x: float = 0.006  # m
    pad_half_y: float = 0.011  # m
    pad_gap: float = 0.002  # m between the two closed fingertips
    button_radius: float = 0.006  # m
    button_mass: float = 0.02  # kg, the cap and what moves with it
    gripper: float = 0.0  # the constant gripper reading of the closed gripper


@dataclass(frozen=True)
class EpisodeParams:
    """Everything that changes from one demonstration to the next."""

    button_xy: tuple[float, float]  # m, button centre on the panel
    activation_force: float  # N needed to click it
    travel: float  # m the button can be pushed in
    start: tuple[float, float, float]  # m, TCP position at the start
    approach_height: float  # m above the cap where the descent begins
    aim_error: tuple[float, float]  # m, where the demonstrator aims, from the button centre
    v_free: float  # m/s, approach and lift
    v_press: float  # m/s, the command's descent while pressing
    hold_s: float  # s held after the click
    slide_force: float  # N, the push the slide is commanded with
    slide_angle: float  # rad, direction from the button to the slide, in the panel plane
    slide_distance: float  # m along the panel
    v_slide: float  # m/s
    lift_height: float  # m above the panel at the end


@dataclass
class Episode:
    """One simulated demonstration: its parameters, outcome and dataset columns."""

    params: EpisodeParams
    clicked: bool
    columns: dict[str, np.ndarray] = field(default_factory=dict)  # key -> [frames, dim] float32
    phases: list[str] = field(default_factory=list)  # demonstrator phase of each frame


def sample_params(rng: np.random.Generator) -> EpisodeParams:
    """Draw one demonstration's parameters; the ranges stand in for the unknown real button."""
    u = rng.uniform
    bx, by = u(-0.03, 0.03), u(-0.03, 0.03)
    aim = np.clip(rng.normal(0.0, 0.0015, size=2), -0.003, 0.003)
    return EpisodeParams(
        button_xy=(bx, by),
        activation_force=u(5.0, 15.0),
        travel=u(0.002, 0.004),
        start=(bx + u(-0.04, 0.04), by + u(-0.04, 0.04), u(0.04, 0.07)),
        approach_height=u(0.008, 0.015),
        aim_error=(float(aim[0]), float(aim[1])),
        v_free=u(0.06, 0.12),
        v_press=u(0.01, 0.03),
        hold_s=u(0.2, 0.5),
        slide_force=u(2.0, 5.0),
        slide_angle=u(0.0, 2 * math.pi),
        slide_distance=u(0.025, 0.04),
        v_slide=u(0.02, 0.04),
        lift_height=u(0.03, 0.05),
    )


def button_spring(params: EpisodeParams) -> dict[str, float]:
    """The button's spring: it pushes back with a preload at rest and clicks at `activation_force`.

    Returns the click depth (m), the stiffness (N/m), the spring's reference
    position (m above the rest position, which gives the preload) and how far
    the cap stands above the panel at rest (m).
    """
    click_depth = CLICK_FRACTION * params.travel
    preload = PRELOAD_FRACTION * params.activation_force
    k = (params.activation_force - preload) / click_depth
    return {
        "click_depth": click_depth,
        "stiffness": k,
        "springref": preload / k,
        "protrusion": params.travel + PROTRUSION_MARGIN,
    }


def _pad_taxels(cfg: ButtonTaskConfig, centre_x: float) -> str:
    """One small sphere per taxel, its lowest point on the fingertip face.

    One collision shape per taxel spreads the contact over the taxels (a single
    shape would touch at one point and light one taxel, see spring_pad.py).
    Spheres rather than boxes: a box pair gives up to four contacts, so the
    144 boxes sliding on the panel gave hundreds of contacts that flickered on
    and off; a sphere gives one.
    """
    sx, sy = cfg.pad_half_x / cfg.nx, cfg.pad_half_y / cfg.ny
    radius = 0.9 * min(sx, sy)
    return "\n      ".join(
        f'<geom type="sphere" size="{radius:.6f}" '
        f'pos="{centre_x - cfg.pad_half_x + sx * (2 * i + 1):.6f} '
        f'{-cfg.pad_half_y + sy * (2 * j + 1):.6f} {radius:.6f}"/>'
        for j in range(cfg.ny)
        for i in range(cfg.nx)
    )


def pad_centres(cfg: ButtonTaskConfig) -> tuple[float, float]:
    """World-x offsets of the left and right fingertip centres from the TCP."""
    offset = cfg.pad_gap / 2 + cfg.pad_half_x
    return -offset, offset


def model_xml(cfg: ButtonTaskConfig, params: EpisodeParams) -> str:
    """The MuJoCo model of one episode as XML text (the button differs per episode)."""
    spring = button_spring(params)
    left, right = pad_centres(cfg)
    damping = 2 * cfg.damping_ratio * math.sqrt(cfg.stiffness * cfg.mass)
    button_damping = 2 * math.sqrt(spring["stiffness"] * cfg.button_mass)
    cap_half = (spring["protrusion"] + 0.005) / 2  # the cap reaches 5 mm below the panel top
    fov_x = math.degrees(math.atan(FOV_MARGIN * cfg.pad_half_x / SITE_HEIGHT))
    fov_y = math.degrees(math.atan(FOV_MARGIN * cfg.pad_half_y / SITE_HEIGHT))
    sensors = "\n    ".join(
        f'<plugin name="skin_{side}" plugin="mujoco.sensor.touch_grid" objtype="site" '
        f'objname="skin_{side}"><config key="size" value="{cfg.nx} {cfg.ny}"/>'
        f'<config key="fov" value="{fov_x:.6f} {fov_y:.6f}"/><config key="gamma" value="0"/>'
        f'<config key="nchannel" value="3"/></plugin>'
        for side in ("left", "right")
    )
    actuators = "\n    ".join(
        f'<position name="spring_{a}" joint="tcp_{a}" kp="{cfg.stiffness}" ctrlrange="-1 1"/>'
        for a in "xyz"
    )
    joints = "\n      ".join(
        f'<joint name="tcp_{a}" type="slide" axis="{axis}" damping="{damping:.6f}"/>'
        for a, axis in zip("xyz", ("1 0 0", "0 1 0", "0 0 1"))
    )
    bx, by = params.button_xy
    return f"""
<mujoco model="button_task">
  <extension><plugin plugin="mujoco.sensor.touch_grid"/></extension>
  <option timestep="{1.0 / (RATE_HZ * STEPS_PER_FRAME):.9f}" gravity="0 0 0"/>
  <default><geom friction="{cfg.friction} 0.005 0.0001" solref="{CONTACT_SOLREF}"
                 solimp="{CONTACT_SOLIMP}"/></default>
  <worldbody>
    <geom name="panel" type="box" pos="0 0 -0.01" size="0.2 0.2 0.01"/>
    <body name="button" pos="{bx:.6f} {by:.6f} 0">
      <joint name="button" type="slide" axis="0 0 1" range="{-params.travel:.6f} 0"
             stiffness="{spring["stiffness"]:.6f}" springref="{spring["springref"]:.6f}"
             damping="{button_damping:.6f}"/>
      <inertial pos="0 0 0" mass="{cfg.button_mass}" diaginertia="1e-7 1e-7 1e-7"/>
      <geom name="cap" type="cylinder" pos="0 0 {spring["protrusion"] - cap_half:.6f}"
            size="{cfg.button_radius:.6f} {cap_half:.6f}"/>
    </body>
    <body name="tcp" pos="0 0 0">
      {joints}
      <inertial pos="0 0 0" mass="{cfg.mass}" diaginertia="1e-3 1e-3 1e-3"/>
      {_pad_taxels(cfg, left)}
      {_pad_taxels(cfg, right)}
      <!-- touch_grid looks along its site's -z axis, like a camera: the sites
           sit above the downward-facing skins with +z up. -->
      <site name="skin_left" pos="{left:.6f} 0 {SITE_HEIGHT}" zaxis="0 0 1"/>
      <site name="skin_right" pos="{right:.6f} 0 {SITE_HEIGHT}" zaxis="0 0 1"/>
    </body>
  </worldbody>
  <contact><exclude body1="world" body2="button"/></contact>
  <actuator>
    {actuators}
  </actuator>
  <sensor>
    {sensors}
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


class _Sim:
    """The compiled model of one episode with named access to what the recorder reads."""

    def __init__(self, cfg: ButtonTaskConfig, params: EpisodeParams) -> None:
        self.mj = _mujoco()
        self.cfg, self.params = cfg, params
        self.model = self.mj.MjModel.from_xml_string(model_xml(cfg, params))
        self.data = self.mj.MjData(self.model)
        self.tcp_q = [int(self.model.joint(f"tcp_{a}").qposadr[0]) for a in "xyz"]
        self.button_q = int(self.model.joint("button").qposadr[0])
        self.tcp_body = int(self.model.body("tcp").id)
        self.spring = button_spring(params)
        self.data.qpos[self.tcp_q] = params.start
        self.data.ctrl[:] = params.start
        self.mj.mj_forward(self.model, self.data)

    def tcp(self) -> np.ndarray:
        return np.array(self.data.qpos[self.tcp_q], dtype=np.float64)

    def depth(self) -> float:
        """How far the button is pushed in, m (0 at rest)."""
        return max(0.0, -float(self.data.qpos[self.button_q]))

    def cap_top(self) -> np.ndarray:
        bx, by = self.params.button_xy
        return np.array([bx, by, self.spring["protrusion"] - self.depth()])

    def taxels(self, side: str) -> np.ndarray:
        """Normal force on each taxel of one fingertip, N, as [ny, nx]."""
        sensor = self.model.sensor(f"skin_{side}")
        values = self.data.sensordata[int(sensor.adr[0]) : int(sensor.adr[0] + sensor.dim[0])]
        return np.array(values).reshape(3, self.cfg.ny, self.cfg.nx)[0]

    def sensed(self) -> np.ndarray:
        """Left taxels, right taxels and the contact force on the fingertips, as one vector.

        The force is what the panel and the button exert on the fingertips, in
        the world frame, N (MuJoCo's `cfrc_ext`, the external force on the body).
        """
        self.mj.mj_rnePostConstraint(self.model, self.data)
        force = np.array(self.data.cfrc_ext[self.tcp_body, 3:6])
        return np.concatenate(
            [self.taxels("left").reshape(-1), self.taxels("right").reshape(-1), force]
        )

    def step(self, command: np.ndarray) -> None:
        self.data.ctrl[:] = command
        self.mj.mj_step(self.model, self.data)


class _Demonstrator:
    """Moves the command like a teleoperator: approach, press, hold, slide beside it, lift.

    After the press the fingertips lift clear of the cap before they move aside
    and come down on the panel. Sliding straight off the cap's edge instead
    drops them onto the panel while part of them is still over the cap, which
    MuJoCo resolves by pushing the button through its stop: an artefact.
    """

    def __init__(self, cfg: ButtonTaskConfig, params: EpisodeParams, dt: float) -> None:
        self.p, self.dt = params, dt
        protrusion = button_spring(params)["protrusion"]
        bx, by = params.button_xy
        self.command = np.array(params.start, dtype=np.float64)
        self.aim = np.array(
            [
                bx + params.aim_error[0],
                by + params.aim_error[1],
                protrusion + params.approach_height,
            ]
        )
        # the push while sliding, counted from the panel's top face
        self.slide_z = -params.slide_force / cfg.stiffness
        # give up far past where the click force would have been reached
        self.deepest = protrusion - 2 * params.activation_force / cfg.stiffness - 0.02
        direction = np.array([math.cos(params.slide_angle), math.sin(params.slide_angle), 0.0])
        # far enough that no part of either fingertip is over the cap
        reach = math.hypot(cfg.pad_gap / 2 + 2 * cfg.pad_half_x, cfg.pad_half_y)
        beside = np.array([bx, by, 0.0]) + (cfg.button_radius + reach + 0.003) * direction
        self.clear_z = protrusion + CLEARANCE
        self.beside = beside + np.array([0.0, 0.0, self.clear_z])
        self.slide_start = beside + np.array([0.0, 0.0, self.slide_z])
        self.slide_end = self.slide_start + params.slide_distance * direction
        self.phase, self.elapsed = "approach", 0.0
        self.clicked = False

    def _move_to(self, target: np.ndarray, speed: float) -> bool:
        """Move the command toward `target` at `speed`; True once it is there."""
        delta = target - self.command
        dist = float(np.linalg.norm(delta))
        step = speed * self.dt
        if dist <= step:
            self.command = target.astype(np.float64).copy()
            return True
        self.command = self.command + delta / dist * step
        return False

    def _enter(self, phase: str) -> None:
        self.phase, self.elapsed = phase, 0.0

    def step(self, pressed: bool) -> bool:
        """Advance one physics step; True when the episode is over."""
        p = self.p
        self.elapsed += self.dt
        if self.phase == "approach":
            if self._move_to(self.aim, p.v_free):
                self._enter("press")
        elif self.phase == "press":
            if pressed:
                self.clicked = True
                self._enter("hold")
            elif self.command[2] <= self.deepest:
                self._enter("lift")
            else:
                self.command[2] -= p.v_press * self.dt
        elif self.phase == "hold":
            if self.elapsed >= p.hold_s:
                self._enter("release")
        elif self.phase == "release":
            if self._move_to(np.array([*self.command[:2], self.clear_z]), p.v_free):
                self._enter("aside")
        elif self.phase == "aside":
            if self._move_to(self.beside, p.v_free):
                self._enter("touch")
        elif self.phase == "touch":
            if self._move_to(self.slide_start, p.v_press):
                self._enter("slide")
        elif self.phase == "slide":
            if self._move_to(self.slide_end, p.v_slide):
                self._enter("lift")
        elif self.phase == "lift":
            if self._move_to(np.array([*self.command[:2], p.lift_height]), p.v_free):
                self._enter("rest")
        elif self.phase == "rest":
            return self.elapsed >= REST_S
        return False


def _observe(sim: _Sim, command: np.ndarray, sensed: np.ndarray) -> dict[str, np.ndarray]:
    """What one frame records, before it is turned into dataset columns.

    `sensed` is `_Sim.sensed()` averaged over the frame; the poses, the command
    and the button are read at the frame's instant, as the recorder reads them.
    """
    tcp = sim.tcp()
    n = sim.cfg.nx * sim.cfg.ny
    return {
        "tcp": tcp,
        "command": command.copy(),
        "button_in_tcp": TCP_ROT.T @ (sim.cap_top() - tcp),
        "left": sensed[:n],
        "right": sensed[n : 2 * n],
        "contact_force": sensed[2 * n :],
        "depth": np.array([sim.depth()]),
        "pressed": np.array([float(sim.depth() >= sim.spring["click_depth"])]),
    }


def to_columns(raw: list[dict[str, np.ndarray]], cfg: ButtonTaskConfig) -> dict[str, np.ndarray]:
    """Turn the observed frames into the dataset's columns; the last frame only gives an action."""
    if len(raw) < 2:
        raise ValueError("an episode needs at least two frames: one to keep and one for its action")

    def stack(key: str) -> np.ndarray:
        return np.stack([r[key] for r in raw]).astype(np.float64)

    n = len(raw)
    rot = np.broadcast_to(TCP_ROT6D, (n, 6))
    pose = np.concatenate([stack("tcp"), rot], axis=1)
    gripper = np.full((n, 1), cfg.gripper)
    button = stack("button_in_tcp")
    columns = {
        "observation.state.cartesian": pose[:-1],
        "observation.state.gripper": gripper[:-1],
        "observation.state.button_in_tcp": button[:-1],
        "observation.state": np.concatenate([pose, gripper, button], axis=1)[:-1],
        "observation.environment_state": button[:-1],
        "action": np.concatenate([pose[1:], gripper[1:]], axis=1),
        "extra.target_cartesian": np.concatenate([stack("command"), rot], axis=1)[:-1],
        "extra.tactile.left": stack("left")[:-1],
        "extra.tactile.right": stack("right")[:-1],
        "extra.contact_force": stack("contact_force")[:-1],
        "extra.button_depth": stack("depth")[:-1],
        "extra.button_pressed": stack("pressed")[:-1],
    }
    return {k: v.astype(np.float32) for k, v in columns.items()}


def run_episode(cfg: ButtonTaskConfig, params: EpisodeParams) -> Episode:
    """Simulate one demonstration and return its columns at RATE_HZ.

    The skins and the contact force are AVERAGED over each frame, as a sensor's
    filter would. Sliding with friction makes MuJoCo's fingertips hop by a few
    micrometres, so the force at one instant is either zero or a multiple of
    the push (seen at every contact setting tried, gone without friction); its
    average over a frame equals the push.
    """
    sim = _Sim(cfg, params)
    demo = _Demonstrator(cfg, params, float(sim.model.opt.timestep))
    raw, phases = [], []
    total, count = sim.sensed() * 0.0, 0
    max_steps = int(MAX_EPISODE_S * RATE_HZ * STEPS_PER_FRAME)
    for step in range(max_steps):
        if step % STEPS_PER_FRAME == 0:
            raw.append(_observe(sim, demo.command, total / count if count else sim.sensed()))
            phases.append(demo.phase)
            total, count = total * 0.0, 0
        if demo.step(sim.depth() >= sim.spring["click_depth"]):
            break
        sim.step(demo.command)
        total, count = total + sim.sensed(), count + 1
    raw.append(_observe(sim, demo.command, total / count if count else sim.sensed()))
    return Episode(params, demo.clicked, to_columns(raw, cfg), phases)
