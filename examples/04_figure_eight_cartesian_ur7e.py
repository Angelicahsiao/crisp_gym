"""Draw a figure-eight on the YZ plane using ManipulatorCartesianEnv with a UR7e robot.

Config: crisp_gym/config/envs/ur7e.yaml
Controller: cartesian_impedance_controller (ur_cartesian_impedance.yaml)
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.envs.manipulator_env_config import make_env_config

# === Parameters ===
RADIUS = 0.15         # [m]
CENTER = np.array([0.4, 0.0, 0.4])
CTRL_FREQ = 50        # Hz
SIN_FREQ_Y = 0.25     # Hz — y oscillation (full figure-eight period = 4 s)
SIN_FREQ_Z = 0.125    # Hz — z oscillation at half frequency → figure eight
MAX_TIME = 8.0        # seconds (two full figure-eights)

# === Environment ===
env_config = make_env_config("ur7e", control_frequency=CTRL_FREQ)
env = ManipulatorCartesianEnv(namespace="", config=env_config)

print("Moving to center position...")
env.move_to(position=CENTER, speed=0.10)
obs, _ = env.reset()

# === Pre-compute trajectory ===
steps = int(MAX_TIME * CTRL_FREQ)
t_arr = np.arange(steps) / CTRL_FREQ

y_traj = RADIUS * np.sin(2 * np.pi * SIN_FREQ_Y * t_arr)
z_traj = RADIUS * np.sin(2 * np.pi * SIN_FREQ_Z * t_arr)

# Finite-difference deltas (action space is positional increments)
dy = np.diff(np.concatenate([[y_traj[-1]], y_traj]))
dz = np.diff(np.concatenate([[z_traj[-1]], z_traj]))

# === Execute figure-eight ===
ee_poses = []
target_poses = []

print("Drawing figure eight...")
for i in range(steps):
    action = np.zeros(7)
    action[1] = dy[i]
    action[2] = dz[i]
    obs, _, _, _, _ = env.step(action, block=True)
    ee_poses.append(env.robot.end_effector_pose.copy())
    target_poses.append(env.robot.target_pose.copy())

# Settle for 1 s
print("Settling...")
for _ in range(int(CTRL_FREQ)):
    obs, _, _, _, _ = env.step(np.zeros(7), block=True)
    ee_poses.append(env.robot.end_effector_pose.copy())
    target_poses.append(env.robot.target_pose.copy())

t_full = np.arange(len(ee_poses)) / CTRL_FREQ

# === Extract positions ===
y_ee = [p.position[1] for p in ee_poses]
z_ee = [p.position[2] for p in ee_poses]
y_tgt = [p.position[1] for p in target_poses]
z_tgt = [p.position[2] for p in target_poses]

# === Plot ===
fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 3, figure=fig)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])

ax0.plot(y_tgt, z_tgt, "--", label="target")
ax0.plot(y_ee, z_ee, label="EE")
ax0.set_xlabel("y [m]")
ax0.set_ylabel("z [m]")
ax0.set_title("YZ plane")
ax0.legend()
ax0.grid()

ax1.plot(t_full, y_tgt, "--", label="target")
ax1.plot(t_full, y_ee, label="EE")
ax1.set_xlabel("t [s]")
ax1.set_ylabel("y [m]")
ax1.set_title("Y vs time")
ax1.legend()
ax1.grid()

ax2.plot(t_full, z_tgt, "--", label="target")
ax2.plot(t_full, z_ee, label="EE")
ax2.set_xlabel("t [s]")
ax2.set_ylabel("z [m]")
ax2.set_title("Z vs time")
ax2.legend()
ax2.grid()

fig.tight_layout()
plt.show()

# === Return home ===
print("Returning home...")
env.home()
env.close()
