"""Draw a circle using the ManipulatorCartesianEnv with a UR robot."""

import numpy as np

from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.envs.manipulator_env_config import UREnvConfig

# %% === Circle Parameters ===
RADIUS = 0.1  # [m]
CENTER = np.array([0.4, 0.0, 0.4])
CTRL_FREQ = 50  # control frequency in Hz
SIN_FREQ = 0.25  # frequency of circular motion in Hz
ITERATIONS = 5  # number of full circles to draw

# %% === Environment Setup ===
env_config = UREnvConfig(control_frequency=CTRL_FREQ)
env = ManipulatorCartesianEnv(namespace="", config=env_config)

# %% === Move to Starting Point ===
start_position = CENTER + [0, RADIUS, 0]
print(f"Moving to start position: {start_position}")
env.move_to(position=start_position, speed=0.15)
obs, _ = env.reset()
import time; time.sleep(15)

# %%=== Generate Circle Trajectory ===
time_period = ITERATIONS / SIN_FREQ  # total time in seconds
steps = int(time_period * CTRL_FREQ)

angles = 2 * np.pi * SIN_FREQ * np.arange(steps) / CTRL_FREQ
x = RADIUS * np.sin(angles)
y = RADIUS * np.cos(angles)

# Velocity (finite difference)
dx = np.diff(np.concatenate([[x[-1]], x]))
dy = np.diff(np.concatenate([[y[-1]], y]))

print(f"Drawing circle for {ITERATIONS} iterations with {steps} steps.")

# %% === Execute Circular Motion ===
for t in range(steps):
    action = np.zeros((7))

    idxs = t % steps
    action[0] = dx[idxs]
    action[1] = dy[idxs]

    obs, _, _, _, _ = env.step(action, block=True)

print("Circle drawing complete.")
print("Going back home.")
env.home()

env.close()
