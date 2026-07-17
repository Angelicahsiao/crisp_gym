![crisp_gym](media/crisp_gym_logo.webp)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![MIT Badge](https://img.shields.io/badge/MIT-License-blue?style=flat)
<a href="https://github.com/utiasDSL/crisp_gym/actions/workflows/ruff_ci.yml"><img src="https://github.com/utiasDSL/crisp_gym/actions/workflows/ruff_ci.yml/badge.svg"/></a>
<a href="https://utiasDSL.github.io/crisp_controllers/"><img alt="Static Badge" src="https://img.shields.io/badge/docs-passing-blue?style=flat&link=https%3A%2F%2FutiasDSL.github.io%2Fcrisp_controllers%2F"></a>
<a href="https://github.com/utiasDSL/crisp_gym/actions/workflows/pixi_ci.yml"><img src="https://github.com/utiasDSL/crisp_gym/actions/workflows/pixi_ci.yml/badge.svg"/></a>
<a href="https://utiasDSL.github.io/crisp_controllers#citing"><img alt="Static Badge" src="https://img.shields.io/badge/arxiv-cite-b31b1b?style=flat"></a>
<img width="60" alt="lerobot-tag" src="https://github.com/user-attachments/assets/441b1d03-43d4-4cb9-bc08-ef56f119933a" />

This repository contains Gymnasium environments to train and deploy high-level learning-based policies from [LeRobot](https://github.com/huggingface/lerobot) using [CRISP_PY](https://github.com/utiasDSL/crisp_py) and the [CRISP controllers](https://github.com/utiasDSL/crisp_controllers).

Check the [docs](https://utiasdsl.github.io/crisp_controllers/getting_started/#4-using-the-gym) to get started.

For the UMI-style data pipeline (handheld/robot recording, dataset alignment, relative-pose training, deployment), see **[USAGE.md](USAGE.md)**; remote model serving is specified in [REMOTE_INFERENCE.md](REMOTE_INFERENCE.md); development conventions live in [HANDOFF.md](HANDOFF.md).

## Workspace layout

Clone `crisp_gym` and its sibling repositories into a common workspace folder
(e.g. `~/workspace`) — several paths assume the repos sit **next to each
other**:

```
workspace/
├── crisp_controllers_demos/   # robot bring-up (Docker); mounts ../crisp_py into its containers
├── crisp_gym/                 # this repository
├── crisp_py/                  # robot/gripper Python client
└── lerobot/                   # created by crisp_gym/scripts/setup_lerobot.sh (../lerobot)
```

```bash
mkdir -p ~/workspace && cd ~/workspace
git clone https://github.com/utiasDSL/crisp_controllers_demos.git
git clone https://github.com/utiasDSL/crisp_gym.git
git clone https://github.com/utiasDSL/crisp_py.git
```

Concretely, the sibling layout matters because:

- `pixi.toml` installs LeRobot editable from `../lerobot` (cloned by
  `scripts/setup_lerobot.sh` — see Installation below),
- `pixi.toml` installs `crisp_py` editable from `../crisp_py` (the local
  checkout satisfies the `crisp_python` requirement instead of the lagging
  PyPI wheel),
- `crisp_controllers_demos/docker-compose.yaml` bind-mounts `../crisp_py`
  into the robot containers.

## Installation

The environments run inside a [pixi](https://pixi.sh) workspace. The
`humble-lerobot` environment (ROS 2 Humble + LeRobot) needs a **local LeRobot
clone next to this repo** — it is installed editable from `../lerobot`, so a
sibling of `crisp_gym` (e.g. `/workspace/lerobot` alongside `/workspace/crisp_gym`).

**Set up LeRobot first, then install the environment:**

```bash
cd /workspace/crisp_gym              # the crisp_gym repo root
bash scripts/setup_lerobot.sh        # clones LeRobot v0.4.4 to ../lerobot and
                                     # patches its pyproject for ROS 2 Humble
rm -f pixi.lock
pixi install -e humble-lerobot
```

`scripts/setup_lerobot.sh` does two things you must not skip:

1. **Clones** LeRobot to `../lerobot` (override the version with
   `LEROBOT_REV=v0.5.1 bash scripts/setup_lerobot.sh`). If you already cloned it
   manually, the script detects the existing directory, skips the clone, and
   still applies the patches below — so run it anyway.
2. **Patches** LeRobot's `pyproject.toml` for the Humble stack, most importantly
   **removing the `rerun-sdk` dependency**. `rerun-sdk>=0.24` requires
   `numpy>=2`, which conflicts with the `numpy==1.26.4` that ROS 2 Humble
   (robostack) pins — without this patch `pixi install -e humble-lerobot` fails
   with:

   > Because rerun-sdk>=0.24.0,<=0.26.2 depends on numpy>=2 and numpy==1.26.4,
   > we can conclude that rerun-sdk … cannot be used … lerobot==0.4.4 cannot be used.

   `crisp_gym` does not use `rerun`; only LeRobot's standalone
   `visualize_dataset.py` does, which is unaffected by recording/training/inference.

> If you cloned LeRobot manually and hit the `rerun-sdk` / `numpy` conflict above,
> just run `bash scripts/setup_lerobot.sh` (it will patch your existing clone),
> then `rm -f pixi.lock && pixi install -e humble-lerobot`.

## Deploying a relative-pose (rot6d) model

Models trained with `scripts/lerobot_relative_pose.py` output UMI-style
RELATIVE poses that must be composed with the TCP pose captured at
observation time. Two deployment paths:

**Local (verification)** — robot machine's lerobot matches the training
version (0.4.4). The `relative_lerobot_policy` runs inference in a worker
process and handles the composition, gripper unit conversion, and obs
history:

```bash
python -m crisp_gym.scripts.deploy_policy \
    --env-config ur7e_robotiq_deploy_umi \
    --policy-config relative_lerobot_policy \
    --path outputs/train/<run>/checkpoints/last/pretrained_model \
    --repo-id my_org/deploy_eval --fps 15
```

Before running: set `device_max_width` in
`config/policy/relative_lerobot_policy.yaml` to YOUR gripper (0.140 for a
Robotiq 2F-140), and point the deploy env config's `primary` camera at the
topics you recorded with. The deploy env must keep
`orientation_representation: rotation_6d` and `use_relative_actions: false`
(see `config/envs/ur7e_robotiq_deploy_umi.yaml` for why).

At startup the worker logs the checkpoint's input features and, on the first
inference, the exact `observation.state` fed to the policy with an
absolute/relative heuristic — use it to sanity-check what a checkpoint was
trained on.

**Remote (canonical)** — inference on the GPU machine over websocket, robot
machine stays torch-free. Contract and wire protocol: [REMOTE_INFERENCE.md](REMOTE_INFERENCE.md).

**Checkpoint generations.** Training stamps `pose_repr.json` next to the
checkpoints recording the pose conventions and (critically) what the
policy's `observation.state` input was. Three generations exist:
ABSOLUTE 10-D (checkpoints trained before the wrapper converted the
concatenated state — includes any checkpoint without the stamp),
RELATIVE 10-D (converted state, no wrt-start), and RELATIVE 16-D
(current wrapper — UMI parity: `[rel_pose9, gripper1, rot_wrt_start6]`,
with the episode-start relative rotation appended). Deployment
auto-detects this from the stamp (`state_input: auto`); don't mix them up
manually. Remote-inference contract templates per generation:
`config/policy/remote_umi_absolute_state.yaml` (gen 1) and
`config/policy/remote_umi_relative_state.yaml` (gen 3).

Check the [docs](https://utiasdsl.github.io/crisp_controllers/getting_started/#4-using-the-gym) to get started.
