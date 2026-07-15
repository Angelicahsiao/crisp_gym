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

## Installation

The environments run inside a [pixi](https://pixi.sh) workspace. The
`humble-lerobot` environment (ROS 2 Humble + LeRobot) needs a **local LeRobot
clone next to this repo** — it is installed editable from `../lerobot`, so a
sibling of `crisp_gym` (e.g. `/workspace/lerobot` alongside `/workspace/crisp_gym`).

**Set up LeRobot first, then install the environment:**

```bash
# from the crisp_gym repo root
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

Check the [docs](https://utiasdsl.github.io/crisp_controllers/getting_started/#4-using-the-gym) to get started.
