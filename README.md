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

## Documentation

Each document answers one question. Start with the one that matches what you are doing.

| document | answers | typical entry point |
|---|---|---|
| this README | what is this, how do I install it | first time here |
| **[USAGE.md](USAGE.md)** | *how do I do X?* — ordered recipes: record → align/merge datasets → train → deploy → diagnose | the day-to-day reference |
| [crisp_gym/scripts/README.md](crisp_gym/scripts/README.md) | *what does this script or flag do?* — training and deployment script reference | you know the step, you need the flags |
| [REMOTE_INFERENCE.md](REMOTE_INFERENCE.md) | the websocket contract and wire protocol for serving a policy from the GPU machine | building or debugging the server |
| [HANDOFF.md](HANDOFF.md) | *what must I not break?* — pose/data conventions, invariants, verification procedure | before changing pose math, recording, or the data schema |
| [CHANGELOG.md](CHANGELOG.md) | released versions | — |

The UMI-style data pipeline — handheld and robot recording, dataset alignment
and merging, relative-pose training, deployment — is covered end to end in
[USAGE.md](USAGE.md).

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
bash scripts/setup_lerobot.sh        # clones LeRobot v0.6.1 to ../lerobot
rm -f pixi.lock
pixi install -e humble-lerobot
```

`scripts/setup_lerobot.sh` clones **v0.6.1**, which is what `pixi.toml`'s
`humble-lerobot` environment expects (Python 3.12 + numpy 2). At that version
**no patching is needed**: LeRobot's own `opencv-python-headless` bound (<4.14)
accepts the conda-provided opencv, and `pixi.toml` pins numpy and packaging to
LeRobot's own ranges. If you already cloned LeRobot yourself the script detects
it, skips the clone, and warns when the checkout is a different revision.

Clone a different revision with `LEROBOT_REV=v0.4.4 bash scripts/setup_lerobot.sh`
— but note 0.4.x has **no `dataset` extra** (added in 0.6.x), so `pixi.toml`'s
`extras = ["dataset"]` will not resolve against it and must be pinned back at the
same time.

> **Do not set `LEROBOT_PATCH_NUMPY1=1` on this environment.** Those patches
> relax the clone's `requires-python` to 3.11, `numpy` to >=1.26 and drop
> `rerun-sdk`; they are correct only when the pixi env is itself on Python 3.11
> + numpy 1.26. Applied to the current numpy-2 environment they silently undo
> the requirements it was migrated to. They are opt-in for exactly that reason.

## Data pipeline: record → train → deploy

```mermaid
flowchart TD
    subgraph RECORD["1 — Record (robot PC, ROS 2 Humble)"]
        H["Handheld UMI gripper<br/>scripts/record_umi_handheld.py"]
        R["Robot teleop (FACTR / leader-follower)<br/>scripts/record_lerobot_format_leader_follower.py<br/>--record-config umi_robot_record.yaml"]
        OLD["LEGACY dataset<br/>(old recorder: Euler pose obs +<br/>delta-command actions)"]
        MIG["ONE-TIME MIGRATION<br/>scripts/migrate_euler_delta_to_rot6d.py<br/>--input old --output old_rot6d<br/>(videos byte-identical; Euler(6)→rot6d(9);<br/>action rebuilt as next_tcp_pose)"]
        DS[("LeRobot dataset — ABSOLUTE poses<br/>obs: [x,y,z, rot6d(6)] + gripper + images<br/>action: absolute TCP pose t+1 + gripper<br/>+ meta/record_config.json")]
        H --> DS
        R --> DS
        OLD --> MIG --> DS
    end

    CHK["VERIFY before training<br/>scripts/check_relative_pose.py<br/>(identity, round-trip, rot6d sanity)"]
    DS --> CHK

    subgraph TRAIN["2 — Train (GPU PC, lerobot 0.4.x or ≥0.5)"]
        TR["scripts/lerobot_relative_pose.py<br/>+ any lerobot-train args"]
        W["RelativePoseDataset (in __getitem__):<br/>abs → RELATIVE wrt last obs frame;<br/>state = [rel_pose9, gripper1, rot_wrt_start6] (16-D);<br/>relative stats recomputed"]
        CKPT[("checkpoint<br/>+ pose_repr.json<br/>(generation stamp)")]
        TR --> W --> CKPT
    end
    CHK --> TR

    subgraph DEPLOY["3 — Deploy (robot PC)"]
        GEN{"pose_repr.json?<br/>(state_input: auto)"}
        G1["missing / absolute → gen-1:<br/>ABSOLUTE 10-D state"]
        G3["state_includes_wrt_start → gen-3:<br/>RELATIVE 16-D state (wrt-start appended,<br/>noise off, episode start = first obs)"]
        LOC["LOCAL verification<br/>scripts/deploy_policy.py<br/>--policy-config relative_lerobot_policy<br/>--env-config *_deploy_umi (rot6d,<br/>use_relative_actions: false)"]
        REM["REMOTE (canonical)<br/>RemotePolicy ⇆ websocket server<br/>contracts: config/policy/remote_umi_*.yaml"]
        ACT["every action chunk composed<br/>T_cmd = T_tcp(obs time) ∘ T_rel"]
        CKPT --> GEN
        GEN --> G1 --> LOC
        GEN --> G3 --> LOC
        G1 -.-> REM
        G3 -.-> REM
        LOC --> ACT
        REM --> ACT
    end
```

Notes:

- **Never store relative poses on disk** — datasets hold ABSOLUTE poses; the
  relative conversion happens inside the training wrapper and (mirrored) at
  inference. Details/invariants: [HANDOFF.md](HANDOFF.md) §1.
- **Migrated legacy data caveat (gripper units):** the old recorder stored the
  device-normalized gripper value, not the UMI reference-width scale. Don't
  mix migrated and UMI-recorded datasets in one training, and when deploying a
  model trained on migrated data set `reference_width: == device_max_width`
  in `config/policy/relative_lerobot_policy.yaml` so the unit conversion is
  identity.
- **Merging datasets recorded on different rigs** (two arms, different codecs,
  different extra states) needs the schemas, the video codecs and the
  statistics reconciled first — `aggregate_datasets` applies four separate
  compatibility gates and two of them fail only at the very end of the merge.
  Step-by-step, with a table of what blocks a merge and what fixes it:
  [USAGE.md §6](USAGE.md#6-post-process-align-and-merge-datasets).
- Full step-by-step commands: [USAGE.md](USAGE.md).

## Deploying a trained policy

Models trained with `scripts/lerobot_relative_pose.py` output UMI-style
RELATIVE poses that must be composed with the TCP pose captured at observation
time. Two paths:

- **Remote (canonical)** — inference on the GPU machine over websocket, the
  robot machine stays torch-free. Contract and wire protocol:
  [REMOTE_INFERENCE.md](REMOTE_INFERENCE.md).
- **Local** — in-process on the robot machine, for verification and only for
  checkpoints trained against that machine's own lerobot:

  ```bash
  python -m crisp_gym.scripts.deploy_policy \
      --env-config ur7e_robotiq_deploy_umi \
      --policy-config relative_lerobot_policy \
      --path outputs/train/<run>/checkpoints/last/pretrained_model \
      --repo-id my_org/deploy_eval --fps 15
  ```

Gripper widths, camera topics and the `rotation_6d` / `use_relative_actions`
requirements of the deploy env are prerequisites, not defaults — and a
checkpoint carries a *generation* that decides what `observation.state` it
expects. Both are in [USAGE.md §9](USAGE.md#9-deploy-a-trained-policy); every
flag and policy config is in
[crisp_gym/scripts/README.md](crisp_gym/scripts/README.md#deployment).

Check the [docs](https://utiasdsl.github.io/crisp_controllers/getting_started/#4-using-the-gym) to get started.
