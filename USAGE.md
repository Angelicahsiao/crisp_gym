# Usage Guide — Recording, Post-Processing, Training, Deployment

Step-by-step guides for every supported workflow. Read `HANDOFF.md` first for
the data conventions (rot6d, absolute-on-disk / relative-at-train). All
commands run inside the ROS 2 container/environment unless noted otherwise.

Contents:
1. [Record with the UMI handheld gripper (OptiTrack)](#1-record-with-the-umi-handheld-gripper-optitrack)
2. [Record with a robot arm + leader teleop (UMI contract)](#2-record-with-a-robot-arm--leader-teleop-umi-contract)
3. [Record with a FACTR leader arm (joint teleop, UMI contract)](#3-record-with-a-factr-leader-arm-joint-teleop-umi-contract)
4. [Classic teleop recording (delta-pose commands)](#4-classic-teleop-recording-delta-pose-commands)
5. [Record all robot states (full-state recording)](#5-record-all-robot-states-full-state-recording)
6. [Post-process: align datasets for mixed training](#6-post-process-align-datasets-for-mixed-training)
7. [Post-process: promote extra states to policy inputs](#7-post-process-promote-extra-states-to-policy-inputs)
8. [Train (LeRobot 0.4.4, UMI-style relative pose)](#8-train-lerobot-044-umi-style-relative-pose)
9. [Deploy a trained policy](#9-deploy-a-trained-policy)
10. [Write your own record config](#10-write-your-own-record-config)
11. [Migrate legacy Euler + delta-command data to rot6d](#11-migrate-legacy-euler--delta-command-data-to-rot6d)

The keyboard recording manager is the same everywhere:
**r** start/stop episode · **s** save episode · **d** delete episode · **q** quit.

---

## 1. Record with the UMI handheld gripper (OptiTrack)

No robot required. Pose comes from OptiTrack, gripper width from a ROS 2 topic.

**Prerequisites**
- OptiTrack streaming via `mocap4ros2_optitrack`, plus a relay node publishing
  the rigid body as `geometry_msgs/PoseStamped` on `/optitrack/umi_gripper/pose`
  (RigidBodies → PoseStamped relay is outside this repo).
- Gripper width in meters as `std_msgs/Float32` on `/umi/gripper_width`.
- Wrist camera publishing (name `primary` in the env config).
- In Motive, define the rigid body so its axes match the robot TCP convention
  (z = approach through fingertips, y = jaw axis) or set `tx_body_tcp` in
  `crisp_gym/config/envs/umi_handheld.yaml`. This transform does NOT cancel in
  training — see HANDOFF.md §1.4.

**Run**
```bash
python crisp_gym/scripts/record_umi_handheld.py \
    --repo-id my_org/umi_handheld_demo \
    --tasks "pick the lego block" \
    --num-episodes 50 --fps 15
```
Uses `config/envs/umi_handheld.yaml` (env: topics, transforms, rot6d) and
`config/recording/umi_handheld_record.yaml` (data contract) by default;
override with `--env-config` / `--record-config`.

**Output**: LeRobot dataset; obs = absolute 9D rot6d TCP pose + gripper +
image; action = 10D absolute measured pose[t+1] + gripper. The contract is
stamped to `meta/record_config.json`.

**Sanity check after one episode**
```python
import pandas as pd, numpy as np
a = np.array(pd.read_parquet(".../episode_000000.parquet")["action"].iloc[0])
assert a.shape == (10,)
r1, r2 = a[3:6], a[6:9]      # rot6d rows: unit norm, orthogonal
print(np.linalg.norm(r1), np.linalg.norm(r2), np.dot(r1, r2))
```

---

## 2. Record with a robot arm + leader teleop (UMI contract)

The leader arm drives the follower; the dataset stores the follower's
**measured** TCP pose (action = pose[t+1]) — byte-compatible with handheld data.

**Prerequisites**
- Robot bringup with `cartesian_impedance_controller` and
  `joint_trajectory_controller` loaded (any crisp_controllers bringup).
- Wrist-mounted camera comparable to the handheld GoPro (same key `primary`).
- Set `device_max_width` in `config/recording/umi_robot_record.yaml` to YOUR
  gripper (0.140 Robotiq 2F-140, 0.08 Franka Hand).
- UR only: `URConfig.target_frame` defaults to `tool0` (the flange). Override
  it to a fingertip TCP frame, or absorb the offset into the handheld's
  `tx_body_tcp` — see the note in `config/envs/ur7e_robotiq.yaml`.

**Run**
```bash
python crisp_gym/scripts/record_lerobot_format_leader_follower.py \
    --record-config crisp_gym/config/recording/umi_robot_record.yaml \
    --follower-config <your_env> --follower-namespace <ns> \
    --leader-config <leader> --leader-namespace <ns> \
    --repo-id my_org/umi_robot_demo --fps 15
```
`--fps` must equal the record config's `rate_hz` (both 15 by default).

---

## 3. Record with a FACTR leader arm (joint teleop, UMI contract)

FACTR drives the follower in joint space; the dataset still stores measured
TCP poses (the FACTR stream itself is never recorded).

**Prerequisites**: UR bringup with Robotiq attached; FACTR node publishing
`/factr_teleop/{name}/cmd_ur_pos` and `/factr_teleop/{name}/cmd_gripper_pos`.

**Run**
```bash
python crisp_gym/scripts/record_lerobot_format_leader_follower.py \
    --use-factr --factr-name right --joint-control \
    --follower-config ur7e_robotiq \
    --record-config crisp_gym/config/recording/umi_robot_record.yaml \
    --repo-id my_org/umi_ur7e_factr --fps 15
```
`--use-factr` requires `--joint-control` and `--record-config`. Note the
teleop runs at the recording rate (15 Hz), slightly laggier than the 50 Hz
live example (`examples/09_factr_ur7e_teleop.py`).

---

## 4. Classic teleop recording (delta-pose commands)

The original crisp_gym behavior: action = the delta command sent to the robot.
**NOT mixable** with UMI-contract datasets (the alignment script will refuse).

```bash
python crisp_gym/scripts/record_lerobot_format_leader_follower.py \
    --follower-config <env> --leader-config <leader> \
    --repo-id my_org/classic_demo --fps 15
    # no --record-config = legacy path; or pass
    # config/recording/teleop_classic_record.yaml to stamp the contract
```

Already have data recorded this way (Euler pose + delta-command action) and
want to train the UMI rot6d + relative policy on it? Convert it once with the
migration script in [§11](#11-migrate-legacy-euler--delta-command-data-to-rot6d).

---

## 5. Record all robot states (full-state recording)

Same UMI contract plus joint positions/velocities/efforts, target pose and raw
gripper — saved as `extra.*` columns that policies ignore (LeRobot treats every
`observation.*` key as a policy input, so extras deliberately live outside
that prefix).

```bash
python crisp_gym/scripts/record_lerobot_format_leader_follower.py \
    --record-config crisp_gym/config/recording/umi_robot_full_record.yaml \
    ... (as in usage 2 or 3)
```
Adjust the joint `shape:` entries (6 = UR, 7 = Franka) and `device_max_width`
in `umi_robot_full_record.yaml`. Joint efforts need `has_effort_feedback: true`
in the robot config.

Recommended default for robot recording: storage is cheap, re-recording isn't.

---

## 6. Post-process: align datasets for mixed training

Makes datasets from different devices schema-identical so LeRobot can
concatenate them. Verifies the stamped contracts first and REFUSES unfixable
mixes (e.g. classic-teleop vs UMI actions). Originals are never modified.

Runs on any machine: `pip install crisp_gym[postprocess]` (pandas + pyarrow).

```bash
python crisp_gym/scripts/postprocess_align_datasets.py \
    --datasets ~/.cache/huggingface/lerobot/my_org/umi_ur7e_full \
               ~/.cache/huggingface/lerobot/my_org/umi_handheld_demo \
    [--dry-run]
# -> umi_ur7e_full_aligned + umi_handheld_demo_aligned (extra.* stripped)
```
Optional fixup for a wrongly-scaled gripper:
`--rescale-gripper OLD_REF NEW_REF` (e.g. `0.08 0.09`).

---

## 7. Post-process: promote extra states to policy inputs

For a ROBOT-ONLY model that should consume joint states etc. Renames
`extra.*` → `observation.state.*` on disk so LeRobot picks them up as STATE
inputs with proper temporal windowing and stats.

```bash
python crisp_gym/scripts/postprocess_align_datasets.py \
    --datasets .../umi_ur7e_full --output-suffix _promoted \
    --promote extra.joints extra.joint_efforts
```
Promotion is refused if any dataset in the mix lacks the column — promoted
datasets cannot be mixed with handheld data, by construction. At deployment
the policy then also needs real joint states in its observation.

---

## 8. Train (LeRobot 0.4.4, UMI-style relative pose)

On the GPU PC (no ROS needed): copy `crisp_gym/scripts/lerobot_relative_pose.py`
(self-contained: lerobot + torch + numpy only) and run it exactly like
`lerobot-train`:

```bash
python lerobot_relative_pose.py \
    --dataset.repo_id=my_org/umi_handheld_demo_aligned \
    --policy.type=diffusion \
    --output_dir=outputs/train/umi \
    --dataset.video_backend=pyav \
    --batch_size=64 --steps=200000
```

**`--dataset.video_backend=pyav` (important).** LeRobot's default video
backend is `torchcodec`, whose frame-accurate *seek* path can fail on AV1
(`libsvtav1`) videos with:

```
RuntimeError: Could not push packet to decoder: Invalid data found when processing input
```

The data is fine (sequential decode works) — it's a torchcodec random-access
bug, and the shuffled training dataloader does random access, so it crashes at
the first step. `pyav` seeks through libav directly and decodes these videos
correctly. Keep this flag on **every** train/eval command for AV1 datasets
(all datasets recorded/migrated by this repo — see the `video.codec: av1`
default). It is a native `lerobot-train` flag, so it passes straight through
`lerobot_relative_pose.py`. If you must use torchcodec (it is faster),
re-encode the videos all-keyframe first: `ffmpeg -c:v libsvtav1 -g 1 ...`.

What it does at load time (disk data stays absolute):
- converts obs window + 16-step action horizon to poses **relative to the
  current TCP frame** (UMI's `pose_rep='relative'`);
- adds `observation.state.rot_wrt_start` (rotation-only, noised start pose) —
  declare it as a policy input feature in the policy config if you use it;
- recomputes normalization stats on the relative values.

Smoke-test with `--steps=100` first and check the
"Recomputed relative-pose stats" log lines appear.

Any native `lerobot-train` argument works unchanged (the script only patches
`make_dataset` then calls `lerobot_train.main()`), e.g. `--policy.device=cuda`,
`--policy.push_to_hub=false`, `--wandb.enable=false`, `--num_workers=8`.

### Full example (dual-camera diffusion, the params used for the UR/franka runs)

```bash
python lerobot_relative_pose.py \
    --dataset.repo_id=/path/to/franka_dual_open_panel_..._rot6d \
    --dataset.video_backend=pyav \
    --policy.type=diffusion \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.use_separate_rgb_encoder_per_camera=true \
    --policy.spatial_softmax_num_keypoints=64 \
    --output_dir=outputs/train/franka_dual_open_panel_..._rot6d_diffusion \
    --batch_size=64 --steps=500000 \
    --log_freq=10000 --wandb.enable=false
```

Parameter notes:
- `--policy.use_separate_rgb_encoder_per_camera=true` — one ResNet encoder per
  camera instead of a shared one. Use for **multi-camera** setups (env_cam +
  wrist_cam) so each view learns its own features; costs more params/VRAM.
- `--policy.spatial_softmax_num_keypoints=64` — spatial-softmax keypoints per
  camera feature map (diffusion default 32). 64 gives the vision head more
  spatial detail; must be a plain integer (a stray char, e.g. `64ls`, makes
  draccus raise `invalid literal for int()`).
- `--batch_size=64 --steps=500000` — full run (~days on one GPU at ~1.2 step/s);
  smoke-test with `--steps=100` first. Diffusion often converges well before
  500k, so a mid checkpoint may suffice.
- `--log_freq` = loss-print interval; `--save_freq` (default 20000) = checkpoint
  interval — first checkpoint at step `save_freq`, under
  `<output_dir>/checkpoints/`.
- `--output_dir` is resolved **relative to the launch cwd**, and LeRobot appends
  the job name (`_diffusion`); pass an absolute path to place it exactly.
- Resume an interrupted run with `--resume=true` and the same `--output_dir`.

---

## 9. Deploy a trained policy

The policy outputs **relative** rot6d poses. At each step compose with the
robot's current TCP pose (captured at OBSERVATION time, not receive time):

```
T_cmd = T_base_tcp_current @ T_rel      # then send to the CIC
```
No OptiTrack-to-robot calibration is needed (the world frame cancels in the
relative representation); what must match is the TCP frame convention and the
gripper `reference_width` (0.09 m) used at recording.

Canonical mode: REMOTE serving — training and rollout live on the GPU
machine (any lerobot version), crisp_gym is only the websocket client. See
**REMOTE_INFERENCE.md** (incl. the version policy) and the contract config
`crisp_gym/config/policy/remote_policy_example.yaml`.
Local in-process deployment (`crisp_gym/scripts/deploy_policy.py`) is legacy:
only for checkpoints trained with the robot machine's own lerobot (0.4.4).

---

## 10. Write your own record config

The record config is the dataset's data contract, independent of how the
robot is driven. Every parameter is documented in
`crisp_gym/config/recording/record_config_example.yaml`. Rules of thumb:

- `action.definition: next_tcp_pose` for anything that should mix with
  handheld data; `command` only for classic teleop datasets.
- Policy inputs live under `observation.*`; debug/analysis extras MUST use the
  `extra.` prefix with `include_in_state: false` (validated at load).
- `reference_width` must be identical across every device recording for the
  same policy; `device_max_width` is per-device.
- `rate_hz` must equal the recording `--fps`; changing it creates a different,
  non-mixable contract.

Use it via `--record-config path/to/your.yaml` on either recording script, or
programmatically:

```python
from crisp_gym.record.record_config import RecordConfig
from crisp_gym.record.record_functions import make_record_fn

cfg = RecordConfig.from_yaml("my_record.yaml")
fn = make_record_fn(env, cfg, drive_fn=...)   # drive_fn None = passive
```

---

## 11. Migrate legacy Euler + delta-command data to rot6d

Datasets recorded with the **old** collection scripts store the orientation as
Euler and the action as a **delta-pose command** (from the teleop `stream_fn`).
The rot6d + relative-pose trainer ([§8](#8-train-lerobot-044-umi-style-relative-pose))
requires the UMI convention instead:

| | Legacy data | Needed for training |
|---|---|---|
| `observation.state.cartesian` | `[x,y,z,roll,pitch,yaw]` (6d) | `[x,y,z,rot6d(6)]` (9d) |
| `action` | delta pose command (relative) | absolute `next_tcp_pose` (`pos+rot6d[+grip]`) |

Feeding the legacy data straight into `lerobot_relative_pose.py` would misread
the Euler angles as rot6d and double-relativise the already-delta action.
Convert it **once** with the migration script, then train normally:

`--input`/`--output` are dataset **root directories** (each with `data/` and
`meta/`), not repo ids. The script does *file surgery*: it copies the dataset
directory and rewrites only the low-dim Parquet columns + `meta` — the camera
**videos are copied byte-for-byte (no AV1 re-encode)**, so they never corrupt.

```bash
# In the lerobot environment; no ROS needed.
# Inspect the planned schema change first:
python crisp_gym/scripts/migrate_euler_delta_to_rot6d.py \
    --input  /path/to/old_euler_delta_demo \
    --output /path/to/old_euler_delta_demo_rot6d \
    --dry-run

# Then run it for real (output must not already exist):
python crisp_gym/scripts/migrate_euler_delta_to_rot6d.py \
    --input  /path/to/old_euler_delta_demo \
    --output /path/to/old_euler_delta_demo_rot6d
```

What it does, per frame:
- converts `observation.state.cartesian` Euler(6) → rot6d(9) using the repo's
  `from_euler("xyz")` → first-two-rows convention;
- rebuilds the concatenated `observation.state`;
- **discards the delta action** and sets `action[t] = absolute measured TCP at
  t+1` (`next_tcp_pose`, lookahead 1) from the converted pose, with the gripper
  taken from `observation.state.gripper` at `t+1` (the last frame of each
  episode repeats its own pose). Pass `--no-action-gripper` for a pose-only
  (9d) action.

Videos, sensors and every other column/file are copied through unchanged, and
stats for the three rewritten keys are recomputed. The result is a standard
absolute-on-disk rot6d dataset — train it with
`lerobot_relative_pose.py` exactly as in [§8](#8-train-lerobot-044-umi-style-relative-pose),
and (if needed) mix it with other UMI-contract datasets via the alignment
script ([§6](#6-post-process-align-datasets-for-mixed-training)).
