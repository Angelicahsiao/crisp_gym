# Training & deployment scripts

Reference for the training, dataset-preprocessing, and deployment scripts in
this directory. They run on the **GPU/training PC** (lerobot + torch, no ROS)
except `deploy_policy.py`, which runs on the **robot PC** (crisp_gym + ROS).

All are crisp-import-free where they run on the GPU PC, so you can copy a single
file to the training machine. lerobot 0.4.x and ≥0.5 (verified 0.4.4 / 0.6.1)
are both supported — each launcher patches whichever dataset factory that
version exposes and **raises** rather than silently training unwrapped.

---

## Pipeline at a glance

```
record  ──►  (optional preprocess)  ──►  train  ──►  deploy
             swap_action_offline.py       one of:      deploy_policy.py
                                          lerobot_relative_pose.py      + policy config:
                                          train_absolute_next_pose.py     relative_lerobot_policy   (relative ckpt)
                                          train_action_from_target_...    absolute_lerobot_policy   (absolute ckpt)
```

| checkpoint outputs | train with | deploy policy config |
|---|---|---|
| **relative** pose (`T_rel`) | `lerobot_relative_pose.py` | `relative_lerobot_policy` |
| **absolute** next pose | `train_absolute_next_pose.py` | `absolute_lerobot_policy` |
| absolute **commanded** pose | swap offline → `train_absolute_next_pose.py` | `absolute_lerobot_policy` |

---

## Common lerobot-train flags

Every launcher forwards all native `lerobot-train` args to draccus. The ones you
almost always need:

| flag | meaning |
|---|---|
| `--dataset.repo_id=<name>` | dataset identifier (a label when `--dataset.root` is set) |
| `--dataset.root=<dir>` | **load a LOCAL dataset** — the directory holding `meta/info.json`. Without it lerobot goes to the Hugging Face Hub and 404s on a local-only dataset. |
| `--policy.type=diffusion` | policy architecture |
| `--policy.push_to_hub=false` | don't push the trained policy to the Hub (otherwise lerobot demands a hub repo_id and aborts) |
| `--output_dir=<dir>` | where checkpoints + provenance JSON are written |
| `--batch_size` / `--steps` | training size; smoke-test with `--steps=100` first |
| `--num_workers=0` | fallback if DataLoader workers crash decoding video |
| `--dataset.video_backend=pyav` | alternative video decoder (same fallback) |

---

## Training scripts

### `lerobot_relative_pose.py` — relative-pose training (UMI)

Re-expresses every pose (obs history + action horizon) **relative to the current
TCP frame** at dataloader level; the dataset on disk stays absolute. Recomputes
normalization stats on the relative values and stamps `pose_repr.json` next to
the checkpoint so deployment knows the convention.

```bash
python lerobot_relative_pose.py \
    --dataset.repo_id=franka_electricbox \
    --dataset.root=datasets/franka_electricbox/lerobot \
    --policy.type=diffusion --policy.push_to_hub=false \
    --output_dir=outputs/train/rel --batch_size=64 --steps=200000
```

**Script-specific flag:**

| flag | meaning |
|---|---|
| `--wrt-start` (default) | append `rot_wrt_start` → **16-D UMI-parity** `observation.state`. Deploy with `state_input: relative_wrt_start`. |
| `--no-wrt-start` | plain **10-D relative** state (pose9 + gripper), no wrt-start. Deploy with `state_input: relative`. |

Deploy: `relative_lerobot_policy` (see below). The `state_input` in that config
must match the flag you trained with.

### `train_absolute_next_pose.py` — absolute-pose baseline

Trains on the dataset **as recorded**: no obs/action transform. The policy
predicts the absolute next TCP pose (`action` column). Stamps
`action_repr.json` (absolute). Same flags as above **minus** `--wrt-start`.

```bash
python train_absolute_next_pose.py \
    --dataset.repo_id=franka_electricbox \
    --dataset.root=datasets/franka_electricbox/lerobot \
    --policy.type=diffusion --policy.push_to_hub=false \
    --output_dir=outputs/train/abs --batch_size=64 --steps=200000
```

Deploy: `absolute_lerobot_policy`.

### `train_action_from_target_cartesian.py` — commanded-pose ablation (online)

Replaces the **arm** dims of `action` with `extra.target_cartesian` (the pose
commanded to the CIC), keeping the gripper — the policy learns the command
stream, not the achieved trajectory. Only valid for **Cartesian-driven** data;
refuses on FACTR/JIC data where `target_cartesian` is constant.

```bash
python train_action_from_target_cartesian.py \
    --dataset.repo_id=franka_electricbox \
    --dataset.root=datasets/franka_electricbox/lerobot \
    --policy.type=diffusion --policy.push_to_hub=false \
    --output_dir=outputs/train/cmd --batch_size=64 --steps=200000
```

If it raises `… window (H,10) and … window (9,) disagree`, your lerobot won't
window a non-policy key at load time — use the **offline** path instead:

### `swap_action_offline.py` — commanded-pose ablation (offline preprocess)

Writes a **copy** of the dataset whose `action` arm dims are
`extra.target_cartesian` (gripper kept). Videos are copied byte-for-byte; only
the `action` column + its stats are rewritten. Then train the copy with
`train_absolute_next_pose.py`. This is also the way to get a **relative
commanded-pose** model: swap offline (command → `action`), then run
`lerobot_relative_pose.py` on the copy.

```bash
python swap_action_offline.py \
    --input  datasets/franka_electricbox/lerobot \
    --output datasets/franka_electricbox_cmd/lerobot
```

| flag | meaning |
|---|---|
| `--input` / `--output` | source dataset root / destination (must not exist) |
| `--atol` | per-episode spread below which `target_cartesian` is "constant" → refuse (default 1e-4) |
| `--dry-run` | validate + variance-check only, write nothing |

---

## Deployment

### `deploy_policy.py` (robot PC, ROS)

Loads a checkpoint, runs the control loop, optionally records the rollout.
Select the policy behavior with `--policy-config`.

```bash
python -m crisp_gym.scripts.deploy_policy \
    --env-config dric_dual_rscam_franka_deploy_umi \
    --policy-config <relative_lerobot_policy | absolute_lerobot_policy> \
    --path outputs/train/<run>/checkpoints/<step>/pretrained_model
```

| flag | meaning |
|---|---|
| `--env-config` | deploy env (must be `rotation_6d` + `use_relative_actions: false`) |
| `--policy-config` | `relative_lerobot_policy` or `absolute_lerobot_policy` |
| `--path` | checkpoint `pretrained_model` dir (prompts if omitted) |
| `--num-inference-steps` | diffusion denoising steps (lower = faster loop) |
| `--n-action-steps` | chunk steps executed before re-planning |
| `--scheduler ddim\|ddpm` | sampler override (use `ddim` + low steps to speed a DDPM checkpoint) |
| `--evaluate` | prompt success/failure per episode, write a CSV |

**Policy configs** (`crisp_gym/config/policy/`):

- `relative_lerobot_policy.yaml` — composes `T_cmd = T_base @ T_rel`. Set
  `state_input` to match training (`relative` for `--no-wrt-start`,
  `relative_wrt_start` for the default).
- `absolute_lerobot_policy.yaml` — same class with `action_repr: absolute`;
  sends the model pose to the CIC directly. Auto-detects "absolute" from
  `action_repr.json` next to the checkpoint.

Both require `device_max_width` (0.085 for the Robotiq 2F-85) and
`reference_width` (0.09) to match the record config's gripper scaling.

> **Gripper convention (both configs):** one convention everywhere — the device
> value, `0=closed / 1=open`. The env observation, the record source
> `gripper.width_normalized`, and the command path all agree, so no inversion
> happens at deploy. Keep `invert_gripper: false`; it exists only for legacy
> datasets whose *action* gripper was stored inverted.

### `check_policy_openloop.py` (GPU PC, no robot)

Open-loop diagnostic that runs on the **training PC**: feeds the policy the
exact training-time observations (via the same `RelativePoseDataset` wrapper +
the policy's `delta_timestamps` window) and compares the predicted action to the
**recorded** action, frame by frame. Answers "did the model learn the task?"
before you touch the robot:

- **low** error → the policy reproduces the demos; a drifting real rollout is a
  deploy problem (images OOD, control rate ≪ training fps, timing).
- **high** error → retrain; no deploy tweak will help.

```bash
python3 check_policy_openloop.py \
    --path outputs/train/<run>/checkpoints/<step>/pretrained_model \
    --repo-id datasets/franka_electricbox/lerobot \
    --episodes 0 1 2 --stride 5 --max-frames 200
```

It expects a local copy of `lerobot_relative_pose.py` in the same folder and is
oriented to the relative pipeline.
