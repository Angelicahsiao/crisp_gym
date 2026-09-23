# Training & deployment scripts

Reference for the training, dataset-preprocessing, and deployment scripts in
this directory. They run on the **GPU/training PC** (lerobot + torch, no ROS)
except `deploy_policy.py`, which runs on the **robot PC** (crisp_gym + ROS).

All are crisp-import-free where they run on the GPU PC, so you can copy a single
file to the training machine — with one exception: `scripts/train_groot.sh` is a
launcher that resolves two siblings by path and needs them beside it (see
[below](#copying-it-to-a-training-server)). lerobot 0.4.x and ≥0.5 (verified 0.4.4 / 0.6.1)
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
| **GR00T N1.7** (dataset stays absolute) | `scripts/train_groot.sh` | `groot_lerobot_policy` |

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

### `scripts/train_groot.sh` — GR00T N1.7 fine-tuning (launcher)

In the **repo-root** `scripts/`, not this directory. Wraps `lerobot-train` for
`--policy.type=groot --policy.embodiment_tag=new_embodiment` and pins three
lerobot defaults that are wrong for a rot6d pose dataset and each fail
**silently**:

| lerobot default | what it does to a rot6d dataset | pinned to |
|---|---|---|
| `use_relative_actions=false` | the base N1.7 checkpoint declares `use_relative_action=true` in its `processor_kwargs`, so the DECODE step relativizes while the relative STATISTICS are computed only when this config flag is set — the model decodes relative actions against absolute stats | `true` |
| `relative_exclude_joints=[]` | every action dim is treated as relative, the gripper included | `["gripper"]` |
| `push_to_hub=true` | a local run with no `repo_id` fails **after** the dataset has loaded | `false` |

Two gates run before anything launches, so a bad run costs seconds instead of
GPU-hours: `groot_preflight.py` on the dataset, then a Hub reachability check on
**both** repos GR00T pulls — `nvidia/GR00T-N1.7-3B` and its
`nvidia/Cosmos-Reason2-2B` backbone, which is **gated** (a 401 there is an
unaccepted licence or a missing `HF_TOKEN`, not a network fault).

It also **refuses a relative-converted dataset**: GR00T builds its own relative
actions from absolute ones, so feeding it a `lerobot_relative_pose.py` output
makes it learn deltas of deltas, and nothing errors.

```bash
bash scripts/train_groot.sh \
    --dataset datasets/franka_electricbox/lerobot \
    --output  outputs/groot_electricbox
```

#### Copying it to a training server

Unlike every other script here, this one is **not** self-contained. It locates
itself and then reaches for two siblings by path:

```bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
...
"$REPO/crisp_gym/scripts/groot_preflight.py"        # the preflight gate
"$REPO/crisp_gym/scripts/lerobot_relative_pose.py"  # --se3 only
```

So `train_groot.sh` must sit in a directory called `scripts/`, and the two
python files must sit at `../crisp_gym/scripts/` relative to it. Four files in
two directories are enough — no git clone, no installed `crisp_gym` package,
no `PYTHONPATH`:

```
<anywhere>/
├── scripts/
│   ├── train_groot.sh            # the launcher
│   └── check_trained_groot.sh    # optional; standalone, run it after
└── crisp_gym/scripts/
    ├── groot_preflight.py        # needed unless --skip-preflight
    └── lerobot_relative_pose.py  # needed only for --se3
```

Both python files are import-clean by design — `groot_preflight.py` uses numpy
and pandas, `lerobot_relative_pose.py` numpy, torch and lerobot. Neither
imports `crisp_gym`; `lerobot_relative_pose.py` carries its own copy of the
rot6d helpers for exactly this reason (its header says to keep them identical
to `crisp_gym/util/rot6d.py`).

Copy the launcher on its own and it fails in two different ways:

| missing | symptom |
|---|---|
| `groot_preflight.py` | `python3: can't open file '.../groot_preflight.py'`, then **`Preflight FAILED — not starting training. Fix the dataset`** — exit 1. The advice is wrong: the dataset is fine, the gate script is absent |
| `lerobot_relative_pose.py` (with `--se3`) | `--se3 needs <path>, which is missing.` — exit 2 |

`check_gpu_groot.sh` and `check_trained_groot.sh` really are single files and
can be scp'd anywhere on their own.

| flag | meaning |
|---|---|
| `--dataset DIR` | dataset root holding `meta/` and `data/` — **required** |
| `--output DIR` | training output directory — **required** |
| `--repo-id ID` | dataset repo id; default = the last two path components of `--dataset` |
| `--batch-size N` | default 32. An H200 has room; a 32 GB card does not |
| `--steps N` | default 100000. Smoke-test with `--steps=2000` first |
| `--save-freq N` | default 20000 |
| `--num-workers N` | default 8 |
| `--job-name NAME` | default: the output directory's name |
| `--se3` | cross-embodiment mode — see below |
| `--no-wrt-start` | with `--se3`: 10-D relative state instead of the 16-D UMI-parity `[rel_pose9, gripper1, wrt_start6]` |
| `--skip-preflight` | launch without gating on `groot_preflight.py` |
| `--dry-run` | print the command and exit |
| `--` | everything after this is appended to the lerobot command |
| `SKIP_HUB_CHECK=1` | environment variable, not a flag: skips the Hub gate (a fully cached offline node) |

**Two modes.** By default GR00T relativizes the **action** itself and the
observation stays absolute — and for a fine-tune that conversion is
componentwise subtraction, not SE(3): the synthesized `new_embodiment` action
config is hardcoded `NON_EEF`/`DEFAULT`, and `relative_eef_to_absolute` runs
only for `eef` + `xyz+rot6d`. With `--se3`, training instead goes through
`lerobot_relative_pose.py`, which re-expresses the **observation** relative to
the current frame (`T_base⁻¹ @ T`). Neither GR00T path does that — in lerobot
and Isaac-GR00T alike the state is only ever a reference, never transformed —
and an absolute TCP pose is in the robot's own base frame, so the same motion is
different numbers on a Franka and a UR. GR00T's own relative conversion is
switched off in that mode so the action is not relativized twice. **The dataset
on disk stays absolute either way**, so the preflight applies unchanged.

**Artefacts.** `train.log` and `launch_command.txt` end up inside the output
directory. They are written *beside* it first and moved in afterwards, because
lerobot refuses to start when `output_dir` already exists
(`configs/train.py:259`). A run that dies before its first checkpoint leaves
them at `<output>.log` and `<output>.launch_command.txt` with the traceback —
lerobot only creates the directory when it writes a checkpoint.

**Watch the log for** `using the generic RelativeActionsProcessorStep fallback`.
That step subtracts, which is wrong for rot6d: if it appears, stop, because the
checkpoint is not carrying native relative statistics.

Deploy with `groot_lerobot_policy`. That config is written for a **default-mode**
checkpoint (`action_repr: absolute`, `state_input: absolute` — GR00T's own
postprocessor already decodes the action back to absolute); an `--se3`
checkpoint needs the opposite on both.

---

## GR00T N1.7 checks

Three standalone checkers bracket a run. `groot_preflight.py` and
`check_trained_groot.sh` need no torch, no lerobot and no GPU, so they run on a
laptop; `check_gpu_groot.sh` is the one that wants the training or deploy
machine.

| script | when | answers |
|---|---|---|
| `scripts/check_gpu_groot.sh` | before anything | does this machine's torch drive the card, and are GR00T's run-time packages installed? |
| `groot_preflight.py` (this directory) | before training | is this dataset fit for GR00T at all? |
| `scripts/check_trained_groot.sh` | after training | did the flags you passed survive into the saved config? |

### `scripts/check_gpu_groot.sh` — machine readiness

```bash
pixi run -e humble-lerobot bash scripts/check_gpu_groot.sh [gpu_check.log]
```

Two stages. **GPU**: the torch build's architecture list against each visible
card's compute capability, VRAM, and a real timed bf16 matmul per device.
**lerobot / GR00T**: that `GrootPolicy` imports, that the packages GR00T needs
at run time are present, and the config defaults that matter for a rot6d
dataset. Skipped cleanly when lerobot is absent, so the same file is useful on a
deploy machine that only has torch.

The run-time package stage exists because a clean `GrootPolicy` import proves
almost nothing: lerobot guards those packages with `require_package()` at the
point of use, and `dm-tree`'s sits inside `GR00TN17.prepare_input` — it fires on
the **first `get_action`**, long after the checkpoint has loaded and the robot
has homed.

| exit | meaning |
|---|---|
| 0 | GO — every visible GPU ran the bf16 matmul |
| 2 | torch missing or not importable |
| 3 | no usable CUDA device |
| 4 | a GPU failed the matmul — this torch build cannot drive it |
| 5 | GPU is fine but `GrootPolicy` will not import |
| 6 | `GrootPolicy` imports but a run-time package is missing |

VRAM shortfalls are WARN, not failure: bf16 inference needs far less than
fine-tuning, so a card too small to train on may still deploy.

### `groot_preflight.py` — is the dataset fit?

```bash
python groot_preflight.py <dataset_root>
python groot_preflight.py <dataset_root> --exclude gripper --chunk-size 40
```

Reads parquet and `info.json` only. `<dataset_root>` is the directory holding
`meta/` and `data/`. Six groups of checks:

| check | catches |
|---|---|
| **G1** action names | an `action` feature with no per-**dimension** `names`. lerobot's `_infer_n1_7_action_groups()` returns `[]` when they are absent, so `--policy.relative_exclude_joints='["gripper"]'` silently does nothing and the gripper trains as a delta. crisp_gym writes them; a hand-built, migrated or aggregated dataset may not |
| **G2** absolute actions | a dataset already converted to relative (median \|xyz\| near zero, rot6d clustered on identity), plus whether `action[t] == state[t+1]` |
| **G3** pose layout | a pose block that is not xyz(3) + rot6d(6) — Euler and quaternion do not map |
| **G4** cameras | no `observation.images.*` features |
| **G5** task / language | a missing or empty task string |
| **G6** scale & chunk period | episodes shorter than the action chunk, and dataset size against the chunk |

`--exclude gripper --chunk-size 40` is what `train_groot.sh` passes on your
behalf. Exit 0 = fit (warnings possible), 1 = a check FAILED, 2 = the dataset
could not be read.

### `scripts/check_trained_groot.sh` — did the flags survive?

```bash
bash scripts/check_trained_groot.sh <output_dir>
bash scripts/check_trained_groot.sh <output_dir>/checkpoints/020000/pretrained_model
```

Given a training `output_dir` it finds the newest
`checkpoints/*/pretrained_model/` itself. It compares what you **asked** for
(`train_config.json`) against what the policy **saved** (`config.json`),
resolving keys at any depth, and reports the image geometry the checkpoint will
apply — `image_target_size`, `image_crop_size` and the `crop_fraction` the
runtime derives from their ratio — which live in the processor sidecars, not
`config.json`.

It exists because the two flags that matter have no runtime symptom. A run that
silently kept `use_relative_actions=false` decodes relative actions against
absolute statistics; one that lost `relative_exclude_joints` trains the gripper
as a delta. Both produce a checkpoint that loads, deploys, and behaves badly.
Exit 0 = the flags survived, 1 = one was lost or disagrees, 2 = no `config.json`
found.

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
