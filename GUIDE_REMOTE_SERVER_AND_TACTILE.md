# Guide: remote policy server and tactile scripts

Step-by-step use of two things that are newer than the rest of the docs:

1. **The remote policy server** — the model runs on a GPU machine, the robot
   machine only talks to it.
2. **The tactile scripts** — a report on the tactile signal, an audit of the
   command-minus-measured residual, and a loader for the high-rate tactile CSVs.

For the design and the wire protocol see [REMOTE_INFERENCE.md](REMOTE_INFERENCE.md);
for the reasoning behind the tactile work see [HANDOFF.md](HANDOFF.md) §7.

Both parts are on branch `claude/determined-faraday-1z6rx0` (the policy server
was merged in from `claude/vigilant-tesla-VZJT6`).

---

## Part 1 — Remote policy server

```
robot machine (ROS 2, no GPU)                  GPU machine (torch + lerobot, no ROS)
deploy_policy.py                               scripts/policy_server.py
  observations, command composition  ── ws ──►   checkpoint, normalization, inference
  env.step()                         ◄───────    action chunk
```

The server runs the same inference code that `deploy_policy.py` runs locally,
so a checkpoint that works locally gives the same actions through the server.

### 1.1 GPU machine: start the server

A clone of this repository is enough; crisp_gym does not need to be installed
there. Use the Python that has torch and the **same lerobot version the
checkpoint was trained with**.

```bash
pip install websockets msgpack rich
```

```bash
python scripts/policy_server.py --path outputs/train/RUN/checkpoints/125000/pretrained_model --host 0.0.0.0 --port 8765
```

It checks its requirements, loads the checkpoint, runs two warm-up inferences
and then listens. The second warm-up time in the log is roughly what the robot
will wait for each chunk.

Check that it is ready, from any machine on the network:

```bash
curl http://GPU_HOST:8765/healthz
```

`OK` means it is listening. Ctrl-C stops it and frees the GPU.

| option | meaning |
|---|---|
| `--path` | the checkpoint's `pretrained_model` directory (required) |
| `--host`, `--port` | where to listen; default `0.0.0.0:8765` |
| `--image-encoding raw\|jpeg`, `--jpeg-quality` | how the robot sends images. `raw` (default) is exact; `jpeg` is smaller on a slow link |
| `--state-input` | `auto` (read from the checkpoint), `absolute`, `relative`, `relative_wrt_start` |
| `--num-inference-steps`, `--scheduler ddim\|ddpm` | denoising speed, diffusion policy only |
| `--set KEY=VALUE` | lerobot policy-config override, repeatable |
| `--no-warmup` | skip the warm-up inferences |

Model settings (`--state-input`, `--num-inference-steps`, `--scheduler`,
`--set`) belong to the server's command line, because the server holds the
checkpoint.

### 1.2 Robot machine: point the policy at the server

The robot machine needs no torch, no lerobot and no checkpoint:

```bash
pip install websockets msgpack
```

In `crisp_gym/config/policy/relative_lerobot_policy.yaml` set the address:

```yaml
server: "ws://192.168.1.50:8765"     # null = run the model on this machine
async_inference: true                # recommended with a server, see 1.3
```

or, with the timeouts spelled out:

```yaml
server:
  host: "192.168.1.50"
  port: 8765
  infer_timeout_s: 5.0       # must be longer than the slowest inference + transfer
  connect_timeout_s: 10.0
  reconnect_attempts: 3
```

Then deploy as usual. `--path` is only a label in this mode; the log names the
checkpoint the server is serving.

```bash
python -m crisp_gym.scripts.deploy_policy --env-config YOUR_DEPLOY_ENV --policy-config relative_lerobot_policy --path remote
```

### 1.3 Settings worth knowing

| setting (robot side) | what it does with a server |
|---|---|
| `async_inference: true` | the next chunk is requested while the current one is still executing, so the arm does not pause for the round trip. With `false` (the default) every new chunk blocks the control loop. The very first chunk always blocks. |
| `prefetch_lead` | how many steps before the end of a chunk the next one is requested; `null` sizes it from the measured latency |
| `n_action_steps` | steps executed per chunk. A round trip longer than this many steps still stalls the arm, async or not |
| `infer_timeout_s` | on a timeout the robot holds its pose and asks again. Set it **longer** than the slowest inference plus transfer; a shorter value never recovers |
| `state_input: auto` | the server's mode is used. An explicit value that contradicts the server refuses to start |
| `num_inference_steps`, `scheduler`, `overrides` | ignored on the robot side and logged as NOT APPLIED — set them on the server |

### 1.4 When something goes wrong

| what you see | cause and fix |
|---|---|
| `curl .../healthz` does not answer | the server is still loading or warming up, or the port is blocked |
| the robot side is refused with `not_ready` | another client is connected — the server takes one robot at a time. After a robot vanished without closing (power, cable) the server needs up to ~40 s to notice |
| the policy refuses to start, naming `state_input` | the YAML's `state_input` contradicts the server's. Set it to `auto` |
| the arm holds its pose now and then | a request timed out. Raise `infer_timeout_s`, or lower the inference time on the server (`--num-inference-steps`) |
| the arm pauses at every chunk | `async_inference` is `false`, or the round trip is longer than `n_action_steps` steps |
| the connection dropped during an episode | the client reconnects by itself. With a `relative_wrt_start` checkpoint the start rotation is then measured from the next observation — restart the episode |
| the server exits with status 1 | its inference worker died; read the lines above the exit in the server log |

### 1.5 Limits

- One robot per server. A second model needs a second server process on
  another port.
- No TLS and no authentication: use it on a private wired link only and do not
  expose the port.
- Checked with a real checkpoint on one machine (loopback). **The round trip on
  the real robot-to-GPU link has not been measured yet** — the server logs
  decode and inference time per request, the robot side logs the total.

---

## Part 2 — Tactile scripts

All of these run anywhere with `numpy`, `pandas`, `pyarrow` and `pyyaml`. They
need no ROS, no lerobot and no install of crisp_gym, and they only read the
dataset.

### 2.1 The data they expect

In the LeRobot dataset, per frame (15 Hz):

| column | content |
|---|---|
| `extra.tactile.left`, `extra.tactile.right` | one tactile sample per finger, any number of taxels |
| `<tactile>.sync_valid` | optional flag; frames where it is false are left out |
| `<tactile>.source_stamp_ns` | optional; the timestamp of the sample that was stored. It joins a frame to the high-rate CSV |
| `observation.state.cartesian`, `extra.target_cartesian` | measured and commanded TCP pose (the audit only) |

Next to the dataset, the full tactile stream (about 200 Hz):

```
<dataset>/tactile_raw/episode_000000_left.csv
<dataset>/tactile_raw/episode_000000_right.csv
```

with the columns `source_stamp_ns` and `taxel_0 … taxel_N`. If only the `.dvc`
pointer files are there, pull the CSVs first.

Older column names (`left_tactile` / `right_tactile`) are found as well; any
other name goes in `--tactile-keys`.

### 2.2 Tactile report — works on handheld and robot data

```bash
python crisp_gym/scripts/audit_command_residual.py /path/to/dataset --tactile-only --out tactile.json
```

It needs no command column, so it runs on handheld data too. Per finger it
prints:

| line | meaning |
|---|---|
| `taxels`, `sync_valid false` | size of the skin; frames left out |
| `dead taxels` | taxels whose value never changes in the whole dataset. In a small dataset this also lists taxels that were simply never touched |
| `raw level` | smallest, median and largest raw value |
| `noise` | spread of one taxel at rest; the norm at rest; the contact threshold derived from it |
| `baseline drift` | how much the zero moves from one episode to the next |
| `in contact` | share of frames above the threshold, overall and per episode |
| `contact norm`, `strongest taxel`, `active taxels` | how hard and how wide a contact is |
| `layout hint` | a guess at the grid width — **not reliable yet**, do not use it |
| `high-rate CSV` | stream rate, interval median / p99 / max, share of time inside gaps, and whether every frame is found in its CSV by timestamp with identical values |

The zero of each episode is the mean of its first 10 valid frames
(`--baseline-frames`), so an episode that starts in contact hides that contact.

**Run it on the handheld dataset and on the robot dataset and compare the two
reports.** A model trained on one only carries over to the other if noise,
contact share, contact strength and active-taxel counts are similar.

### 2.3 Residual audit — robot data only

Under Cartesian impedance control a contact force sits in the gap between the
commanded and the measured pose (`force ≈ stiffness × gap`). The audit measures
that gap.

```bash
python crisp_gym/scripts/audit_command_residual.py /path/to/robot_dataset --env-config crisp_gym/config/envs/ur7e_robotiq.yaml --out audit.json
```

`--env-config` takes the env YAML the data was recorded with and reads the
controller gains from the control file it names. The report prints which file
it used.

| if | then |
|---|---|
| the env YAML sets `cartesian_control_param_config: null` (the dric Franka envs) | the gains come from the robot bring-up. Pass that file: `--control-config .../config/fr3/controllers.yaml` |
| the robot PC keeps its configs in another directory | add `--config-path <dir>`; `CRISP_CONFIG_PATH` is honoured too |
| you just want to type the gains | `--k-pos 200 --k-rot 60` (overrides both files) |

What the steps need and answer:

| step | question | needs |
|---|---|---|
| A | how big is the gap in contact and in free motion? | commanded + measured pose |
| B | how much of the free-motion gap is tracking lag? | commanded + measured pose |
| C | what stiffness does the controller really deliver? | `extra.joints` + `extra.ext_torque`, **Franka only** |
| D | can tactile predict the gap on held-out episodes? | tactile columns |
| E | the same gap for the gripper | `extra.gripper_target` |
| F | how different are the two action labels (command vs measured pose `L` frames ahead)? | commanded + measured pose |

Things to know:

- **Handheld data is refused** with "no command was recorded": a handheld
  device has no command. Use `--tactile-only` there. Joint-driven data (FACTR)
  is refused for the same reason.
- **Step D's tactile rows are not trustworthy when contact is rare.** With
  about 10 % of the frames in contact it printed R2 values of −300. That is the
  probe failing, not a statement about the sensor. The `velocity` row is fine.
- **Step C is skipped on a UR** (6 joints, built-in kinematics are Franka's).
- Contact frames are picked from tactile when it is present, otherwise from
  the external force.

### 2.4 Loading the high-rate tactile CSVs

`crisp_gym/util/tactile_raw.py` turns the CSVs into fixed-length windows, one
per dataset frame:

```python
from crisp_gym.util import tactile_raw as tr

stamps, taxels = tr.load_csv(tr.csv_path(dataset / "tactile_raw", episode, "left"))

# frame_stamps: this episode's extra.tactile.left.source_stamp_ns, as int64
values, age_ms = tr.windows(stamps, taxels, frame_stamps, length=32, rate_hz=200.0)
```

- `values` has shape `[frames, 32, taxels]`. The last column is the sample the
  dataset stored for that frame; the columns before it go back in time in
  steps of `1 / rate_hz`.
- `age_ms` has shape `[frames, 32]` and says how old each value was at its grid
  time. A large age is a gap in the stream; a negative age means the grid point
  lies before the stream started.
- A window only ever holds past samples, never a later one. The stream has
  gaps, so windows are cut by time and not by row count.
- Keep the timestamps as int64. Through float32 they lose all precision and no
  frame is found; `tr.match_frames` returns −1 then.

### 2.5 Recording the extra columns on a robot

Both entries go under `observations:` in the record config.

The gripper command, in the same units as `observation.state.gripper` (use the
same two width values as that entry):

```yaml
  - key: extra.gripper_target
    source: gripper.target_width_normalized
    reference_width: 0.09
    device_max_width: 0.085        # 0.140 for a Robotiq 2F-140
    include_in_state: false
```

A wrist force-torque sensor. Any topic of type `geometry_msgs/WrenchStamped`
works with the existing `force_torque` sensor type. In the env YAML:

```yaml
sensor_configs:
  - sensor_type: "force_torque"
    name: "ft_sensor"
    shape: [6]
    data_topic: "/your/wrench/topic"
```

and in the record config:

```yaml
  - key: extra.ft_wrench
    source: robot.sensor
    name: ft_sensor
    shape: [6]
    include_in_state: false
```

`config/recording/umi_robot_full_record.yaml` already records the command, the
gripper command, the joints and the external joint torques for a Franka.

### 2.6 Limits

- Tactile is not a model input yet. It is stored under `extra.`, which neither
  training nor deployment reads.
- The robot dataset recorded so far has no `source_stamp_ns` column, so its
  CSVs cannot be joined to frames. The handheld dataset has it.
- The audit has no option yet to use a recorded wrench for step C.
- Controller gains are not stored in the dataset. They are read from the
  config files, so the files on the PC that recorded the data are the ones
  that count.

### 2.7 Tests

```bash
python tests/test_audit_command_residual.py
```

```bash
python tests/test_tactile_raw.py
```

Both run on synthetic data with known answers and need no robot.
