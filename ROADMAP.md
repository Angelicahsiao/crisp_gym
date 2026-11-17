# Roadmap: unfinished and untested features, and how to finish them

For any assistant, including one with a small context, asked to take a feature
of this repository from "written" to "works on the robot". It is deliberately
high level: what each feature is for, what exists, what has actually been
proven, the order to finish it in, and how you will know it works. File-level
detail is in the documents each section names.

Written 2026-11-17 at commit `36d133e`. `GUIDANCE.md` holds the working rules
and the small clean-up packages; **features come first**, and a clean-up
package is worth doing when it stands in a feature's way.

---

## 1. How to finish a feature here

### 1.1 Three levels of "done"

| level | means | evidence |
|---|---|---|
| **BUILT** | the code exists | a file |
| **FAKES** | it passes tests that replace ROS, torch, lerobot or the robot with stand-ins | `python -m pytest` |
| **REAL** | it ran on the GPU PC with the real lerobot, or on the robot, and the output was looked at | output pasted by the owner, recorded in `HANDOFF.md` |

Almost everything in §2 is at FAKES. **Finishing a feature means moving it to
REAL.** Writing more code on top of a FAKES layer is how this repository got
several features deep without knowing whether the first one works.

### 1.2 You cannot run the real thing. The owner can.

A cloud session has no robot, no GPU, no lerobot, no ROS. So the unit of work
for a REAL step is a **run sheet**, not code:

1. one command per block, copyable, with real paths;
2. what the output should show if it works, and what it shows if it does not;
3. exactly what to paste back.

A run sheet looks like this:

```
RUN SHEET: F2 part A, step 1 (GPU PC)
1. python crisp_gym/scripts/add_environment_state.py --input IN --output OUT \
       --columns extra.ext_torque --dry-run
   works: a line "[  0:  7]  extra.ext_torque  min ... max ..." and
          "--dry-run: the request is valid"
   fails: "Refused, nothing written: ..." (the sentence says why)
Paste back: everything the command printed.
```

Then stop and wait. When the output comes back, read it before explaining it.
This loop (you prepare, the owner runs, you interpret) is how every finding in
`HANDOFF.md` §7 was made.

### 1.3 Thinnest path first

Take ONE feature through the whole chain before widening it:

```
dataset → training smoke run (100 steps) → open-loop check → one robot rollout
```

One new thing per run. If the smoke run fails, nothing after it was worth
writing yet.

### 1.4 Plug into what exists

| to add | plug it in here | pattern to copy |
|---|---|---|
| a quantity to record | a `@register_source` function in `record/record_config.py`, and a line in the record YAML with an `extra.` key | `gripper.target_width_normalized`; `tests/test_gripper_target_source.py` |
| a ROS sensor | crisp_py's `@register_sensor`, then `sensor_configs` in the env YAML | `ext_torque` in `dric_franka_oakw_umi.yaml` |
| a derived column or label | an offline script that writes a COPY, adds statistics, and leaves a small `meta/*.json` note | `swap_action_offline.py`, `add_environment_state.py` |
| a model input | `observation.state` (every policy reads it) or `observation.environment_state` (diffusion and ACT only) | `add_environment_state.py`; the wrt-start append in `lerobot_relative_pose.py` |
| something done to every training sample | `RelativePoseDataset.convert_item` in `lerobot_relative_pose.py`, plus recomputed statistics | wrt-start |
| what a checkpoint must tell deployment | a field in `pose_repr.json`, written by `stamp_pose_repr`, read by `find_pose_repr` | `action.source` |
| what the robot sends the model | `build_obs_frame` (robot side) and `inference_worker` (model side), both in `relative_lerobot_policy.py` | the `state_input` handling |
| a check before a rollout | `scripts/deploy_preflight.py`; `_verify_state_dim`, `_verify_image_keys` | — |
| a limit on commands | a pure function in `util/safety.py`, called from `ManipulatorCartesianEnv.step` | `limit_target_step` |
| a look at recorded data | a step in `audit_command_residual.py`, or a small read-only script | `check_tactile_sync.py` |
| a simulation | `crisp_gym/sim/` | `spring_pad.py` |

Do not build a second mechanism beside one of these.

### 1.5 Make silent failures loud

Every expensive mistake so far was silent: a "promoted" input the model never
read, an all-zero placeholder column that looks like data, a command column
that is frozen under joint teleop, a finetune that kept the base model's input
list, a bound of 0.0 read as "no bound". So for every new data path:

- find out what the CONSUMER actually reads (open its source or grep it), do
  not trust a name or a document;
- make the tool REFUSE the cases that would train or command on nothing
  (missing, constant, NaN, wrong width), with a message that says what to do;
- record what was produced in a small file next to it, and have the next stage
  read that file instead of guessing.

### 1.6 When a result surprises you

Say "candidate", name the measurement that would settle it, and get it. Twice
in this project a confident explanation was wrong (gripper weight; a thin
collision plate) and only a measurement showed it. After a fix, break it on
purpose once and watch its test fail.

### 1.7 Report

What was done; what the tests showed; what was NOT run and where it must be
run; the run sheet; the questions. Update the level in §2 and the note in the
feature's section in the same commit.

---

## 2. Where everything stands

| # | feature | level | what is missing | needs |
|---|---|---|---|---|
| F1 | Measured-pose vs command action (models M and C) | FAKES | no training run, no rollout | GPU PC, robot |
| F2 | Torque, force or tactile as a model input | tool FAKES; deployment not built | real dataset run, training, the robot side, `multi_task_dit` | GPU PC, then code, then robot |
| F3 | Safety limits on the commanded pose | FAKES | never acted on a real arm; no workspace box chosen | robot |
| F4 | One-camera Franka configs (`dric_franka_oakw_*`) | FAKES (YAML checks) | never started on the robot | robot |
| F5 | Policy server and asynchronous inference | REAL on one machine | the robot-to-GPU link; async on the robot; speed options for `multi_task_dit` | robot + GPU PC |
| F6 | Gripper command as a label | FAKES | no dataset has it; the swap tool ignores the gripper | robot, then code |
| F7 | Simulation | first step REAL in CI | a task and a dataset writer; `tactile_envs`; the real controller in MuJoCo | cloud, then owner's PC |
| F8 | Residual model (relabel handheld data) | not started | everything; depends on F1 and F7 | research |
| F9 | Tactile | ON HOLD (hardware) | recording on the robot, time stamps, the probe | owner |
| F10 | The test suite in the real environments | never run there | one pytest run each on robot PC and GPU PC | owner |
| F11 | Open measurements | — | delivered stiffness, duplicated configs, loop left-overs | robot |
| F12 | Other trained-policy paths (GR00T, SmolVLA) | unknown to this document | ask the owner what has run | owner |

Order: **F10 → F4 → F3 → F1**, because F1's rollouts need the configs, the
limits and trustworthy tests. F2 in parallel on the GPU PC. F6 with the next
data collection. F7 whenever no robot time is available.

---

## 3. Feature guides

### F1 · Measured-pose vs command action

**For.** Deciding whether the pose the operator COMMANDED is a better action
label in contact than the pose the arm REACHED. If it is, handheld data (which
has no command) is missing something, and F8 is worth building.

**Exists.** `swap_action_offline.py` makes the command copy;
`lerobot_relative_pose.py` trains both and writes `action.source` into
`pose_repr.json`; `check_policy_openloop.py` warns `ACTION SOURCE MISMATCH`;
the deploy log prints the source. The audit says the two labels differ by a
median of 48 mm in contact on `franka_electricbox`. The protocol, with
commands, data amounts and trial counts, is in USAGE §14 "Measured-pose action
vs command action".

**To finish.**
1. [GPU PC] Make the command copy. Confirm `meta/action_repr.json` exists in it.
2. [GPU PC] Smoke-train each for 100 steps. Confirm each output folder's
   `pose_repr.json` says `measured` and `command`. This is the first time the
   stamp runs with a real lerobot.
3. [GPU PC] Train both fully, same settings and seed. Pin `--policy.horizon`
   and `--policy.n_action_steps` (lerobot 0.6.1 defaults are 64 and 32).
4. [GPU PC] Open-loop check, each model against its OWN dataset.
5. [robot] Roll out both, alternating, with `dric_franka_oakw_deploy_umi`
   (after F4 and F3). Record the rollouts.
6. [cloud] Run the audit on the recorded rollouts; compare the contact gap.

**It works when.** Both stamps are right; both open-loop checks pass; a table
of success, time and contact gap exists for 20–30 trials each.

**Traps.** The gripper is NOT swapped: both models learn the measured width,
so neither commands a squeeze (F6). Do not compare the two open-loop errors
with each other. Recordings use ONE stiffness; the answer may change with it.

**Ask the owner.** Diffusion, `multi_task_dit`, or both? How many trials?

---

### F2 · Torque, force or tactile as a model input

**For.** Letting the policy feel contact, not only see it.

**The fact everything hangs on** (lerobot source, confirmed on the owner's GPU
PC): `diffusion` and `act` read `observation.state` and
`observation.environment_state`. `multi_task_dit` reads `observation.state`
only. Nothing reads `extra.*` or `observation.state.<name>`.

**Exists.** `add_environment_state.py` writes a dataset copy with the chosen
columns joined into `observation.environment_state` (USAGE §7.1). Tested on
synthetic datasets only.

**To finish, part A: training** [GPU PC].
1. `--dry-run` on `franka_electricbox` with `extra.ext_torque`; then for real.
2. `check_schema_drift.py` on the result.
3. Smoke-train 100 steps with diffusion, from scratch. Confirm
   `observation.environment_state` is in the checkpoint's `config.json`
   `input_features` (command in USAGE §7.1).
4. Only then a full training.

**Part B: the robot side** [code, then robot]. Not built: nothing in
`relative_lerobot_policy.py`, `policy_server.py`, `check_policy_openloop.py` or
`deploy_preflight.py` mentions the key, so a checkpoint trained in part A can
neither be checked open loop nor deployed yet. Design, in this order:
1. *Stamp.* At training time copy the dataset's `meta/environment_state.json`,
   and for each of its columns the `source` and parameters listed in
   `meta/record_config.json`, into `pose_repr.json` (as `action.source` is).
2. *Open loop first.* Make `check_policy_openloop.py` pass the column through
   from the dataset. No robot needed; proves the model side.
3. *Robot side.* Rebuild each column with the SAME record source that recorded
   it (`SOURCE_REGISTRY[source](env, **params)`), join in the stamped order,
   and send it as `observation.environment_state`. Refuse to start when a
   source is missing or the width differs from the stamp.
4. *The trap.* A sensor in the env's `sensor_configs` is added to
   `observation.state` automatically (as `observation.state.sensors_<name>`),
   which is why the deploy env has `sensor_configs: []`. Adding the torque
   sensor there would widen the state from 10 to 17 and break the checkpoint.
   `build_obs_frame` must leave out of the state what goes to the environment
   state, and the state-width check must still pass.

**Part C: `multi_task_dit`.** It cannot use the environment state. The only
route is to append the columns to `observation.state` itself. That changes the
state layout (GUIDANCE §3 item 4): it needs the owner's yes, the wrt-start
append must stay last, and deployment gets the same trap as B4 in reverse.

**It works when.** A checkpoint lists the key; the open-loop check runs with
it; on the robot, pressing the gripper by hand changes what the log shows
being sent to the model.

**Traps.** A model given force may ignore it (FACTR, TA-VLA): compare against
the same model without it, under occlusion, before believing it helps. A
finetune from `--policy.path` keeps the old input list. Handheld data has no
torque: this is a robot-only input unless tactile is used on both.

**Ask the owner.** Part C: append to `observation.state` for `multi_task_dit`?

---

### F3 · Safety limits on the commanded pose

**For.** A wrong frame, decode or scale must not reach the controller as a jump.

**Exists.** `max_target_step_m` / `max_target_step_deg` (0.03 m and 5° in the
three deploy envs; off elsewhere) and a workspace box `min_x … max_z` (none set
anywhere). USAGE §9 "Safety limits on the commanded pose".

**To finish** [robot].
1. Deploy any working checkpoint. Look for `Target step limit acted` in the
   log. In the recorded Franka data the command exceeds the limit in 0.21 % of
   steps, so expect it rarely; if it fires constantly, something is wrong
   upstream, not with the limit.
2. Choose box values for the rig with the owner (the arm may be on a mobile
   base and reach below its own base, so there is no default floor).
3. Record what was seen in `HANDOFF.md`.

**Traps.** The step limit compares with the previous TARGET, never the measured
pose; do not "fix" that. The box limits position only, not orientation.

---

### F4 · One-camera Franka configs

**For.** The rig as it is now: one OAK-W wrist camera.

**Exists.** `dric_franka_oakw_umi` (streamed teleop recording),
`factr_dric_franka_oakw` (FACTR recording), `dric_franka_oakw_deploy_umi`,
`dric_franka_oakw_deploy_groot`, and `umi_robot_full_record.yaml`. Written from
the owner's description of the camera; checked only as YAML
(`tests/test_franka_oakw_configs.py`).

**To finish** [robot]. Start one recording with each recording env and one
deployment with each deploy env. The first failure is usually a topic name, a
camera that publishes nothing, or a sensor the record config expects and the
env lacks; the error message names it. Record one short episode and run
`inspect_dataset.py` on it: the columns and sizes must match the existing
`franka_electricbox` data, plus the new `extra.gripper_target`.

**It works when.** A dataset recorded with the new configs merges with, or
trains like, the old one.

---

### F5 · Policy server and asynchronous inference

**For.** Running the model on the GPU PC, and removing the pause between
chunks.

**Exists.** `scripts/policy_server.py` and the `server:` setting of
`relative_lerobot_policy`; `async_inference: true` (default `false`). Checked
with a real checkpoint over loopback on one machine.
`GUIDE_REMOTE_SERVER_AND_TACTILE.md` part 1.

**To finish** [robot + GPU PC].
1. Serve a checkpoint; deploy with `server:` set. Read the round-trip time both
   sides log.
2. Turn `async_inference` on. The arm should no longer pause at each chunk; if
   it still does, the round trip is longer than `n_action_steps` steps.
3. For `multi_task_dit`: `--num-inference-steps` and `--scheduler` do not
   reach it yet. Find its own speed setting before writing code.

**Traps.** One robot per server. No encryption: private wired link only. The
first chunk always blocks. `_start_worker` is the only place that knows local
from remote; keep it so.

---

### F6 · The gripper command as a label

**For.** The measured width stalls at the object, as the measured pose stalls
at a surface. A model trained on it never commands a squeeze.

**Exists.** The record source `gripper.target_width_normalized`, already listed
in `umi_robot_full_record.yaml` as `extra.gripper_target`. No dataset has the
column yet. `swap_action_offline.py` swaps the arm only. Audit step E reads it.

**To finish.**
1. [robot] Record new data with `umi_robot_full_record.yaml`; confirm the
   column exists and moves (`inspect_dataset.py`, audit step E).
2. [cloud] Add an option to the swap tool that also swaps the gripper
   dimension, refusing when the column is missing or constant. Test it like the
   arm swap.

**Trap.** It gives squeeze or no squeeze, not a graded force: the Robotiq
applies its fixed effort once blocked.

---

### F7 · Simulation

**For.** Trying label and input ideas with a known ground truth, without robot
time. Plan and simulators: `RESEARCH_FORCE_TACTILE.md`.

**Exists.** `crisp_gym/sim/spring_pad.py`: a 72-taxel pad on a spring pressed
onto a peg; spring force equals taxel force; 11 tests in CI.

**To finish.**
1. [cloud] Grow it into a small task (reach, press with a wanted force, slide)
   driven by a scripted expert, and a writer that stores episodes as a LeRobot
   dataset with the REAL column names. Proof: `inspect_dataset.py`,
   `check_schema_drift.py` and the residual steps of the audit (A, F, G) run on
   it unchanged and report the gap the simulation was given.
2. [cloud] Run F1 in simulation at several stiffness values.
3. [owner's PC] `tactile_envs` (a Robotiq 2F-85 with taxel pads); it could not
   be executed in a cloud session.
4. [owner's PC] The real controller in MuJoCo (`FRANKA_FAKE_HARDWARE=true`).
   That scene has the arm only: no gripper, camera or torque topic, so the
   usual configs do not start on it as they are.

**Traps** (each pinned by a test in `tests/test_sim_spring_pad.py`). The touch
sensor looks along its site's −z axis. Move the command, do not jump it. Use
the simulator to compare variants, never for absolute numbers.

---

### F8 · Residual model: a command label for handheld data

**For.** Handheld demonstrations have tactile and motion but no command. If F1
shows the command matters, predict it.

**Not started. Do not start before F1 has an answer.**

**Shape of the work.**
1. In simulation (F7), where the true contact force is known: fit
   `stiffness × (command − measured)` from tactile and velocity on
   spring-driven episodes; test on episodes moved kinematically, which is what
   a handheld device is.
2. On robot data, where the command is known: the same fit, scored on held-out
   EPISODES, contact frames only.
3. Relabel handheld data into a NEW column. Never overwrite `action`.

**Traps.** It is a labelling model, not a controller: applied online it only
sustains whatever force is already there. The residual means something only
together with the stiffness it was recorded at; use `K × residual`. Tactile is
on hold (F9), so on the Franka the input today is joint torque, which a
handheld device does not have.

---

### F9 · Tactile (on hold)

Held by the owner until the skins stop reading when the gripper closes on
itself. When that is fixed, in this order:

1. Recording on the robot goes through this repository: a sensor config and a
   record source for the taxels AND their time stamp (`source_stamp_ns`). Today
   an external tool records them and the robot dataset has no stamp, so its
   high-rate CSV files cannot be joined to frames.
2. `audit_command_residual.py --tactile-only` on handheld and robot data;
   compare noise, dead taxels and contact share.
3. Fix the probe of audit step D before reading it: fit on contact frames, a
   few components instead of raw taxels (HANDOFF §7 "KNOWN WEAKNESS").
4. Then F2 with the taxels as columns, and F8.

Loader for the high-rate CSVs: `util/tactile_raw.py` (cut windows by time,
never by row count).

---

### F10 · The test suite in the real environments

**For.** Knowing that "tests pass" means the same on the machines that matter.

**State.** Run only in a ROS-free environment. `tests/conftest.py` gives every
test file its own fake modules; with the real ROS and lerobot installed that
isolation has never been exercised. Tests marked `lerobot` have not run under
pytest anywhere.

**To finish** [owner]. One run on each machine, pasted back:

```bash
python -m pytest -rs -q
```

Expect failures. Sort each one: (a) a fake that now collides with the real
package: fix the test; (b) the isolation: try `CRISP_TEST_NO_ISOLATION=1`
on that file to confirm, then fix the conftest; (c) a real bug: that is the
find.

---

### F11 · Open measurements

| question | how to settle it | why it matters |
|---|---|---|
| The arm delivers 150–280 N/m against 400 configured | hold a fixed target, push the gripper with a scale to 10 N and 20 N, note the displacement; or the same in the MuJoCo stack (F7 step 4) | the size of every command-minus-measured gap depends on it |
| Which copy of the four duplicated configs is in effect | `echo "CRISP_CONFIG_PATH=$CRISP_CONFIG_PATH"` in the robot PC's pixi shell | decides which copy is kept (HANDOFF §6.1) |
| Skipped frames per episode | count the "returned None" lines of one episode | confirms the tested explanation (2 relative, 1 otherwise) |
| Single-frame jumps of up to 138 mm in the measured pose | look at those frames' timing in a recorded episode | possibly dropped frames (HANDOFF §4b) |
| FACTR absolute-joint teleop on the Franka | ask the owner whether it has been used since HANDOFF §6.3 was written | marked "not yet validated on hardware" there |

---

### F12 · Other trained-policy paths

`dric_franka_oakw_deploy_groot.yaml`, the `groot_*` policy configs,
`train_groot*.sh`, `groot_preflight.py`, and the SmolVLA findings of HANDOFF
§4c exist. This document does not know which of them has run for real. Ask
the owner before planning work on them, and read §4c first: it explains why a
from-scratch diffusion policy beat a finetuned VLA on this rig.

---

## 4. First-run checklist

Things that exist and have never run for real. One line each; the commands are
in the USAGE section named.

**GPU PC**

| run | where | look for |
|---|---|---|
| the test suite | F10 | failures to sort |
| `add_environment_state.py --dry-run` on real data | USAGE §7.1 | the slice table; no refusal |
| `swap_action_offline.py` | USAGE §14 | `meta/action_repr.json` in the copy |
| a 100-step relative training on each | USAGE §8 | `pose_repr.json` with the right `action.source` |
| `check_policy_openloop.py` on each | USAGE §14 | no `ACTION SOURCE MISMATCH` |
| `deploy_preflight.py` on a new checkpoint | USAGE §9 | every check green |
| `policy_server.py` | GUIDE part 1 | `/healthz` answers `OK` |

**Robot PC**

| run | where | look for |
|---|---|---|
| the test suite | F10 | failures to sort |
| one recording with `dric_franka_oakw_umi` | USAGE §5 | starts; `inspect_dataset.py` shows the expected columns, incl. `extra.gripper_target` |
| one deployment with `dric_franka_oakw_deploy_umi` | USAGE §9 | starts; the action source line; step-limit warnings rare |
| the same through the server, then with `async_inference: true` | GUIDE part 1 | round-trip time; no pause between chunks |
| `echo "CRISP_CONFIG_PATH=$CRISP_CONFIG_PATH"` | F11 | which config copy is live |

---

## 5. Keeping this true

When a feature changes level, change §2 and its section in the same commit,
and write the measured result in `HANDOFF.md`. When the owner answers a
question listed here, write the answer into the section. Keep this file under
500 lines; detail goes to the document a section names.
