# crisp_gym code review — structure, functions, and where to improve

Reviewed: branch `claude/determined-faraday-1z6rx0` at commit `afb4a18`, 2026-11-16.

**What this file is.** A record of findings, for a later session that will turn
them into work guidance. It changes nothing. Per `CLAUDE.md`, no item here is
to be fixed without the owner's go-ahead; several need an owner decision first
(marked **DECIDE**).

**How it was produced.** Reading the code, plus measurements (AST outlines,
grep counts, `ruff`, the ROS-free test suite). Nothing was run against ROS,
lerobot, torch, a GPU or a robot. Each finding carries one status:

| status | meaning |
|---|---|
| VERIFIED | read in the code at the cited line, and where possible reproduced |
| INFERRED | follows from reading; not executed |
| FROM HANDOFF | already recorded in `HANDOFF.md`; listed here so the backlog is in one place, details stay there |

Line numbers are for commit `afb4a18` (after the one-off `ruff format`).
Scope is `crisp_gym` only; `crisp_py` and `crisp_controllers_demos` are cited
where a finding depends on them.

---

## 0. Owner decisions that constrain any fix

Recorded in this conversation or in `HANDOFF.md`; a guidance document should
repeat them, not re-open them.

- Only `relative_lerobot_policy` is used for deployment. The owner wants
  asynchronous inference to remove the pause between chunks; that is
  `async_inference: true` in `relative_lerobot_policy.yaml`, not the old
  `async_lerobot_policy` class.
- Remote inference = `relative_lerobot_policy` with `server:` set, behind
  `scripts/policy_server.py`. `RemotePolicy` stays a minimal client for other
  servers (HANDOFF §4 item 1). `_start_worker` is the only place that knows
  where the model runs.
- Controller gains are NOT stored in datasets. The env YAML's
  `cartesian_control_param_config` is their single home; no second copy in the
  crisp_py robot config.
- Training/GPU-side files stay free of crisp_py / ROS imports. Recording machine:
  ROS 2 Humble, Python 3.11, lerobot 0.4.4. GPU machine: lerobot 0.6.1.
- Tactile work is on hold until the closed-gripper reading is fixed in hardware.
- The measured-pose vs command-action comparison runs on the Franka first.
- Process: plan first, change only what was approved, ask (`CLAUDE.md`); develop
  on the designated `claude/...` branch; no pull request unless asked.

---

## 1. Architecture as it is

### 1.1 Machines and processes

```
ROBOT PC (ROS 2, Python 3.11)                         GPU PC (no ROS)
┌──────────────────────────────────────────────┐      ┌──────────────────────────────┐
│ record: record_lerobot_format_leader_follower│      │ train: lerobot_relative_pose │
│   / record_umi_handheld                      │      │   train_absolute_next_pose   │
│   ├ env (crisp_gym.envs) ─ crisp_py ─ ROS    │      │   train_action_from_target_* │
│   ├ teleop (FACTR / streamed pose / mouse)   │      │ dataset tools, diagnostics   │
│   └ RecordingManager ─ writer subprocess     │      │   (audit, open-loop, ...)    │
│        └ LeRobotDataset on disk ─────────────┼─copy─┼─►                            │
│ deploy: deploy_policy                        │      │ scripts/policy_server.py     │
│   └ RelativeLerobotPolicy ── pipe ──┐        │      │   └ inference_worker (subproc)│
│        or ── websocket ─────────────┼────────┼──────┼──►                            │
│      inference_worker (local subproc)┘       │      └──────────────────────────────┘
└──────────────────────────────────────────────┘
        controllers: crisp_controllers (CIC / JIC, 1 kHz) via crisp_py at 50 Hz
```

### 1.2 Package layers

| package | lines | role | heavy imports at module top |
|---|---|---|---|
| `crisp_gym/config` | 220 py + 56 YAML | config search (`find_config`), home poses, all YAML configs | — |
| `crisp_gym/util` | 2,000 | rot6d math, features, loop timing, epochs, tactile CSV, prompts | gymnasium (features) |
| `crisp_gym/envs` | 2,500 | `ManipulatorBaseEnv` + Cartesian/Joint envs, env configs, handheld env, wrappers | rclpy, crisp_py, scipy |
| `crisp_gym/teleop` | 1,000 | FACTR / streamed-pose / spacemouse leaders | rclpy, crisp_py |
| `crisp_gym/record` | 2,700 | `RecordConfig` + record sources, record/drive functions, `RecordingManager` | rclpy, lerobot (manager) |
| `crisp_gym/policy` | 3,800 | policy registry, relative policy + inference worker, remote protocol/client/transport, two legacy policies | torch/lerobot only in the legacy ones |
| `crisp_gym/scripts` | 9,700, 21 files | robot entry points, GPU training wrappers, dataset tools, diagnostics | varies per file |
| `scripts/` (root) | 1,500 py + shell | `policy_server.py`, GR00T/DiT launchers, `dataset_conversions/` | lerobot (conversions) |
| `tests` | 13,100, 34 files | mostly synthetic-data tests; ROS-free suite runs in CI | fakes for torch/rclpy/lerobot |

Import direction is mostly clean (scripts → record/policy → envs/teleop →
util/config). Cross-layer imports are guarded with `TYPE_CHECKING` in the
relative and remote policies, `record_functions` and `lerobot_features`.
Exceptions: the two legacy policies import `crisp_gym.envs` (and so ROS) at
module top; `recording_manager.py` imports lerobot and rclpy at module top
(its own TODO at `recording_manager.py:19`).

### 1.3 Data flow and the contracts that hold it together

```
record ──► dataset (+ meta/record_config.json) ──► [swap_action_offline: + meta/action_repr.json]
       ──► train ──► checkpoint (+ pose_repr.json | action_repr.json) ──► deploy / policy_server
```

| contract file | written by | read by | carries |
|---|---|---|---|
| `meta/record_config.json` | `RecordConfig.to_metadata` (both record scripts) | `postprocess_align_datasets`, `lerobot_relative_pose.read_action_source` | action definition + lookahead, observation sources, robot type |
| `meta/action_repr.json` (dataset) | `swap_action_offline.py` | `read_action_source` | the action column is the command |
| `pose_repr.json` (checkpoint) | `lerobot_relative_pose.stamp_pose_repr` | `relative_lerobot_policy.find_pose_repr`, `policy_server.checkpoint_spec` | relative conventions, state layout, wrt-start, action source |
| `action_repr.json` (checkpoint) | the two absolute launchers | `find_action_repr`, open-loop check | "this checkpoint outputs absolute poses" |

These are the right idea, but each file has its own ad-hoc reader in 2–3
places (finding A2) and the key names they describe are string literals
everywhere (A3).

### 1.4 Extension points that exist

- Policies: `@register_policy` + `make_policy` (`policy/policy.py`).
- Record sources: `@register_source` (`record/record_config.py:66`).
- Sensors: crisp_py's `@register_sensor` (`crisp_py/sensors/sensor.py`).
- Inference transport: `WorkerConnection` protocol (`policy/worker_connection.py`).
- Env configs: `STRING_TO_CONFIG` dict of dataclasses **and** YAML files under
  `config/envs/` — two mechanisms for one thing (C1, A9).

---

## 2. Findings

### 2.1 Summary

| ID | finding | priority | effort | status |
|---|---|---|---|---|
| S1 | Safety box silently drops any bound equal to 0.0; the default z floor never applies | P0 | S | VERIFIED |
| S2 | No limit on how far the commanded pose may jump in one tick | P0 | M | VERIFIED (absence) |
| C1 | Env names defined both as a Python class and a YAML: the class wins, the YAML is never read | P1 | S | VERIFIED |
| C2 | Config files shadowed between crisp_py and crisp_gym; search order flips with an env var | P1 | S–M | FROM HANDOFF |
| C3 | `observation.state` concatenated in dict insertion order | P1 | S | FROM HANDOFF |
| C4 | `data_fn` returns `None` twice per episode, uninvestigated | P1 | S | FROM HANDOFF |
| C5 | `control_frequency`, `block`, `max_episode_steps` are dead; `step()` builds an observation every caller discards | P1 | M | VERIFIED |
| C6 | Recording with `--follower-config franka` bypasses the config system with hard-coded values | P1 | S | VERIFIED |
| C7 | Interactive config prompts default to whatever the filesystem lists first | P1 | S | VERIFIED (code) |
| C8 | `make_policy` raises `ValueError` with a tuple message | P3 | XS | VERIFIED |
| C9 | 14 test files use `SourceFileLoader.load_module()`, removed in Python 3.15 | P2 | S | VERIFIED |
| C10 | `scripts/dataset_conversions/*` use an old lerobot API | P2 | S | INFERRED |
| A1 | `crisp_gym/scripts` mixes four kinds of program; a second `scripts/` folder at the root | P2 | L | VERIFIED |
| A2 | GPU-side code is duplicated so scripts stay single-file | P2 | M–L | VERIFIED |
| A3 | Dataset schema lives in string literals across 20+ files | P2 | M | VERIFIED |
| A4 | 14 functions over 150 lines, 35 over 80 | P2 | L | VERIFIED |
| A5 | Two unused legacy policies with their own inference workers | P2 | S | VERIFIED + owner |
| A6 | `RelativeLerobotPolicy` file: 1,718 lines, one class of 858 lines | P2 | L | VERIFIED |
| A7 | `RecordingManager`: one class of 1,031 lines doing five jobs | P2 | L | VERIFIED |
| A8 | Five places spin their own ROS executor in a daemon thread | P2 | M | VERIFIED |
| A9 | Two mechanisms for env configs (classes and YAML) | P2 | M | VERIFIED |
| A10 | 86 broad `except Exception` handlers in 22 files; `print` in library code | P2 | M | VERIFIED |
| K1 | Packaging: wheel has no configs, installs tests as a package, dependencies undeclared | P2 | M | VERIFIED |
| K2 | CI runs only the ROS-free tests; ROS, lerobot and hardware paths never run in CI | P2 | M–L | VERIFIED |
| K3 | Test fakes installed at module level and duplicated across files | P3 | M | VERIFIED |
| K4 | Lint baseline of 234 findings in 25 files | P3 | ongoing | VERIFIED |
| K5 | Version and changelog follow upstream; badges point at upstream | P3 | S | VERIFIED |
| D1 | Six overlapping docs, about 4,000 lines; HANDOFF mixes invariants and logs | P3 | M | VERIFIED |
| D2 | 12 config files referenced nowhere in the repository | P3 | S | VERIFIED |
| T1–T4 | Tactile and audit tooling (on hold) | P3 | — | VERIFIED |

Priority: **P0** can move the robot somewhere it should not go; **P1** silently
wrong data, training or configuration; **P2** maintainability with a real cost
today; **P3** hygiene. Effort: XS < 1 h, S < ½ day, M 1–3 days, L > 3 days.

### 2.2 P0 — safety

#### S1. Safety box drops any bound equal to 0.0 — VERIFIED

`manipulator_env_config.py:224-239` builds the workspace box as
`self.min_x or -float("inf")` (same for all six bounds). In Python `0.0 or -inf`
is `-inf`, so a bound of exactly 0.0 is replaced by "no bound". The default is
`min_z: None | float = 0.0  # e.g., table height` (`:120`), and no shipped env
YAML sets any bound, so **the table-height floor that every env appears to have
has never been applied**. Any user bound of 0.0 (a common value for y or z) is
dropped the same way. The box is applied in `ManipulatorCartesianEnv.step`
(`manipulator_env.py:855`, `:860`, via `clip_position_for_safety` at `:732`).

Fix: `-inf if self.min_x is None else self.min_x`, with a unit test on the
config alone (no ROS needed). **Behaviour change**: once fixed, targets below
z = 0 in the robot base frame are clipped. Confirm on hardware that z = 0 is the
intended floor for each rig (Franka base frame vs UR base frame, and the TCP
frame in use) before enabling it; otherwise default `min_z` to `None` and make
the floor an explicit per-rig setting.

#### S2. No limit on the per-tick jump of the commanded pose — VERIFIED (absence)

Deployment path: model output → `RelativeLerobotPolicy._to_env_action`
(`relative_lerobot_policy.py:939`) → `env.step(block=False)` → crisp_py
`Robot.set_target` (`crisp_py/robot/robot.py:442`), which stores the pose as is.
The only guard in crisp_gym is the safety box (S1, broken for 0-bounds, and
position only). The bring-up's `cartesian_impedance_controller` block in
`crisp_controllers_demos/.../config/fr3/controllers.yaml` sets no explicit
limit (the joint impedance block has `max_delta_tau: 0.5`); whether
crisp_controllers clips internally was **not checked**. A wrong composition
base, frame, rot6d decode or gripper scaling therefore reaches the controller
as a jump.

Suggested: a configurable maximum translation and rotation per tick, applied
in the env's Cartesian `step` (so every policy and teleop path gets it), that
clips and logs once per episode when it acts. `max_startup_joint_offset`
already exists for joint control and is the model to follow.

### 2.3 P1 — silently wrong data or configuration

#### C1. Class beats YAML for the same env name — VERIFIED

`make_env_config` (`manipulator_env_config.py:602`) looks in `STRING_TO_CONFIG`
first and reads a YAML only if the name is not there. `no_cam_franka` and
`only_wrist_cam_franka` exist as both. `deploy_policy.py:299` and
`record_lerobot_format_leader_follower.py:250` call `make_env` without
`config_path`, so `config/envs/no_cam_franka.yaml` is never read: that YAML
sets `cartesian_control_param_config: null` and `gripper_mode: none`, while
the class (via `FrankaEnvConfig`, `:350`) pushes
`control/default_cartesian_impedance.yaml` — whose crisp_py copy wins (C2) —
and keeps the default gripper mode. Editing the YAML has no effect, silently.

Fix now: raise when a name exists in both places. Later: A9.

#### C2. Config shadowing between packages — FROM HANDOFF §6.1

`find_config` returns the first match in `[crisp_py/config, crisp_gym/config]`.
Four files still exist in both with different content
(`control/default_cartesian_impedance.yaml`, `cameras/wrist_camera.yaml`,
`control/gravity_compensation_on_plane.yaml`,
`control/joint_velocity_control.yaml`); crisp_py's copy wins. Setting
`CRISP_CONFIG_PATH` reverses the whole list (`config/path.py:54`), so precedence
depends on an unrelated environment variable. The audit's `--env-config`
reproduces this order on purpose.

Fix: give each file name one owner (rename or delete the crisp_gym copy, as
was done for `fr3_joint_control.yaml`), and make the built-in order not depend
on whether `CRISP_CONFIG_PATH` is set. HANDOFF §6.1 warns that applying
crisp_gym's nullspace values would change Cartesian behaviour — hardware check
first. **DECIDE** which copy is right for each.

#### C3. State vector built from dict order — FROM HANDOFF §4b

`concatenate_state_features` (`util/lerobot_features.py:260`) concatenates
`observation.state` in obs-dict insertion order, not the declared feature
order. It is correct today only because `_collect_obs` iterates
`record_config.observations` in order. A reorder anywhere upstream silently
permutes the state a model sees. Fix: concatenate in the declared order and
test it.

#### C4. `data_fn` returns `None` twice per episode — FROM HANDOFF §4b item 4

Reproducible in every field run (`recording_manager.py:953`, the
`skipped=True` path). No row is written, so no desync, but the cause is
unknown. Investigate before relying on frame counts per episode.

#### C5. Dead configuration and wasted observations — VERIFIED

- `control_frequency` (`manipulator_env_config.py:69`) and the `block` path of
  `step` (`manipulator_env.py:870`, `:1088`, both marked FIXME) are unused: all
  seven call sites pass `block=False` (relative/legacy/remote policies,
  `record_functions.py:68`, `:167`, `leader_follower_teleop.py`). Loop rate is
  owned by the recording/deploy loop.
- `max_episode_steps` (`:125`, FIXME) is read only in the base `step` to set
  `truncated`; the Gym wrappers in `envs/env_wrapper.py` pass it through, and
  no recording or deploy loop acts on it.
- Every `step` ends with `obs = self._get_obs()` (`manipulator_env.py:876`,
  `:1094`); recording discards it and re-reads (HANDOFF §4b "STILL NOT DONE" 1),
  costing a full observation (images included) per tick.

Fix: remove `block` and `control_frequency`'s pretence, or document them as
"not the loop rate"; make `step` not build an observation unless asked.
This changes the Gym `step` contract — check examples and the wrappers in
`envs/env_wrapper.py` first.

#### C6. Hard-coded Franka path in the recording script — VERIFIED

`record_lerobot_format_leader_follower.py:232-240`: when
`--follower-config franka`, the script builds
`FrankaEnvConfig(control_frequency=50, gripper_config=None, camera_configs=[])`
itself instead of going through `make_env`. Cameras and gripper are dropped
and no YAML can change that. Fix: remove the special case and use an env YAML.

#### C7. Interactive prompt defaults depend on filesystem order — VERIFIED (code), impact INFERRED

`list_configs_in_folder` (`config/path.py`) returns `Path.iterdir()` order
(unsorted) across config paths, without removing a name that exists in two
paths. `deploy_policy.py:262-268` offers `policy_configs[0]` as the default
when `--policy-config` is omitted. The `0_default_config.yaml` name suggests
sorted order was expected; in fact pressing Enter can select the legacy
`lerobot_policy` or any other file. The env prompt (`deploy_policy.py:254`)
defaults to the first entry of `list_env_configs()`, which is the first class
in `STRING_TO_CONFIG` (`right_aloha_franka`), not a rig the owner uses. Fix: sort, de-duplicate by stem (first path wins, as `find_config`
does), and default to the documented choice.

#### C8. `make_policy` error message is a tuple — VERIFIED

`policy/policy.py:74` passes two f-strings to `ValueError`, so the message
prints as `('Policy ... not registered', 'Make sure ...')`. Join them.

#### C9. Deprecated test loader — VERIFIED

14 test files load modules with `SourceFileLoader(...).load_module()`, removed
in Python 3.15 (the DeprecationWarning shows in every run). Replace with one
helper in `tests/conftest.py` using `importlib.util.spec_from_file_location` +
`exec_module` (as `test_pose_math.py` already does).

#### C10. Old dataset conversion scripts — INFERRED

`scripts/dataset_conversions/*.py` import `lerobot.envs.utils.einops` and the
pre-0.5 `LeRobotDataset` API (e.g. `fix_rotation_vectors_in_dataset.py:12`).
Not run here; likely broken on lerobot ≥ 0.5. Newer tools cover much of it
(`migrate_euler_delta_to_rot6d.py`, `postprocess_align_datasets.py`,
`swap_action_offline.py`). **DECIDE**: delete, or port and test.

### 2.4 P2 — structure

#### A1. What `crisp_gym/scripts` holds — VERIFIED

21 files, 9,700 lines, four kinds of program side by side:

| kind | runs on | files |
|---|---|---|
| robot entry points | robot PC (ROS) | `deploy_policy`, `record_lerobot_format_leader_follower`, `record_umi_handheld`, `leader_follower_teleop`, `deploy_preflight`, `check_config` |
| training wrappers | GPU PC | `lerobot_relative_pose`, `train_absolute_next_pose`, `train_action_from_target_cartesian` |
| dataset tools | either | `swap_action_offline`, `migrate_euler_delta_to_rot6d`, `postprocess_align_datasets`, `fix_dataset_task`, `reencode_dataset_videos`, `check_schema_drift` |
| diagnostics | GPU PC | `check_policy_openloop`, `audit_command_residual`, `groot_preflight`, `check_relative_pose`, `viz_encoder_attention`, `check_available_policies` |

Plus root `scripts/`: `policy_server.py`, shell launchers, `dataset_conversions/`.
46 path references in the docs point at these files, so any move needs thin
shims at the old paths. **DECIDE** whether a layout change is worth that.

#### A2. Duplicated GPU-side code — VERIFIED

The training and dataset scripts must import neither crisp_py nor ROS, and are
written to be copyable as single files, so shared code is copied:

| code | copies | similarity |
|---|---|---|
| rot6d / pose9 math | `util/rot6d.py:31`, `lerobot_relative_pose.py:97`, `audit_command_residual.py:124` | same convention, pinned by `test_pose_math.py` |
| `_setup_epochs`, `_load_epoch_steps` | the three training launchers | byte-identical |
| `_installed_lerobot_version` | the three launchers | three variants (~70 % similar) |
| `_disable_video_query`, `convert_item`, `_to_np` | `lerobot_relative_pose`, `train_action_from_target_cartesian` | 55 %, different, identical |
| `_assert_features`, `_assert_target_varies` | `swap_action_offline`, `train_action_from_target_cartesian` | parallel implementations |
| parquet / `meta/info.json` walking | 7 scripts | each its own |
| checkpoint stamp readers | `relative_lerobot_policy.action_source_from_stamps:232`, `lerobot_relative_pose.checkpoint_action_source:459` | identical behaviour, pinned by `test_both_readers_of_the_stamps_agree` |

Suggested: one ROS-free subpackage (e.g. `crisp_gym/offline/`) needing only
numpy/pandas/pyarrow: pose math, dataset I/O (data files, info, stats for v2.x
and v3.0), and the contract files of §1.3. The GPU PC already needs a clone for
`policy_server.py`, so single-file copying buys less than it did. **DECIDE**:
give up single-file copying or keep the copies and pin them with tests.

#### A3. Schema as string literals — VERIFIED

Files containing each literal: `"observation.state"` 20,
`"observation.state.cartesian"` 10, `"observation.state.gripper"` 7,
`"action"` 24, `"extra.target_cartesian"` 3. Gripper scaling values (`0.09`,
`0.085`, `reference_width`) appear 37 times across three files. The tactile
columns (`extra.tactile.<side>` + sync columns) are defined by an external
recorder (`umi_data_collection`, not in these repos). Suggested: one schema
module (names, shapes, units, rot6d layout, gripper semantics) used by record,
train, deploy and the audit, and a dataset validator built on
`check_schema_drift.py`.

#### A4. Long functions — VERIFIED

| function | location | lines |
|---|---|---|
| `_run` | `scripts/check_policy_openloop.py:439` | 535 |
| `main` | `scripts/record_lerobot_format_leader_follower.py:33` | 507 |
| `main` | `scripts/reencode_dataset_videos.py:716` | 438 |
| `main` | `scripts/deploy_policy.py:20` | 417 |
| `analyze` | `scripts/audit_command_residual.py:654` | 297 |
| `inference_worker` | `policy/relative_lerobot_policy.py:1423` | 296 |
| `align` | `scripts/postprocess_align_datasets.py:291` | 237 |
| `main` | `scripts/record_umi_handheld.py:30` | 234 |
| `__init__` | `policy/relative_lerobot_policy.py:585` | 201 |
| `_writer_proc` | `record/recording_manager.py:665` | 176 |
| `make_data_fn` | `policy/relative_lerobot_policy.py:1160` | 174 |

The three entry-point `main`s each do argument parsing, interactive prompts,
env/policy/recorder construction and the loop in one body, which is why their
tests replace whole modules (K3). Split points: parse → build → run, with the
build step returning plain objects a test can construct.

#### A5. Legacy policies — VERIFIED, owner confirmed unused

`lerobot_policy.py` (511 lines) and `async_lerobot_policy.py` (289) each carry
their own `inference_worker`. The async one hard-codes `n_obs=2`, `n_act=5`,
replan 3 (`async_lerobot_policy.py:48`), uses `print`, and has no relative-pose
handling. `relative_lerobot_policy` already implements asynchronous prefetch
(`async_inference`, `prefetch_lead`, with index offset on swap). Their configs
`config/policy/0_default_config.yaml` and `1_async_default_config.yaml` are
referenced nowhere and can become the prompt default (C7). Suggested: remove
both classes and both YAMLs, keep `crisp_gym.policy` importable without them
(it already tolerates their absence, `policy/__init__.py:33`). **DECIDE**.

The `Policy` base class (`policy/policy.py`) defines `make_data_fn()` returning
a closure that both infers and steps the env. Inference and actuation are
mixed, and the observation/action contract is untyped. Worth revisiting when
A6 is done, not before.

#### A6. `relative_lerobot_policy.py` — VERIFIED

1,718 lines: helpers, `build_obs_frame`, server settings, the 858-line
`RelativeLerobotPolicy` (`__init__` 201, `make_data_fn` 174 with the sync and
async paths interleaved, `_to_env_action` 121), and the 296-line
`inference_worker` that `policy_server.py` also runs. Suggested split, keeping
`_start_worker` as the only local/remote seam:

1. observation building and checks (`build_obs_frame`, `_verify_*`);
2. a chunk scheduler (sync / async / prefetch / swap offset) as plain Python
   with no torch — the part most worth unit-testing;
3. action composition (`_to_env_action`, gripper conversion, euler autofix);
4. the inference worker in its own module, imported by the server.

#### A7. `RecordingManager` — VERIFIED

`recording_manager.py:57`, 1,031 lines, 32 methods: writer subprocess
(`_writer_proc` 176 lines, pickled state via `__getstate__`), video encoder
preflight and streaming checks, ROS and keyboard episode control, the episode
loop, and dataset creation. Suggested: writer side (everything the subprocess
needs) in its own module with no rclpy import; control side separate. HANDOFF
§4b holds the measured timing behaviour any refactor must keep.

#### A8. ROS executors in daemon threads — VERIFIED

Each of `recording_manager.py:1112`, `teleop/teleop_factr_stream.py:130`,
`teleop/teleop_sensor_stream.py:66`, `envs/umi_handheld_env.py:174` and
`teleop/spacemouse_node.py:117` starts its own executor (single- or
multi-threaded) in a daemon thread, in addition to crisp_py's own nodes. Many
nodes per process, no ordered shutdown, threads killed at interpreter exit.
Suggested: one helper that owns a shared executor and shuts nodes down in
order. INFERRED risk: callbacks running during interpreter teardown.

#### A9. Env configs in two mechanisms — VERIFIED

10 dataclass subclasses in `STRING_TO_CONFIG` (`manipulator_env_config.py:626`)
and 16 YAMLs in `config/envs/`; two names collide (C1). HANDOFF §4 item 4
plans decorator registries. Suggested: YAML-only env configs, with the
dataclasses reduced to the base `ManipulatorEnvConfig`.

#### A10. Error handling and output — VERIFIED

86 `except Exception` handlers in 22 files (most in
`relative_lerobot_policy.py` 15, `recording_manager.py` 9,
`remote_policy.py` 7, `spacemouse_node.py` 6, `deploy_preflight.py` 6). Some
are deliberate ("provenance must never kill a training run"). On the robot
side each should do one of: re-raise, hold pose and log with traceback, or be
narrowed to the exception it expects. `print` in library code:
`async_lerobot_policy.py` 6, `recording_manager.py` 3, `spacemouse_node.py` 1.

### 2.5 P2/P3 — packaging, tests, CI

#### K1. Packaging — VERIFIED

A wheel built from `pyproject.toml` contains no YAML configs (so `find_config`
finds nothing in a wheel install), and installs `tests`, `scripts` and
`examples` as top-level packages. `ruff` is a runtime dependency. Imported but
undeclared: `numpy`, `msgpack`, `websockets` (server path), `opencv-python`,
`av`, `torch`/`lerobot` (deploy and training), `typing_extensions`. Suggested:
package data for `config/**`, `packages.find.include = ["crisp_gym*"]`, extras
(`server`, `dataset`, `train`, `video`); ROS packages stay system-provided and
documented as such. pixi remains the robot-PC install path.

#### K2. CI coverage — VERIFIED

`.github/workflows/tests.yml` runs the ROS-free suite on 3.11/3.12; tests
needing lerobot/torch (`@pytest.mark.lerobot`) or ROS skip. Never run in CI:
any ROS node, `RecordingManager` with a real dataset, a real inference worker,
the fake-hardware Franka end-to-end test (HANDOFF §4 item 3, blocked here by
the network). Suggested, cheapest first: an import + `--help` smoke test of
every entry point in a pixi environment; then the fake-hardware test on a
self-hosted runner.

#### K3. Test fakes — VERIFIED

Test files install fake `torch` / `rclpy` / `lerobot` / `crisp_py` modules at
import time, with stub lists repeated across files. `tests/conftest.py`
isolates files from each other as a workaround (HANDOFF §5). The root fix is
shared fixtures that install and remove fakes. Not yet run under pytest in
the real ROS + lerobot environment.

#### K4. Lint baseline — VERIFIED

`pyproject.toml` lists 234 missing docstrings/annotations in 25 files as a
per-file baseline. Shorten as files are touched; never extend. CI pins ruff
0.12.0 (as `.pre-commit-config.yaml`); a newer local ruff can disagree on
formatting.

#### K5. Version and provenance — VERIFIED

`pyproject.toml` 4.2.1 and `CHANGELOG.md` come from release-please on the
upstream `main`; work on this fork's branches is unversioned. README badges
point at the upstream repository. **DECIDE** whether the fork needs its own
versioning.

### 2.6 P3 — documentation and config hygiene

#### D1. Documentation — VERIFIED

README 200 lines, USAGE 1,646, HANDOFF 1,024, REMOTE_INFERENCE 367,
GUIDE_REMOTE_SERVER_AND_TACTILE 324, `crisp_gym/scripts/README.md` 499 — about
4,000 lines — plus this file. HANDOFF mixes stable invariants (§1), decisions (§4, §6), field
measurements and investigation logs (§4b, §7) and process rules (§5). Stale:
the §2 heading still says "all on branch `claude/vigilant-tesla-VZJT6` unless
merged" (it is merged). Suggested: a short CONVENTIONS document (invariants a
change must not break), a decisions log, and an open-issues list (this file),
with USAGE split per workflow.

#### D2. Config files referenced nowhere — VERIFIED (repository only)

`control/clipped_cartesian_impedance.yaml`,
`control/default_operational_space_controller.yaml`,
`control/gravity_compensation_on_fix_position.yaml`,
`control/no_friction_cartesian_impedance.yaml`, `envs/right_robot_env.yaml`,
`grippers/gripper_left_v2.yaml`, `policy/0_default_config.yaml`,
`policy/1_async_default_config.yaml`, `recording/umi_robot_dual_cam_record.yaml`,
`teleop/example_teleop.yaml`, `teleop/left_leader.yaml`, `teleop/single_arm.yaml`.
They may still be chosen by name on a command line or by the owner's scripts;
ask before deleting.

### 2.7 Tactile and audit tooling (on hold, written in this conversation)

- **T1** `audit_command_residual.py` is 1,717 lines doing five jobs (residual
  audit, Franka stiffness identification with built-in DH kinematics, tactile
  report, CSV checks, gains lookup). Split after A2 exists.
- **T2** Step D's linear probe is unreliable when contact is rare (printed
  R2 ≈ −300 on `ur_electricbox`); the layout hint misread a 6-wide grid as
  36 × 2. Both documented in HANDOFF §7.
- **T3** Tactile is not a model input (`extra.*` is read by neither training
  nor deployment); the robot tactile dataset has no `source_stamp_ns`; the
  schema is owned by the external recorder.
- **T4** `swap_action_offline.py` swaps the arm only; the gripper action stays
  the measured width, so a "command" model never commands a squeeze.
  `extra.gripper_target` must be recorded first.
- Step C (stiffness) needs `extra.ext_torque` and works only on the Franka; on
  the UR it would need a recorded wrist wrench (`--wrench-key`, not written).

---

## 3. Suggested order of work

Each step: write the test first, keep existing command lines working, and run
in the ROS + lerobot environment before calling it done.

| phase | items | why first |
|---|---|---|
| 0 — small, safe | S1 (after the hardware check), C8, C9, C7, C1 guard (raise on collision) | robot safety and silent misconfiguration, each under a day |
| 1 — safety and data | S2, C3, C5, C6, investigate C4 | protects the robot and the datasets the comparison will use |
| 2 — needs decisions | A5 retire legacy policies, C2 one owner per config, C10, A2 + A3 offline subpackage and schema, K1 packaging | each needs an owner answer marked DECIDE |
| 3 — refactors | A6, A7, A8, A9, A4 splits, A1 layout, K2 CI in pixi, D1 docs | large; safer once phases 0–2 add tests around them |

---

## 4. Blind spots of this review

- No code was executed against ROS, crisp_py, lerobot, torch, a GPU or a robot.
  Runtime claims (C4, C5 cost, A8 teardown) come from HANDOFF field notes or
  reading.
- crisp_py and crisp_controllers were read only where a finding depends on
  them; whether the controller clips large targets internally (S2) is open.
- "Referenced nowhere" (D2) means inside this repository only.
- Duplication figures come from name matching and `difflib` ratios, not a clone
  detector; there may be more copies under different names.

## 5. Reproducing the measurements

Run from the repository root (ROS-free; `pip install -r tests/requirements.txt`).

```bash
python -m pytest -q -rs
```

```bash
ruff check . && ruff format --check .
```

```bash
python - <<'EOF'
import ast, pathlib
rows = []
for p in [*pathlib.Path("crisp_gym").rglob("*.py"), *pathlib.Path("scripts").rglob("*.py")]:
    for n in ast.walk(ast.parse(p.read_text())):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rows.append((n.end_lineno - n.lineno + 1, f"{p}:{n.lineno}", n.name))
for r in sorted(rows, reverse=True)[:15]:
    print(*r)
EOF
```
