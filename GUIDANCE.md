# Working guide for crisp_gym

For the owner and for any AI assistant asked to change this repository,
including assistants with a small context window. It says what to read, what
must not break, how to make a change safely, and which improvements to do in
which order. Each work package in §6 is written to be done on its own, without
re-deriving why.

Written 2026-11-16 at commit `12f6b29`, branch `claude/determined-faraday-1z6rx0`;
status columns last updated 2026-11-17 at `36d133e`. State then:
`python -m pytest` gives 756 passed, 13 skipped in 30 s (755 and 14 without
PyAV); `ruff check .` is clean with ruff 0.12.0. The rule is zero failures, not
a particular count.

**If the code does not match what this guide says, the guide is out of date.
Stop, say what differs, and ask. Do not force the steps onto different code.**

| document | holds | read it |
|---|---|---|
| `CLAUDE.md` | the owner's three process rules | always |
| `GUIDANCE.md` (this file) | how to work here; ordered work packages | always, first |
| `HANDOFF.md` | data conventions and why; investigation records | the sections §1.2 below names for your task |
| `ROADMAP.md` | the unfinished and untested FEATURES: what each is for, what is proven, how to finish it | when asked to finish, test or extend a feature; before picking a clean-up package |
| `RUN_SHEETS.md` | commands for the owner to run on the GPU PC and the robot PC, with what to expect and what to paste back; their results | when a step needs a machine you do not have |
| `CODE_REVIEW.md` | the problem list with evidence (IDs such as C3, A6) | when a work package cites an ID |
| `USAGE.md` | command lines for every workflow | when you run or document a workflow |
| `RESEARCH_FORCE_TACTILE.md` | force / tactile research plan, simulators | when planning model experiments |
| `REMOTE_INFERENCE.md`, `GUIDE_REMOTE_SERVER_AND_TACTILE.md` | server protocol; server and tactile how-to | when touching the policy server |

---

## 1. Start here

### 1.1 Rules

1. First request of a session: write a plan, change nothing, ask how to
   proceed (`CLAUDE.md`).
2. Change only what was approved. A work package marked **DECISION** needs the
   owner's answer before any edit.
3. One work package at a time. No "while I am here" edits in the same commit.
4. Test first: write or extend the test, see it fail, then change the code.
5. Before every commit, both must pass:
   `python -m pytest` and `ruff check . && ruff format --check .`
   with **ruff 0.12.0 exactly** (a newer ruff reformats 7 files differently).
6. Work on the designated branch only. No pull request unless asked.
7. Say what was not run. A cloud session has no ROS, torch, lerobot, GPU or
   robot: a change to the robot path is "tested on fakes, not run on the robot"
   until the owner runs it.
8. Never delete a config file, a dataset column or a script without the
   owner's yes. "Referenced nowhere in the repository" is not "unused".

### 1.2 What to read, by task

`CLAUDE.md` asks you to read `HANDOFF.md` before work on the pipeline. It is
long (81,000 characters); this table says where to read slowly.

| task touches | read |
|---|---|
| any pose, rot6d, relative or action math | HANDOFF §1.1–1.4 and §3; run `tests/test_pose_math.py` |
| what a dataset row contains | HANDOFF §1.2, §1.5, §4 item 3; docstring of `record/record_config.py`; `config/recording/record_config_example.yaml` |
| recording loop speed, dropped frames | HANDOFF §4b; USAGE §12–13 |
| training wrappers | HANDOFF §1.2, §1.3, §1.6; the lerobot-version table at the end of HANDOFF §2 |
| deployment, the policy class, the server | HANDOFF §2 row `relative_lerobot_policy.py`, §4 item 1; `REMOTE_INFERENCE.md` |
| controller configs, gains | HANDOFF §6 and the "Where the CIC gains come from" item of §7 |
| force, torque, tactile, the audit | HANDOFF §7; USAGE §14; `RESEARCH_FORCE_TACTILE.md` |
| GR00T or another VLA | HANDOFF §4c |
| tests, CI, lint | HANDOFF §5; the docstring of `tests/conftest.py` |
| merging or repairing datasets | USAGE §6; the tool's row in HANDOFF §2 |

### 1.3 Commands

```bash
pip install -r tests/requirements.txt "ruff==0.12.0"
```

```bash
python -m pytest
```

```bash
python -m pytest tests/test_pose_math.py -q
```

```bash
ruff check . && ruff format --check .
```

```bash
python scripts/code_metrics.py
```

### 1.4 Stop and ask when

- the change touches anything listed in §3;
- a test you did not write starts failing and your change does not obviously
  explain it (suspect the test isolation in `tests/conftest.py` first);
- the fix needs the robot, a GPU or the real lerobot to be verified;
- two documents disagree, or a document disagrees with the code;
- you are about to delete or rename a file, a config or a dataset key;
- the work package says DECISION.

---

## 2. The system in one page

Two machines. The **robot PC** (ROS 2 Humble, Python 3.11) records and
deploys. The **GPU PC** (no ROS; lerobot 0.6.1, 0.4.4 still supported) trains,
runs the dataset tools, and can serve the model over a websocket.

```
record ─► dataset (+ meta/record_config.json) ─► train ─► checkpoint (+ pose_repr.json)
                                                              │
robot PC: deploy_policy ─ RelativeLerobotPolicy ─ worker (local subprocess, or policy_server.py on the GPU PC)
controllers: crisp_controllers (1 kHz) ◄─ crisp_py (50 Hz re-publish) ◄─ env.step (15 Hz)
```

| package | role | may import |
|---|---|---|
| `crisp_gym/config` | config search, home poses, all YAML | stdlib |
| `crisp_gym/util` | rot6d math, safety limits, loop timing, tactile CSV loader | numpy, scipy, gymnasium (`lerobot_features`) |
| `crisp_gym/envs` | gym envs on top of crisp_py, env configs, handheld env | rclpy, crisp_py |
| `crisp_gym/teleop` | FACTR, streamed-pose and spacemouse leaders | rclpy, crisp_py |
| `crisp_gym/record` | what is recorded (`RecordConfig`) and the writer (`RecordingManager`) | rclpy, lerobot (manager only) |
| `crisp_gym/policy` | policy registry, relative policy, inference worker, remote transport | no ROS, torch or lerobot at import time |
| `crisp_gym/scripts` | entry points: robot, training, dataset tools, diagnostics | depends on the file; see CODE_REVIEW A1 |
| `crisp_gym/sim` | small MuJoCo models to try ideas without the robot (`spring_pad.py`) | numpy; mujoco only when a simulation runs |
| `scripts/` (root) | `policy_server.py`, shell launchers | lerobot on the GPU PC |
| `tests` | 40 test files, run without ROS; fakes for torch / rclpy / lerobot | — |

### Where is it decided?

| question | answer |
|---|---|
| What goes into a dataset row? | the record YAML in `config/recording/` → `record/record_config.py::RecordConfig`; built each tick by `record/record_functions.py::make_record_fn` |
| What is `action`? | `ActionConfig.definition`. Default `next_tcp_pose`, lookahead 1: the MEASURED pose one frame later, plus the measured gripper |
| Where is the commanded pose? | `extra.target_cartesian` (source `robot.target_pose`). It moves only under Cartesian control; under FACTR joint teleop it is constant |
| Who sets the loop rate? | `record/recording_manager.py::RecordingManager.record_episode`, for recording and for deployment. Not `env.step(block=...)`, not `control_frequency` |
| Where are safety limits applied? | `envs/manipulator_env.py::ManipulatorCartesianEnv.step`: step limit, then workspace box. The maths is in `util/safety.py` |
| Where do controller gains come from? | the env YAML's `cartesian_control_param_config`. When it is `null` (every `dric_*` Franka env) the gains are the bring-up's `controllers.yaml` in crisp_controllers_demos. Never the dataset |
| Which file does a config name load? | `config/path.py::find_config`: first match in `[crisp_py/config, crisp_gym/config]`. With `CRISP_CONFIG_PATH` set: its folders, then crisp_gym, then crisp_py |
| Which config does an env name use? | `envs/manipulator_env_config.py::make_env_config`: a Python class in `STRING_TO_CONFIG` wins; otherwise `config/envs/<name>.yaml` |
| Absolute → relative for training | `scripts/lerobot_relative_pose.py::RelativePoseDataset.convert_item` |
| The same at deployment | worker: `policy/relative_lerobot_policy.py::convert_window_state_to_relative`; robot side: `_to_env_action`, `compose_relative_pose` |
| How does a checkpoint say what it was trained on? | `pose_repr.json` (relative) or `action_repr.json` (absolute) beside it; written by `stamp_pose_repr`, read by `find_pose_repr` |
| Where does the model run? | decided only in `RelativeLerobotPolicy._start_worker` |
| Gripper units | device value 0 = closed, 1 = open. Datasets store width / `reference_width` (0.09 m). `gripper_device_to_ref`, `gripper_ref_to_device` |
| What does a lerobot diffusion policy read? | `observation.state`, the image keys, and `observation.environment_state` if declared. Nothing else (finding C11) |

---

## 3. What must never break

Each line names the test that fails when it does. If you must change one of
these, stop and ask.

| # | invariant | pinned by |
|---|---|---|
| 1 | rot6d is the first two ROWS of the rotation matrix; decoding uses Gram-Schmidt | `tests/test_pose_math.py` |
| 2 | Datasets hold ABSOLUTE poses. Relative poses exist only inside the training wrapper and at deployment | `tests/test_pose_math.py`, `tests/test_relative_deploy.py` |
| 3 | A relative action chunk is composed with the pose captured when its OBSERVATION was taken, never the pose at execution time | `tests/test_pose_math.py` |
| 4 | `observation.state` is `[pose 9, gripper 1, (extras), (wrt-start 6)]`, in the same order at record, train and deploy | `tests/test_relative_deploy.py`, `tests/test_franka_oakw_configs.py` |
| 5 | wrt-start is rotation only; noise at training, none at deployment | `tests/test_relative_deploy.py` |
| 6 | One gripper convention everywhere (0 closed, 1 open), one reference width (0.09) for handheld and robot | `tests/test_relative_deploy.py`, `tests/test_gripper_target_source.py` |
| 7 | The `action` column of a dataset is never rewritten in place. A derived label goes to a new column or to a dataset COPY | `tests/test_swap_action_offline.py` |
| 8 | Recorded extras use the `extra.` prefix, never `observation.` | `RecordConfig.validate` |
| 9 | Training and dataset files import neither crisp_py nor rclpy | by convention; see WP-13 |
| 10 | Controller gains are not stored in datasets | owner decision |
| 11 | `_start_worker` is the only place that knows local from remote. `RemotePolicy` stays a minimal client | `tests/test_policy_server.py` |
| 12 | The step limit compares target to PREVIOUS TARGET, never target to measured pose | `tests/test_safety_limits.py` |
| 13 | Normalisation statistics are recomputed on relative values, and that pass opens no video decoder in the main process | HANDOFF §1.6 |

---

## 4. How to make a change

### 4.1 Procedure

1. **Plan** (send it, then wait):
   - goal in one sentence;
   - files and functions to change;
   - the test that will prove it;
   - what cannot be verified here (robot, GPU);
   - questions.
2. **Test first.** Add the test. Run it. It must fail for the right reason.
3. **Change the code.** Smallest change that makes the test pass.
4. **Other copies.** Check §4.3: some code exists in several files on purpose.
5. **Run everything**: the two commands of rule 5.
6. **Documents.** Update what the change makes untrue: `USAGE.md` for a
   command line, `HANDOFF.md` for a convention or a file's role, the progress
   table in `CODE_REVIEW.md` for a finished finding, the status column in §6
   of this file.
7. **Commit** one logical change, message `type(scope): what and why`.
8. **Report**: what changed, what the tests showed, what was NOT run, what the
   owner should run on the robot or GPU PC and what output to expect.

### 4.2 Writing a new test

- Put it in `tests/test_<topic>.py`. Use plain `assert`. No docstrings or type
  annotations are required in tests.
- Code that needs ROS, crisp_py, torch or lerobot is tested with small fake
  modules placed in `sys.modules` at the top of the test file. Copy the block
  from a neighbour: `tests/test_lerobot_record.py` (stubs for crisp_py and
  rclpy), `tests/test_relative_deploy.py` (policy without torch).
- Two lighter techniques, to prefer when they are enough:
  `tests/test_franka_oakw_configs.py` checks config YAMLs as plain data, and
  `tests/test_safety_limits.py` parses the env source and executes only the
  methods under test, so no fake is needed.
- Best of all: put the logic in a pure function in a ROS-free module (as
  `util/safety.py` does) and test that function directly.
- `tests/conftest.py` gives every test FILE its own fakes and its own fresh
  copy of each `crisp_gym` module. Never rely on a module imported by another
  test file.
- A test that needs the real lerobot/torch or ROS gets `@pytest.mark.lerobot`
  or `@pytest.mark.ros`; it then skips where they are missing.
- To load a script by path, copy `_load_by_path` from any test that has it
  (for example `tests/test_tactile_raw.py`). It registers the module under its
  name before executing it; dataclasses and pickle look the module up by that
  name and fail without it. Keep the helper local to the test file: the files
  must still run as scripts.
- After writing a test for a fix, break the fix on purpose once and watch the
  test fail. A test that still passes proves nothing about that fix.

### 4.3 Code that exists in several copies on purpose

The GPU-side scripts must run without crisp_py, so shared code is copied and
tests keep the copies equal. Edit **all** copies in one commit.

| code | copies | kept equal by |
|---|---|---|
| rot6d / pose-9 math | `util/rot6d.py`, `scripts/lerobot_relative_pose.py`, `scripts/audit_command_residual.py` | `tests/test_pose_math.py`, `tests/test_audit_command_residual.py` |
| reading a checkpoint's action source | `policy/relative_lerobot_policy.py::action_source_from_stamps`, `scripts/lerobot_relative_pose.py::checkpoint_action_source` | `test_both_readers_of_the_stamps_agree` |
| `_setup_epochs`, `_load_epoch_steps`, `_installed_lerobot_version` | `lerobot_relative_pose.py`, `train_absolute_next_pose.py`, `train_action_from_target_cartesian.py` | `tests/test_launcher_epochs.py` |
| `_disable_video_query` | `lerobot_relative_pose.py`, `train_action_from_target_cartesian.py` | — (differs on purpose, HANDOFF §1.6) |
| target-column checks | `swap_action_offline.py`, `train_action_from_target_cartesian.py` | `tests/test_swap_action_offline.py`, `tests/test_action_swap.py` |
| config search order | `config/path.py`, `scripts/audit_command_residual.py::config_search_paths` | `tests/test_audit_command_residual.py` |

WP-13 removes most of this table.

---

## 5. Is the code easy to manage? An assessment

**Short answer: the design is sound and the risky maths is well tested, but
the knowledge is spread over too much text and a few very large units. A
person or a model can work safely on one area with this guide; nobody can
hold the whole repository in mind at once.**

What already helps:

- Layers import in one direction (scripts → record / policy → envs → util).
- Extension points are registries: policies, record sources, sensors, and the
  worker connection.
- Every dataset and checkpoint carries a small file saying what it contains.
- The invariants of §3 have tests, and the suite runs in 25 s without ROS.
- Docstrings explain why, not only what.

What makes it hard, in order of cost:

| # | problem | size of it | work package |
|---|---|---|---|
| 1 | Required reading is too long to hold. HANDOFF mixes rules that never change with dated investigation logs, and its file table has cells up to 2,900 characters | 8 documents, about 270,000 characters | WP-17 |
| 2 | Dataset keys are string literals | `"observation.state"` in 20 files, `"action"` in 24 | WP-12 |
| 3 | Seven files over 800 lines, 14 functions over 150 lines. The three entry-point `main` functions parse, prompt, build and loop in one body, so tests must fake whole modules | see `scripts/code_metrics.py` | WP-14, WP-15, WP-16 |
| 4 | Copied code (§4.3) | 6 sets | WP-13 |
| 5 | Two ways to define an env config, and a config search order that an environment variable reverses | 10 classes + 12 YAMLs; 4 files in both repos | WP-03, C2 |
| 6 | Tests install fakes at import time; a shared conftest hides the side effects | 40 files | §4.2, later K3 |
| 7 | Comments that tell history ("this used to return 1 − value") beside the code | throughout | move to commit messages when touching a file |

Rules that keep it from getting worse (apply from now on):

- A new function stays under 80 lines; a new file under 500.
- Do not add to a file that is already over 800 lines unless the work package
  is to split it. Put new code in a new module and import it.
- A new dataset key, or a new use of an existing one, goes through the schema
  module once WP-12 exists.
- Every new file starts with a docstring whose first lines say: what it is,
  which machine it runs on, what it may import, and which test covers it.
- New knowledge goes to the smallest document that fits: a command line to
  USAGE, a rule to HANDOFF §1 or §5, a dated measurement to HANDOFF §7 or a
  note file. Do not add rows to the HANDOFF §2 table with paragraphs in them.

---

## 6. Work packages

Size: XS under 1 h, S under half a day, M 1–3 days, L more.
"Robot" says whether a run on the robot is needed before it counts as done.

**These are clean-up packages. Features come first: `ROADMAP.md` lists what is
unfinished or untested and how to finish it. Do a package from this section
when it stands in a feature's way, or when the owner asks for it.**

### 6.1 Order

| step | packages | why this order |
|---|---|---|
| 1. No decision needed | WP-01, WP-04, WP-06, WP-08, WP-09 — **all done 2026-11-17** | small, safe, each proven by a test here |
| 2. After one owner answer each | WP-02, WP-03, WP-05, WP-07 (WP-10 continues as ROADMAP F2) | silent misconfiguration and silently unused inputs |
| 3. Needs the robot | WP-11, and the robot checks in §7 | measure before changing the loop |
| 4. Structure, in this order | WP-12 → WP-13 (both unblocked by D-7) → WP-14 → WP-15 → WP-16 | each makes the next smaller and safer |
| 5. Documents | WP-17, WP-18 | after the structure settles, or the map goes stale twice |

Not recommended now: moving the script folders (CODE_REVIEW A1: 46 document
references to update for no gain in safety), own versioning for the fork (K5),
a shared ROS executor (A8) unless shutdown problems are seen.

| ID | finding | title | size | robot | decision | status |
|---|---|---|---|---|---|---|
| WP-01 | C8 | `make_policy` error message | XS | no | — | done `d284f37` |
| WP-02 | C7 | sorted, de-duplicated config lists and prompt defaults | S | no | D-3 | open |
| WP-03 | C1 | one definition per env name | S | no | D-1 | open |
| WP-04 | C9 | replace the deprecated test loader | S | no | — | done `47aa704` |
| WP-05 | C3 | check the state vector order | S | no | D-2 | open |
| WP-06 | C4 | explain and pin the skipped frames | XS | no | — | done `6e78c53` |
| WP-07 | C6 | remove the hard-coded `franka` branch | S | yes | D-4 | open |
| WP-08 | A10 | commented-out prints, `print` in library code | XS | no | — | done `ce16193` |
| WP-09 | K2 | `--help` smoke test for ROS-free scripts | S | no | — | done `b468ce0` |
| WP-10 | C11 | make promoted columns reach the model | M | no | D-5 | dataset tool done `013e8f6`; robot side open, see ROADMAP F2 |
| WP-11 | C5 | `step` builds an observation nobody reads | M | yes | D-6 | open |
| WP-12 | A3 | one schema module | M | no | D-7 | open |
| WP-13 | A2 | a ROS-free `offline` subpackage | M–L | no | D-7 | open |
| WP-14 | A4 | split the three entry-point `main`s | L | yes | — | open |
| WP-15 | A6 | split `relative_lerobot_policy.py` | L | yes | — | open |
| WP-16 | A7, T1 | split `RecordingManager`; split the audit | L | yes | — | open |
| WP-17 | D1 | conventions separate from logs | M | no | D-8 | open |
| WP-18 | — | generated module map and size check | S | no | — | open |

### 6.2 Cards

#### WP-01 · `make_policy` error message (C8) — DONE

- **Goal.** The error for an unknown policy reads as one sentence, not a tuple.
- **Where.** `crisp_gym/policy/policy.py::make_policy`, the `raise ValueError(`
  that is given two f-strings separated by a comma.
- **Test first.** New `tests/test_policy_registry.py`: call
  `make_policy("no_such_policy")`, catch `ValueError`, assert
  `len(err.args) == 1` and `"not registered" in str(err)`.
- **Change.** Join the two strings into one (keep both pieces of information).
- **Done when.** The new test passes; nothing else changes.

#### WP-02 · sorted, de-duplicated config lists (C7) — DECISION D-3

- **Goal.** Pressing Enter at a prompt never picks a config by filesystem
  accident.
- **Where.** `crisp_gym/config/path.py::list_configs_in_folder`. Its callers:
  `policy/policy.py::list_policy_configs`,
  `envs/manipulator_env_config.py::list_env_configs`, and the prompts in
  `scripts/deploy_policy.py` and
  `scripts/record_lerobot_format_leader_follower.py`.
- **Test first.** New `tests/test_config_listing.py`. Point
  `crisp_gym.config.path.CRISP_CONFIG_PATHS` (monkeypatch) at two temporary
  folders that both contain `policy/b.yaml` and one of which contains
  `policy/a.yaml`. Expect: names sorted; `b` listed once; the `b` returned is
  the one from the FIRST folder (the precedence `find_config` uses).
- **Change.** Sort by file name; keep the first path's file for a repeated
  name. Policy prompt default: `relative_lerobot_policy` when present. Env
  prompt default: the owner's answer to D-3.
- **Do not** change `find_config` or the search order here (that is C2).

#### WP-03 · one definition per env name (C1) — DECISION D-1

- **Problem.** `make_env_config` uses a Python class when the name is in
  `STRING_TO_CONFIG` and never reads a YAML of the same name. Today
  `no_cam_franka` and `only_wrist_cam_franka` exist as both; editing those two
  YAML files has no effect.
- **After D-1** (recommended: delete the two YAML files, which are never
  read): add a pure helper to the ROS-free `crisp_gym/config/path.py`,
  `check_single_definition(name, class_names, folder="envs")`, that raises
  `ValueError` naming both when `name` is in `class_names` AND
  `find_config(f"{folder}/{name}.yaml")` finds a file. Call it at the top of
  `make_env_config` when no `config_path` was passed.
- **Test first.** New `tests/test_config_listing.py` (shared with WP-02); it
  imports only `crisp_gym.config.path`, so it needs no fake module. Point
  `CRISP_CONFIG_PATHS` at a temporary folder containing `envs/a.yaml`; expect
  the error for `("a", {"a"})` and no error for `("a", {"b"})` and
  `("c", {"c"})`.
- **Do not** remove the classes in this package (that is A9).

#### WP-04 · replace the deprecated test loader (C9) — DONE

- **Problem.** 15 calls in 14 test files use
  `SourceFileLoader(...).load_module()`, removed in Python 3.15.
- **Files.** `test_action_swap`, `test_align_shape_and_robot_type`,
  `test_audit_command_residual`, `test_check_policy_openloop` (2 calls),
  `test_gripper_target_source`, `test_loop_timing`,
  `test_record_config_robot_type`, `test_reencode_dataset_videos`,
  `test_reencode_resize_e2e`, `test_sensor_source`, `test_swap_action_offline`,
  `test_tactile_raw`, `test_target_joint_source`, `test_writer_throughput`.
- **Change.** In each file replace the call with the `_load` helper of §4.2,
  **keeping the same module name** (the first argument). Keep the helper local
  to the file: the files must still run as scripts.
- **One file per commit is fine.** After each file: run that file with pytest,
  and with `python tests/<file>` if it ends in an `if __name__` block.
- **As done.** Two of the calls sit inside driver scripts that tests run in a
  subprocess (`test_check_policy_openloop`, `test_reencode_resize_e2e`); there
  the four lines are written inline. The re-encode one runs only where PyAV
  is installed, so it was checked with PyAV 15.
- **Done when.** `grep -c "load_module()" tests/*.py` prints only zeros and the
  suite still reports 709 passed.

#### WP-05 · check the state vector order (C3) — DECISION D-2

- **Problem.** `util/lerobot_features.py::concatenate_state_features` joins
  the `observation.state.*` entries in the order the dictionary happens to
  have. A reorder upstream would permute what the model sees, and the only
  existing check (total length) would not notice.
- **What makes a cheap check possible.** Every sub-key has its own `names` in
  the feature dictionary, and `RecordConfig.to_features` builds
  `features["observation.state"]["names"]` by joining them in declared order.
- **Change.** In `concatenate_state_features`, when `features` is given and
  every sub-key present has `names`: join the sub-keys' names in the order
  they are being concatenated and compare with
  `features["observation.state"]["names"]`. Raise `ValueError` showing both
  lists when they differ. When a sub-key has no `names`, keep today's
  behaviour and log once at DEBUG.
- **Test first.** Extend `tests/test_lerobot_record.py`: same keys in a
  different order must raise; the declared order must pass; total length
  equal in both cases.
- **D-2.** Raise (recommended: a permuted state is never right) or warn.
- **Do not** sort the keys to "fix" the order: the dataset on disk defines it.

#### WP-06 · explain and pin the skipped frames (C4) — DONE

- **What reading shows, reproduced on a fake env (2026-11-16: 2 / 1 / 1).**
  `make_record_fn` is built once per episode. Its
  `_fn` returns `(None, None)` in exactly two places: the drive warm-up tick
  (a relative drive function has no previous pose to subtract yet), and while
  the lookahead buffer fills (`lookahead` ticks, 1 by default). So: 2 skipped
  frames per episode with a relative drive function, 1 with an absolute one
  (FACTR with `use_relative_actions: false`) or with none (handheld).
- **Test first.** Extend `tests/test_loop_timing.py` (it already builds
  `make_record_fn` on fakes): count the `(None, None)` results over the first
  five calls for (a) a drive function whose first call returns `None`,
  (b) one that always returns a vector, (c) no drive function. Expect 2, 1, 1.
- **Change.** None to behaviour. Give each of the two returns a one-line
  comment naming its reason. Replace HANDOFF §4b item 4 "not yet investigated"
  with the explanation.
- **As done.** `tests/test_loop_timing.py`: 2 / 1 / 1, and 4 / 3 with a
  lookahead of 3.
- **Still for the owner.** Confirm on the robot that the count matches for the
  rig in use.

#### WP-07 · remove the hard-coded `franka` branch (C6) — DECISION D-4

- **Where.** `scripts/record_lerobot_format_leader_follower.py`, the block
  starting `if args.follower_config == "franka":`. It builds
  `FrankaEnvConfig(control_frequency=50, gripper_config=None, camera_configs=[])`
  by hand, so cameras and gripper are dropped and no YAML can change that.
- **After D-4.** Delete the block so every name goes through `make_env`.
- **Risk.** Whoever still types `--follower-config franka` gets a different
  env (with gripper and cameras). The default is `dric_franka_oakw_umi` since
  2026-11, so this should affect nobody; the owner confirms.
- **Robot.** One recording start with the usual command line.

#### WP-08 · leftover prints (A10, part) — DONE

- **Where.** `envs/manipulator_env.py::_set_gripper_action`: eight
  commented-out `# print(...)` lines, and one in
  `scripts/leader_follower_teleop.py`; delete them.
  `record/recording_manager.py::ROSRecordingManager._callback_recording_trigger`
  and `teleop/spacemouse_node.py`: `print(...)` → `logger.warning` / `logger.info`.
- **Keep** `print(Panel(...))` in `RecordingManager.__enter__`: it is the
  operator's on-screen instruction, written with rich on purpose.
- **Done when.** `grep -n "# print(" crisp_gym -r` prints nothing.

#### WP-09 · `--help` smoke test for the ROS-free scripts (K2, part) — DONE

- **Goal.** An import error in a GPU-side script fails CI instead of failing on
  the GPU PC.
- **As done.** `tests/test_cli_help.py` starts 14 scripts with `--help` in a
  subprocess: `audit_command_residual`, `inspect_dataset`,
  `check_tactile_sync`, `check_schema_drift`, `fix_dataset_task`,
  `swap_action_offline`, `migrate_euler_delta_to_rot6d`,
  `postprocess_align_datasets`, `viz_encoder_attention`, `groot_preflight`,
  `deploy_preflight`, `scripts/code_metrics.py`, `scripts/policy_server.py`,
  `crisp_gym/sim/spring_pad.py`. Two more carry the `lerobot` marker
  (`check_policy_openloop`, `reencode_dataset_videos`); they have not run
  anywhere yet, so expect to look at them on the GPU PC.
- **Left out on purpose.** The ROS entry points; and the three training
  launchers, `check_relative_pose` and `check_available_policies`, which have
  no argument parser of their own (`--help` there is lerobot's).
- **A second test** fails when a file is added to `crisp_gym/scripts` without
  being listed or excluded with a reason.

#### WP-10 · make promoted columns reach the model (C11) — tool DONE, rest in ROADMAP F2

- **Done 2026-11-17.** `crisp_gym/scripts/add_environment_state.py` (route (a)
  below), 15 tests, USAGE §7 rewritten. Not run on a real dataset or with a
  real training. The robot side and `multi_task_dit` are ROADMAP F2 parts B
  and C. The text below is the original card.

- **Problem (read in lerobot v0.4.4 and main; not run).**
  `postprocess_align_datasets.py --promote extra.foo` renames the column to
  `observation.state.foo`. A lerobot diffusion policy reads `observation.state`,
  the images and `observation.environment_state` only. The renamed column is
  loaded and normalised but never enters the network. USAGE §7 says otherwise.
- **This blocks** every "force / torque / tactile as a model input" experiment
  on recorded data.
- **D-5, two ways to fix.**
  - (a) *Environment state.* Write the chosen `extra.*` columns, joined, to
    one new column `observation.environment_state`. `observation.state` and
    all relative-pose code stay untouched. Recommended.
  - (b) *Append to the state.* Rebuild `observation.state` as
    `[pose 9, gripper 1, extras]`, with names, shape and statistics. The
    training wrapper already passes dimensions 9 and up through, and
    `build_obs_frame` already concatenates every `observation.state.*` the
    deploy env produces. Changes the state width, so §3 item 4 applies.
- **Steps for (a).**
  1. On the GPU PC confirm the policy in use reads the key:
     `grep -n "OBS_ENV_STATE" <lerobot>/policies/<policy>/modeling_*.py`.
     Diffusion and ACT do. For `multi_task_dit` this is unknown: check first.
  2. New ROS-free script modelled on `scripts/swap_action_offline.py`: copy
     the dataset, add the column, add the feature to `meta/info.json`, compute
     its statistics with the helpers the swap script already reuses, write
     `meta/environment_state.json` listing the source columns in order.
  3. Test on a synthetic dataset as `tests/test_swap_action_offline.py` does.
  4. Deployment is a second package: `build_obs_frame` must emit the same
     vector from the env's sensors and the worker must pass it on.
- **Do not** change `--promote` silently. Correct USAGE §7 in the same commit.

#### WP-11 · `step` builds an observation nobody reads (C5) — DECISION D-6

- **Problem.** `ManipulatorCartesianEnv.step` and `ManipulatorJointEnv.step`
  end with `obs = self._get_obs()`. `make_record_fn` throws that away and reads
  the sources itself; the relative policy calls `env.get_obs()` separately.
  `block` and `control_frequency` are dead (both `FIXME`).
- **First measure.** On the robot, record 30 s with `timing_report` on (USAGE
  §12) and read the `step` phase. If it is under 2 ms, close this package as
  "not worth changing".
- **If worth it.** Add a keyword `return_obs: bool = True` to both `step`
  methods; with `False` return `{}`. Pass `False` only from `make_record_fn`
  and from the relative policy's loop. Default behaviour unchanged, so the Gym
  wrappers in `envs/env_wrapper.py` and the examples keep working.
- **D-6.** Remove `block` and `control_frequency`, or keep them and say in
  their docstrings that they do not set the loop rate.

#### WP-12 · one schema module (A3) — DECISION D-7

- **Goal.** Dataset key names, sizes and gripper constants are written once.
- **Change.** New `crisp_gym/schema.py`, standard library only: constants for
  the keys (`observation.state`, `.cartesian`, `.gripper`, `action`,
  `extra.target_cartesian`, `extra.gripper_target`, `extra.ext_torque`, the
  tactile keys and their four sync suffixes), `POSE_DIM = 9`,
  `ACTION_DIM = 10`, `DEFAULT_REFERENCE_WIDTH = 0.09`.
- **Migrate one file per commit**, starting with `record/record_config.py`
  and `policy/relative_lerobot_policy.py`. Replace literals with the
  constants; change no behaviour. The existing tests are the proof.
- **D-7** decides whether the GPU-side scripts may import `crisp_gym.schema`
  (see WP-13). Until then leave those scripts alone.

#### WP-13 · a ROS-free `offline` subpackage (A2) — DECISION D-7

- **D-7.** The training and dataset scripts were kept copyable as single
  files. The GPU PC already needs a clone of this repository for
  `scripts/policy_server.py`. Recommended: give up single-file copying, keep
  the rule "no crisp_py, no rclpy".
- **Change.** New `crisp_gym/offline/` holding: pose math (from
  `util/rot6d.py`), dataset reading (parquet files, `meta/info.json`,
  statistics for layouts v2.x and v3.0), the readers and writers of the
  contract files, the launcher helpers of §4.3. Then make each script import
  from it, one script per commit, deleting its copy.
- **Guard test.** Import every module of `crisp_gym/offline` in a subprocess
  in which `rclpy` and `crisp_py` are made unimportable; assert it works.
- **Order.** Pose math first: `tests/test_pose_math.py` already proves the
  copies are equal, so replacing them is mechanical.

#### WP-14 · split the three entry-point `main`s (A4)

- **Files.** `scripts/deploy_policy.py::main` (417 lines),
  `scripts/record_lerobot_format_leader_follower.py::main` (507),
  `scripts/record_umi_handheld.py::main` (234).
- **Shape.** `parse_args(argv) → fill_missing_by_prompt(args) → build(args)
  → run(objects)`. `build` returns plain objects; `run` holds the loop.
- **Rule.** Move code, do not rewrite it. One function extracted per commit.
  Command lines and prompts stay exactly as they are.
- **Robot.** One recording and one deployment before it counts as done.

#### WP-15 · split `relative_lerobot_policy.py` (A6)

1,718 lines. Target modules, keeping `_start_worker` as the only local /
remote switch: (1) observation building and its checks, (2) the chunk
scheduler (sync, async, prefetch, offset on swap) as plain Python with no
torch, (3) action composition and gripper conversion, (4) `inference_worker`
in its own module, imported by `scripts/policy_server.py`. Keep the old
import paths working by re-exporting from the old file. The scheduler gets new
unit tests first; it is the part with the most states and the fewest tests.

#### WP-16 · split `RecordingManager` and the audit (A7, T1)

- `record/recording_manager.py`: the writer subprocess and everything it
  needs go to a module that does not import rclpy. HANDOFF §4b holds the
  timing behaviour that must stay the same; `tests/test_writer_throughput.py`
  and `tests/test_loop_timing.py` are the proof.
- `scripts/audit_command_residual.py` (2,034 lines): after WP-13, split into
  kinematics, residual and fits, tactile, report text, command line. The
  1,023-line test file is the proof; the script must still run as one file
  path for the owner's existing command lines.

#### WP-17 · conventions separate from logs (D1) — DECISION D-8

- **Target.** `CONVENTIONS.md` under 250 lines: HANDOFF §1, §3 and §5, nothing
  dated. `docs/notes/<year-month>-<topic>.md` for the investigation records
  now in HANDOFF §4b, §4c, §6 and §7, one topic per file. HANDOFF keeps a
  short index. USAGE split per workflow only if the owner wants it.
- **Rule.** Move text, do not rewrite it, and leave a one-line pointer where
  each section was. `CLAUDE.md` then names `CONVENTIONS.md` (owner's edit).

#### WP-18 · generated module map and size check

- Extend `scripts/code_metrics.py` with `--map`: for every package file print
  path, lines, and the first line of its docstring, as a Markdown table.
  Commit the output as `docs/MODULE_MAP.md`; a test regenerates it and fails
  when it differs. This replaces the hand-written table in HANDOFF §2.
- Add `--check`: fail when a function not on a recorded baseline exceeds 150
  lines. Same idea as the lint baseline: shorten, never extend.

---

## 7. Open owner decisions and robot checks

| ID | question | recommendation |
|---|---|---|
| D-1 | `no_cam_franka` and `only_wrist_cam_franka` exist as class and YAML; the YAML is never read. Delete the two YAMLs, or the classes? | delete the YAMLs |
| D-2 | Wrong state order: raise or warn? | raise |
| D-3 | Default env at the deploy and record prompts? | the rig's env, e.g. `dric_franka_oakw_deploy_umi` / `dric_franka_oakw_umi` |
| D-4 | Remove the hard-coded `--follower-config franka` branch? | yes |
| D-5 | Extra model inputs through `observation.environment_state` (a) or appended to `observation.state` (b)? | **ANSWERED 2026-11-17: (a)**, then superseded as the main route by D-10 the same day. The tool for (a) stays, for diffusion / ACT experiments |
| D-6 | Remove `block` and `control_frequency`, or document them as dead? | document now, remove with WP-14 |
| D-7 | May GPU-side scripts import ROS-free `crisp_gym` modules (ending single-file copying)? | **ANSWERED 2026-11-17: yes.** The rule "no crisp_py, no rclpy, no ROS" stays |
| D-8 | Split HANDOFF into conventions and dated notes? | yes |
| D-9 | Add one line to `CLAUDE.md` naming `GUIDANCE.md` as the first file to read, and an `AGENTS.md` for tools that read that name? | **ANSWERED 2026-11-17: yes to both, done** |
| D-10 | `multi_task_dit` and GR00T read `observation.state` only. Put extra inputs INTO `observation.state`, after the gripper (changes the state layout)? | **ANSWERED 2026-11-17: yes.** One route for every policy; the robot side already adds env sensors to the state. Missing: `--promote` must rebuild `observation.state`. ROADMAP F2 |
| D-11 | Which policies for the measured-vs-command comparison? | **ANSWERED 2026-11-17: `multi_task_dit` and diffusion first; GR00T later** (the owner trains GR00T in SE3 mode, which uses the same wrapper) |
| D-12 | When to make `--promote` rebuild `observation.state`? | **ANSWERED 2026-11-17: a later session.** It is step 1 of ROADMAP F2 |

Robot-side checks still owed (from earlier sessions):

| check | how | why |
|---|---|---|
| Which copy of the 4 duplicated configs is in effect | `echo "CRISP_CONFIG_PATH=$CRISP_CONFIG_PATH"` in the pixi shell on the robot PC | decides which copy moves to crisp_py (C2) |
| The test suite in the ROS + lerobot environment | `python -m pytest -rs` | the isolation in `tests/conftest.py` has only run without ROS |
| The step limit on the robot | deploy once, look for "Target step limit acted" in the log | S2 was tested on fakes only |
| Delivered stiffness | hold a fixed target, push with a scale to 10 N and 20 N, note the displacement | the audit measures 150–280 N/m against 400 configured (HANDOFF §7) |
| Skipped frames per episode | WP-06 | confirms the explanation |

---

## 8. Mistakes already made once

| mistake | what to do instead |
|---|---|
| A force threshold of 3 N flagged 99 % of Franka frames as contact | joint torques rest at a bias (joint 1 about −1.3 Nm); compare with each episode's resting level |
| Force / gap printed as "stiffness" was dominated by a 7 N floor | fit the slope between gap bands, not the ratio |
| A hypothesis (gripper weight sag) stated as a finding | say "candidate"; name the measurement that would settle it |
| Tests passed alone and failed together | fakes in `sys.modules` leak between files; see `tests/conftest.py` |
| A newer ruff reformatted unrelated files | pin ruff 0.12.0 |
| A text replacement missed because the file had been reformatted | re-read the lines before editing; never edit from memory |
| A workflow file with everything commented out showed as a failed run | delete a workflow that should not run |
| "Promote" was documented as feeding the model; it does not | read what the consumer reads (C11) before documenting a data path |
| A synthetic test cancelled the effect it was meant to show | print the intermediate numbers of a new test once and look at them |
| `0.0 or inf` turned a bound of zero into "no bound" | compare with `is None`, never truthiness, for numbers that may be 0 |
| A simulator failure was explained by a guess (thin collision plate); the regression test passed with the "fix" removed | break the fix on purpose and watch the test fail before believing the explanation; then measure the mechanism (here: contact directions) |
| A skip condition was guessed ("needs av"); the script needed lerobot as well | install the optional package once and run the test, instead of assuming what it needs |
| A text tool replaced a call that lived inside a string holding another script | after a scripted edit, check with the parser that every replacement is real code |

---

## 9. Keeping this guide true

- When a work package is finished, set its status in §6.1 and add a row to
  the progress table in `CODE_REVIEW.md`.
- When a decision of §7 is answered, write the answer in the table.
- When §2 "Where is it decided?" becomes wrong, fix the row in the same
  commit as the code.
- Keep this file under 700 lines. Details belong in the document §1 names.
