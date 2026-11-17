# Force and tactile in the policy: research options, and simulators to try them in

Written 2026-11-16. For the owner's direction in HANDOFF §7: contact-aware
policies on a Franka FR3 + Robotiq 2F-85 under a Cartesian impedance
controller, with a handheld device and 72-taxel skins on both.

How each statement here was checked:

| mark | meaning |
|---|---|
| RAN | executed in a cloud session (MuJoCo 3.2.7 and 4.0.1 from pip, CPU, no display) |
| READ | source code or README read; not executed |
| PAPER | abstract or summary only |

Nothing here was run on the robot, with lerobot, or on a GPU.

---

## 1. The questions

| # | question | where it stands |
|---|---|---|
| Q1 | **Label.** In contact, is the COMMAND a better action label than the measured pose? | Planned on the Franka (models M and C, USAGE §14). The audit says the two labels differ by a median of 48 mm in contact |
| Q2 | **Relabel.** Can a model predict the command (or the force) from tactile + motion, so handheld data gets a command label? | The planned residual model. Needs Q1 to be "yes" first |
| Q3 | **Input.** Does the policy do better when it also SEES torque, force or tactile? | Not started. Blocked by finding C11 (GUIDANCE WP-10): promoted columns do not reach a lerobot diffusion policy today |
| Q4 | **Speed.** Is a 15 Hz chunked policy fast enough in contact, or is a fast inner loop needed? | Not started. Today only the 1 kHz impedance controller reacts between ticks, and only mechanically |
| Q5 | **Compliance as action.** Should the policy also output stiffness or a target force? | Not started. Needs a controller that accepts gains at run time |

Q1 and Q2 are about labels and need no model change. Q3 needs a new model
input. Q4 and Q5 change the control stack and are the most work.

---

## 2. What others have done, by idea

| idea | work | what it does | code | meaning for this rig |
|---|---|---|---|---|
| Force-informed target labels | **ACP** (Adaptive Compliance Policy) | From the measured wrench, computes offline a "virtual target" beyond the surface and a stiffness (200–5000 N/m by force level); the policy outputs both | local repo, `PyriteEnvSuites/scripts/postprocess_add_virtual_target_label.py` (READ) | Q2 is this, with skin instead of a wrench sensor |
| | **DexForce** | Kinesthetic demos with fingertip force sensors; target = observed position + stiffness⁻¹ × force; summaries report near-zero success without the force term (PAPER) | — | same relabel formula; shows position-only labels fail in contact |
| | **UMI-FT** | Handheld UMI gripper with a 6-axis sensor in each finger; ACP extended to grasp force; three loops: policy, wrist admittance, gripper force (PAPER) | project page says open | the closest published system to the planned handheld + skin |
| Force or torque as INPUT | **FACTR** | A policy given force tends to ignore it and overfit to vision; a curriculum that blurs images early makes it attend to force (PAPER + local repo) | local repo `FACTR/` (READ: real robot only) | expect "adding torque changed nothing" unless training forces attention; reuse its curriculum |
| | **TA-VLA** | Where to feed joint torque into a VLA: the action decoder beats the vision encoder; a torque history as one token; predicting future torque as an extra output helps (PAPER) | project page | joint torque only, no wrist sensor: the Franka's situation |
| Slow policy + fast reactive layer | **RDP** (Reactive Diffusion Policy) | Slow latent diffusion for chunks, fast tactile/force closed loop at 20–30 Hz inside the chunk | released; real robot only; its README lists FR3 and ROS 2 Humble (READ) | the design to study for Q4 |
| Residual on a base policy | **CR-DAgger** | A compliant residual policy with force input, trained on human corrections; outputs residual motion and target force (PAPER) | — | a different residual from Q2: it corrects a policy online, Q2 relabels data offline |
| Compliance as output | **Comp-ACT** | ACT that predicts the trajectory and a time-varying stiffness | released, with a robosuite task (READ) | Q5 in a simulator, runnable without a robot |
| Force from joint torque | **FILIC** | MuJoCo peg-in-hole; end-effector force estimated from joint torques via the Jacobian, as the audit's step C does (PAPER) | the repository link in the paper returned 404 when checked | confirms the approach; nothing to run yet |
| Tactile representation | **M3L** | Masked visuo-tactile pre-training with RL, in MuJoCo with a 2F-85 and taxel grids | `tactile_envs` + `M3L` (READ) | the ready-made simulator of §3 |
| | **ManiFeel**, **ManiSkill-ViTac** | Benchmarks for visuo-tactile policies (insertion, assembly, lock opening) | released; heavier simulators | later, if a camera-based tactile sensor is considered |

Two warnings that come out of this reading:

- **A labelling model is not a controller.** `command = measured + force / K`
  applied online keeps whatever force is already there; it regulates nothing
  (HANDOFF §7). Online, a fast layer needs a force TARGET from the policy
  (RDP, UMI-FT, CR-DAgger all do this).
- **Giving the model force is not enough.** FACTR and TA-VLA both report that
  the way it is fed in, and how training is scheduled, decide whether it is
  used at all.

---

## 3. Simulators and runnable projects

Lightest first.

| | project | simulator | what you get | fit | checked |
|---|---|---|---|---|---|
| A | **Plain MuJoCo + `touch_grid`** (`pip install mujoco`) | MuJoCo | Any taxel grid on any body; force sensors; contact forces. The plugin ships in the pip wheel | exact taxel count (6 × 12 = 72), any stiffness; you write the scene | RAN, §4 |
| B | **tactile_envs** (`carlosferrazza/tactile_envs`, MIT, last commit 2024-07) | MuJoCo 3.1.2 pinned | `tactile_envs/Insertion-v0`: a floating **Robotiq 2F-85** with a taxel grid on each pad, a camera, six peg shapes; the base is driven by position actuators (`kp=100`), i.e. a spring, so command ≠ measured in contact. Also door and Shadow-hand tasks | closest ready-made match: same gripper, same kind of skin, same spring physics. No arm | READ. Not executed here: running downloaded code was blocked in this session |
| C | **robosuite** (and the Comp-ACT fork) | MuJoCo | Panda / UR5e arms, a force-torque sensor in every gripper wrist (not a default observation; add it as an observable), an operational-space controller with a variable-impedance mode where the action carries stiffness, tasks such as Door, Wipe, NutAssembly | best for Q5 and for wrist-force input; no taxels | docs and README read |
| D | **Your own stack in MuJoCo** (`crisp_controllers_demos/crisp_mujoco_sim`, `FRANKA_FAKE_HARDWARE=true`) | MuJoCo behind `ros2_control` | The REAL crisp controllers, crisp_py and crisp_gym, on a simulated FR3. Today: arm only, zero gravity, a floor, no gripper, no objects, no camera | the only option that tests the real controller and the real deploy code; needs ROS and scene work | READ (local) |
| E | **MuJoCo Menagerie** | MuJoCo | Models: `franka_fr3`, `franka_fr3_v2`, `robotiq_2f85`, `robotiq_2f85_v4`, `umi_gripper`, UR arms | parts to build your rig in A or D | README read |
| F | **LeRobot `gym-hil`** | MuJoCo | Franka Panda pick-cube, made for LeRobot | LeRobot-native loop; no force or tactile out of the box | docs read |
| G | **IsaacLab**: `contrib/forge` tasks and the TacSL visuo-tactile sensor | Isaac Sim (GPU) | Force-aware insertion tasks; camera-based tactile simulation | already in your `IsaacLab` checkout; heavy | READ (local: both folders exist) |

`mujoco.readthedocs.io` and `robosuite.ai` could not be opened from this
session (hosts outside the allowed network), so C and the plugin's parameter
list rest on search results and on what ran in §4.

---

## 4. What was run here

`crisp_gym/sim/spring_pad.py`: a 72-taxel pad on a 400 N/m position servo,
pressed onto a fixed peg. It is the smallest model of the real situation: a
spring-driven tool, a command that runs past the surface, a skin that feels
the result. Depth is counted from the peg's surface, positive = commanded
into it.

```bash
pip install mujoco numpy
```

```bash
python crisp_gym/sim/spring_pad.py --show-grid
```

```
pad 6 x 12 = 72 taxels, 3 values each | spring 400 N/m
command(mm)  measured(mm)  gap(mm)  spring force(N)  taxel force(N)  lit taxels
     -10.00        -10.00     0.00             0.00            0.00           0
       2.00          0.00     2.00             0.80            0.80           9
      12.00          0.01    11.99             4.80            4.80           9
      22.00          0.02    21.98             8.79            8.79           9
      42.00          0.04    41.96            16.78           16.78           9
```

What it shows:

- In steady contact, **stiffness × (command − measured) equals the summed
  taxel force**, to two decimals. This is the relation the residual model
  (Q2) has to learn on real data.
- The measured depth stays within 0.04 mm of the surface whether the command
  is 2 mm or 42 mm past it. A measured-pose label would replay "stay at the surface",
  which commands no force. That is Q1 in one table.
- The grid gives 3 values per taxel (normal and two shear) and a spatial
  pattern: the lit taxels lie under the off-centre peg.

Options: `--stiffness`, `--taxels NX NY`, `--depths-mm`, `--peg-xy-mm`. The
functions `press`, `model_xml` and `read_taxels` can be imported to build on.
Eleven tests (`tests/test_sim_spring_pad.py`) check it against statics and pass
on MuJoCo 3.2.7 and 4.0.1; the two versions differ in how far the contact gives
way (0.16 mm against 0.04 mm at 17 N), not in the result.

Four things learned while building it, each pinned by a test or stated in the
script:

- The sensor's site looks along its **−z** axis, like a camera. With the site
  pointing the other way every taxel reads zero and nothing warns you.
- A rigid box pad gives ONE contact point, so one lit taxel. To spread contact
  over taxels, build the pad from one small box per taxel. `tactile_envs` does
  the same (`generate_pad_collisions.py --nx 6 --ny 12` gives 72).
- The sensor's `fov` is a half angle and the taxels are bins of ANGLE as seen
  from the site. Put the site far from the pad (the script uses 1 m), or the
  outer taxels cover more surface than the inner ones.
- **Move the command, do not jump it.** After a jump, a spring of about
  800 N/m or more drives the pad into the peg fast enough to break through;
  every contact normal then points sideways (measured at 1600 N/m) and the pad
  stays sunk at a fraction of the force. The script moves the command at 5 cm/s and flags a sunk
  result. Thicker collision boxes do not prevent it (tried).

---

## 5. Recommended path

**Stage 1 — one file, one day, CPU.** Grow `crisp_gym/sim/spring_pad.py` into a small task:
a 2-D or 3-D spring-driven pad (or two pads as a gripper) that must reach a
surface, press with a wanted force, and slide. Write the episodes as a LeRobot
dataset with **the same column names as the real data**:
`observation.state.cartesian`, `observation.state.gripper`, `action`
(measured pose one frame later), `extra.target_cartesian`,
`extra.tactile.left/right`. Then `audit_command_residual.py`,
`swap_action_offline.py` and the training wrappers run on it unchanged, and
every number they print can be compared with a known truth.

**Stage 2 — one week.** `tactile_envs/Insertion-v0` with `--nx 6 --ny 12`
pads: a scripted expert that inserts the peg, the same dataset columns, plus
the camera. This adds vision, a real task and a real success rate. Its
requirements pin mujoco 3.1.2 and Python 3.11; install it in its own
environment.

**Stage 3 — when the controller matters.** Your own stack in MuJoCo (option
D). Two uses:

- *The open stiffness question.* The audit measures 150–280 N/m delivered
  against 400 configured. Add a fixed box to
  `crisp_controllers_robot_demos/config/fr3/scene.xml`, command a target a few
  centimetres inside it, record a short dataset, and run the audit's step C
  with `--ext-torque-key extra.joint_efforts` (gravity is off in that scene, so
  in steady contact the motor torque is the contact torque; check the sign).
  400 N/m in simulation means the real arm loses it to friction or hardware;
  150–280 in simulation means the controller does it.
- *Deploy code without the robot.* The step limit, the workspace box and the
  relative policy loop can run end to end.

Expect configuration work first: the simulated Franka has no gripper, no
camera and no external-torque topic, so the usual env and record configs do
not start on it as they are (CODE_REVIEW N4).

---

## 6. Experiments

| | question | setup | measure | decides |
|---|---|---|---|---|
| E1 | Q1 | Stage 1 or 2. Train M (action = measured pose) and C (action = command via `swap_action_offline.py`) on the same episodes; repeat at K = 200, 400, 800 | success; contact force reached vs wanted | whether the command label is needed, and how the answer scales with stiffness (real data has one gain only) |
| E2 | Q2 | Fit `K × (command − measured)` from tactile + velocity on spring-driven episodes; test on "handheld-like" episodes where the pad is moved kinematically (no spring), so no command exists but the truth does | error of the predicted residual against the simulator's contact force; then train a policy on relabelled handheld-like data | whether relabelling handheld data is feasible, and with which inputs |
| E3 | Q3 | Policy inputs: vision; + summed force per finger (2 numbers); + full grids (144); + joint torque. With and without a FACTR-style curriculum | success under visual occlusion or misleading vision | which signal to wire into the real model first |
| E4 | Q4 | Same policy at 15 Hz chunks, against the same policy with a fast inner loop that tracks the policy's force target | force overshoot at first contact; success | whether a fast layer is worth building |
| E5 | Q5 | robosuite with the variable-impedance controller, or Stage 1 with K as an action | success and peak force against fixed-K | whether stiffness as an output is worth a controller change |

Order: E1 → E2 → E3. E4 and E5 only if E1–E3 leave contact failures.

---

## 7. What a simulator will not tell you

- MuJoCo contacts are rigid and point-like; a real skin spreads load, has
  noise, drift and dead taxels, and today reads when the fingers touch each
  other (the reason tactile is on hold, HANDOFF §7).
- The real arm delivers less stiffness than configured, for a reason not yet
  known. A simulated spring does exactly what its gain says.
- Friction, cable drag and the 15 Hz / 50 Hz / 1 kHz timing stack are absent
  unless modelled.

Use the simulator for logic, pipeline checks and comparisons between
variants. Do not carry absolute numbers (forces, success rates) to the robot.

---

## 8. Links to the code work

| research step | needs in this repository |
|---|---|
| E1 | nothing new: `swap_action_offline.py`, `lerobot_relative_pose.py`, the audit |
| a simulated dataset | a small writer producing the real column names (new, ROS-free), next to `crisp_gym/sim/spring_pad.py` |
| E3 on real data | GUIDANCE WP-10 (model inputs that actually reach the network) |
| squeeze as a command | record `extra.gripper_target` (the source exists; the Franka data lacks it) |
| E4, E5 on the robot | a second worker connection for a fast model (HANDOFF §7 "timing stack"); run-time gains in crisp_py |

---

## Sources

- [ACP, local repository `adaptive_compliance_policy`](https://github.com/yifan-hou/adaptive_compliance_policy)
- [DexForce, arXiv 2501.10356](https://arxiv.org/pdf/2501.10356)
- [In-the-Wild Compliant Manipulation with UMI-FT, arXiv 2601.09988](https://arxiv.org/html/2601.09988)
- [FACTR, arXiv 2502.17432](https://arxiv.org/pdf/2502.17432v2.pdf)
- [TA-VLA, arXiv 2509.07962](https://arxiv.org/html/2509.07962v1)
- [Reactive Diffusion Policy, arXiv 2503.02881](https://arxiv.org/html/2503.02881v3) and its [code](https://github.com/xiaoxiaoxh/reactive_diffusion_policy)
- [Compliant Residual DAgger, arXiv 2506.16685](https://arxiv.org/html/2506.16685v5)
- [Comp-ACT, arXiv 2406.14990](https://arxiv.org/html/2406.14990v2) and its [code](https://github.com/omron-sinicx/CompACT)
- [FILIC, arXiv 2509.17053](https://arxiv.org/html/2509.17053v1)
- [M3L, arXiv 2311.00924](https://arxiv.org/pdf/2311.00924), [M3L code](https://github.com/carlosferrazza/M3L), [tactile_envs](https://github.com/carlosferrazza/tactile_envs)
- [ManiFeel, arXiv 2505.18472](https://arxiv.org/abs/2505.18472)
- [ManiSkill-ViTac 2025, arXiv 2411.12503](https://arxiv.org/abs/2411.12503)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [robosuite sensors](https://robosuite.ai/docs/modules/sensors.html) and [controllers](https://robosuite.ai/docs/modules/controllers.html)
- [LeRobot HIL-SERL simulation (`gym-hil`)](https://huggingface.co/docs/lerobot/en/hilserl_sim)
- [lerobot diffusion model source, v0.4.4](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/diffusion/modeling_diffusion.py)
