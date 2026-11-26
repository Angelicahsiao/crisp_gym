# Run sheets

Commands for the owner to run where an assistant cannot: on the GPU PC and on
the robot PC. Each sheet says what the output should show and what to paste
back. `ROADMAP.md` §1.2 explains the idea; results are written into the
"Result" line of each sheet once they are known.

Paths are the owner's: the repository at `/workspace/team/anglica/crisp_gym`,
datasets under `/workspace/datasets`. One command per block. If `git pull`
refuses because of local changes, paste what it says and stop there.

---

## Sheet 1 · The test suite on the GPU PC (ROADMAP F10)

**Why.** The tests have only ever run without lerobot and torch. With the real
packages installed some may behave differently; that is what this finds.

1. Get the newest code.

```bash
cd /workspace/team/anglica/crisp_gym && git pull && git log --oneline -1
```

2. Is pytest there? (If this prints a version, skip step 3.)

```bash
python -m pytest --version
```

3. Only if step 2 failed. This adds one package and changes nothing else. Do
   NOT install `tests/requirements.txt` here: it pins numpy and could change
   the versions lerobot runs on.

```bash
python -m pip install pytest
```

4. Run.

```bash
cd /workspace/team/anglica/crisp_gym && python -m pytest -rs -q 2>&1 | tail -60
```

**Works:** a last line like `7xx passed, N skipped`. The simulator tests skip
unless `mujoco` is installed; that is fine. An error about a missing package
(`websockets`, `msgpack`, `gymnasium`, ...) is part of the answer too.
**Does not work:** lines starting with `FAILED` or `ERROR`. That is a result,
not a problem: do not try to fix anything.
**Paste back:** everything step 4 printed.

**Result:** not run yet.

---

## Sheet 2 · The test suite on the robot PC (ROADMAP F10)

**Why.** Same reason, with real ROS and crisp_py present.

Run these in the shell you record in (the pixi environment, inside the
container). Change the path if the repository is elsewhere on that machine.

1. Get the newest code.

```bash
cd /workspace/team/anglica/crisp_gym && git pull && git log --oneline -1
```

2. Is pytest there? (If this prints a version, skip step 3.)

```bash
python -m pytest --version
```

3. Only if step 2 failed. One package, into the current environment.

```bash
python -m pip install pytest
```

4. Run. No robot motion: the tests start no controller and send no command.

```bash
cd /workspace/team/anglica/crisp_gym && python -m pytest -rs -q 2>&1 | tail -60
```

5. While you are there, one line for another open question (which copy of the
   duplicated configs is in effect):

```bash
echo "CRISP_CONFIG_PATH=$CRISP_CONFIG_PATH"
```

**Paste back:** everything steps 4 and 5 printed.

**Result:** not run yet.

---

## Sheet 3 · Comparison smoke run on the GPU PC (ROADMAP F1, steps 1–2)

**Why.** Before spending GPU hours on four full trainings (diffusion and
`multi_task_dit`, each on the recording and on the command copy), prove in a
few minutes that the command copy is made correctly and that every checkpoint
records which action it learned. Nothing here has run with a real lerobot.

### Part A · the command copy

1. What the dataset says about itself, and how big it is (step 3 duplicates
   the videos).

```bash
ls /workspace/datasets/franka_electricbox/lerobot/meta/ && du -sh /workspace/datasets/franka_electricbox/lerobot && df -h /workspace/datasets | tail -1
```

2. Dry run. Writes nothing.

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/swap_action_offline.py --input /workspace/datasets/franka_electricbox/lerobot --output /workspace/datasets/franka_electricbox_cmd/lerobot --dry-run
```

**Works:** `--dry-run: features OK, target varies. No data written.`
**Does not work:** an error naming a missing column, or saying the target is
constant. Stop and paste it.

3. Make the copy.

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/swap_action_offline.py --input /workspace/datasets/franka_electricbox/lerobot --output /workspace/datasets/franka_electricbox_cmd/lerobot
```

**Works:** `Done: 103040 frames, 293 episodes.` and a file
`/workspace/datasets/franka_electricbox_cmd/lerobot/meta/action_repr.json`.

### Part B · diffusion, 100 steps each

4. On the recording (the "measured" model).

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/lerobot_relative_pose.py \
    --dataset.repo_id=franka_electricbox \
    --dataset.root=/workspace/datasets/franka_electricbox/lerobot \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
    --policy.horizon=16 --policy.n_action_steps=8 --seed=1000 \
    --output_dir=/workspace/team/anglica/outputs/train/smoke_diffusion_measured \
    --batch_size=8 --steps=100 --save_freq=100 --num_workers=4 --wandb.enable=false
```

5. On the command copy.

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/lerobot_relative_pose.py \
    --dataset.repo_id=franka_electricbox_cmd \
    --dataset.root=/workspace/datasets/franka_electricbox_cmd/lerobot \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
    --policy.horizon=16 --policy.n_action_steps=8 --seed=1000 \
    --output_dir=/workspace/team/anglica/outputs/train/smoke_diffusion_command \
    --batch_size=8 --steps=100 --save_freq=100 --num_workers=4 --wandb.enable=false
```

**Works (steps 4 and 5):** the log shows `Wrapped dataset with
RelativePoseDataset`, then `Stamped .../pose_repr.json`, then loss lines, and
the run ends by itself.
**Does not work:** `Could not stamp pose_repr.json`, or a traceback. If the
output folder already exists, lerobot refuses: delete that folder or change
the name.

### Part C · `multi_task_dit`, 100 steps each

The launcher must sit in the folder that CONTAINS the `crisp_gym` clone (it
looks for `crisp_gym/crisp_gym/scripts` beside itself).

6. Is it there, and is it the repository's version?

```bash
ls -l /workspace/team/anglica/train_multi_task_dit_server.sh && diff -q /workspace/team/anglica/train_multi_task_dit_server.sh /workspace/team/anglica/crisp_gym/scripts/train_multi_task_dit_server.sh && echo "same as the repository"
```

   If it is missing, copy it there. If it is there but differs, do not
   overwrite it: paste what step 6 printed and keep going with your copy.

```bash
cp /workspace/team/anglica/crisp_gym/scripts/train_multi_task_dit_server.sh /workspace/team/anglica/
```

7. Dry run on the recording: runs the dataset gate and the hub check, prints
   the training command, launches nothing.

```bash
cd /workspace/team/anglica && DRY_RUN=1 DATASET=/workspace/datasets/franka_electricbox/lerobot DATASET_NAME=franka_electricbox OUT=/workspace/team/anglica/outputs/train/smoke_dit_measured STEPS=100 SAVE_FREQ=100 BATCH_SIZE=16 NUM_WORKERS=4 ./train_multi_task_dit_server.sh --seed=1000
```

**Works:** gate lines, then `==> python .../lerobot_relative_pose.py --wrt-start
--dataset.repo_id=franka_electricbox ...`, then `(dry run — nothing launched)`.
**Does not work:** `Preflight FAILED` or `Hub check FAILED`. Paste it.

8. The same without `DRY_RUN=1`: the measured model.

```bash
cd /workspace/team/anglica && DATASET=/workspace/datasets/franka_electricbox/lerobot DATASET_NAME=franka_electricbox OUT=/workspace/team/anglica/outputs/train/smoke_dit_measured STEPS=100 SAVE_FREQ=100 BATCH_SIZE=16 NUM_WORKERS=4 ./train_multi_task_dit_server.sh --seed=1000
```

9. The command model. Its dataset gate prints a warning that the action "is
   not the next measured pose". That is what a command copy is: expected.

```bash
cd /workspace/team/anglica && DATASET=/workspace/datasets/franka_electricbox_cmd/lerobot DATASET_NAME=franka_electricbox_cmd OUT=/workspace/team/anglica/outputs/train/smoke_dit_command STEPS=100 SAVE_FREQ=100 BATCH_SIZE=16 NUM_WORKERS=4 ./train_multi_task_dit_server.sh --seed=1000
```

### Part D · what each run recorded

10. Read the action source of all four.

```bash
python -c "
import glob, json
for f in sorted(glob.glob('/workspace/team/anglica/outputs/train/smoke_*/pose_repr.json')):
    print(f.split('/')[-2], '->', json.load(open(f))['action']['source'])
"
```

**Works:** four lines. The two `..._command` ones say
`{'kind': 'command', 'column': 'extra.target_cartesian', ...}`; the two
`..._measured` ones say `{'kind': 'measured', 'column': 'next_tcp_pose',
'lookahead': 1, ...}`. `'kind': 'unknown'` for a measured one means the
dataset has no `meta/record_config.json` (step 1 shows whether it does); say
so when pasting.

**Paste back:** the output of steps 1, 2, 3, 6, 7 and 10, and the last 20
lines of steps 4, 5, 8 and 9.

**Result:** not run yet.

**After it works.** Delete the four `smoke_*` folders; keep the command copy.
The full trainings are ROADMAP F1 step 3 (USAGE §14): your usual batch size
and steps, and for each policy the same settings and `--seed` on both
datasets.

### Part E · a FIRST pair (planned; the training part cannot run yet)

**Why.** FACTR 2 (arXiv 2606.12406v2, read in full) changes one thing in
training, which it calls FIRST: frames in the second before each contact
starts (pre-contact) and during contact are drawn more often. The model, the
loss and the data stay the same. Its contact labels come from the size of the
external joint torque, which this dataset records (`extra.ext_torque`). Its
best weights, free : pre-contact : contact, are 1 : 5 : 1, and 1 : 3 : 3 on its
one task with sustained contact (cap screwing); weights above about 5 gave
nothing more or made things worse, and up-sampling contact alone did not help.
Its policies also take the torque as input, but the re-sampling mostly helped
policies without any force input too (its Table 9), and that is the case for
the two models here.

**The rule.** FIRST is an extra PAIR beside the four runs above, never a swap
inside them: measured + FIRST and command + FIRST, with the same sampler,
settings and `--seed` as each other. A FIRST model against a plain one answers
a different question (does re-sampling help), so read the two pairs apart.

**Why the training part cannot run yet.** lerobot 0.6.1 (the GPU PC's) and
0.7.0 have no weighted frame sampler: the training loop always builds a
uniform `EpisodeAwareSampler`. Their `sample_weighting` option scales the loss
per frame instead, and only for policies whose `forward` takes `reduction`;
diffusion, `multi_task_dit` and GR00T do not, so it fails on all three. FACTR 2
also tried weighting the loss by phase and found it worse than re-sampling (its
Table 11), so that option is the wrong tool anyway. The pair needs a sampler in
`lerobot_relative_pose.py` first. Not built; postponed by the owner.

11. What runs now: how many frames each phase would get. The audit with the
    joint-torque contact flag (each joint against its resting level at the
    start of the episode, 2 Nm), keeping its per-frame flags.

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/audit_command_residual.py /workspace/datasets/franka_electricbox/lerobot --contact-from torque --out /workspace/team/anglica/outputs/first_audit.json --dump-npz /workspace/team/anglica/outputs/first_frames.npz
```

**Works:** the report, with `Contact flag: torque, ...% of frames in contact`
(the second audit run found 22.6 %), then `wrote .../first_frames.npz`.

12. Contact, pre-contact (the second before each contact starts, as in
    FACTR 2) and free frames. Reads the two files of step 11, writes nothing.

```bash
python -c "
import json, numpy as np
d = np.load('/workspace/team/anglica/outputs/first_frames.npz')
fps = json.load(open('/workspace/team/anglica/outputs/first_audit.json'))['fps']
ep, c = d['episode'], d['contact'].astype(bool)
pre, w, onsets = np.zeros_like(c), int(round(fps)), 0
for e in np.unique(ep):
    i = np.flatnonzero(ep == e)
    s = np.flatnonzero(c[i] & ~np.r_[False, c[i][:-1]])
    onsets += len(s)
    for k in s:
        pre[i[max(0, k - w):k]] = True
pre &= ~c
print(f'{len(c)} frames at {fps:g} Hz | contact {100*c.mean():.1f}% | pre-contact (1 s before) {100*pre.mean():.1f}% | free {100*(~c & ~pre).mean():.1f}% | {onsets / len(np.unique(ep)):.1f} contact onsets per episode')
"
```

**Works:** one line: `103040 frames at ... Hz | contact ...% | pre-contact
(1 s before) ...% | free ...% | ... contact onsets per episode`. The owner's
torque plot shows three contact phases per episode, so about 3 onsets per
episode is expected. Many more means the flag flickers around 2 Nm: rerun
step 11 with `--torque-threshold 3` and compare. (The audit has one threshold;
FACTR 2 uses two, a higher one to switch contact on and a lower one to switch
it off, which avoids that flicker.)

**Paste back:** the `Contact flag` line of step 11 and the line of step 12.
With them, the paper's 1 : 5 : 1 turns into the share of each phase per batch.

**Result:** not run yet.

---

## Sheet 4 · Simulated press-button data on the GPU PC

**Why.** Three datasets of the same 200 simulated demonstrations, each pressing
a spring button and then sliding on the panel beside it, at spring stiffness
200, 400 and 800 N/m (`crisp_gym/sim/button_task.py`). They have the Franka
recordings' column names, so the audit, the command copy and both trainings
run on them unchanged, and the simulator knows the true contact force
(`extra.contact_force`). No camera: the button's position is in
`observation.state` after the gripper. Tested in a cloud session with lerobot
0.6.1 on CPU: 12 episodes written and read back, the audit, the command copy,
20 diffusion steps and 10 `multi_task_dit` steps (through the launcher) on
both copies. Never run on the GPU PC.

1. Get the newest code.

```bash
cd /workspace/team/anglica/crisp_gym && git pull && git log --oneline -1
```

2. Is MuJoCo there? (If this prints a version, skip step 3.)

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

3. Only if step 2 failed. First see what the install would change:

```bash
uv pip install --dry-run mujoco
```

   **Works:** it adds `mujoco` and a few small packages and changes nothing
   that is already installed. If it would change numpy or torch, stop and
   paste what it printed. Otherwise install it:

```bash
uv pip install mujoco
```

4. One demonstration, printed. Writes nothing.

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/sim/button_dataset.py --preview --stiffness 400
```

**Works:** `clicked: yes`, and a line `command below measured, largest: X mm
(clicking needs Y mm at this spring)` with X a little above Y.

5. The three datasets, one per stiffness. Each takes a few minutes. The same
   `--seed` makes the same 200 demonstrations (button, start, speeds) at each
   stiffness.

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/sim/button_dataset.py --root /workspace/datasets/sim_button_k200/lerobot --stiffness 200 --episodes 200 --seed 0
```

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/sim/button_dataset.py --root /workspace/datasets/sim_button_k400/lerobot --stiffness 400 --episodes 200 --seed 0
```

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/sim/button_dataset.py --root /workspace/datasets/sim_button_k800/lerobot --stiffness 800 --episodes 200 --seed 0
```

**Works:** each ends with `Done: 200 episodes, ... frames, 0 skipped without a
click, at ... N/m`. A few skipped is fine (they are replaced). If the folder
already exists it refuses: delete it or pick another `--root`.

6. The audit on the 400 N/m dataset. Here the right answers are known.

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/audit_command_residual.py /workspace/datasets/sim_button_k400/lerobot --k-pos 400 --out /workspace/team/anglica/outputs/sim_audit_k400.json
```

**Works:** in section B, `tau` close to 141 ms on x, y and z: that is the
simulator's true lag, 2·√(2 kg / 400 N/m). Section C says skipped (the
simulation has no joints). Section F shows the contact gap that a
measured-pose label does not carry.

7. The command copies.

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/swap_action_offline.py --input /workspace/datasets/sim_button_k200/lerobot --output /workspace/datasets/sim_button_k200_cmd/lerobot
```

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/swap_action_offline.py --input /workspace/datasets/sim_button_k400/lerobot --output /workspace/datasets/sim_button_k400_cmd/lerobot
```

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/swap_action_offline.py --input /workspace/datasets/sim_button_k800/lerobot --output /workspace/datasets/sim_button_k800_cmd/lerobot
```

**Works:** `Done: ... frames, 200 episodes.` for each.

8. Diffusion, 100 steps, on the 400 N/m recording and its command copy.

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/lerobot_relative_pose.py \
    --dataset.repo_id=sim_button_k400 \
    --dataset.root=/workspace/datasets/sim_button_k400/lerobot \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
    --policy.horizon=16 --policy.n_action_steps=8 --seed=1000 \
    --output_dir=/workspace/team/anglica/outputs/train/sim_smoke_diffusion_measured \
    --batch_size=8 --steps=100 --save_freq=100 --num_workers=4 --wandb.enable=false
```

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/lerobot_relative_pose.py \
    --dataset.repo_id=sim_button_k400_cmd \
    --dataset.root=/workspace/datasets/sim_button_k400_cmd/lerobot \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
    --policy.horizon=16 --policy.n_action_steps=8 --seed=1000 \
    --output_dir=/workspace/team/anglica/outputs/train/sim_smoke_diffusion_command \
    --batch_size=8 --steps=100 --save_freq=100 --num_workers=4 --wandb.enable=false
```

**Works:** `observation.state: widened 13 -> 19 dims`, `Wrapped dataset with
RelativePoseDataset`, `Stamped .../pose_repr.json`, loss lines, and the run
ends by itself.

9. `multi_task_dit`, 100 steps each, through the launcher of sheet 3 (step 6
   there puts it in `/workspace/team/anglica`). `SKIP_PREFLIGHT=1` because the
   dataset gate fails on "no cameras", which is expected here; it is the gate's
   only failure on these datasets.

```bash
cd /workspace/team/anglica && SKIP_PREFLIGHT=1 DATASET=/workspace/datasets/sim_button_k400/lerobot DATASET_NAME=sim_button_k400 OUT=/workspace/team/anglica/outputs/train/sim_smoke_dit_measured STEPS=100 SAVE_FREQ=100 BATCH_SIZE=16 NUM_WORKERS=4 ./train_multi_task_dit_server.sh --seed=1000
```

```bash
cd /workspace/team/anglica && SKIP_PREFLIGHT=1 DATASET=/workspace/datasets/sim_button_k400_cmd/lerobot DATASET_NAME=sim_button_k400_cmd OUT=/workspace/team/anglica/outputs/train/sim_smoke_dit_command STEPS=100 SAVE_FREQ=100 BATCH_SIZE=16 NUM_WORKERS=4 ./train_multi_task_dit_server.sh --seed=1000
```

**Works:** loss lines, `End of training`, then `==> Finished.`

10. What each run recorded.

```bash
python -c "
import glob, json
for f in sorted(glob.glob('/workspace/team/anglica/outputs/train/sim_smoke_*/pose_repr.json')):
    print(f.split('/')[-2], '->', json.load(open(f))['action']['source']['kind'])
"
```

**Works:** four lines; the two `..._command` say `command`, the two
`..._measured` say `measured`.

**Paste back:** the output of steps 2 and 4, the `Done` lines of steps 5 and
7, sections B and F of step 6, the last 20 lines of steps 8 and 9, and the
output of step 10.

**Result (2026-11-26, owner, GPU PC):** works.
- Step 5: `sim_button_k400` written, 200 episodes, 20,111 frames. The 200 and
  800 N/m `Done` lines were not pasted.
- Step 6, the audit, against the values the simulator was given: tracking lag
  141.4 / 141.6 / 142.0 ms (x / y / z) against 141.4 expected; largest contact
  spring force 15.2 N against the hardest button's 15 N; in contact the command
  leads along base z only (−12.4 mm mean, x and y about 0); no step over the
  30 mm limit. Section F: 10.5 mm median gap in contact at the best lookahead,
  0.6 mm in free motion. Section D: tactile alone predicts the vertical
  residual with R² 0.95 on contact frames (x and y 0.0: the skins measure
  normal force only, the sideways gap is sliding friction); tactile + velocity
  0.77 / 0.77 / 0.98.
- Step 10: the four smoke checkpoints stamped `command` / `measured`
  correctly; the folders were deleted afterwards. The logs of steps 8 and 9
  were not pasted.

**After it works.** Delete the four `sim_smoke_*` folders and keep the six
datasets. The full comparison is the same pair (recording against command
copy) at each stiffness. Its answer needs the rollout script that runs a
trained policy inside the simulation and counts clicks and force; that script
is not built yet (owner: later).

---

## Sheet 5 · `multi_task_dit` on the simulated data at 400 N/m (full trainings)

**Why.** The first pair of the measured-vs-command comparison in simulation
(ROADMAP F7 step 4): two models, the same settings and seed, differing only in
the action label, trained 200 epochs (UMI's schedule for its transformer). The
owner chose this one pair first (2026-11-26): it shows the answer at one
stiffness before GPU time goes into 200 and 800 N/m. Which model presses the
button better needs the rollout script (F7 step 3, not built yet); these are
the checkpoints it will be tested on. The open-loop check here only shows that
each model learned its own label, on demonstrations it never saw.

1. Get the newest code, and check that the launcher beside the clone is the
   repository's (sheet 3, step 6).

```bash
cd /workspace/team/anglica/crisp_gym && git pull && git log --oneline -1
```

```bash
diff -q /workspace/team/anglica/train_multi_task_dit_server.sh /workspace/team/anglica/crisp_gym/scripts/train_multi_task_dit_server.sh && echo "same as the repository"
```

   If it differs, copy the repository's version over it:

```bash
cp /workspace/team/anglica/crisp_gym/scripts/train_multi_task_dit_server.sh /workspace/team/anglica/
```

2. The measured model. `SKIP_PREFLIGHT=1` as in sheet 4: the gate fails only
   on "no cameras".

```bash
cd /workspace/team/anglica && SKIP_PREFLIGHT=1 EPOCHS=200 DATASET=/workspace/datasets/sim_button_k400/lerobot DATASET_NAME=sim_button_k400 OUT=/workspace/team/anglica/outputs/train/sim_dit_k400_measured ./train_multi_task_dit_server.sh --seed=1000
```

3. The command model: the same, on the command copy.

```bash
cd /workspace/team/anglica && SKIP_PREFLIGHT=1 EPOCHS=200 DATASET=/workspace/datasets/sim_button_k400_cmd/lerobot DATASET_NAME=sim_button_k400_cmd OUT=/workspace/team/anglica/outputs/train/sim_dit_k400_command ./train_multi_task_dit_server.sh --seed=1000
```

**Works (steps 2 and 3):** near the start, `[epochs]` lines: `--epochs 200 ->
epoch target = ceil(200 x N / 320) = ... steps` and `target checkpoint at step
...`, with the same N in both runs (the copy has the same frames). Then loss
lines, `End of training` and `==> Finished.` There are no camera images, so
these should run much faster than your image trainings.

4. Twenty demonstrations the models never saw (seed 1; the trainings used
   seed 0), and their command copy.

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/sim/button_dataset.py --root /workspace/datasets/sim_button_k400_heldout/lerobot --stiffness 400 --episodes 20 --seed 1
```

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/swap_action_offline.py --input /workspace/datasets/sim_button_k400_heldout/lerobot --output /workspace/datasets/sim_button_k400_heldout_cmd/lerobot
```

5. Print the two open-loop commands, each pointing at its run's epoch-target
   checkpoint (read from its `train.log`), and each against the held-out copy
   with its own label.

```bash
cd /workspace/team/anglica && python -c "
import glob, re
for run in ('measured', 'command'):
    out = '/workspace/team/anglica/outputs/train/sim_dit_k400_' + run
    step = int(re.findall(r'target checkpoint at step (\d+)', open(out + '/train.log').read())[0])
    ckpt = [d for d in glob.glob(out + '/checkpoints/*') if d.split('/')[-1].isdigit() and int(d.split('/')[-1]) == step][0]
    data = 'sim_button_k400_heldout' + ('_cmd' if run == 'command' else '')
    print('python crisp_gym/crisp_gym/scripts/check_policy_openloop.py --path ' + ckpt + '/pretrained_model --repo-id ' + data + ' --root /workspace/datasets/' + data + '/lerobot --episodes 0 1 2 3 4')
"
```

**Works:** two lines starting `python crisp_gym/crisp_gym/scripts/check_policy_openloop.py`.

6. Run the two printed lines, one after the other, from
   `/workspace/team/anglica`.

**Works:** each ends with a `[openloop] verdict:` line. GOOD (err/move ≤ 0.3)
means the model reproduces demonstrations it never saw. Either model may be
weaker in contact; the rollouts decide which presses better, not this.

**Paste back:** the `[epochs]` lines and the last 20 lines of steps 2 and 3,
the `Done` lines of step 4, and the table and the two `[openloop]` lines of
each run in step 6.

**Result:** not run yet.
