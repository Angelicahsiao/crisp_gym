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
