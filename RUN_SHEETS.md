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

**Why.** Before spending GPU hours on two full trainings, prove in a few
minutes that the command copy is made correctly and that each checkpoint
records which action it learned. Nothing here has run with a real lerobot.

It uses the diffusion policy because it starts fastest; the check is the same
for any policy trained through `lerobot_relative_pose.py`.

1. What the dataset says about itself, and how big it is (the copy in step 3
   duplicates the videos).

```bash
ls /workspace/datasets/franka_electricbox/lerobot/meta/ && du -sh /workspace/datasets/franka_electricbox/lerobot && df -h /workspace/datasets | tail -1
```

2. Dry run of the command copy. Writes nothing.

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

4. Smoke-train on the recording (the "measured" model). About 100 steps.

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/lerobot_relative_pose.py \
    --dataset.repo_id=franka_electricbox \
    --dataset.root=/workspace/datasets/franka_electricbox/lerobot \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
    --policy.horizon=16 --policy.n_action_steps=8 --seed=1000 \
    --output_dir=/workspace/team/anglica/outputs/train/smoke_measured \
    --batch_size=8 --steps=100 --save_freq=100 --num_workers=4 --wandb.enable=false
```

5. The same on the command copy.

```bash
cd /workspace/team/anglica && python crisp_gym/crisp_gym/scripts/lerobot_relative_pose.py \
    --dataset.repo_id=franka_electricbox_cmd \
    --dataset.root=/workspace/datasets/franka_electricbox_cmd/lerobot \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
    --policy.horizon=16 --policy.n_action_steps=8 --seed=1000 \
    --output_dir=/workspace/team/anglica/outputs/train/smoke_command \
    --batch_size=8 --steps=100 --save_freq=100 --num_workers=4 --wandb.enable=false
```

**Works (steps 4 and 5):** the log shows `Wrapped dataset with
RelativePoseDataset`, then `Stamped .../pose_repr.json`, then loss lines, and
the run ends by itself.
**Does not work:** `Could not stamp pose_repr.json`, or a traceback. If the
output folder already exists, lerobot refuses: delete that folder or change
the name.

6. Read what each run recorded about its action.

```bash
python -c "
import glob, json
for f in sorted(glob.glob('/workspace/team/anglica/outputs/train/smoke_*/pose_repr.json')):
    print(f.split('/')[-2], '->', json.load(open(f))['action']['source'])
"
```

**Works:** `smoke_command -> {'kind': 'command', 'column': 'extra.target_cartesian', ...}`
and `smoke_measured -> {'kind': 'measured', 'column': 'next_tcp_pose', 'lookahead': 1, ...}`.
`'kind': 'unknown'` for the measured one means the dataset has no
`meta/record_config.json` (step 1 shows whether it does); say so when pasting.

**Paste back:** the output of steps 1, 2, 3 and 6, and the last 20 lines of
steps 4 and 5.

**Result:** not run yet.

**After it works.** Delete the two `smoke_*` folders; keep the command copy.
The full trainings are ROADMAP F1 step 3 (USAGE §14), with your usual batch
size and steps, and the same `--policy.horizon` / `--policy.n_action_steps` /
`--seed` for both.
