# LeRobot Migration Guide: v0.3.x → v0.5.x

> Generated May 2026 · Sources: PyPI, GitHub Releases, HuggingFace Blog

---

## Version History

| Version | Release Date | Notes |
|---------|-------------|-------|
| **0.5.1** | Apr 7, 2026 | Latest stable |
| 0.5.0 | Mar 9, 2026 | Python 3.12+ required |
| 0.4.4 | Feb 27, 2026 | Bug fixes |
| 0.4.3 | Jan 22, 2026 | Bug fixes |
| 0.4.2 | Nov 27, 2025 | Bug fixes |
| 0.4.1 | Nov 10, 2025 | Bug fixes |
| 0.4.0 | Oct 23, 2025 | **Dataset v3.0, VLA models, plugin system** |
| 0.3.3 | Aug 6, 2025 | |
| 0.3.2 | Aug 1, 2025 | |
| 0.1.0 | Mar 9, 2024 | Initial release |

---

## Dataset Format Versions

| LeRobot version | Dataset format version |
|---|---|
| ≤ 0.3.x | **v2.1** — one Parquet + one MP4 file per episode |
| ≥ 0.4.0 | **v3.0** — chunked multi-episode files, streaming support |

### v2.1 Directory Structure (old)
```
dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── CAMERA_KEY/
│           ├── episode_000000.mp4
│           └── ...
└── meta/
    └── episodes.jsonl
```

### v3.0 Directory Structure (new)
```
dataset/
├── data/
│   └── chunk-000/
│       └── file-000.parquet       # Multiple episodes per file
├── videos/
│   └── CAMERA_KEY/
│       └── chunk-000/
│           └── file-000.mp4       # Consolidated video chunks
└── meta/
    ├── episodes/
    │   └── chunk-000/
    │       └── file-000.parquet   # Structured episode metadata
    ├── tasks.parquet
    ├── stats.json
    └── info.json
```

### v3.0 Performance Improvements
- 3–5× faster dataset initialization
- Better RAM usage through memory mapping
- Scales to millions of episodes (OXE-level, 400GB+)
- Native streaming support (no local download required)

---

## Installation

```bash
# Latest stable
pip install lerobot==0.5.1

# Or upgrade
pip install --upgrade lerobot

# With optional extras
pip install 'lerobot[all]'
pip install 'lerobot[aloha,pusht]'
pip install 'lerobot[feetech]'
```

> **Note (v0.5.0+):** Python 3.12+ is now required as the minimum version.

---

## Code Migration: v0.3.x → v0.4+

### Loading a Dataset

The import path is unchanged:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Standard load (same as before)
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access data
sample = dataset[0]
print(sample['action'].shape)
```

**New in v0.4+** — Streaming (no local download):

```python
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

dataset = StreamingLeRobotDataset("your-user/your-large-dataset")
```

**Access old v2.1 revision explicitly:**

```python
dataset = LeRobotDataset("lerobot/svla_so101_pickplace", revision="v2.1")
```

### Training CLI

```bash
# Old (v0.3.x) — Python script
python lerobot/scripts/train.py \
    policy=act \
    dataset_repo_id=lerobot/aloha_mobile_cabinet

# New (v0.4+) — CLI entry point
lerobot-train \
    --policy=act \
    --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

### Evaluation CLI

```bash
# New (v0.4+)
lerobot-eval \
    --policy.path=lerobot/pi0_libero_finetuned \
    --env.type=libero \
    --env.task=libero_object \
    --eval.n_episodes=10
```

### Recording a Dataset

```bash
# New (v0.4+)
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem585A0076841 \
    --robot.id=my_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --dataset.num_episodes=50
```

### Dataset Editing Tools (new in v0.4+)

```bash
# Merge multiple datasets
lerobot-edit-dataset \
    --repo_id user/merged_dataset \
    --operation.type merge \
    --operation.repo_ids "['user/ds1', 'user/ds2']"

# Delete specific episodes
lerobot-edit-dataset \
    --repo_id user/my_dataset \
    --new_repo_id user/my_dataset_cleaned \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 2, 5]"

# Split dataset
lerobot-edit-dataset \
    --repo_id user/my_dataset \
    --operation.type split \
    --operation.fraction 0.8
```

### Image Transforms (new in v0.4+)

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig

transforms_config = ImageTransformsConfig(
    enable=True,
    max_num_transforms=3,
    random_order=False,
)
transforms = ImageTransforms(transforms_config)

dataset = LeRobotDataset(
    repo_id="your-username/your-dataset",
    image_transforms=transforms
)
```

---

## Dataset Migration: v2.1 → v3.0

### Convert a single dataset

```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id=<HFUSER/DATASET_ID>
```

### What the conversion does automatically
- Converts per-episode Parquet files → chunked multi-episode files
- Migrates metadata from JSON Lines → Parquet
- Aggregates statistics and creates per-episode stats
- Updates `codebase_version` in `meta/info.json`
- Consolidates video files into chunks

---

## New Features in v0.4.0

### VLA Models
- **PI0 / PI0.5** — vision-language-action model
- **GR00T N1.5** — NVIDIA Isaac integration

### Simulation Environments
- **LIBERO** — 130+ tasks benchmark for VLA policies
- **Meta-World** — 50 diverse manipulation tasks

### Hardware Plugin System
New extensible plugin system for hardware integration:
- Decoupled `Robot` class interface
- Easy custom robot implementation

### Multi-GPU Training
```bash
lerobot-train \
    --policy=act \
    --dataset.repo_id=lerobot/aloha_mobile_cabinet \
    --training.num_gpus=4
```

---

## Key Resources

| Resource | URL |
|---|---|
| GitHub | https://github.com/huggingface/lerobot |
| PyPI | https://pypi.org/project/lerobot/ |
| Official Docs | https://huggingface.co/docs/lerobot |
| Dataset v3.0 Blog | https://huggingface.co/blog/lerobot-datasets-v3 |
| v0.4.0 Release Blog | https://huggingface.co/blog/lerobot-release-v040 |
| Dataset v3.0 Docs | https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3 |
| Porting Large Datasets | https://huggingface.co/docs/lerobot/porting_datasets_v3 |
| GitHub Releases | https://github.com/huggingface/lerobot/releases |
