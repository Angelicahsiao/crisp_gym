r"""Visualize what a LeRobot DiffusionPolicy's visual encoder looks at.

STANDALONE — runs on the GPU/training PC with only lerobot + torch + numpy +
matplotlib (no crisp_gym import), like check_policy_openloop.py. Verified on
lerobot 0.4.4 and 0.6.1 (image- and video-backed datasets, separate and shared
encoders).

DiffusionPolicy's image path is  ResNet backbone -> SpatialSoftmax(K) -> Linear.
SpatialSoftmax turns the backbone feature map into K learned "keypoints": each
keypoint is a 1x1-conv detector whose softmax over the HxW grid IS a heat map,
and whose expected (x, y) under that heat map is what the policy receives.
This script reads those heat maps straight out of the model.

  1. ATTENTION (always): the K spatial-softmax heat maps, captured with a
     forward hook on rgb_encoder.pool. Exact, not an approximation: the
     keypoints are re-derived from the maps and asserted equal to the
     module's own output.
  2. OCCLUSION (--occlusion): slide a neutral patch over the image and measure
     how much the encoder's output feature changes. Model-agnostic check of
     which pixels the visual feature depends on.

Reading the results:
  * a keypoint that MOVES across an episode tracks something (object, arm);
  * a keypoint that is STATIC in a fixed camera is usually background or a
    fixture; in the WRIST camera, static often means "locked onto the gripper
    fingers", which is legitimately useful (it encodes gripper opening);
  * a DIFFUSE keypoint (entropy near uniform) attends to nothing in particular;
  * many keypoints on the same cell = keypoint collapse (wasted capacity).
  The static/tracking/diffuse labels in the CSV are heuristics, not ground truth.
  The keypoint marker is the softmax's EXPECTED position, so a diffuse or
  two-peaked map puts it between the peaks (the spatial-softmax midpoint artifact).

Usage (GPU PC, lerobot env):
    python3 viz_encoder_attention.py \
        --path .../checkpoints/100000/pretrained_model \
        --repo-id /abs/path/to/dataset \
        --episodes 0 3 --stride 10 --max-frames 30 --occlusion \
        --out-dir viz_out

    # Same, on loose image files (e.g. a robot-camera snapshot vs a handheld
    # frame, to compare what the encoder sees under the visual domain gap):
    python3 viz_encoder_attention.py --path ... \
        --image-files robot.png handheld.png --camera observation.images.wrist
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch
    from torch import nn

logger = logging.getLogger(__name__)

# Heuristic thresholds for the per-keypoint labels in the CSV.
DIFFUSE_ENTROPY = 0.90  # normalized entropy (1.0 = uniform over the grid)
STATIC_MOTION = 0.05  # std of keypoint position, in [-1, 1] grid units


# ── model access ──────────────────────────────────────────────────────────────


def load_policy(path: str, device: torch.device) -> tuple[Any, Any]:
    """Load the policy + preprocessor the same way check_policy_openloop.py does."""
    from lerobot.policies.factory import get_policy_class

    ptype = "diffusion"
    cfg_file = Path(path) / "config.json"
    if cfg_file.exists():
        ptype = json.loads(cfg_file.read_text()).get("type", ptype)
    if ptype != "diffusion":
        raise SystemExit(
            f"policy type is '{ptype}': this script reads DiffusionPolicy's "
            "SpatialSoftmax encoder. (Occlusion is model-agnostic in principle, "
            "but the encoder lookup here is diffusion-specific.)"
        )
    policy = get_policy_class(ptype).from_pretrained(path)
    policy.to(device).eval()

    pre = None
    try:
        from lerobot.policies.factory import make_pre_post_processors

        pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=path)
    except ImportError:
        pass  # older lerobot: normalization lives inside the policy
    return policy, pre


def encoders_by_camera(policy: Any) -> dict[str, nn.Module]:
    """Map image key -> DiffusionRgbEncoder, honouring use_separate_rgb_encoder_per_camera."""
    import torch.nn as nn

    keys = list(policy.config.image_features)
    if not keys:
        raise SystemExit("policy has no image features")
    enc = policy.diffusion.rgb_encoder
    if isinstance(enc, nn.ModuleList):  # one encoder per camera, in image_features order
        return dict(zip(keys, enc, strict=True))
    return {k: enc for k in keys}  # shared encoder


def normalize_image(
    img: torch.Tensor, key: str, policy: Any, pre: Any, device: torch.device
) -> torch.Tensor:
    """Turn a (1, C, H, W) image in [0, 1] into exactly what the encoder receives."""
    frame = {key: img.to(device)}
    if pre is not None:
        out = pre(frame)
    elif hasattr(policy, "normalize_inputs"):
        out = policy.normalize_inputs(frame)
    else:
        out = frame
    return out[key].float()


def display_crop(img: torch.Tensor, enc: nn.Module) -> torch.Tensor:
    """Apply the encoder's eval-time resize/centre-crop to the RAW image.

    This keeps overlays aligned with what the backbone actually saw.
    """
    x = img
    resize = getattr(enc, "resize", None)  # newer lerobot only
    if resize is not None:
        x = resize(x)
    if getattr(enc, "do_crop", False):
        x = enc.center_crop(x)
    return x


def spatial_softmax_maps(
    enc: nn.Module, x_norm: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the encoder once; return (attn (K,h,w), keypoints (K,2), feature (D,)).

    The heat maps are recomputed from the pool's INPUT exactly as
    SpatialSoftmax.forward does, then checked against the pool's OUTPUT.
    """
    import torch
    import torch.nn.functional as F

    cap: dict[str, torch.Tensor] = {}

    def hook(_m: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        cap["feat"] = inp[0].detach()
        cap["kp"] = out.detach()

    pool = enc.pool
    handle = pool.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            feature = enc(x_norm)
            logits = pool.nets(cap["feat"]) if pool.nets is not None else cap["feat"]
            b, k, hh, ww = logits.shape
            attn = F.softmax(logits.reshape(b, k, hh * ww), dim=-1)
            kp = attn @ pool.pos_grid
    finally:
        handle.remove()

    if not torch.allclose(kp, cap["kp"].reshape(b, k, 2), atol=1e-4):
        raise RuntimeError(
            "re-derived keypoints != SpatialSoftmax output — this lerobot's "
            "SpatialSoftmax differs from the one this script mirrors"
        )
    return (
        attn.reshape(k, hh, ww).cpu().numpy(),
        kp.reshape(k, 2).cpu().numpy(),
        feature.reshape(-1).cpu().numpy(),
    )


def occlusion_map(
    enc: nn.Module, x_norm: torch.Tensor, patch: int, stride: int, batch: int = 64
) -> np.ndarray:
    """Relative change of the encoder feature when each patch is masked.

    The patch is filled with the image's own per-channel mean in NORMALIZED
    space (neutral regardless of the normalization mode). Works on the
    pre-resize/pre-crop image, so the map is in raw-image pixels; patches
    outside the centre crop correctly score ~0.
    """
    import torch

    _, _, height, width = x_norm.shape
    fill = x_norm.mean(dim=(-2, -1), keepdim=True)
    with torch.inference_mode():
        base = enc(x_norm)
        ys = list(range(0, max(height - patch, 0) + 1, stride))
        xs = list(range(0, max(width - patch, 0) + 1, stride))
        boxes = [(y, x) for y in ys for x in xs]
        scores = []
        for i in range(0, len(boxes), batch):
            chunk = boxes[i : i + batch]
            xb = x_norm.repeat(len(chunk), 1, 1, 1)
            for j, (y, x) in enumerate(chunk):
                xb[j, :, y : y + patch, x : x + patch] = fill[0]
            f = enc(xb)
            scores.append((f - base).norm(dim=-1) / (base.norm(dim=-1) + 1e-8))
        scores = torch.cat(scores).cpu().numpy()

    heat = np.zeros((height, width))
    count = np.zeros((height, width))
    for s, (y, x) in zip(scores, boxes, strict=True):
        heat[y : y + patch, x : x + patch] += s
        count[y : y + patch, x : x + patch] += 1
    return heat / np.maximum(count, 1)


# ── data access ───────────────────────────────────────────────────────────────


def episode_range(ds: Any, ep: int) -> tuple[int, int]:
    """Return the [from, to) global frame indices of an episode, across lerobot versions.

    0.4.x exposes episode_data_index; 0.6.x dropped it in favour of
    meta.episodes[ep]["dataset_from_index"/"dataset_to_index"].
    """
    idx = getattr(ds, "episode_data_index", None) or getattr(
        getattr(ds, "meta", None), "episode_data_index", None
    )
    if idx is not None:
        return int(idx["from"][ep]), int(idx["to"][ep])
    eps = getattr(getattr(ds, "meta", None), "episodes", None)
    if eps is not None:
        row = eps[ep]
        return int(row["dataset_from_index"]), int(row["dataset_to_index"])
    raise SystemExit("cannot locate episode boundaries in this lerobot version")


def iter_dataset_frames(
    args: argparse.Namespace, image_keys: list[str]
) -> Iterator[tuple[str, dict[str, torch.Tensor]]]:
    """Yield (tag, {image_key: (1, C, H, W) in [0, 1]}) for the requested frames."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = args.root or (args.repo_id if os.path.isdir(args.repo_id) else None)
    ds = LeRobotDataset(args.repo_id, root=root)
    for ep in args.episodes:
        lo, hi = episode_range(ds, ep)
        for n, idx in enumerate(range(lo, hi, args.stride)):
            if n >= args.max_frames:
                break
            item = ds[idx]
            imgs = {
                k: torch.as_tensor(item[k]).float().unsqueeze(0) for k in image_keys if k in item
            }
            yield f"ep{ep:03d}_f{idx:06d}", imgs


def iter_image_files(
    args: argparse.Namespace, policy: Any
) -> Iterator[tuple[str, dict[str, torch.Tensor]]]:
    """Yield loose image files, resized to the policy's input shape for one camera."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    key = args.camera or next(iter(policy.config.image_features))
    _, h, w = policy.config.image_features[key].shape
    for p in args.image_files:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        yield Path(p).stem, {key: t}


# ── plotting ──────────────────────────────────────────────────────────────────


def kp_to_px(
    kp: np.ndarray, grid_hw: tuple[int, int], img_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert [-1, 1] keypoint coords to pixel coords of the cropped image.

    pos_grid spans grid-cell CENTRES, so x=-1 is the centre of cell 0.
    """
    (gh, gw), (ih, iw) = grid_hw, img_hw
    px = ((kp[:, 0] + 1) * (gw - 1) / 2 + 0.5) * iw / gw
    py = ((kp[:, 1] + 1) * (gh - 1) / 2 + 0.5) * ih / gh
    return px, py


def plot_frame(
    out_png: Path,
    title: str,
    crop_img: np.ndarray,
    attn: np.ndarray,
    kp: np.ndarray,
    order: list[int],
    n_show: int,
    occ: np.ndarray | None = None,
    raw_img: np.ndarray | None = None,
) -> None:
    """Write one figure: crop, max-over-keypoints map, occlusion, top individual maps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k, gh, gw = attn.shape
    ih, iw = crop_img.shape[:2]
    ext = (0, iw, ih, 0)
    n_show = min(n_show, k)
    ncols = 2 + (1 if occ is not None else 0) + int(np.ceil(n_show / 2))
    fig = plt.figure(figsize=(3.2 * ncols, 6.6))
    gs = fig.add_gridspec(2, ncols)

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(crop_img)
    ax.set_title("input (eval crop)")
    ax.axis("off")

    ax = fig.add_subplot(gs[:, 1])
    ax.imshow(crop_img)
    ax.imshow(attn.max(axis=0), extent=ext, cmap="jet", alpha=0.5, interpolation="bilinear")
    px, py = kp_to_px(kp, (gh, gw), (ih, iw))
    ax.scatter(px, py, c=np.arange(k), cmap="hsv", s=18, edgecolors="white", linewidths=0.5)
    distinct = len({int(a) for a in attn.reshape(k, -1).argmax(1)})
    ax.set_title(f"max over {k} keypoints\n{distinct}/{k} distinct peak cells")
    ax.axis("off")

    col = 2
    if occ is not None:
        ax = fig.add_subplot(gs[:, col])
        ax.imshow(raw_img)
        ax.imshow(occ, cmap="jet", alpha=0.5)
        ax.set_title("occlusion: feature change\n(raw frame)")
        ax.axis("off")
        col += 1

    for i, kidx in enumerate(order[:n_show]):
        ax = fig.add_subplot(gs[i % 2, col + i // 2])
        ax.imshow(crop_img)
        ax.imshow(attn[kidx], extent=ext, cmap="jet", alpha=0.55, interpolation="bilinear")
        ax.plot(px[kidx], py[kidx], "w+", ms=10, mew=2)
        ax.set_title(f"kp {kidx}  peak {attn[kidx].max():.2f}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse args, collect heat maps for every frame/camera, write figures + CSVs."""
    p = argparse.ArgumentParser(
        description="Spatial-softmax heat maps of a DiffusionPolicy encoder"
    )
    p.add_argument("--path", required=True, help="checkpoint pretrained_model dir")
    p.add_argument("--repo-id", help="dataset repo id or local path")
    p.add_argument("--root", default=None, help="dataset root (if repo-id is a plain name)")
    p.add_argument("--episodes", type=int, nargs="*", default=[0])
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=30, help="per episode")
    p.add_argument("--image-files", nargs="*", default=None, help="run on loose images instead")
    p.add_argument("--camera", default=None, help="image key for --image-files (default: first)")
    p.add_argument("--top-k", type=int, default=8, help="individual keypoint maps per figure")
    p.add_argument("--occlusion", action="store_true")
    p.add_argument("--patch", type=int, default=32)
    p.add_argument("--patch-stride", type=int, default=16)
    p.add_argument("--out-dir", default="viz_encoder")
    args = p.parse_args()
    if not args.repo_id and not args.image_files:
        p.error("give --repo-id or --image-files")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, pre = load_policy(args.path, device)
    encs = encoders_by_camera(policy)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[viz] cameras: %s | separate encoders: %s | K=%d",
        list(encs),
        len({id(e) for e in encs.values()}) > 1,
        policy.config.spatial_softmax_num_keypoints,
    )

    if args.image_files:
        frames = iter_image_files(args, policy)
    else:
        frames = iter_dataset_frames(args, list(encs))

    # Pass 1: collect maps (tiny: frames x K x h x w).
    records: dict[str, list] = {k: [] for k in encs}
    for tag, imgs in frames:
        for key, img in imgs.items():
            enc = encs[key]
            x = normalize_image(img, key, policy, pre, device)
            attn, kp, _ = spatial_softmax_maps(enc, x)
            crop = display_crop(img, enc)[0].permute(1, 2, 0).clamp(0, 1).numpy()
            occ = occlusion_map(enc, x, args.patch, args.patch_stride) if args.occlusion else None
            raw = img[0].permute(1, 2, 0).clamp(0, 1).numpy()
            records[key].append((tag, attn, kp, crop, occ, raw))
    if not any(records.values()):
        raise SystemExit("no frames found — check --repo-id/--episodes/--image-files")

    # Per-keypoint statistics across all frames, per camera.
    for key, recs in records.items():
        if not recs:
            continue
        cam = key.split(".")[-1]
        maps = np.stack([r[1] for r in recs])  # (F, K, h, w)
        kps = np.stack([r[2] for r in recs])  # (F, K, 2)
        nf, k, gh, gw = maps.shape
        flat = maps.reshape(nf, k, -1)
        peak = flat.max(-1).mean(0)
        ent = (-(flat * np.log(flat + 1e-12)).sum(-1) / np.log(gh * gw)).mean(0)
        if nf > 1:
            motion = np.sqrt(kps[..., 0].std(0) ** 2 + kps[..., 1].std(0) ** 2)
        else:
            motion = np.zeros(k)
        # Motion is undefined for a single frame — don't call a sharp keypoint "static".
        moving_label = np.where(motion < STATIC_MOTION, "static", "tracking") if nf > 1 else "n/a"
        label = np.where(ent > DIFFUSE_ENTROPY, "diffuse", moving_label)
        distinct = np.mean([len(set(f.argmax(-1).tolist())) for f in flat])

        with open(out / f"keypoint_stats_{cam}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "keypoint",
                    "mean_peak",
                    "mean_norm_entropy",
                    "motion_std",
                    "label",
                    "mean_x",
                    "mean_y",
                ]
            )
            for i in range(k):
                w.writerow(
                    [
                        i,
                        f"{peak[i]:.4f}",
                        f"{ent[i]:.4f}",
                        f"{motion[i]:.4f}",
                        label[i],
                        f"{kps[:, i, 0].mean():.3f}",
                        f"{kps[:, i, 1].mean():.3f}",
                    ]
                )

        counts = {
            name: int((label == name).sum())
            for name in ("tracking", "static", "diffuse", "n/a")
            if (label == name).any()
        }
        logger.info(
            "[viz] %s: %d frames, grid %dx%d | %s | avg %.1f/%d distinct peak cells%s",
            cam,
            nf,
            gh,
            gw,
            counts,
            distinct,
            k,
            "  (motion needs >1 frame)" if nf == 1 else "",
        )

        # Show the most informative keypoints first: tracking (by motion), then sharpest.
        order = sorted(range(k), key=lambda i: (label[i] != "tracking", -motion[i], -peak[i]))
        for tag, attn, kp, crop, occ, raw in recs:
            plot_frame(
                out / f"{tag}_{cam}.png",
                f"{tag}  [{key}]",
                crop,
                attn,
                kp,
                order,
                args.top_k,
                occ,
                raw,
            )
    logger.info("[viz] wrote figures + keypoint_stats_*.csv to %s", out.resolve())


if __name__ == "__main__":
    main()
