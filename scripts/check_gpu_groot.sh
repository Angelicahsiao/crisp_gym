#!/usr/bin/env bash
# Is this machine able to run GR00T N1.7 (3B + a Cosmos-Reason2-2B backbone)?
#
# WHY THIS EXISTS
#   Two things can stop a 3B model on a recent card, and neither reports itself
#   clearly. The torch build may have no compiled kernels for the GPU's compute
#   capability (Blackwell consumer cards are sm_120; a torch built before that
#   support landed has no sm_120 in get_arch_list()), and there may simply not
#   be enough VRAM. Checking the arch list alone is NOT decisive: if the build
#   embeds PTX for a lower arch, the driver can JIT forward onto a newer one --
#   it works, but the first kernel launch stalls for a long time. So this script
#   ends with an actual bf16 matmul, which is the only answer that settles it.
#
# STANDALONE ON PURPOSE
#   No crisp_gym / crisp_py / lerobot imports -- only torch. Copy it to a
#   training server with scp and run it, or git pull and run it in place.
#
# WHAT IT CHECKS
#   1. GPU    torch build vs the card's compute capability, VRAM, and a real
#             bf16 matmul on every visible device.
#   2. GR00T  whether GrootPolicy imports, and whether the config defaults that
#             matter for a relative-pose rot6d dataset are set. Skipped cleanly
#             when lerobot is not installed, so the same script is useful on a
#             deploy machine that only has torch.
#
# USAGE
#   bash scripts/check_gpu_groot.sh                 # print to stdout
#   bash scripts/check_gpu_groot.sh gpu_check.log   # also tee to a file
#
#   Run it with whatever python will do the training. If that is a container
#   with lerobot already installed, plain `bash scripts/check_gpu_groot.sh` is
#   right. Inside a pixi env, go through pixi so the env's torch is the one
#   tested:  pixi run -e humble-lerobot bash scripts/check_gpu_groot.sh
#
# EXIT CODES (so it can gate a script)
#   0  GO            every visible GPU ran the bf16 matmul
#   2  torch missing or not importable
#   3  no usable CUDA device
#   4  a GPU failed the matmul -- this torch build cannot drive it
#   5  GPU is fine but GrootPolicy will not import
#
#   VRAM shortfalls are reported as WARN, not failure: inference in bf16 needs
#   far less than fine-tuning, so a card too small to train on may still deploy.

set -u

LOG="${1:-}"

run() {
    PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" python3 - <<'PYEOF'
import sys

BAR = "-" * 68

def line(k, v):
    print(f"  {k:<18} {v}")

print(BAR)
print("  GR00T N1.7 GPU readiness")
print(BAR)

try:
    import torch
except Exception as exc:  # noqa: BLE001
    print(f"  FAIL  torch is not importable: {type(exc).__name__}: {exc}")
    print("        Run this inside the environment that will do the training,")
    print("        e.g. pixi run -e humble-lerobot bash scripts/check_gpu_groot.sh")
    sys.exit(2)

line("torch", torch.__version__)
line("built for CUDA", torch.version.cuda or "CPU-only build")
try:
    import platform
    line("python", platform.python_version())
except Exception:  # noqa: BLE001, S110
    pass

if not torch.cuda.is_available():
    print()
    print("  FAIL  torch.cuda.is_available() is False.")
    if torch.version.cuda is None:
        print("        This is a CPU-only torch build -- reinstall a CUDA build.")
    else:
        print("        The build has CUDA support, so this is usually a driver")
        print("        problem or no GPU visible. Check `nvidia-smi` and whether")
        print("        CUDA_VISIBLE_DEVICES excludes everything.")
    sys.exit(3)

arch_list = list(torch.cuda.get_arch_list())
line("kernel archs", " ".join(arch_list) or "(none)")
ptx = [int(a.split("_", 1)[1]) for a in arch_list if a.startswith("compute_")]
line("embedded PTX", ", ".join(f"compute_{p}" for p in sorted(ptx)) or "none")
line("visible GPUs", torch.cuda.device_count())

# GR00T N1.7: ~3B params. configuration_groot.py sets model_params_fp32=True,
# so every parameter is resident in fp32 whether trainable or not.
PARAMS_FP32_GB = 3.0 * 4        # ~12 GB, before any gradient or optimizer state
PARAMS_BF16_GB = 3.0 * 2        # ~6 GB, inference

print()
failures, warnings = [], []

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    cap = f"sm_{p.major}{p.minor}"
    vram = p.total_memory / 1e9
    print(f"  GPU {i}: {p.name}")
    line("  capability", cap)
    line("  VRAM", f"{vram:.1f} GB")

    if cap in arch_list:
        line("  kernels", f"native ({cap} present)")
    elif ptx and max(ptx) <= p.major * 10 + p.minor:
        line("  kernels", f"NO {cap} -- will JIT from compute_{max(ptx)} (slow first launch)")
        warnings.append(f"GPU {i} has no native {cap}; relying on PTX JIT")
    else:
        line("  kernels", f"NO {cap} and no usable PTX -- expect failure")

    try:
        import time
        torch.cuda.set_device(i)
        x = torch.randn(4096, 4096, device=f"cuda:{i}", dtype=torch.bfloat16)
        torch.cuda.synchronize(i)
        (x @ x).sum().item()          # discard: warm-up may include JIT
        torch.cuda.synchronize(i)
        t0 = time.perf_counter()
        for _ in range(10):
            y = x @ x
        torch.cuda.synchronize(i)
        dt = (time.perf_counter() - t0) / 10
        tflops = 2 * 4096 ** 3 / dt / 1e12
        line("  bf16 matmul", f"OK  ({dt * 1e3:.1f} ms, ~{tflops:.0f} TFLOP/s)")
        line("  result", f"{float(y.sum()):.4g}")
        del x, y
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        line("  bf16 matmul", f"FAILED: {type(exc).__name__}: {exc}")
        failures.append(f"GPU {i} ({p.name}, {cap}) cannot run a bf16 matmul")
        print()
        continue

    if vram < PARAMS_BF16_GB:
        warnings.append(
            f"GPU {i}: {vram:.0f} GB cannot even hold 3B params in bf16 (~{PARAMS_BF16_GB:.0f} GB)"
        )
    elif vram < PARAMS_FP32_GB:
        warnings.append(
            f"GPU {i}: {vram:.0f} GB holds 3B in bf16 (~{PARAMS_BF16_GB:.0f} GB) but not the "
            f"fp32 residency fine-tuning uses (~{PARAMS_FP32_GB:.0f} GB)"
        )
    else:
        headroom = vram - PARAMS_FP32_GB
        line("  train headroom", f"~{headroom:.0f} GB above the {PARAMS_FP32_GB:.0f} GB fp32 params")
        if headroom < 8:
            warnings.append(
                f"GPU {i}: only ~{headroom:.0f} GB above params for gradients, optimizer "
                "state and activations -- start at batch_size 1-2, not the default 32"
            )
    print()

print(BAR)
if failures:
    print("  VERDICT: NO-GO")
    for f in failures:
        print(f"    - {f}")
    print()
    print("    This torch build cannot drive the GPU. Install a build with")
    print("    kernels for the card's compute capability; nothing else above")
    print("    matters until that is fixed.")
    print(BAR)
    sys.exit(4)

print("  GPU: GO -- every visible GPU ran the bf16 matmul.")
for w in warnings:
    print(f"    WARN  {w}")
if not warnings:
    print("    No warnings.")
print()
print("  The fp32 figures assume configuration_groot.py's model_params_fp32=True.")
print("  The backbone is frozen by default (tune_llm=False, tune_visual=False), so")
print("  only the projector, diffusion model and VL LayerNorm carry gradients --")
print("  the real optimizer footprint is smaller than the parameter count")
print("  suggests, and worth measuring rather than estimating.")
print(BAR)

# ── stage 2: lerobot / GR00T, skipped cleanly when lerobot is absent ─────────
print()
print(BAR)
print("  lerobot / GR00T")
print(BAR)

try:
    import lerobot
except Exception as exc:  # noqa: BLE001
    print(f"  SKIP  lerobot not importable here ({type(exc).__name__}).")
    print("        Only the GPU section above applies on this machine.")
    print(BAR)
    sys.exit(0)

line("lerobot", getattr(lerobot, "__version__", "unknown"))

try:
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy  # noqa: F401
except Exception as exc:  # noqa: BLE001
    text = f"{exc}"
    print(f"  FAIL  GrootPolicy will not import: {type(exc).__name__}: {text}")
    if "CXXABI" in text or "libstdc++" in text:
        print()
        print("        This is the libstdc++ ordering problem, not a GR00T problem.")
        print("        torch is a PyPI wheel with no RPATH into the conda/pixi env, so")
        print("        it maps the SYSTEM libstdc++ first; cv2 then needs a newer")
        print("        CXXABI than that copy provides. Preload the env's copy:")
        print()
        print("          export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6")
        print()
        print("        (scripts/set_env.sh.example carries a guarded version.)")
    print(BAR)
    sys.exit(5)

print("  GrootPolicy imports OK -- no NVIDIA Isaac-GR00T package required.")
print()

# PreTrainedConfig.__post_init__ resolves an unset device and logs
# "Device 'None' is not available. Switching to 'cuda'." We only want the
# dataclass defaults, so silence that one logger rather than print noise.
import logging
logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
cfg = GrootConfig()
line("base model", cfg.base_model_path)
line("embodiment_tag", cfg.embodiment_tag)
line("chunk_size", f"{cfg.chunk_size}  (n_action_steps {cfg.n_action_steps}, n_obs_steps {cfg.n_obs_steps})")
line("denoise steps", cfg.num_inference_timesteps if cfg.num_inference_timesteps is not None
     else "None (checkpoint default: 4)")
line("batch_size", f"{cfg.batch_size}   <- a datacenter default; drop it hard on one card")

print()
print("  Defaults that are WRONG for a relative-pose rot6d dataset:")
ok = True
if not cfg.use_relative_actions:
    ok = False
    print("    use_relative_actions       False  -> pass --policy.use_relative_actions=true")
    print("      The base N1.7 checkpoint declares use_relative_action=True in its")
    print("      processor_kwargs, and the DECODE step follows the checkpoint while")
    print("      relative STATS are computed only when this config flag is set. Leave")
    print("      it False and the model decodes relative against absolute stats.")
if not cfg.relative_exclude_joints:
    ok = False
    print("    relative_exclude_joints    []     -> pass --policy.relative_exclude_joints='[\"gripper\"]'")
    print("      Empty means EVERY action dim is treated as relative, gripper included.")
if ok:
    print("    none -- this config is already set for it.")

print()
print("  Also: feed GR00T the ABSOLUTE dataset. It takes absolute action + state")
print("  and does the SE(3) relative conversion itself (GrootN17PackInputsStep")
print("  caches the raw state; GrootN17ActionDecodeStep composes back). Running")
print("  crisp_gym's relative conversion first would double-convert.")
print()
print("  If training logs 'using the generic RelativeActionsProcessorStep")
print("  fallback', STOP: that step subtracts, which is wrong for rot6d.")
print(BAR)
PYEOF
}

if [ -n "$LOG" ]; then
    run 2>&1 | tee "$LOG"
    status=${PIPESTATUS[0]}
    echo "(saved to $LOG)"
    exit "$status"
fi
run
