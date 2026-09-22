"""Camera-key contract for the local deployment path (RelativeLerobotPolicy).

Two failures produced the SAME bare `KeyError: 'observation.images.oakw_cam'`:

  1. GR00T. `_prepare` runs the lerobot preprocessor and THEN stacks the
     per-camera keys into the single `observation.images` tensor that
     diffusion/ACT/VQ-BeT consume. GR00T's pack step packs every camera into
     one `video` tensor and POPS every `observation.images.*` key, so the
     stack blew up on a key the checkpoint legitimately declares -- every
     frame, for every GR00T checkpoint.
  2. A deploy-env `camera_name` that disagrees with the training dataset.
     `build_obs_frame` only forwards an image whose (renamed) key is one the
     checkpoint declared, so a typo silently yields a frame without it.

The fix makes (1) a no-op and (2) a named error. These tests pin both, plus the
invariant that the diffusion/ACT path is unchanged: where the image keys
survive the preprocessor, the stack still happens, in `image_features` order.

`_prepare` is a closure inside `inference_worker`, so it is exec'd out of the
shipped source rather than imported -- the assertions stay on the real code.

Run: python -m pytest tests/test_deploy_image_keys.py -v
"""

import ast
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SOURCE = REPO / "crisp_gym" / "policy" / "relative_lerobot_policy.py"
OBS_IMAGES = "observation.images"

# ── torch stand-in: _prepare only needs stack, and numpy's semantics for it
# are identical. Deliberately NOT registered in sys.modules -- it is handed to
# the exec'd closure directly, so this file has no import side effects on the
# rest of the suite (tests/test_pose_math.py installs its own, richer stub).
try:  # pragma: no cover - the robot/training envs have the real thing
    import torch as _torch
    _stack = _torch.stack
except ImportError:
    _stack = lambda seq, dim=0: np.stack([np.asarray(x) for x in seq], axis=dim)  # noqa: E731

torch = types.SimpleNamespace(stack=_stack)


def _load_prepare(preprocessor, image_features):
    """Exec the real `_prepare` closure out of inference_worker()."""
    tree = ast.parse(SOURCE.read_text())
    worker = next(
        fn for fn in tree.body
        if isinstance(fn, ast.FunctionDef) and fn.name == "inference_worker"
    )
    fn = next(
        node for node in ast.walk(worker)
        if isinstance(node, ast.FunctionDef) and node.name == "_prepare"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    ns: dict = {
        # the frame is already array-valued here; the real permute is irrelevant
        # to which KEYS exist, which is all these tests are about
        "numpy_obs_to_torch": dict,
        "preprocessor": preprocessor,
        "image_features": image_features,
        "torch": torch,
        "OBS_IMAGES": OBS_IMAGES,
    }
    exec(compile(module, str(SOURCE), "exec"), ns)  # noqa: S102
    return ns["_prepare"]


def _img(h: int = 4, w: int = 6) -> np.ndarray:
    return np.zeros((3, h, w), np.uint8)


def _groot_preprocessor(batch: dict) -> dict:
    """GrootN17PackInputsStep: cameras -> one `video`, image keys popped.

    Mirrors lerobot policies/groot/processor_groot.py (`obs["video"] = video`
    then `obs.pop(k)` for every key starting with OBS_IMAGES).
    """
    out = dict(batch)
    cams = sorted(k for k in out if k.startswith(OBS_IMAGES))
    if cams:
        out["video"] = np.stack([out[k] for k in cams], axis=0)
        for k in cams:
            out.pop(k, None)
    return out


def _passthrough_preprocessor(batch: dict) -> dict:
    """Diffusion/ACT: normalizes in place, keeps every image key."""
    return dict(batch)


# ── 1. the GR00T regression ───────────────────────────────────────────────────

def test_groot_preprocessor_consuming_the_image_keys_is_not_an_error():
    keys = [f"{OBS_IMAGES}.oakw_cam"]
    prepare = _load_prepare(_groot_preprocessor, keys)
    out = prepare({keys[0]: _img(), "observation.state": np.zeros(10, np.float32)})
    assert "video" in out, "GR00T's packed tensor must survive"
    assert OBS_IMAGES not in out, (
        "stacking into observation.images for GR00T is meaningless -- the model "
        "reads `video` -- and the keys it would stack are gone"
    )


def test_the_groot_batch_really_loses_the_declared_key():
    """Why the guard is needed: the unguarded stack has nothing to stack."""
    key = f"{OBS_IMAGES}.oakw_cam"
    batch = _groot_preprocessor({key: _img()})
    assert key not in batch
    with pytest.raises(KeyError):
        torch.stack([batch[k] for k in [key]], dim=-4)


def test_groot_with_several_cameras_also_survives():
    keys = [f"{OBS_IMAGES}.oakw_cam", f"{OBS_IMAGES}.wrist"]
    prepare = _load_prepare(_groot_preprocessor, keys)
    out = prepare({k: _img() for k in keys})
    assert OBS_IMAGES not in out
    assert out["video"].shape[0] == 2


# ── 2. the diffusion/ACT path must be unchanged ───────────────────────────────

def test_diffusion_path_still_stacks_into_observation_images():
    keys = [f"{OBS_IMAGES}.oakw_cam"]
    prepare = _load_prepare(_passthrough_preprocessor, keys)
    out = prepare({keys[0]: _img(), "observation.state": np.zeros(10, np.float32)})
    assert OBS_IMAGES in out, "guard must not skip a policy that keeps its keys"
    assert np.asarray(out[OBS_IMAGES]).shape == (1, 3, 4, 6)
    assert keys[0] in out, "the per-camera key stays (the model re-stacks it)"


def test_diffusion_multi_camera_stacks_in_image_features_order():
    """The stack order IS the checkpoint's camera order -- swapping it feeds the
    wrist frames into the base-camera encoder slot."""
    keys = [f"{OBS_IMAGES}.oakw_cam", f"{OBS_IMAGES}.wrist"]
    prepare = _load_prepare(_passthrough_preprocessor, keys)
    frame = {keys[0]: np.full((3, 4, 6), 1, np.uint8),
             keys[1]: np.full((3, 4, 6), 2, np.uint8)}
    out = np.asarray(prepare(frame)[OBS_IMAGES])
    assert out.shape == (2, 3, 4, 6)
    assert out[0].max() == 1 and out[1].max() == 2


def test_no_preprocessor_still_stacks():
    keys = [f"{OBS_IMAGES}.oakw_cam"]
    prepare = _load_prepare(None, keys)
    out = prepare({keys[0]: _img()})
    assert OBS_IMAGES in out


def test_state_only_checkpoint_adds_no_image_key():
    prepare = _load_prepare(_passthrough_preprocessor, [])
    out = prepare({"observation.state": np.zeros(10, np.float32)})
    assert OBS_IMAGES not in out


# ── 3. the camera-name preflight ──────────────────────────────────────────────

def _load_policy_module():
    """Load relative_lerobot_policy with the crisp_gym package chain stubbed
    (the real chain reaches rclpy). Same shape as tests/test_relative_deploy.py,
    kept local so the two files do not share import-order side effects.
    """
    import importlib.util

    pkg = types.ModuleType("crisp_gym")
    pol = types.ModuleType("crisp_gym.policy")
    polpol = types.ModuleType("crisp_gym.policy.policy")
    polpol.Policy = object
    polpol.Action = object
    polpol.Observation = dict
    polpol.register_policy = lambda name: (lambda cls: cls)
    saved = {k: sys.modules.get(k) for k in
             ("crisp_gym", "crisp_gym.policy", "crisp_gym.policy.policy")}
    sys.modules.update(
        {"crisp_gym": pkg, "crisp_gym.policy": pol, "crisp_gym.policy.policy": polpol}
    )
    try:
        spec = importlib.util.spec_from_file_location("rlp_imgkeys", SOURCE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)


rlp = _load_policy_module()


def _policy(image_keys, rename_map=None):
    obj = object.__new__(rlp.RelativeLerobotPolicy)  # __init__ needs a live env
    obj.meta = {"image_keys": image_keys, "rename_map": rename_map or {}}
    obj._image_keys_checked = False
    return obj


def test_missing_camera_names_the_mismatch():
    pol = _policy([f"{OBS_IMAGES}.oakw_cam"])
    frame = {"observation.state": np.zeros(10), f"{OBS_IMAGES}.oakw": _img()}
    with pytest.raises(ValueError) as exc:
        pol._verify_image_keys(frame)
    msg = str(exc.value)
    assert f"{OBS_IMAGES}.oakw_cam" in msg, "must name what the checkpoint wants"
    assert f"{OBS_IMAGES}.oakw" in msg, "must name what the env produced"
    assert "camera_name" in msg, "must name the knob that fixes it"


def test_matching_camera_passes():
    key = f"{OBS_IMAGES}.oakw_cam"
    _policy([key])._verify_image_keys({key: _img()})


def test_check_runs_once_like_the_state_dim_check():
    key = f"{OBS_IMAGES}.oakw_cam"
    pol = _policy([key])
    pol._verify_image_keys({key: _img()})
    pol._verify_image_keys({})  # a dropped frame later must not kill the rollout


def test_state_only_checkpoint_is_not_flagged():
    _policy([])._verify_image_keys({"observation.state": np.zeros(10)})
    _policy(None)._verify_image_keys({"observation.state": np.zeros(10)})


def test_rename_map_is_reported_when_set():
    pol = _policy([f"{OBS_IMAGES}.camera1"],
                  rename_map={f"{OBS_IMAGES}.oakw_cam": f"{OBS_IMAGES}.camera1"})
    with pytest.raises(ValueError) as exc:
        pol._verify_image_keys({f"{OBS_IMAGES}.oakw_cam": _img()})
    assert "rename_map" in str(exc.value)


def test_a_padded_slot_counts_as_produced():
    """pad_missing_images fills the key in build_obs_frame, so the preflight
    (which runs after it) must accept the padded frame."""
    keys = [f"{OBS_IMAGES}.camera1", f"{OBS_IMAGES}.camera2"]
    _policy(keys)._verify_image_keys({keys[0]: _img(), keys[1]: _img()})


def test_the_preflight_is_actually_wired_into_the_obs_loop():
    """A checker nothing calls is not a checker. make_data_fn must run it on
    the built frame, beside the observation.state width check."""
    tree = ast.parse(SOURCE.read_text())
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "make_data_fn"
    )
    called = {
        node.func.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_verify_image_keys" in called, (
        "make_data_fn does not call _verify_image_keys, so a camera-name "
        "mismatch still reaches the worker as a bare KeyError"
    )
    assert "_verify_state_dim" in called, "the state-width check went missing"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} deploy image-key tests passed.")
