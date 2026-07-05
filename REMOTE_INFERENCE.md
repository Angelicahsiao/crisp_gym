# Remote Inference — WebSocket Policy Client

How crisp_gym runs a policy that lives on another machine (the GPU/training
PC), so the robot machine stays on ROS 2 Humble / Python 3.11 while the model
runs on any Python with torch + lerobot.

```
┌─ Robot machine (ROS 2, Py 3.11) ────────┐      ┌─ GPU machine (any Python) ──┐
│ crisp_gym env  (obs: pose/gripper/image)│      │ lerobot 0.4.4 + torch       │
│ RemotePolicy   @register_policy("remote")      │ policy server (YOUR side)   │
│   collects obs history ── msgpack/ws ───┼──────┼→ relative conversion        │
│   composes T_cmd = T_obs_tcp @ T_rel  ◄─┼──────┼─ normalize → select_action  │
│   → env.step()                          │      │ → action chunk (raw units)  │
└─────────────────────────────────────────┘      └─────────────────────────────┘
```

The client half lives in crisp_gym; the **server is not part of this repo**
(you run it on the GPU machine). This document defines the contract both
sides must satisfy. Status: the client (`crisp_gym/policy/remote_policy.py`)
is the top roadmap item in HANDOFF.md §4; the config format below is final.

---

## 1. The contract config

Everything the client needs to know — what data the model wants and what its
output means — lives in one YAML:

**`crisp_gym/config/policy/remote_policy_example.yaml`** (fully documented;
copy it per model). It answers three questions:

| Section | Question | Key fields |
|---|---|---|
| `server` | how to connect | host/port, protocol, `infer_timeout_s` (on timeout: hold pose) |
| `observation` | what the model wants | keys + shapes + dtypes, `n_obs_steps` history window, pose `layout`/`frame`, gripper `reference_width`, `task_required` |
| `action` | what the output means | `dim`, `layout`, **`pose_repr`** (relative/abs/delta), **`compose_base`** (obs_time!), `orthogonalize_rot6d`, `chunk_size`/`n_action_steps`/`dt`, gripper semantics |

Provenance chain — the same facts travel through the whole pipeline, and the
contract config is just their deploy-time form:

```
recording  → dataset meta/record_config.json   (what was recorded)
training   → checkpoint pose_repr.json          (what the model learned on)
deployment → config/policy/remote_*.yaml        (what the client sends/receives)
```

On connect the server sends its own metadata (derived from the checkpoint);
the client verifies it field-by-field against the YAML and **refuses to run
on mismatch**. A wrong rotation layout or gripper scale does not crash — it
moves the robot wrongly. Never disable `verify_handshake`.

## 2. Wire protocol (msgpack-numpy over websocket, openpi-style)

Persistent websocket; binary frames are msgpack with the msgpack-numpy
extension (arrays travel with dtype+shape intact). Three message types:

**Client → server**
```python
{"type": "reset"}                          # episode start: clear policy state
{"type": "infer",
 "obs": {
   "observation.state.cartesian": np.ndarray (n_obs_steps, 9),  # ABSOLUTE
   "observation.state.gripper":   np.ndarray (n_obs_steps, 1),
   "observation.images.primary":  np.ndarray (n_obs_steps, 224, 224, 3) uint8,
 },
 "task": "pick the lego block",
 "obs_timestamp": float}                   # monotonic time of capture
```

**Server → client**
```python
# on connect (handshake):
{"type": "meta", "model_id": ..., "observation": {...}, "action": {...},
 "normalization": "server_side", "protocol_version": "1"}
# per infer:
{"type": "actions", "actions": np.ndarray (chunk_size, 10)}   # raw units
{"type": "error", "message": str}
```

Division of labor (fixed by design, do not move it):
- **Client sends absolute poses / raw units.** The server owns the relative
  conversion — it must byte-match training
  (`RelativePoseDataset.convert_item` in `scripts/lerobot_relative_pose.py`)
  including `rot_wrt_start` generation if the policy uses it.
- **Server owns normalization** (stats live with the checkpoint). The client
  never sees normalized values.
- **Client owns command composition** (it is the only side that knows the
  robot's live pose): for `pose_repr: relative`,

  ```
  T_cmd = T_base_tcp(at obs capture) @ GramSchmidt(action[3:9]), pos = action[0:3]
  gripper_cmd = clip(action[9], 0, 1) * reference_width / device_max_width
  ```

## 3. Control-loop timing

With `chunk_size: 16`, `n_action_steps: 8`, `dt: 1/15`:

1. tick: capture obs (+ remember the TCP pose at this instant), append to the
   n_obs_steps history buffer;
2. when the executed queue drops below `chunk_size - n_action_steps` steps,
   send `infer` with the current history;
3. execute queued actions at `dt` spacing, each composed against the TCP pose
   remembered **at its observation's capture time** (`compose_base: obs_time`)
   — the robot moves during the round-trip, and composing against the arrival
   pose would double-count that motion;
4. on `infer_timeout_s`: hold the last commanded pose and keep retrying —
   never block the control loop, never extrapolate.

Chunking hides network latency: with ~100 ms round-trips and 8 executed steps
(~533 ms) per request, inference happens fully in the background.

## 4. Server-side checklist (for whoever builds the server)

The server is out of scope here, but to satisfy the contract it must:

1. Load the checkpoint (`TrainPipelineConfig.from_pretrained`, policy class
   `from_pretrained`), warm up before accepting connections.
2. Re-implement (or import) the training-time transforms verbatim:
   relative conversion wrt the last obs frame, `rot_wrt_start` (noise OFF at
   inference), pre/post processors, normalization with the recomputed
   relative stats.
3. Answer the handshake from ground truth, not hardcoding: derive obs/action
   spec from the checkpoint's `train_config.json`, the stamped
   `pose_repr.json`, and the dataset's `record_config.json`.
4. Return DEnormalized, raw-unit action chunks.
5. Handle `reset` by clearing `policy.reset()` / obs queues.

## 5. Failure modes the client must handle

| Event | Behavior |
|---|---|
| handshake mismatch | refuse to start, print both sides' contract |
| infer timeout | hold last commanded pose, retry next tick |
| socket drop | hold pose, reconnect with backoff, send `reset` on reconnect |
| stale obs source (camera/pose older than 2× period) | skip tick + warn |
| server `error` message | stop episode, surface message |

## 6. Promoted-state models

If the model was trained on a dataset with promoted extras (usage 7 in
USAGE.md — e.g. `observation.state.joints` as a policy input), the contract
config must list those keys under `observation.keys` too (source
`robot.joint_positions`, shape [N]) — and such a model can only run on the
robot, never from handheld-style observation sources.
