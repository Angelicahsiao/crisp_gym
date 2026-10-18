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

The client half lives in crisp_gym (`crisp_gym/policy/remote_policy.py`) and
speaks **protocol v1**, defined below. The wire format itself is a single
shared module, `crisp_gym/policy/remote_protocol.py`, which BOTH sides import
so they cannot drift — it needs only numpy and msgpack, no torch, no ROS.
`tests/test_remote_protocol.py` runs the real client against a real server
over a loopback socket and pins every rule in §2.

**v1 is not compatible with the earlier untagged format**, and deliberately
so: see "Why v1" below.

**Version policy (canonical deployment mode).** Training AND rollout both run
only on the GPU machine; crisp_gym acts purely as the websocket client. The
GPU machine may run any lerobot version (train + serve with the SAME version
so the server's transforms/normalization match the checkpoint). The robot
machine's lerobot 0.4.4 is used only for RECORDING datasets — never for
inference. `deploy_policy.py` (local, in-process inference) is legacy: only
valid for checkpoints trained with the robot machine's own lerobot version;
do not use it with checkpoints from a newer lerobot. When upgrading lerobot
on the GPU, re-verify the training wrapper's seams (lerobot_train import,
make_dataset patch, and that recompute_relative_stats' patched meta.stats are
actually used by normalization) with a --steps=100 smoke run.

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

## 2. Wire protocol — v1

Persistent websocket, binary frames, msgpack. Arrays travel as a tagged map
carrying dtype and shape, so they arrive bit-exact rather than as nested
lists. Build and parse frames with `remote_protocol.py`; do not hand-roll
them.

### Why v1

The earlier format was an untagged request/reply: the client sent
`{"type": "infer", "obs": ...}` and read `result["actions"]` off whatever came
back next. Nothing said **which request a reply belonged to**, and that is not
cosmetic. After an infer timeout the late reply is still queued, so the next
`recv()` returns the PREVIOUS observation's chunk — which the client then
composes against the base pose captured for the NEW observation. That is
exactly the error `RemotePolicy` caches `_chunk_base` to avoid; its own
comment puts the cost at "~doubles velocity, tens of mm / tens of degrees by
step 8". One dropped deadline was enough, silently.

Two more things the old format described but never did: `verify_handshake`
existed in three config files and in this document and in **zero lines of
Python**, and the contract configs could not be loaded at all —
`RemotePolicy.__init__` took five scalars while the YAMLs declared `server`,
`observation`, `action`, `normalization`, `reset_on_episode_start` and
`verify_handshake`, so `make_policy` raised `TypeError`.

### Every frame

Three mandatory fields. An unknown `v`, a missing field, or a frame arriving
from the wrong direction is a `ProtocolError` — refuse, do not guess.

| field | type | meaning |
|---|---|---|
| `v` | int | protocol version; `1` |
| `t` | str | frame type |
| `id` | uint64 | client-monotonic request id, **echoed by the server on every reply** |

### Client → server

| `t` | fields | notes |
|---|---|---|
| `hello` | `contract`, `contract_digest` | first frame on every connection |
| `reset` | `episode` | episode start; must be acked before any `infer` |
| `infer` | `obs`, `task`, `obs_t` | `obs_t` = client monotonic clock at capture |
| `bye` | — | clean close |

### Server → client

| `t` | fields | notes |
|---|---|---|
| `hello_ok` | `model_id`, `observation`, `action`, `chunk_size`, `normalization`, `image`, `contract`, `contract_digest` | derived from the checkpoint, never hardcoded |
| `reset_ok` | — | echoes the `reset` id |
| `actions` | `actions` (chunk, dim) float32, `server_t` | echoes the `infer` id; raw units |
| `error` | `code`, `message`, `fatal` | `fatal` is derived from `code`, so the two cannot disagree |

### Error codes

| code | fatal | meaning |
|---|---|---|
| `bad_version` | yes | peer speaks another protocol version |
| `bad_frame` | yes | malformed or wrong-direction frame |
| `contract_mismatch` | yes | the server rejected the client's contract |
| `not_ready` | no | `infer` arrived before `reset` was acked |
| `stale_obs` | no | the observation queued too long; resend a fresh one |
| `infer_failed` | no | the model raised on this observation |

### The rules both sides must obey

1. **One outstanding id.** The client discards any `actions` whose `id` is not
   the one it is waiting for, and never executes a chunk whose id does not
   match the `_chunk_base` captured with it.
2. **The reset barrier.** No `infer` until `reset_ok`. The client sends the
   reset itself if it finds it has not been acked, so a chunk computed from
   the previous episode's state cannot be executed in this one.
3. **The handshake decides.** The client sends its contract and a digest; the
   server answers with its own, derived from the checkpoint. On any difference
   the client **refuses to start** and prints both sides field by field. Only
   fields that change what the robot DOES are compared (layouts, widths,
   `pose_repr`, `normalization`) — timeouts and hostnames are local choices.
   `verify_handshake: false` downgrades this to a warning. Leave it on: a
   wrong rotation layout or gripper scale does not crash, it moves the robot
   wrongly.
4. **The server owns the image encoding.** `hello_ok.image` carries
   `{encoding: raw|jpeg, quality, shape}` and the client obeys. Lossy encoding
   changes what the model sees relative to training, so the side that holds
   the checkpoint decides it. Raw 800x1280x3 is 3.07 MB a frame — 46 MB/s at
   15 Hz, which saturates 1 GbE; that is when `jpeg` earns its place.
5. **Staleness is measured on the client's clock.** The two machines' monotonic
   clocks are not comparable, so the server compares `obs_t` against the
   previous `obs_t` on the same connection, never against its own clock.
6. **A dropped socket is reopened with backoff**, and the handshake and reset
   are redone from scratch — a reconnected server has no memory of the episode.

### Division of labor (fixed by design, do not move it)

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

All of these are implemented in `WebsocketPolicyClient` and pinned by
`tests/test_remote_protocol.py`.

| Event | Behavior |
|---|---|
| handshake mismatch | refuse to start, printing both digests and every differing field |
| infer timeout | raise to `RemotePolicy`, which holds the last commanded pose; the id stays outstanding so the late reply is dropped, not executed |
| late reply after a timeout | discarded with a warning — never returned for the following request |
| socket drop | reconnect with exponential backoff, then re-`hello` and re-`reset` |
| `infer` before `reset_ok` | the client sends the reset itself and waits for the ack |
| retryable server `error` (`stale_obs`, `infer_failed`, `not_ready`) | raise for this tick, hold pose, socket stays up |
| fatal server `error` (`bad_version`, `bad_frame`, `contract_mismatch`) | stop the episode, surface the code and message |
| a v0 text frame | `ProtocolError` saying a v0 server cannot talk to a v1 client |

## 6. Promoted-state models

If the model was trained on a dataset with promoted extras (usage 7 in
USAGE.md — e.g. `observation.state.joints` as a policy input), the contract
config must list those keys under `observation.keys` too (source
`robot.joint_positions`, shape [N]) — and such a model can only run on the
robot, never from handheld-style observation sources.
