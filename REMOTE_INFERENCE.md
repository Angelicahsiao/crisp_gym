# Remote Inference — WebSocket Policy Client

How crisp_gym runs a policy that lives on another machine (the GPU/training
PC), so the robot machine stays on ROS 2 Humble / Python 3.11 while the model
runs on any Python with torch + lerobot.

```
┌─ Robot machine (ROS 2) ─────────────────┐      ┌─ GPU machine (no ROS) ──────┐
│ crisp_gym env  (obs: pose/gripper/image)│      │ lerobot + torch             │
│ robot-side policy (no torch needed)     │      │ scripts/policy_server.py    │
│   collects obs history ── msgpack/ws ───┼──────┼→ relative conversion        │
│   composes T_cmd = T_obs_tcp @ T_rel  ◄─┼──────┼─ normalize → action chunk   │
│   → env.step()                          │      │   (raw units)               │
└─────────────────────────────────────────┘      └─────────────────────────────┘
```

**Both halves exist.** The server is `scripts/policy_server.py` (§4): it puts
this protocol in front of the same `inference_worker` that `deploy_policy.py`
runs in a local subprocess, unmodified, so a checkpoint that deploys
in-process is served with the same numbers. The robot side has the protocol
client and a pipe-compatible transport; which policy class uses them is
described in §4.5.

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
| `infer` | `obs`, `task`, `obs_t` | `obs` is the observation WINDOW: a list of frames, oldest first (one frame alone is also accepted). `obs_t` = client monotonic clock at capture |
| `bye` | — | clean close |

### Server → client

| `t` | fields | notes |
|---|---|---|
| `hello_ok` | `model_id`, `observation`, `action`, `chunk_size`, `normalization`, `image`, `contract`, `contract_digest`, `meta` (optional) | derived from the checkpoint, never hardcoded. `meta` is the inference worker's own description (n_obs_steps, chunk length, image keys and shapes, state width and mode, rename map) |
| `reset_ok` | — | echoes the `reset` id |
| `actions` | `actions` (chunk, dim) float32, `server_t`, `timing` (optional) | echoes the `infer` id; raw units. `timing` = `{decode_ms, infer_ms}` spent on the server |
| `error` | `code`, `message`, `fatal` | `fatal` is derived from `code`, so the two cannot disagree |

### Error codes

| code | fatal | meaning |
|---|---|---|
| `bad_version` | yes | peer speaks another protocol version |
| `bad_frame` | yes | malformed or wrong-direction frame |
| `contract_mismatch` | yes | the server rejected the client's contract |
| `not_ready` | no | `infer` arrived before `reset` was acked |
| `stale_obs` | no | a newer request was already waiting behind this one, or its `obs_t` is older than one already served; resend a fresh observation |
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
7. **The client sends the observation window, not one frame.** Only the robot
   side sees every control tick. It asks for a chunk once per `n_action_steps`
   ticks, so a server that kept the history itself would pair frames
   `n_action_steps` ticks apart — for a 2-step policy that is not the motion it
   was trained on. `remote_protocol.encode_window` / `decode_window`.

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

## 4. The server — `scripts/policy_server.py`

Runs on the GPU machine. It needs no ROS and no install of crisp_gym — a clone
of this repository is enough (the script puts it on `sys.path`).

```bash
# once, in the Python that has torch + the lerobot version the checkpoint was trained with
pip install websockets msgpack rich

python scripts/policy_server.py \
    --path outputs/train/RUN/checkpoints/125000/pretrained_model \
    --host 0.0.0.0 --port 8765
```

It checks its own requirements first and names what is missing, loads the
checkpoint, runs two warm-up inferences (the second is the time a robot will
wait), then listens. `curl http://GPU_HOST:8765/healthz` answers `OK` once it
is ready. Ctrl-C or SIGTERM stops it and releases the GPU.

| option | meaning |
|---|---|
| `--path` | the checkpoint's `pretrained_model` directory (required) |
| `--host`, `--port` | where to listen; default `0.0.0.0:8765` |
| `--image-encoding raw\|jpeg`, `--jpeg-quality` | what the robot side sends (rule 4). Default `raw` |
| `--state-input` | `auto` (reads `pose_repr.json`), `absolute`, `relative`, `relative_wrt_start` |
| `--num-inference-steps`, `--scheduler ddim\|ddpm` | denoising speed, diffusion policy only (they do not reach `multi_task_dit` yet) |
| `--set KEY=VALUE` | lerobot policy-config override, repeatable |
| `--no-warmup` | skip the warm-up inferences |

**What it is.** A thin protocol front (`PolicyServer`) and the inference worker
behind a pipe:

- The worker is `crisp_gym.policy.relative_lerobot_policy.inference_worker` —
  the function `deploy_policy.py` already runs in a local subprocess, not a
  copy of it. Checkpoint loading, the relative / wrt-start conversion (noise
  off), uint8 -> [0, 1] image scaling, normalization and the per-policy-family
  queue handling all stay in that one place. Any lerobot policy it can run
  locally (diffusion, multi_task_dit, act, groot, ...) it can serve.
- The handshake is derived, not typed in: the worker's metadata, the action
  width from `config.json`, the pose representation and gripper semantics from
  `pose_repr.json` / `action_repr.json` next to the checkpoint.
- One robot at a time. A second connection is refused with `not_ready`.
- A request that already has a newer one waiting behind it is answered
  `stale_obs` without running the model: the client keeps one request
  outstanding, so the older one was given up on.
- If the worker process dies, the server stops listening and exits with
  status 1 instead of answering every later request with an error.

**Checked** (`tests/test_policy_server.py`, real client and server on a real
loopback socket, only the model replaced), and with a real lerobot 0.6.1
diffusion image checkpoint on CPU:

| check | result |
|---|---|
| chunk through the server vs. the same worker over the local pipe, same noise seed, `raw` images | identical (max difference 0.0 over 8 windows) |
| the same with `jpeg` quality 90 | differs by <= 1e-5 on that small synthetic model; not measured on a real one |
| robot side run from a Python with no torch and no lerobot | works; 37 ms round trip, 33 ms of it inference |
| protocol cost alone for a real-size window (2 frames of 800x1280x3 = 6.1 MB), instant model, loopback | `raw` 22 ms; `jpeg` q90 52 ms (0.37 MB) |

On a wired 1 GbE link, 6.1 MB adds about 50 ms of wire time to `raw`, so the
two encodings cost about the same there and `raw` is the exact one; on a faster
link `raw` is simply quicker. **Not measured: the round trip on the real GPU
with a real checkpoint and the real cable** — the server logs decode and
inference time per request, and the robot side sees the total.

**Not handled:** TLS and authentication (it is meant for a private wired link —
do not expose the port), more than one robot, and a client that vanishes
without closing its socket (power or cable): the server notices through
websocket pings, up to ~40 s, and refuses a new client until then.

### 4.5 The robot side

Two pieces exist, both free of torch, lerobot and ROS:

- `WebsocketPolicyClient` (`crisp_gym/policy/remote_policy.py`) — the protocol
  client of §2 and §5.
- `RemoteWorkerConnection` (`crisp_gym/policy/remote_worker.py`) — the same
  client behind the four calls of the multiprocessing pipe that
  `RelativeLerobotPolicy` uses to talk to its local worker (`send`, `recv`,
  `poll`, and `send(None)` to stop). A timeout or a dropped socket becomes the
  pipe's `("error", message)` reply, on which that policy already holds pose.
  `tests/test_policy_server.py` runs the unmodified `RelativeLerobotPolicy`
  control loop on it — synchronous, with prefetch, and with a late answer —
  and every commanded pose matches ground truth.

**What is not wired yet** is which policy class a deploy config selects to use
them:

- `RemotePolicy` (`name: remote_policy`) connects and runs, but still sends
  only the current frame (not the window), does no gripper unit conversion,
  has no state/image checks and blocks the control loop during inference.
- `RelativeLerobotPolicy` has all of those, but always starts a local worker.

Pointing `RelativeLerobotPolicy` at `RemoteWorkerConnection` instead of its
pipe is a small change (the constructor's worker start-up, `action_repr: auto`
reading the handshake instead of a local file, and `shutdown`). It has not
been made: it changes the class the robot is deployed with today, and is the
owner's call.

Two consequences of a network that a local pipe does not have:

- `infer_timeout` must be longer than the slowest inference plus transfer. A
  shorter one never recovers: each retry is answered after its own deadline.
- After a dropped socket the server is reset, so a `relative_wrt_start`
  checkpoint measures its start rotation from the next observation, not from
  the episode start. It is logged; restart the episode.

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
