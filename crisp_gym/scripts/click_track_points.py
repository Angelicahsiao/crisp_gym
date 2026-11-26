#!/usr/bin/env python3
"""Click the query points a point tracker should follow, once per episode.

RoboTracker (github.com/allenai/robotrack) tracks the points it is given; it
does not find objects. To train a policy that also predicts where the button
and the door move, every episode needs the SAME physical points as queries —
for example two points on the button and two on the door. This tool shows one
camera frame of an episode in a browser and records where you click each named
point, and on which frame of the episode.

It runs where the dataset is (it decodes the video with lerobot) and serves a
page on that machine. From your own computer, open it through an SSH tunnel.

On the GPU machine, in the Python that has lerobot:

    python crisp_gym/scripts/click_track_points.py \\
        --root /workspace/datasets/umi_electricbox/lerobot \\
        --camera observation.images.oakw_cam \\
        --points button_a button_b door_a door_b \\
        --out /workspace/datasets/umi_electricbox/track_queries.json

On your computer: forward port 8899 when you ssh in (ssh -L 8899:localhost:8899
...), then open http://localhost:8899. In the page: click each named point,
press X for a point that cannot be seen in this episode, Enter to save. Arrow
keys step frames (Shift: 10 at a time). Every saved episode is written to
--out at once, and a restart continues with the first unlabelled episode.

OUTPUT (--out, JSON)

    {"version": 1, "camera": ..., "image_size": [H, W], "points": [names],
     "episodes": {"12": {"button_a": {"frame": 0, "x": 640.0, "y": 410.5},
                         "door_b": null, ...}, ...}}

    frame   index of the frame within the episode (0 = its first frame): the
            point's query frame for the tracker.
    x, y    pixels of the decoded camera frame, x to the right, y down, pixel
            centres at whole numbers — the convention robotrack's track()
            reads a query with (it rounds a query to the pixel it samples).
    null    the point is not visible anywhere in this episode.

Click the same physical spot in every episode, on a part with texture (an
edge, a corner, a printed mark): a tracker cannot hold a point on a plain
surface. RoboTracker tracks FORWARD from the query frame, so the earliest
frame where the point is clearly visible is the best one to click on.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import logging
import math
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

FORMAT_VERSION = 1
COORDINATES = (
    "pixels of the decoded camera frame; x right, y down; pixel centres at "
    "whole numbers (robotrack track() convention)"
)
logger = logging.getLogger("click_track_points")


class LabelError(ValueError):
    """A label that cannot be saved as given. The message is shown on the page."""


# ── the labels file ───────────────────────────────────────────────────────────

class LabelStore:
    """The query points of every episode, kept in one JSON file.

    Rewritten after every saved episode, atomically (a crash leaves the previous
    complete file), so an interrupted session loses nothing that was saved.

    Args:
        path: the JSON file. If it exists it is resumed — but only when it was
            made for the same camera, point names and image size; anything else
            is refused rather than mixed into it.
        dataset_root: recorded in the file, for reference.
        camera: the dataset image key the points were clicked on.
        points: the point names, in the order they are clicked.
        image_size: (H, W) of the decoded frames.
        episode_frames: {episode index: number of frames}.
    """

    def __init__(self, path, *, dataset_root, camera: str, points, image_size,
                 episode_frames: dict):
        self.path = Path(path)
        self.points = [str(p) for p in points]
        if not self.points:
            raise LabelError("give at least one point name")
        if len(set(self.points)) != len(self.points):
            raise LabelError(f"point names must be unique, got {self.points}")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.episode_frames = {int(k): int(v) for k, v in episode_frames.items()}
        self._lock = threading.Lock()
        self.data = {
            "version": FORMAT_VERSION,
            "dataset_root": str(dataset_root),
            "camera": camera,
            "image_size": list(self.image_size),
            "points": self.points,
            "coordinates": COORDINATES,
            "episodes": {},
        }
        if self.path.exists():
            self._resume(json.loads(self.path.read_text()))

    def _resume(self, old: dict) -> None:
        mismatches = [
            f"{key}: the file has {old.get(key)!r}, this run has {self.data[key]!r}"
            for key in ("version", "camera", "points", "image_size")
            if old.get(key) != self.data[key]
        ]
        if mismatches:
            raise LabelError(
                f"{self.path} was made for something else — refusing to mix labels into it:\n  "
                + "\n  ".join(mismatches)
                + "\nUse another --out, or run with the same --camera and --points."
            )
        if old.get("dataset_root") != self.data["dataset_root"]:
            logger.warning("%s was made for dataset %s; continuing it for %s",
                           self.path, old.get("dataset_root"), self.data["dataset_root"])
        episodes = dict(old.get("episodes") or {})
        unknown = sorted(int(e) for e in episodes if int(e) not in self.episode_frames)
        if unknown:
            logger.warning("%s has labels for episodes this dataset does not have: %s "
                           "(kept in the file, not shown)", self.path, unknown)
        self.data["episodes"] = episodes
        logger.info("resuming %s: %d episode(s) already labelled", self.path, len(episodes))

    # ── reading ──────────────────────────────────────────────────────────
    def labelled(self) -> list[int]:
        return sorted(int(e) for e in self.data["episodes"] if int(e) in self.episode_frames)

    def get(self, episode: int) -> dict | None:
        return self.data["episodes"].get(str(int(episode)))

    # ── writing ──────────────────────────────────────────────────────────
    def validate(self, episode, assignment) -> dict:
        """The assignment as it will be stored. Raises LabelError naming the problem."""
        try:
            episode = int(episode)
        except (TypeError, ValueError):
            raise LabelError(f"episode must be a number, got {episode!r}") from None
        if episode not in self.episode_frames:
            raise LabelError(f"episode {episode} is not in this dataset")
        if not isinstance(assignment, dict):
            raise LabelError("points must be a map from point name to a click or null")
        missing = [p for p in self.points if p not in assignment]
        extra = sorted(set(assignment) - set(self.points))
        if missing or extra:
            raise LabelError(
                f"every point needs a click or 'not visible' — missing {missing}, unknown {extra}"
            )
        n_frames = self.episode_frames[episode]
        H, W = self.image_size
        out = {}
        for name in self.points:
            click = assignment[name]
            if click is None:
                out[name] = None
                continue
            try:
                frame, x, y = int(click["frame"]), float(click["x"]), float(click["y"])
            except (KeyError, TypeError, ValueError):
                raise LabelError(f"{name}: a click needs frame, x and y, got {click!r}") from None
            if not 0 <= frame < n_frames:
                raise LabelError(f"{name}: frame {frame} is outside episode {episode} (0..{n_frames - 1})")
            if not (math.isfinite(x) and math.isfinite(y)) or not (0 <= x <= W - 1 and 0 <= y <= H - 1):
                raise LabelError(f"{name}: ({x}, {y}) is outside the {W}x{H} image")
            out[name] = {"frame": frame, "x": round(x, 2), "y": round(y, 2)}
        return out

    def save(self, episode, assignment) -> dict:
        clean = self.validate(episode, assignment)
        with self._lock:
            self.data["episodes"][str(int(episode))] = clean
            self._write()
        return clean

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        episodes = dict(sorted(self.data["episodes"].items(), key=lambda kv: int(kv[0])))
        tmp.write_text(json.dumps({**self.data, "episodes": episodes}, indent=1))
        os.replace(tmp, self.path)


# ── frames ────────────────────────────────────────────────────────────────────

def to_hwc_uint8(image) -> np.ndarray:
    """A camera frame as lerobot returns it (CHW float in [0, 1], or uint8) -> HWC uint8."""
    arr = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D image, got shape {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(np.round(arr.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr)


def encode_jpeg(frame: np.ndarray, quality: int = 90) -> bytes:
    """HWC uint8 RGB -> JPEG bytes (OpenCV if present, else Pillow)."""
    try:
        import cv2

        ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()
    except ImportError:
        from PIL import Image

        out = io.BytesIO()
        Image.fromarray(frame).save(out, format="JPEG", quality=int(quality))
        return out.getvalue()


class LerobotFrameSource:
    """Frames of one camera of a LeRobot dataset, decoded on demand."""

    def __init__(self, root, camera: str, repo_id: str | None = None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Episode lookup shared with the open-loop check (works on v3.0
        # datasets, where meta.episode_data_index no longer exists).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from check_policy_openloop import dataset_episode_column, episode_row_range

        self.camera = camera
        self.ds = LeRobotDataset(repo_id or f"local/{Path(root).name}", root=root)
        cameras = list(getattr(self.ds.meta, "camera_keys", []) or [])
        if camera not in cameras:
            raise SystemExit(f"{camera!r} is not a camera of this dataset; it has {cameras}")
        column = dataset_episode_column(self.ds)
        self.ranges = {int(ep): episode_row_range(column, int(ep)) for ep in np.unique(column)}
        self._lock = threading.Lock()   # one decoder, many request threads

    def episode_frames(self) -> dict[int, int]:
        return {ep: hi - lo for ep, (lo, hi) in self.ranges.items()}

    def frame(self, episode: int, index: int) -> np.ndarray:
        lo, hi = self.ranges[int(episode)]
        if not 0 <= int(index) < hi - lo:
            raise IndexError(f"frame {index} is outside episode {episode} (0..{hi - lo - 1})")
        with self._lock:
            item = self.ds[lo + int(index)]
        return to_hwc_uint8(item[self.camera])


# ── the page ──────────────────────────────────────────────────────────────────

class ClickServer:
    """Serves the click page and its API for one frame source and one label store."""

    def __init__(self, source, store: LabelStore, host: str = "127.0.0.1", port: int = 8899,
                 cache_size: int = 256):
        self.source, self.store = source, store
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._cache_size = int(cache_size)
        self._cache_lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # noqa: ANN001, ANN002
                logger.debug("%s " + fmt, self.address_string(), *args)

            def do_GET(self):  # noqa: N802
                server._get(self)

            def do_POST(self):  # noqa: N802
                server._post(self)

        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.httpd.daemon_threads = True

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def start(self) -> "ClickServer":
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    # ── responses ────────────────────────────────────────────────────────
    @staticmethod
    def _send(handler, status: HTTPStatus, body: bytes, content_type: str) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store" if content_type.startswith("application/json")
                            else "max-age=3600")
        handler.end_headers()
        handler.wfile.write(body)

    def _json(self, handler, payload, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(handler, status, json.dumps(payload).encode(), "application/json")

    def state(self) -> dict:
        frames = self.store.episode_frames
        return {
            "camera": self.store.data["camera"],
            "points": self.store.points,
            "image_size": list(self.store.image_size),
            "episodes": sorted(frames),
            "n_frames": {str(e): n for e, n in frames.items()},
            "labels": {str(e): self.store.get(e) for e in self.store.labelled()},
            "out": str(self.store.path),
        }

    def jpeg(self, episode: int, index: int) -> bytes:
        key = (int(episode), int(index))
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        data = encode_jpeg(self.source.frame(*key))
        with self._cache_lock:
            self._cache[key] = data
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return data

    def _get(self, handler) -> None:
        url = urlparse(handler.path)
        try:
            if url.path in ("/", "/index.html"):
                self._send(handler, HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json(handler, self.state())
            elif url.path == "/api/frame":
                query = parse_qs(url.query)
                episode, index = int(query["ep"][0]), int(query["f"][0])
                if episode not in self.store.episode_frames:
                    raise IndexError(f"episode {episode} is not in this dataset")
                self._send(handler, HTTPStatus.OK, self.jpeg(episode, index), "image/jpeg")
            else:
                self._json(handler, {"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, IndexError) as exc:
            self._json(handler, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001 — a decode failure must reach the page
            logger.exception("request %s failed", handler.path)
            self._json(handler, {"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _post(self, handler) -> None:
        url = urlparse(handler.path)
        if url.path != "/api/label":
            self._json(handler, {"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(handler.headers.get("Content-Length") or 0)
            body = json.loads(handler.rfile.read(length) or b"{}")
            clean = self.store.save(body.get("episode"), body.get("points"))
        except (LabelError, json.JSONDecodeError) as exc:
            self._json(handler, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        logger.info("episode %s saved (%d of %d labelled)", body.get("episode"),
                    len(self.store.labelled()), len(self.store.episode_frames))
        self._json(handler, {"ok": True, "saved": clean, "labelled": self.store.labelled()})


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Click track points</title>
<style>
  :root { --bg:#f6f6f4; --fg:#1d1d1b; --muted:#6b6b66; --line:#d8d8d2; --accent:#1f6feb; --bad:#b42318; }
  body { margin:0; font:14px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; gap:16px; align-items:baseline; flex-wrap:wrap; }
  header b { font-size:16px; }
  #prompt { font-weight:600; color:var(--accent); }
  main { display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:16px; padding:12px 16px; }
  @media (max-width:900px) { main { grid-template-columns:1fr; } }
  #stage { line-height:0; }
  #view { cursor:crosshair; display:block; background:#000; }
  .controls { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:8px 0; }
  button { font:inherit; padding:4px 10px; border:1px solid var(--line); background:#fff; border-radius:6px; cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  button:disabled { opacity:.45; cursor:default; }
  input[type=range] { flex:1; min-width:160px; }
  aside h3 { margin:12px 0 6px; font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
  ol { margin:0; padding-left:20px; }
  li.current { font-weight:600; color:var(--accent); }
  li .where { color:var(--muted); font-weight:400; }
  li a { cursor:pointer; text-decoration:underline; }
  #ref { width:100%; height:auto; display:block; background:#000; }
  #msg { min-height:1.4em; color:var(--bad); }
  kbd { border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:0 4px; background:#fff; font-size:12px; }
  .muted { color:var(--muted); }
</style>
</head>
<body>
<header>
  <b>Click track points</b>
  <span id="where"></span>
  <span id="prompt"></span>
  <span id="progress" class="muted"></span>
</header>
<main>
  <section>
    <div id="stage"><canvas id="view"></canvas></div>
    <div class="controls">
      <button id="m10" title="Shift+Left">&laquo; 10</button>
      <button id="m1" title="Left">&lsaquo; 1</button>
      <input id="slider" type="range" min="0" value="0">
      <button id="p1" title="Right">1 &rsaquo;</button>
      <button id="p10" title="Shift+Right">10 &raquo;</button>
    </div>
    <div class="controls">
      <button id="undo" title="Z">Undo</button>
      <button id="skip" title="X">Not visible</button>
      <button id="save" class="primary" title="Enter">Save &amp; next</button>
      <button id="prev" title="P">Previous episode</button>
      <button id="next" title="N">Next episode</button>
    </div>
    <div id="msg"></div>
    <p class="muted"><kbd>&larr;</kbd><kbd>&rarr;</kbd> frame, <kbd>Shift</kbd> &times;10 &middot;
      <kbd>Z</kbd> undo &middot; <kbd>X</kbd> not visible &middot; <kbd>Enter</kbd> save &middot;
      <kbd>N</kbd>/<kbd>P</kbd> episode. Click the earliest frame where a point is clearly visible:
      the tracker follows it forward from there.</p>
  </section>
  <aside>
    <h3>Points</h3>
    <ol id="list"></ol>
    <h3>Reference</h3>
    <canvas id="ref"></canvas>
    <div id="refcap" class="muted"></div>
  </aside>
</main>
<script>
const COLORS = ["#ff3b30","#34c759","#0a84ff","#ffcc00","#bf5af2","#ff9f0a","#64d2ff","#ff375f"];
let S = null, ep = null, frame = 0, assign = {}, order = [], dirty = false;
const view = document.getElementById("view"), ctx = view.getContext("2d");
const img = new Image();
const $ = id => document.getElementById(id);

function nFrames() { return S.n_frames[String(ep)]; }
function nextPoint() { return S.points.find(p => !(p in assign)); }
function say(text) { $("msg").textContent = text || ""; }

async function api(path, body) {
  const r = await fetch(path, body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : {});
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function firstUnlabelled(after) {
  const eps = S.episodes, start = after == null ? 0 : eps.indexOf(after) + 1;
  for (let i = 0; i < eps.length; i++) {
    const e = eps[(start + i) % eps.length];
    if (!(String(e) in S.labels)) return e;
  }
  return null;
}

function openEpisode(e) {
  if (dirty && !confirm("This episode has unsaved clicks. Leave it?")) return;
  ep = e; assign = {}; order = []; dirty = false; say("");
  const saved = S.labels[String(e)];
  if (saved) { for (const p of S.points) { assign[p] = saved[p]; order.push(p); } }
  const clicked = Object.values(assign).filter(c => c);
  frame = clicked.length ? Math.min(...clicked.map(c => c.frame)) : 0;
  $("slider").max = nFrames() - 1;
  drawReference();
  showFrame(frame);
}

function showFrame(f) {
  frame = Math.max(0, Math.min(nFrames() - 1, f));
  $("slider").value = frame;
  img.onload = () => { view.width = img.naturalWidth; view.height = img.naturalHeight; fit(); draw(); };
  img.onerror = () => say("could not load frame " + frame + " of episode " + ep);
  img.src = "/api/frame?ep=" + ep + "&f=" + frame;
  render();
}

// Image pixels per screen pixel, so markers keep one on-screen size whatever the image size.
function scaleOf(canvas) { const w = canvas.getBoundingClientRect().width; return w ? canvas.width / w : 1; }

// Show the image as large as fits (width of its column, 72% of the window height), aspect kept.
function fit() {
  if (!S) return;
  const H = S.image_size[0], W = S.image_size[1];
  const s = Math.min($("stage").clientWidth / W, window.innerHeight * 0.72 / H);
  view.style.width = (W * s) + "px"; view.style.height = (H * s) + "px";
}
window.addEventListener("resize", () => { fit(); draw(); });

function marker(c, x, y, label, color) {
  const k = scaleOf(c.canvas), r = 7 * k;
  c.lineWidth = 2 * k;
  c.strokeStyle = "#000"; c.beginPath(); c.arc(x, y, r + k, 0, 2 * Math.PI); c.stroke();
  c.strokeStyle = color; c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.stroke();
  c.beginPath(); c.moveTo(x - r * 1.6, y); c.lineTo(x + r * 1.6, y); c.moveTo(x, y - r * 1.6); c.lineTo(x, y + r * 1.6); c.stroke();
  c.font = "bold " + (13 * k) + "px system-ui, sans-serif";
  const tw = c.measureText(label).width;
  const tx = x + r * 1.8 + tw <= c.canvas.width ? x + r * 1.8 : x - r * 1.8 - tw;   // flip left at the right edge
  const ty = Math.max(13 * k, y - r * 1.2);
  c.lineWidth = 3 * k; c.strokeStyle = "#000"; c.strokeText(label, tx, ty);
  c.fillStyle = color; c.fillText(label, tx, ty);
}

function draw() {
  if (!img.complete || !img.naturalWidth) return;
  ctx.drawImage(img, 0, 0);
  S.points.forEach((p, i) => {
    const c = assign[p];
    if (c && c.frame === frame) marker(ctx, c.x + 0.5, c.y + 0.5, p, COLORS[i % COLORS.length]);
  });
}

function render() {
  const idx = S.episodes.indexOf(ep);
  $("where").textContent = "Episode " + ep + " (" + (idx + 1) + " of " + S.episodes.length + ") · frame " + frame + " / " + (nFrames() - 1);
  const p = nextPoint();
  $("prompt").textContent = p ? "Click: " + p : "All points set — press Enter to save";
  $("progress").textContent = Object.keys(S.labels).length + " of " + S.episodes.length + " episodes labelled" + (dirty ? " · unsaved" : "");
  const list = $("list"); list.innerHTML = "";
  S.points.forEach((name, i) => {
    const li = document.createElement("li");
    li.style.color = COLORS[i % COLORS.length];
    if (name === p) li.className = "current";
    li.append(name + " ");
    const where = document.createElement("span"); where.className = "where";
    if (!(name in assign)) where.textContent = "—";
    else if (assign[name] === null) where.textContent = "not visible";
    else {
      const a = document.createElement("a"); a.textContent = "frame " + assign[name].frame;
      a.onclick = () => showFrame(assign[name].frame); where.append(a);
    }
    li.append(where); list.append(li);
  });
  $("save").disabled = !!p;
  $("undo").disabled = order.length === 0;
  $("skip").disabled = !p;
}

function drawReference() {
  const refCanvas = $("ref"), rc = refCanvas.getContext("2d");
  const other = S.episodes.find(e => e !== ep && String(e) in S.labels);
  if (other === undefined) { refCanvas.width = 0; refCanvas.height = 0; $("refcap").textContent = "The first saved episode appears here, to click the same spots again."; return; }
  const lab = S.labels[String(other)], counts = {};
  for (const c of Object.values(lab)) if (c) counts[c.frame] = (counts[c.frame] || 0) + 1;
  const frames = Object.keys(counts).map(Number);
  if (!frames.length) { $("refcap").textContent = "Episode " + other + ": no visible points."; return; }
  const f = frames.sort((a, b) => counts[b] - counts[a] || a - b)[0];
  const ref = new Image();
  ref.onload = () => {
    refCanvas.width = ref.naturalWidth; refCanvas.height = ref.naturalHeight; rc.drawImage(ref, 0, 0);
    S.points.forEach((p, i) => { const c = lab[p]; if (c && c.frame === f) marker(rc, c.x + 0.5, c.y + 0.5, p, COLORS[i % COLORS.length]); });
  };
  ref.src = "/api/frame?ep=" + other + "&f=" + f;
  $("refcap").textContent = "Episode " + other + ", frame " + f;
}

view.addEventListener("click", ev => {
  const p = nextPoint(); if (!p) { say("All points are set. Undo one to move it, or press Enter."); return; }
  const rect = view.getBoundingClientRect();
  const W = S.image_size[1], H = S.image_size[0];
  const x = Math.min(W - 1, Math.max(0, (ev.clientX - rect.left) / rect.width * W - 0.5));
  const y = Math.min(H - 1, Math.max(0, (ev.clientY - rect.top) / rect.height * H - 0.5));
  assign[p] = {frame: frame, x: Math.round(x * 100) / 100, y: Math.round(y * 100) / 100};
  order.push(p); dirty = true; say(""); draw(); render();
});

function undo() { const p = order.pop(); if (p !== undefined) { delete assign[p]; dirty = true; draw(); render(); } }
function skip() { const p = nextPoint(); if (p) { assign[p] = null; order.push(p); dirty = true; render(); } }
async function save() {
  if (nextPoint()) { say("Set every point first (click it, or X for not visible)."); return; }
  try {
    const res = await api("/api/label", {episode: ep, points: assign});
    S.labels[String(ep)] = res.saved; dirty = false;
    const nxt = firstUnlabelled(ep);
    if (nxt === null) { render(); say(""); $("prompt").textContent = "All episodes labelled — saved to " + S.out; }
    else openEpisode(nxt);
  } catch (e) { say("Not saved: " + e.message); }
}
function step(d) { const i = S.episodes.indexOf(ep) + d; if (i >= 0 && i < S.episodes.length) openEpisode(S.episodes[i]); }

$("m10").onclick = () => showFrame(frame - 10); $("m1").onclick = () => showFrame(frame - 1);
$("p1").onclick = () => showFrame(frame + 1); $("p10").onclick = () => showFrame(frame + 10);
$("slider").oninput = ev => showFrame(Number(ev.target.value));
$("undo").onclick = undo; $("skip").onclick = skip; $("save").onclick = save;
$("prev").onclick = () => step(-1); $("next").onclick = () => step(1);
document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT" && ev.key !== "Enter") return;
  const k = ev.key;
  if (k === "ArrowLeft") { showFrame(frame - (ev.shiftKey ? 10 : 1)); ev.preventDefault(); }
  else if (k === "ArrowRight") { showFrame(frame + (ev.shiftKey ? 10 : 1)); ev.preventDefault(); }
  else if (k === "z" || k === "Z") undo();
  else if (k === "x" || k === "X") skip();
  else if (k === "Enter") { save(); ev.preventDefault(); }
  else if (k === "n" || k === "N") step(1);
  else if (k === "p" || k === "P") step(-1);
});

(async () => {
  try {
    S = await api("/api/state");
    const start = firstUnlabelled(null);
    openEpisode(start === null ? S.episodes[0] : start);
  } catch (e) { say("Could not reach the server: " + e.message); }
})();
</script>
</body>
</html>
"""


# ── command line ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Click the query points a point tracker should follow, once per episode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for the output format.",
    )
    p.add_argument("--root", required=True, help="LeRobot dataset folder (the one with meta/, data/, videos/)")
    p.add_argument("--repo-id", default=None, help="dataset repo id (default: local/<folder name>)")
    p.add_argument("--camera", required=True, help="image key, e.g. observation.images.oakw_cam")
    p.add_argument("--points", nargs="+", required=True,
                   help="point names in click order, e.g. button_a button_b door_a door_b")
    p.add_argument("--out", required=True, help="labels JSON; resumed when it already exists")
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to serve on. 127.0.0.1 = reachable only through an SSH tunnel")
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("opening %s ...", args.root)
    source = LerobotFrameSource(args.root, args.camera, repo_id=args.repo_id)
    frames = source.episode_frames()
    first = min(frames)
    image_size = source.frame(first, 0).shape[:2]
    try:
        store = LabelStore(args.out, dataset_root=Path(args.root).resolve(), camera=args.camera,
                           points=args.points, image_size=image_size, episode_frames=frames)
    except LabelError as exc:
        logger.error("%s", exc)
        return 2
    server = ClickServer(source, store, host=args.host, port=args.port)
    logger.info("%d episodes, %d frames, images %dx%d; %d already labelled",
                len(frames), sum(frames.values()), image_size[1], image_size[0], len(store.labelled()))
    logger.info("serving on http://%s:%d  (labels -> %s)", args.host, server.port, store.path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopped; %d of %d episodes labelled in %s",
                    len(store.labelled()), len(frames), store.path)
    finally:
        server.httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
