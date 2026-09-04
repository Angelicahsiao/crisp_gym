"""Configuration classes for recording managers."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from crisp_gym.config.path import find_config, list_configs_in_folder


@dataclass(kw_only=True)
class RecordingManagerConfig:
    """Configuration for recording managers.

    This configuration class contains all parameters needed to initialize
    a recording manager, including dataset configuration, recording settings,
    and system parameters.
    """

    # Dataset configuration
    features: Dict[str, Any]
    repo_id: str
    robot_type: str = "Franka"
    resume: bool = False
    fps: int = 30
    num_episodes: int = 3
    push_to_hub: bool = False

    # System configuration
    use_sound: bool = True
    # Frames of slack between the recording loop and the writer process. Prefer
    # queue_seconds, which expresses the same thing in units that matter (how
    # long a writer hiccup the loop can absorb before teleop stalls).
    queue_size: int = 16
    # When set, OVERRIDES queue_size with ceil(queue_seconds * fps). Size it to
    # cover save_episode, which is not bounded by the queue: the loop blocks on
    # any writer work longer than this. Costs RAM — each queued frame holds its
    # raw uint8 images (a single 800x1280 RGB camera is ~3.1 MB/frame).
    queue_seconds: float | None = None
    # Bounded join for the writer at shutdown (see __exit__). Distinct from
    # shutdown_drain_timeout: this one only covers a writer that is idle and
    # simply has to exit.
    writer_timeout: float = 10.0
    # Start method for the writer process. "spawn" is the default on purpose:
    # the writer is created from a process that has ROS2/DDS threads running
    # and (through lerobot/torch) an initialised CUDA context, and NEITHER
    # survives fork(). Verified on the owner's box — a CUDA-initialised parent
    # forking a child makes h264_nvenc fail with AVERROR_UNKNOWN, while spawn
    # opens it fine; forking a live DDS/multithreaded parent can also deadlock
    # the child on an inherited lock. Spawn costs a few seconds of writer
    # startup (the child re-imports lerobot/torch), covered by
    # writer_startup_timeout. Set "fork" only to reproduce the old behaviour.
    writer_start_method: str = "spawn"
    # Startup budget for the writer, from process creation to "dataset ready".
    # Distinct from writer_timeout: under spawn the child re-imports lerobot
    # and torch before it can even begin, which the old 10 s could not cover.
    writer_startup_timeout: float = 180.0
    # How long shutdown waits for the writer to drain the queue AND finish its
    # final save_episode (video encode + parquet) before giving up and
    # terminating it. Terminating mid-write loses the episode and can leave the
    # parquet footer unwritten, so this must exceed a save_episode.
    shutdown_drain_timeout: float = 300.0

    # ── Writer throughput ────────────────────────────────────────────────────
    # lerobot PNG-encodes every camera frame inside add_frame. With 0 threads
    # that happens synchronously on the writer's only thread and sets the
    # achievable rate. PIL's PNG encoder releases the GIL, so threads scale
    # nearly linearly (measured 4.6x at 4 threads on 800x1280).
    # processes=0 means threads-only — no pickling of multi-MB images.
    image_writer_processes: int = 0
    image_writer_threads: int = 0

    # ── Video encoding (save_episode) ────────────────────────────────────────
    # vcodec: "auto" picks the first available hardware encoder (h264_nvenc on
    #   NVIDIA, vaapi/qsv on Intel/AMD) and falls back to libsvtav1 with a
    #   warning. Availability is probed through PyAV, so the wheel's bundled
    #   ffmpeg must carry the encoder — having the GPU is not sufficient.
    #   None = leave lerobot's default (libsvtav1).
    # video_crf: quality. NOTE the meaning is codec-specific: lerobot feeds it
    #   to libsvtav1 as CRF but to NVENC as constant QP (rc=0, qp=crf). The
    #   lerobot default of 30 is a good AV1 point and a visibly lossy h264 one,
    #   so set this explicitly whenever vcodec changes — it is training data.
    # video_gop: keyframe interval. lerobot defaults to 2 (near all-intra) to
    #   keep random-access seeking fast in the training dataloader. Only worth
    #   raising if you are stuck on a software encoder. None = keep the default.
    # video_preset: speed/quality preset, codec-specific (libx264: ultrafast..
    #   veryslow; nvenc: p1..p7). lerobot only defaults this for libsvtav1, so
    #   software h264 otherwise runs at its "medium" default.
    # video_extra_options: raw codec options merged last by lerobot (never
    #   overriding the structured fields above). Needed for constraints lerobot
    #   has no field for — notably `bf` on NVENC: its presets enable B-frames,
    #   and NVENC requires gop_size > b_frames + 1, so lerobot's g=2 fails to
    #   open with "Gop Length should be greater than number of B frames + 1"
    #   unless B-frames are disabled. `bf: 0` is applied automatically for
    #   NVENC at small GOPs (see RecordingManager._rgb_encoder); set this
    #   explicitly to override.
    vcodec: str | None = None
    video_crf: int | None = None
    video_gop: int | None = None
    video_preset: str | int | None = None
    video_extra_options: Dict[str, Any] | None = None
    encoder_threads: int | None = None

    # ── Streaming encoding ───────────────────────────────────────────────────
    # Default path (False): lerobot writes every camera frame to disk as a PNG
    # during recording, then reads them all back and decodes them at
    # save_episode. Measured at 800x1280: ~1.3 GB written and read per episode,
    # ~49 ms/frame of save time, and enough disk churn to cost the recording
    # loop ~11% of its rate (an episode recorded before the first save_episode
    # held 14.4 FPS; every one after it sat at 13.2-13.5).
    #
    # True: frames go straight to a per-camera encoder thread. No PNG is
    # written or read, image stats come from the encoder, and save_episode
    # drops to roughly its fixed cost.
    #
    # THE RISK, and why the guard below is not optional: lerobot's
    # StreamingVideoEncoder.feed_frame DROPS a frame when its queue is full,
    # while add_frame still appends a parquet row — silently desynchronising
    # rows from video frames. RecordingManager therefore inspects the encoder's
    # dropped-frame counters after every save_episode and fails the recording
    # if any are non-zero (or if it cannot read them at all).
    streaming_encoding: bool = False
    # Seconds of per-camera buffer between the writer and its encoder thread,
    # converted to lerobot's encoder_queue_maxsize. Raw uint8 frames, so a
    # single 800x1280 camera costs ~3.1 MB per buffered frame.
    encoder_queue_seconds: float = 4.0

    # Instrumentation (measurement only — never changes what is recorded).
    # timing_report: log a per-episode phase breakdown of the recording loop
    #   (data_fn / queue.put / sleep) and of the writer process, so a dropped
    #   control rate can be attributed to producer work vs. writer
    #   back-pressure. See util/loop_timing.py.
    # timing_csv_dir: directory for a per-frame CSV trace (one file per
    #   episode, written at episode end). None = no CSV.
    timing_report: bool = True
    timing_csv_dir: str | None = None

    def resolved_encoder_queue_size(self) -> int:
        """Per-camera encoder buffer in frames (lerobot's encoder_queue_maxsize)."""
        return max(1, math.ceil(self.encoder_queue_seconds * self.fps))

    def resolved_queue_size(self) -> int:
        """Queue capacity in frames, honouring queue_seconds when set."""
        if self.queue_seconds is None:
            return self.queue_size
        return max(1, math.ceil(self.queue_seconds * self.fps))

    @classmethod
    def from_yaml(cls, yaml_path: Path | str, **overrides) -> "RecordingManagerConfig":  # noqa: ANN003
        """Create a RecordingManagerConfig from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.
            **overrides: Additional keyword arguments to override config values.

        Returns:
            A RecordingManagerConfig instance.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            yaml.YAMLError: If the YAML file is malformed.
            TypeError: If required fields are missing.
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            config_data = {}

        # Apply overrides
        config_data.update(overrides)

        return cls(**config_data)

    def to_yaml(self, yaml_path: Path | str) -> None:
        """Save the configuration to a YAML file.

        Args:
            yaml_path: Path where to save the YAML configuration file.
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling non-serializable types
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, (dict, list, str, int, float, bool)) or field_value is None:
                config_dict[field_name] = field_value
            else:
                # For complex types, convert to string representation
                config_dict[field_name] = str(field_value)

        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)


def make_recording_manager_config(
    name: str,
    config_path: Path | str | None = None,
    **overrides,  # noqa: ANN003
) -> RecordingManagerConfig:
    """Factory function to create a recording manager configuration based on the type.

    This function allows for both predefined recording manager types and custom YAML configurations.
    It will first check if the type is in the predefined set, and if not, it will look for a YAML config file.

    Args:
        name: Type of recording manager configuration
        config_path: Optional path to YAML config file
        **overrides: Additional parameters to override defaults/YAML values

    Returns:
        RecordingManagerConfig: Configured recording manager instance
    """
    config_class = STRING_TO_CONFIG.get(name.lower())
    if config_class is None:
        # Try to find YAML config if not in predefined types
        config_path = find_config("recording/" + name.lower() + ".yaml")
        if config_path is None:
            raise ValueError(
                f"Unsupported recording manager type: {name}. The list of supported types are: {list_recording_configs()}"
            )
        config_class = RecordingManagerConfig

    if config_path:
        config_path = Path(config_path) if isinstance(config_path, str) else config_path
        return config_class.from_yaml(config_path, **overrides)

    return config_class(**overrides)


def list_recording_configs() -> list[str]:
    """List all available recording manager configurations."""
    predefined = list(STRING_TO_CONFIG.keys())
    other = list_configs_in_folder("recording")
    yaml_configs = [file.stem for file in other if file.suffix == ".yaml"]
    return predefined + yaml_configs


STRING_TO_CONFIG: Dict[str, type[RecordingManagerConfig]] = {}
