"""Keyboard event listener for controlling episode recording."""

import logging
import multiprocessing as mp
import queue as queue_mod
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from inspect import signature
from pathlib import Path
from typing import Callable, Literal
from typing_extensions import override

import sys
import numpy as np
import rclpy

# TODO: make this optional, we do not want to depend on lerobot
try:
    from lerobot.utils.constants import HF_LEROBOT_HOME
except ImportError:
    from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rclpy.executors import SingleThreadedExecutor
from rich import print
from rich.panel import Panel
from std_msgs.msg import String

from crisp_gym.config.path import find_config
from crisp_gym.policy.policy import Action, Observation
from crisp_gym.record.recording_manager_config import RecordingManagerConfig
from crisp_gym.util.lerobot_features import concatenate_state_features
from crisp_gym.util.loop_timing import (
    DeadlinePacer,
    LoopTimingRecorder,
    WriterTimingRecorder,
)

logger = logging.getLogger(__name__)

_ADD_FRAME_HAS_TASK = "task" in signature(LeRobotDataset.add_frame).parameters


class RecordingManager(ABC):
    """Base class for event listener to control episode recording."""

    def __init__(
        self,
        config: RecordingManagerConfig | None = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the recording manager.

        Args:
            config: RecordingManagerConfig instance. If provided, other parameters are ignored.
            **kwargs: Individual parameters for backwards compatibility.
        """
        # Handle config vs individual parameters
        self.config = (
            config
            if config is not None
            else RecordingManagerConfig.from_yaml(
                find_config("recording/default_recording.yaml"), **kwargs
            )
        )

        # State is written by input threads (stdin reader / ROS callback) and
        # read by the recording loop — guard it with a lock via the `state`
        # property. The switch semantics/UX are unchanged.
        self._state_lock = threading.Lock()
        self._state: Literal[
            "is_waiting",
            "recording",
            "paused",
            "to_be_saved",
            "to_be_deleted",
            "exit",
        ] = "is_waiting"

        self.episode_count = 0

        self.queue_capacity = self.config.resolved_queue_size()
        if self.config.queue_seconds is not None:
            logger.info(
                f"Writer queue: {self.queue_capacity} frames "
                f"({self.config.queue_seconds:.1f} s at {self.config.fps} fps). "
                "The recording loop blocks on any writer work longer than this "
                "— size it to cover save_episode."
            )
        self.queue = mp.JoinableQueue(self.queue_capacity)
        self.episode_count_queue = mp.Queue(1)
        self.dataset_ready = mp.Event()
        # Set by the writer process when a FRAME/SAVE/startup error occurs, so
        # the recording loop fails loudly instead of silently producing a
        # truncated/corrupt episode.
        self.writer_error = mp.Event()

        # Start the writer process
        self.writer = mp.Process(
            target=self._writer_proc,
            args=(),
            name="dataset_writer",
            daemon=False,
        )
        self.writer.start()

    @property
    def state(self) -> str:
        """Current recording state (thread-safe)."""
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, value: str) -> None:
        with self._state_lock:
            self._state = value

    @property
    def dataset_directory(self) -> Path:
        """Return the path to the dataset directory."""
        return Path(HF_LEROBOT_HOME / self.config.repo_id)

    @property
    def num_episodes(self) -> int:
        """Return the number of episodes to record."""
        return self.config.num_episodes

    def queue_depth(self) -> int:
        """Frames currently queued for the writer process, or -1 if unknown.

        `mp.Queue.qsize()` reads the underlying semaphore; it is unimplemented
        on platforms without `sem_getvalue` (macOS), hence the guard.
        """
        try:
            return self.queue.qsize()
        except (NotImplementedError, OSError):
            return -1

    def writer_status(self) -> str:
        """One-line liveness description of the writer process, for logs."""
        if self.writer.is_alive():
            return "writer: alive"
        return f"writer: DEAD (exitcode={self.writer.exitcode})"

    def _put_blocking(self, msg: dict, poll_s: float = 1.0, required: bool = True) -> None:
        """Enqueue `msg`, waiting as long as the writer is alive.

        Frames are never dropped — a gap would silently corrupt the fixed-fps
        timestamps LeRobot writes — so this still blocks under back-pressure.
        What it will not do is block FOREVER: `writer_error` only covers
        exceptions the writer catches, so a writer killed outright (OOM,
        SIGSEGV in the encoder, an earlier terminate()) used to wedge the
        recording loop with no diagnosis.

        Args:
            msg: The message to enqueue.
            poll_s: How often to re-check that the writer is still alive.
            required: Raise when the writer is gone (the caller's data would be
                silently lost). Pass False on the shutdown path, where raising
                would mask the exception being handled and skip the cleanup
                that follows — there we only log.
        """
        while True:
            try:
                self.queue.put(msg, timeout=poll_s)
                return
            except queue_mod.Full:
                if self.writer.is_alive():
                    continue
                reason = (
                    "The dataset writer process is gone "
                    f"(exitcode={self.writer.exitcode}) and the queue is full "
                    f"— '{msg.get('type')}' cannot be delivered. Check the "
                    "writer traceback above; a SIGKILL usually means the "
                    "machine ran out of memory."
                )
                if required:
                    raise RuntimeError(reason) from None
                logger.error(reason)
                return

    def wait_until_ready(self, timeout: float | None = None) -> None:
        """Wait until the dataset writer is ready."""
        if timeout is None:
            timeout = self.config.writer_timeout

        original_timeout = timeout
        while not self.dataset_ready.is_set():
            if self.writer_error.is_set():
                raise RuntimeError(
                    "Dataset writer failed during startup — see the writer "
                    "process traceback above."
                )
            logger.debug("Waiting for dataset to be ready...")
            time.sleep(1.0)
            timeout -= 1.0
            if timeout <= 0.0:
                raise TimeoutError(
                    f"Timeout waiting for dataset to be ready after {original_timeout} seconds."
                )

        self.update_episode_count()

    def update_episode_count(self) -> None:
        """Update the episode count from the queue.

        This is useful when resuming from an existing dataset.
        If the queue is empty, it will not change the episode count.
        """
        if not self.episode_count_queue.empty():
            self.episode_count = self.episode_count_queue.get()

    def done(self) -> bool:
        """Return true if we are done recording."""
        return self.state == "exit"

    @abstractmethod
    def get_instructions(self) -> str:
        """Return the instructions to use the recording manager."""
        raise NotImplementedError()

    def _writer_kwargs(self, fn) -> dict:  # noqa: ANN001
        """Async-image-writer + video-encoder kwargs accepted by `fn`.

        Filtered against the callable's own signature because the repo targets
        several lerobot versions: `image_writer_*` exists back to 0.4.x, while
        `rgb_encoder` / `encoder_threads` are 0.6+. Passing an unknown kwarg
        would be a TypeError inside the writer subprocess, which surfaces only
        as a startup timeout in the parent.
        """
        import inspect

        wanted: dict = {
            "image_writer_processes": self.config.image_writer_processes,
            "image_writer_threads": self.config.image_writer_threads,
        }

        encoder = self._rgb_encoder()
        if encoder is not None:
            wanted["rgb_encoder"] = encoder
        if self.config.encoder_threads is not None:
            wanted["encoder_threads"] = self.config.encoder_threads

        try:
            accepted = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            return {}

        kwargs = {k: v for k, v in wanted.items() if k in accepted}
        for name in wanted.keys() - kwargs.keys():
            logger.warning(
                f"This lerobot version does not accept '{name}' — ignoring it. "
                "Recording still works, but the corresponding tuning has no effect."
            )
        return kwargs

    def _rgb_encoder(self):  # noqa: ANN202 — lerobot RGBEncoderConfig, imported lazily
        """Build lerobot's RGBEncoderConfig, or None to keep its defaults."""
        if self.config.vcodec is None and self.config.video_crf is None and self.config.video_gop is None:
            return None
        try:
            from lerobot.configs import RGBEncoderConfig
        except ImportError:
            logger.warning(
                "lerobot.configs.RGBEncoderConfig is unavailable (pre-0.6 "
                "lerobot) — vcodec/video_crf/video_gop are ignored."
            )
            return None

        fields: dict = {}
        if self.config.vcodec is not None:
            fields["vcodec"] = self.config.vcodec
        if self.config.video_crf is not None:
            fields["crf"] = self.config.video_crf
        if self.config.video_gop is not None:
            fields["g"] = self.config.video_gop
        # __post_init__ resolves "auto" against the encoders PyAV actually has
        # and logs the winner (or warns and falls back to libsvtav1).
        encoder = RGBEncoderConfig(**fields)
        logger.info(
            f"Video encoder: vcodec={encoder.vcodec} crf/qp={encoder.crf} g={encoder.g}"
        )
        return encoder

    def _create_dataset(self) -> LeRobotDataset:
        """Factory function to create a dataset object."""
        logger.debug("Creating dataset object.")
        if self.config.resume:
            logger.info(f"Resuming recording from existing dataset: {self.config.repo_id}")
            # v0.5.1+ uses LeRobotDataset.resume(); older versions use the
            # plain constructor (which handles resuming based on existing meta).
            if hasattr(LeRobotDataset, "resume"):
                dataset = LeRobotDataset.resume(
                    repo_id=self.config.repo_id,
                    **self._writer_kwargs(LeRobotDataset.resume),
                )
            else:
                dataset = LeRobotDataset(repo_id=self.config.repo_id)
            if self.config.num_episodes <= dataset.num_episodes:
                # Raise (not exit()): this runs in the writer subprocess, and a
                # bare exit() would leave the parent blocked until the
                # wait_until_ready timeout with a misleading TimeoutError.
                raise ValueError(
                    f"The dataset already has {dataset.num_episodes} episodes recorded; "
                    f"--num-episodes ({self.config.num_episodes}) must be larger to resume."
                )
            logger.info(
                f"Resuming from episode {dataset.num_episodes} with {self.config.num_episodes} episodes to record."
            )
            self.episode_count_queue.put(dataset.num_episodes - 1)
        else:
            logger.info(
                f"[green]Creating new dataset: {self.config.repo_id}", extra={"markup": True}
            )
            # Clean up existing dataset if it exists
            if Path(HF_LEROBOT_HOME / self.config.repo_id).exists():
                logger.error(
                    f"The repo_id already exists. If you intended to resume the collection of data, then execute this script with the --resume flag. Otherwise remove it:\n'rm -r {str(Path(HF_LEROBOT_HOME / self.config.repo_id))}'."
                )
                raise FileExistsError(
                    f"The repo_id already exists. If you intended to resume the collection of data, then execute this script with the --resume flag. Otherwise remove it:\n'rm -r {str(Path(HF_LEROBOT_HOME / self.config.repo_id))}'."
                )
            dataset = LeRobotDataset.create(
                repo_id=self.config.repo_id,
                fps=self.config.fps,
                robot_type=self.config.robot_type,
                features=self.config.features,
                use_videos=True,
                **self._writer_kwargs(LeRobotDataset.create),
            )
            logger.debug(f"Dataset created with meta: {dataset.meta}")
        return dataset

    def _writer_proc(self):
        """Process to write data to the dataset."""
        logger.info("Starting dataset writer process.")
        try:
            dataset = self._create_dataset()
        except Exception:
            self.writer_error.set()
            raise
        self.dataset_ready.set()
        logger.debug(f"Dataset features: {list(self.config.features.keys())}")

        # Consumer-side instrumentation. `idle` (time blocked in queue.get())
        # is the decisive number: near 0% means the writer never runs out of
        # work, i.e. it — not the recording loop — sets the achievable rate.
        writer_timing = WriterTimingRecorder(
            budget_s=1.0 / self.config.fps,
            enabled=self.config.timing_report,
        )
        frames_in_episode = 0

        while True:
            idle_start = time.perf_counter()
            msg = self.queue.get()
            writer_timing.add_idle(time.perf_counter() - idle_start)
            logger.debug(f"Received message: {msg['type']}")
            handler_start = time.perf_counter()
            try:
                mtype = msg["type"]

                if mtype == "FRAME":
                    obs, action, task = msg["data"]

                    logger.debug(f"Received frame with action: {action} and obs: {obs.keys()}")

                    # Build frame directly from observation using feature-based approach
                    frame = {"action": action.astype(np.float32)}

                    # Add all observation features that match our dataset features
                    for feature_name in self.config.features:
                        if feature_name == "action":
                            continue  # Already added above
                        if feature_name in obs:
                            value = obs[feature_name]
                            if isinstance(value, np.ndarray) and feature_name.startswith(
                                "observation.state"
                            ):
                                frame[feature_name] = value.astype(np.float32)
                            else:
                                frame[feature_name] = value

                    # Concatenate state vector
                    frame["observation.state"] = concatenate_state_features(
                        obs, self.config.features
                    )

                    logger.debug(f"Constructed frame with keys: {frame.keys()}")
                    if _ADD_FRAME_HAS_TASK:  # For lerobot versions with explicit `task` parameter (>= v3.0)
                        dataset.add_frame(frame, task=task)
                    else:  # For older lerobot versions without `task` parameter (< v3.0)
                        frame["task"] = task
                        dataset.add_frame(frame)
                    writer_timing.add_frame(time.perf_counter() - handler_start)
                    frames_in_episode += 1

                elif mtype == "SAVE_EPISODE":
                    if self.config.use_sound:
                        try:
                            subprocess.Popen(
                                [
                                    "paplay",
                                    "/usr/share/sounds/freedesktop/stereo/complete.oga",
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to play sound for episode completion: {e}",
                            )

                    logger.info("Saving current episode to dataset.")

                    writer_timing.log_summary("episode")
                    save_start = time.perf_counter()
                    dataset.save_episode()
                    if self.config.timing_report:
                        logger.info(
                            "[timing/writer] save_episode took "
                            f"{time.perf_counter() - save_start:.2f} s "
                            "(stats over sampled frames + video encoding + "
                            "parquet). The recording loop only has "
                            f"{self.queue_capacity} frames "
                            f"({self.queue_capacity / self.config.fps:.1f} s "
                            "at this fps) of slack before it blocks on this."
                        )
                    frames_in_episode = 0

                elif mtype == "DELETE_EPISODE":
                    if self.config.use_sound:
                        try:
                            subprocess.Popen(
                                [
                                    "paplay",
                                    "/usr/share/sounds/freedesktop/stereo/suspend-error.oga",
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to play sound for episode deletion: {e}",
                            )

                    writer_timing.log_summary("discarded episode")
                    frames_in_episode = 0
                    dataset.clear_episode_buffer()

                elif mtype == "PUSH_TO_HUB":
                    logger.info(
                        "Pushing dataset to Hugging Face Hub...",
                    )
                    try:
                        dataset.push_to_hub(repo_id=self.config.repo_id, private=True)
                        logger.info("Dataset pushed to Hugging Face Hub successfully.")
                    except Exception as e:
                        logger.error(
                            f"Failed to push dataset to Hugging Face Hub: {e}",
                            exc_info=True,
                        )
                elif mtype == "SHUTDOWN":
                    logger.info("Shutting down writer process.")
                    writer_timing.log_summary(
                        f"at shutdown ({frames_in_episode} unsaved frames)"
                    )
                    # REQUIRED: finalize() closes the ParquetWriter and writes
                    # the footer. Without it the last data file has no footer
                    # and the dataset is unreadable. DatasetWriter.__del__ is
                    # only a best-effort net and never runs if the process is
                    # terminated, so call it explicitly here.
                    if hasattr(dataset, "finalize"):
                        finalize_start = time.perf_counter()
                        dataset.finalize()
                        logger.info(
                            "Dataset finalized in "
                            f"{time.perf_counter() - finalize_start:.2f} s."
                        )
                    break
            except Exception as e:
                logger.exception("Error occurred: %s", e)
                # A failed add_frame/save_episode means the episode on disk is
                # truncated/corrupt. Flag it so the recording loop raises
                # instead of letting the operator believe the save succeeded.
                if msg.get("type") in ("FRAME", "SAVE_EPISODE"):
                    self.writer_error.set()
            finally:
                # One task_done per get() keeps the JoinableQueue accounting
                # correct (previously a single task_done after the loop).
                self.queue.task_done()

        logger.info("Writer process finished.")

    def record_episode(
        self,
        data_fn: Callable[[], tuple[Observation, Action]],
        task: str,
        on_start: Callable[[], None] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> None:
        """Record a single episode from user-provided data function.

        Args:
            data_fn: A function that returns (obs, action) at each step.
            task: The task label for the episode.
            on_start: Optional hook called at the start of the episode.
            on_end: Optional hook called at the end (before save/delete).
        """
        try:
            self._wait_for_start_signal()
        except StopIteration:
            logger.info("Recording manager is shutting down.")
            return

        if on_start:
            logger.info("Resetting Environment.")
            on_start()

        logger.info("Started recording episode.")

        # Phase timing for THIS episode. The loop below is also the deployment
        # loop (deploy_policy.py hands it policy.make_data_fn()), so these
        # numbers describe policy control as well as teleop recording.
        # Measurement only: no branch below depends on `timing`.
        budget_s = 1.0 / self.config.fps
        timing = LoopTimingRecorder(
            label=f"episode_{self.episode_count:04d}",
            budget_s=budget_s,
            queue_capacity=self.queue_capacity,
            enabled=self.config.timing_report,
            csv_dir=self.config.timing_csv_dir,
        )
        # Sub-phase breakdown of data_fn (drive / step / collect / action) when
        # the data function publishes one — see record_functions.make_record_fn.
        sub_timing = getattr(data_fn, "timing", None)

        # DEADLINE pacing, not duration pacing. Sleeping for `budget - work`
        # makes every late wake permanently shift the schedule: with time.sleep
        # returning ~14 ms late (GIL contention with the ROS executor threads),
        # the field run captured 748 frames at 81 ms spacing while LeRobot
        # stamped them 66.7 ms apart — a 22% error in the dataset's time base,
        # which is what `action[t] = pose[t+1]` is measured against.
        # Targeting absolute deadlines instead makes the next sleep shorter by
        # exactly however late this one woke, so jitter cancels instead of
        # accumulating.
        pacer = DeadlinePacer(budget_s)

        while self.state == "recording":
            t_frame = time.perf_counter()

            t_mark = time.perf_counter()
            obs, action = data_fn()
            data_s = time.perf_counter() - t_mark

            if obs is None or action is None:
                logger.debug("Data function returned None, skipping frame.")
                # If the data function returns None, skip this frame
                sleep_requested, sleep_s = pacer.wait()
                timing.add_frame(
                    data_s=data_s,
                    put_s=0.0,
                    sleep_s=sleep_s,
                    total_s=time.perf_counter() - t_frame,
                    queue_depth=self.queue_depth(),
                    sleep_requested_s=sleep_requested,
                    sub_timing=sub_timing,
                    skipped=True,
                )
                continue

            if self.writer_error.is_set():
                raise RuntimeError(
                    "Dataset writer failed (see writer traceback above) — the "
                    "current episode is not being persisted. Aborting recording."
                )

            # Sampled BEFORE the put: the depth the frame actually met. At
            # capacity, the put below blocks until the writer drains a slot,
            # and that block is what shows up as a dropped teleop rate.
            queue_depth = self.queue_depth()
            t_mark = time.perf_counter()
            self._put_blocking({"type": "FRAME", "data": (obs, action, task)})
            put_s = time.perf_counter() - t_mark

            work_s = time.perf_counter() - t_frame
            sleep_requested, sleep_s = pacer.wait()
            if work_s > budget_s:
                blocked = "queue.put (writer back-pressure)" if put_s > data_s else "data_fn"
                logger.warning(
                    f"Frame work took {1e3 * work_s:.1f} ms, over the "
                    f"{1e3 * budget_s:.1f} ms budget by {1e3 * (work_s - budget_s):.1f} ms "
                    f"(i.e. {1.0 / work_s:.2f} FPS). Dominant phase: {blocked} — "
                    f"data_fn {1e3 * data_s:.1f} ms, queue.put {1e3 * put_s:.1f} ms, "
                    f"writer queue {queue_depth}/{self.queue_capacity} on entry."
                )

            timing.add_frame(
                data_s=data_s,
                put_s=put_s,
                sleep_s=sleep_s,
                total_s=time.perf_counter() - t_frame,
                queue_depth=queue_depth,
                sleep_requested_s=sleep_requested,
                sub_timing=sub_timing,
            )

        logger.debug("Finished recording...")

        extra = self.writer_status()
        if pacer.resyncs:
            extra += (
                f"; pacing resynced {pacer.resyncs}x (fell more than one period "
                "behind — those gaps are real time missing from the episode)"
            )
        timing.log_summary(extra=extra)

        if on_end:
            on_end()

        self._handle_post_episode()

    def _wait_for_start_signal(self) -> None:
        """Wait until the recording state is set to 'recording'."""
        logger.info("Waiting to start recording...")
        while self.state != "recording":
            if self.state == "exit":
                raise StopIteration
            time.sleep(0.05)

    def _handle_post_episode(self) -> None:
        """Handle the state after recording an episode."""
        if self.state == "paused":
            logger.info("Paused. Awaiting user decision to save/delete...")
            while self.state == "paused":
                time.sleep(0.5)

        if self.state == "to_be_saved":
            logger.info("Saving current episode.")
            self._put_blocking({"type": "SAVE_EPISODE"})
            self.episode_count += 1
            self._set_to_wait()
        elif self.state == "to_be_deleted":
            logger.info("Deleting current episode.")
            self._put_blocking({"type": "DELETE_EPISODE"})
            self._set_to_wait()
        elif self.state == "exit":
            pass
        else:
            logger.warning(f"Unexpected state after recording: {self.state}")

    def __enter__(self) -> "RecordingManager":  # noqa: D105
        """Enter the recording manager context."""
        print(Panel(self.get_instructions()))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001, D105
        """Exit the recording manager."""
        if exc_type is not None:
            logger.error(
                "An error occurred during recording. Shutting down the recording manager.",
                exc_info=(exc_type, exc_value, traceback),
            )

        if not self.config.push_to_hub:
            logger.info("Not pushing dataset to Hugging Face Hub.")
        else:
            self._put_blocking({"type": "PUSH_TO_HUB"}, required=False)
        logger.info("Shutting down the record process...")
        self._put_blocking({"type": "SHUTDOWN"}, required=False)

        # The writer still has to drain the queue, run the last save_episode
        # (video encode + parquet) and finalize(). That is minutes of work on a
        # large episode, so it gets its OWN timeout: writer_timeout (10s by
        # default) only ever covered an idle writer exiting, and expiring it
        # mid-encode terminates the process before finalize() writes the
        # parquet footer — losing the episode and leaving the dataset
        # unreadable.
        deadline = time.perf_counter() + self.config.shutdown_drain_timeout
        last_report = 0.0
        while self.writer.is_alive() and time.perf_counter() < deadline:
            self.writer.join(timeout=1.0)
            remaining = self.queue_depth()
            elapsed = self.config.shutdown_drain_timeout - (deadline - time.perf_counter())
            if elapsed - last_report >= 10.0:
                last_report = elapsed
                logger.info(
                    f"Waiting for the dataset writer to finish "
                    f"({remaining if remaining >= 0 else '?'} frames still "
                    f"queued, {elapsed:.0f}/{self.config.shutdown_drain_timeout:.0f}s). "
                    "It is encoding video and writing parquet — do not kill it."
                )

        if self.writer.is_alive():
            logger.error(
                "Dataset writer did not finish within "
                f"{self.config.shutdown_drain_timeout}s — terminating it. THE "
                "LAST EPISODE IS PROBABLY CORRUPT: finalize() never ran, so "
                "the parquet footer may be missing. Raise "
                "shutdown_drain_timeout above the time one save_episode takes."
            )
            self.writer.terminate()
            self.writer.join(timeout=self.config.writer_timeout)
        elif self.writer.exitcode not in (0, None):
            logger.error(
                f"Dataset writer exited with code {self.writer.exitcode} — the "
                "last episode may be incomplete."
            )

    def _set_to_wait(self) -> None:
        """Set to wait if possible."""
        if self.state not in ["to_be_saved", "to_be_deleted"]:
            raise ValueError("Can not go to wait state if the state is not to be saved or deleted!")
        if self.episode_count >= self.config.num_episodes:
            self.state = "exit"
        else:
            self.state = "is_waiting"


class ROSRecordingManager(RecordingManager):
    """ROS-based recording manager for controlling episode recording."""

    def __init__(self, config: RecordingManagerConfig | None = None, **kwargs) -> None:  # noqa: ANN003
        """Initialize ROS recording manager.

        Args:
            config: RecordingManagerConfig instance. If provided, **kwargs are ignored except for backwards compatibility.
            **kwargs: Individual parameters for backwards compatibility.
        """
        super().__init__(config=config, **kwargs)
        if not rclpy.ok():
            raise RuntimeError(
                "ROS2 is not initialized. Please initialize ROS2 before using the RecordingManager."
            )
        self.allowed_actions = ["record", "save", "delete", "exit"]
        self.node = rclpy.create_node("recording_manager")
        self._subscriber = self.node.create_subscription(
            String, "record_transition", self._callback_recording_trigger, 10
        )
        logger.debug("ROS2 node created and subscriber initialized.")

        threading.Thread(target=self._spin_node, daemon=True).start()

    def _spin_node(self):
        """Spin the ROS2 node in a separate thread."""
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    @override
    def get_instructions(self) -> str:
        """Returns the instructions to use the recording manager."""
        return (
            "[b]Published messages for recording state:[/b]\n"
            "<record> to start/stop recording.\n"
            "<save> to save the current recorded episode.\n"
            "<delete> to delete the current episode.\n"
            "<exit> to exit the recording manager."
        )

    def _callback_recording_trigger(self, msg: String) -> None:
        """Callback for recording state trigger.

        Args:
            msg: The message containing the recording state
        """
        if msg.data not in self.allowed_actions:
            print(f"[red]Invalid action received: {msg.data}[/red]")
            print("[yellow]Allowed actions are: record, save, delete, exit[/yellow]")
            return

        logger.debug(f"Received message: {msg.data}")
        logger.debug(f"Current state: {self.state}")

        if self.state == "is_waiting":
            if msg.data == "record":
                logger.debug("Transitioning to recording state.")
                self.state = "recording"
            if msg.data == "exit":
                logger.debug("Transitioning to exit state.")
                self.state = "exit"
        elif self.state == "recording":
            if msg.data == "record":
                logger.debug("Transitioning to paused state.")
                self.state = "paused"
        elif self.state == "paused":
            if msg.data == "exit":
                logger.debug("Transitioning to exit state.")
                self.state = "exit"
            if msg.data == "save":
                logger.debug("Transitioning to to_be_saved state.")
                self.state = "to_be_saved"
            if msg.data == "delete":
                logger.debug("Transitioning to to_be_deleted state.")
                self.state = "to_be_deleted"


# class KeyboardRecordingManager(RecordingManager):
#     """Keyboard-based recording manager for controlling episode recording."""

#     def __init__(self, config: RecordingManagerConfig | None = None, **kwargs) -> None:  # noqa: ANN003
#         """Initialize keyboard recording manager.

#         Args:
#             config: RecordingManagerConfig instance. If provided, **kwargs are ignored except for backwards compatibility.
#             **kwargs: Individual parameters for backwards compatibility.
#         """
#         super().__init__(config=config, **kwargs)
#         self.listener = keyboard.Listener(on_press=self._on_press)

#     @override
#     def get_instructions(self) -> str:
#         """Returns the instructions to use the recording manager."""
#         return "[b]Keys for recording:[/b]\n<r> To start/stop [b]R[/b]ecording.\n<s> To [b]S[/b]ave the current recorded episode.\n<d> to [b]D[/b]elete the current episode.\n<q> To [b]Q[/b]uit the recording."

#     def _on_press(self, key: keyboard.KeyCode | keyboard.Key | None) -> None:
#         """Handle keyboard press events.

#         Args:
#             key: The keyboard key that was pressed
#         """
#         if key is None:
#             return

#         if isinstance(key, keyboard.Key):
#             return

#         try:
#             if self.state == "is_waiting":
#                 if key.char == "r":
#                     self.state = "recording"
#                 if key.char == "q":
#                     self.state = "exit"
#             elif self.state == "recording":
#                 if key.char == "r":
#                     self.state = "paused"
#             elif self.state == "paused":
#                 if key.char == "q":
#                     self.state = "exit"
#                 if key.char == "s":
#                     self.state = "to_be_saved"
#                 if key.char == "d":
#                     self.state = "to_be_deleted"
#         except AttributeError:
#             pass

#     def stop(self) -> None:
#         """Stop the keyboard listener."""
#         self.listener.stop()

#     def __enter__(self) -> "RecordingManager":  # noqa: D105
#         self.listener.start()
#         return super().__enter__()

class KeyboardRecordingManager(RecordingManager):
    """Stdin-based recording manager for controlling episode recording."""

    def __init__(self, config: RecordingManagerConfig | None = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._running = False
        self._thread: threading.Thread | None = None

    @override
    def get_instructions(self) -> str:
        return (
            "[b]Keys for recording:[/b]\n"
            "<r> To start/stop [b]R[/b]ecording.\n"
            "<s> To [b]S[/b]ave the current recorded episode.\n"
            "<d> To [b]D[/b]elete the current episode.\n"
            "<q> To [b]Q[/b]uit the recording.\n"
            "\nPress key then ENTER."
        )

    def _input_loop(self) -> None:
        """Background thread reading stdin (key + ENTER; stdlib only, no pynput).

        Uses select() with a short timeout so the thread actually exits when
        stop()/__exit__ clears _running (a bare readline() blocks forever), and
        detects EOF (non-TTY / closed stdin) instead of busy-looping on it.
        """
        import select

        while self._running:
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not readable:
                    continue  # timeout: re-check _running
                line = sys.stdin.readline()
                if line == "":
                    # EOF — stdin closed / not a TTY; no input will ever come.
                    logger.warning("stdin closed — keyboard control disabled.")
                    break
                key = line.strip().lower()[:1]
                if key:
                    self._handle_key(key)
            except Exception:
                break

    def _handle_key(self, key: str) -> None:
        if self.state == "is_waiting":
            if key == "r":
                self.state = "recording"
            elif key == "q":
                self.state = "exit"

        elif self.state == "recording":
            if key == "r":
                self.state = "paused"

        elif self.state == "paused":
            if key == "q":
                self.state = "exit"
            elif key == "s":
                self.state = "to_be_saved"
            elif key == "d":
                self.state = "to_be_deleted"

    def stop(self) -> None:
        """Stop stdin listener thread."""
        self._running = False

    def __enter__(self) -> "RecordingManager":
        self._running = True
        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        super().__exit__(exc_type, exc_value, traceback)

def make_recording_manager(
    recording_manager_type: Literal["keyboard", "ros"],
    config: RecordingManagerConfig | None = None,
    config_path: Path | str | None = None,
    **kwargs: dict,
) -> RecordingManager:
    """Factory function to create a recording manager.

    Args:
        recording_manager_type: Type of recording manager to create.
        config: RecordingManagerConfig instance. Takes precedence over config_path.
        config_path: Path to YAML config file to load.
        **kwargs: Additional arguments to override config values or for backwards compatibility.

    Returns:
        A RecordingManager instance of the specified type.
    """
    if config is not None:
        if kwargs:
            config_dict = config.__dict__.copy()
            config_dict.update(kwargs)
            final_config = RecordingManagerConfig(**config_dict)
        else:
            final_config = config
    elif config_path is not None:
        final_config = RecordingManagerConfig.from_yaml(config_path, **kwargs)
    else:
        final_config = None

    if recording_manager_type == "keyboard":
        return KeyboardRecordingManager(config=final_config, **kwargs)
    elif recording_manager_type == "ros":
        return ROSRecordingManager(config=final_config, **kwargs)
    else:
        raise ValueError(f"Unknown recording manager type: {recording_manager_type}")
