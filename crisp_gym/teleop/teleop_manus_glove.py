"""Teleoperation input from a Manus glove via a retargeting strategy."""

import logging
import threading
import time
from typing import Optional

import numpy as np
import rclpy
import rclpy.executors
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data

from crisp_gym.teleop.retargeting.base import GloveRetargeter

logger = logging.getLogger(__name__)


class ManusGloveTeleop:
    """Subscribe to a Manus glove topic, apply a retargeter, expose joint targets.

    The retargeter chooses both the ROS2 message type to subscribe to and the
    mapping from glove data to normalized [0, 1] gripper joint targets.
    ``last_gripper_joints`` returns a copy of the most recent retargeted vector.
    """

    def __init__(
        self,
        retargeter: GloveRetargeter,
        topic: str = "/manus_glove_0",
        namespace: str = "",
    ):
        if not rclpy.ok():
            rclpy.init()

        self._retargeter = retargeter
        self._topic = topic
        self._last_joints: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        self.node = rclpy.create_node("manus_glove_teleop", namespace=namespace)
        self._sub = self.node.create_subscription(
            retargeter.topic_type,
            topic,
            self._callback,
            callback_group=ReentrantCallbackGroup(),
            qos_profile=qos_profile_sensor_data,
        )
        logger.info(
            f"Subscribed to {topic} ({retargeter.topic_type.__name__}), "
            f"num_joints={retargeter.num_joints}"
        )

        threading.Thread(target=self._spin, daemon=True).start()

    def _spin(self):
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
        executor.add_node(self.node)
        try:
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.01)
        except Exception as e:
            logger.error(f"Executor error: {e}", exc_info=True)

    def _callback(self, msg):
        try:
            joints = self._retargeter.retarget(msg)
        except Exception as e:
            logger.error(f"Retargeting failed: {e}", exc_info=True)
            return
        with self._lock:
            self._last_joints = joints

    @property
    def num_joints(self) -> int:
        return self._retargeter.num_joints

    @property
    def last_gripper_joints(self) -> np.ndarray:
        with self._lock:
            if self._last_joints is None:
                raise RuntimeError(
                    f"No glove message received yet. "
                    f"Check with: ros2 topic echo {self._topic}"
                )
            return self._last_joints.copy()

    def is_ready(self) -> bool:
        with self._lock:
            return self._last_joints is not None

    def wait_until_ready(self, timeout: float = 5.0):
        start = time.time()
        while not self.is_ready() and rclpy.ok():
            time.sleep(0.05)
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Timed out after {timeout}s waiting for glove data on {self._topic}"
                )
        if not rclpy.ok():
            raise RuntimeError("ROS2 has been shut down.")
