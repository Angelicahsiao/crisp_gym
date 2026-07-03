"""FACTR leader arm ROS subscriber — mirrors TeleopStreamedPose for joint-based teleop."""

import logging
import threading
import time

import numpy as np
import rclpy
import rclpy.executors
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

logger = logging.getLogger(__name__)


class FACTRStreamedJoints:
    """Subscribe to FACTR leader arm joint and gripper topics.

    FACTR publishes (both as sensor_msgs/JointState):
      /factr_teleop/{name}/cmd_ur_pos      — 6-DOF arm joint positions (position[0:6])
      /factr_teleop/{name}/cmd_gripper_pos — gripper trigger position (position[0])

    The gripper trigger is expected in [0, 1] where 0 = open, 1 = fully squeezed.
    It is inverted to match the Robotiq convention (set_target: 0 = closed, 1 = open),
    so squeezing the leader closes the follower.
    """

    def __init__(self, name: str = "right", namespace: str = ""):
        if not rclpy.ok():
            rclpy.init()

        self._name = name
        self._prefix = f"{namespace}_" if namespace else ""
        self.node = rclpy.create_node("factr_stream", namespace=namespace)

        self._joint_topic = f"/factr_teleop/{name}/cmd_ur_pos"
        self._gripper_topic = f"/factr_teleop/{name}/cmd_gripper_pos"

        self._last_joint_pos: np.ndarray | None = None
        self._last_gripper: float | None = None

        logger.info(f"Subscribing to: {self._joint_topic}, {self._gripper_topic}")

        self.node.create_subscription(
            JointState,
            self._joint_topic,
            self._callback_joints,
            callback_group=ReentrantCallbackGroup(),
            qos_profile=qos_profile_sensor_data,
        )
        self.node.create_subscription(
            JointState,
            self._gripper_topic,
            self._callback_gripper,
            callback_group=ReentrantCallbackGroup(),
            qos_profile=qos_profile_sensor_data,
        )

        threading.Thread(target=self._spin_node, daemon=True).start()

    def _spin_node(self):
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
        executor.add_node(self.node)
        try:
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.01)
        except Exception as e:
            logger.error(f"Executor error: {e}", exc_info=True)

    def _callback_joints(self, msg: JointState):
        self._last_joint_pos = np.array(msg.position, dtype=np.float64)

    def _callback_gripper(self, msg: JointState):
        # FACTR trigger (position[0]): 0 = released, 1 = squeezed (can overshoot
        # to ~[-1, 2]). Invert so squeezing the leader closes the follower, then
        # clamp into the [0, 1] range Gripper.set_target expects (0 = closed,
        # 1 = open).
        if not msg.position:
            return
        self._last_gripper = float(np.clip(1.0 - msg.position[0], 0.0, 1.0))

    @property
    def last_joint_pos(self) -> np.ndarray:
        if self._last_joint_pos is None:
            raise RuntimeError(
                f"No joint states received yet. Is FACTR running? "
                f"Check: ros2 topic echo {self._joint_topic}"
            )
        return self._last_joint_pos.copy()

    @property
    def last_gripper(self) -> float:
        if self._last_gripper is None:
            raise RuntimeError(
                f"No gripper value received yet. Is FACTR running? "
                f"Check: ros2 topic echo {self._gripper_topic}"
            )
        return self._last_gripper

    def is_ready(self) -> bool:
        return self._last_joint_pos is not None and self._last_gripper is not None

    def wait_until_ready(self, timeout: float = 10.0):
        start = time.time()
        logger.info("Waiting for first FACTR joint + gripper messages...")
        while not self.is_ready() and rclpy.ok():
            time.sleep(0.01)
            if time.time() - start > timeout:
                raise TimeoutError(
                    "Timed out waiting for FACTR stream. "
                    f"Check topics: {self._joint_topic}, {self._gripper_topic}"
                )
        logger.info("FACTR stream ready.")
