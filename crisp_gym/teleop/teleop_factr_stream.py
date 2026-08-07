"""FACTR leader arm ROS subscriber — mirrors TeleopStreamedPose for joint-based teleop."""

import logging
import threading
import time

import numpy as np
import rclpy
import rclpy.executors
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

logger = logging.getLogger(__name__)


class FACTRStreamedJoints:
    """Subscribe to FACTR leader arm joint and gripper topics.

    FACTR publishes (both as sensor_msgs/JointState):
      /factr_teleop/{name}/cmd_arm_pos     — arm joint positions, one per follower
                                             joint (position[0:N]; N=6 on the UR7e,
                                             7 on the FR3)
      /factr_teleop/{name}/cmd_gripper_pos — gripper trigger position (position[0])

    This class additionally PUBLISHES:
      /factr_teleop/{name}/follow_mode (std_msgs/Bool)
          — whether the FACTR leader should FOLLOW the follower arm rather than
            command it (sent by set_follow_mode()).

            data=True  — entered between recorded episodes, while the follower
                homes to its (per-episode randomized, see --home-config-noise)
                pose. The leader tracks the follower there, so both arms end up
                in the same configuration without crisp_gym having to tell the
                leader which pose that is.
            data=False — teleoperation resumes; the leader commands again.

            The FACTR node must subscribe and implement the mode switch itself
            — this is only the request. Nothing is published unless
            set_follow_mode() is called, and no state is latched here.

    The gripper trigger is expected in [0, 1] ALREADY in the follower's
    convention (Gripper.set_target: 0 = closed, 1 = open) and is passed through
    unchanged — only clamped, since the trigger can overshoot to roughly
    [-1, 2]. The FACTR node owns the mapping from its physical trigger to this
    range; if squeezing the leader OPENS the follower, invert it there, not here.
    """

    def __init__(self, name: str = "right", namespace: str = ""):
        if not rclpy.ok():
            rclpy.init()

        self._name = name
        self._prefix = f"{namespace}_" if namespace else ""
        self.node = rclpy.create_node("factr_stream", namespace=namespace)

        self._joint_topic = f"/factr_teleop/{name}/cmd_arm_pos"
        self._gripper_topic = f"/factr_teleop/{name}/cmd_gripper_pos"
        self._follow_mode_topic = f"/factr_teleop/{name}/follow_mode"

        self._last_joint_pos: np.ndarray | None = None
        self._last_gripper: float | None = None

        # Message counters, so callers can tell a FRESH sample from the stale
        # one cached before follow mode silenced the stream. See
        # wait_for_new_data().
        self._joint_msg_count = 0
        self._gripper_msg_count = 0

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
        self._follow_mode_publisher = self.node.create_publisher(
            Bool,
            self._follow_mode_topic,
            qos_profile_system_default,
            callback_group=ReentrantCallbackGroup(),
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
        self._joint_msg_count += 1

    def _callback_gripper(self, msg: JointState):
        # FACTR trigger (position[0]) is already in the follower's convention
        # (Gripper.set_target: 0 = closed, 1 = open), so it is passed through
        # unchanged — only clamped, because the trigger can overshoot its
        # nominal range to roughly [-1, 2].
        if not msg.position:
            return
        self._last_gripper = float(np.clip(msg.position[0], 0.0, 1.0))
        self._gripper_msg_count += 1

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

    def wait_for_new_data(self, timeout: float = 5.0) -> bool:
        """Block until a joint AND gripper message arrive that postdate this call.

        The FACTR node stops publishing while it is following, so
        last_joint_pos / last_gripper keep returning whatever was cached before
        follow mode began. Commanding that stale pose is dangerous in absolute
        mode: it is the leader's position from BEFORE it moved to track the
        follower, so the arm would be sent back there. Call this after leaving
        follow mode, before the first command, so the values are known-fresh.

        Args:
            timeout: Seconds to wait for a new sample on both topics.

        Returns:
            True if both topics produced a new message, False on timeout.
        """
        joints_at_call = self._joint_msg_count
        gripper_at_call = self._gripper_msg_count
        start = time.time()
        while time.time() - start < timeout:
            if (
                self._joint_msg_count > joints_at_call
                and self._gripper_msg_count > gripper_at_call
            ):
                return True
            time.sleep(0.01)
        return False

    def wait_for_follow_mode_subscriber(self, timeout: float = 5.0) -> bool:
        """Block until the FACTR node has subscribed to the follow_mode topic.

        The publisher is VOLATILE, so anything sent before the FACTR node has
        discovered it is dropped silently. Call this before a set_follow_mode()
        whose message MUST arrive — notably the one sent at startup, before
        wait_until_ready(), which is what puts the leader into follow mode in
        the first place.

        Args:
            timeout: Seconds to wait for a subscriber to appear.

        Returns:
            True if a subscriber appeared, False if the timeout elapsed.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self._follow_mode_publisher.get_subscription_count() > 0:
                return True
            time.sleep(0.05)
        return False

    def set_follow_mode(self, enabled: bool) -> None:
        """Ask the FACTR leader node to enter or leave follow mode.

        Publishes std_msgs/Bool(data=enabled) on
        /factr_teleop/{name}/follow_mode. Equivalent to:

            ros2 topic pub --once /factr_teleop/right/follow_mode \\
                std_msgs/msg/Bool "{data: true}"

        In follow mode the leader TRACKS the follower instead of commanding it.
        Recording turns it on between episodes, while the follower homes to a
        per-episode randomized pose, so both arms converge without crisp_gym
        having to send the pose itself; it is turned off again when the next
        episode starts.

        Fire-and-forget: the FACTR node owns the actual mode switch. If it does
        not subscribe, the message goes nowhere (a warning is logged).

        Args:
            enabled: True to follow the follower, False to resume commanding it.
        """
        if self._follow_mode_publisher.get_subscription_count() == 0:
            logger.warning(
                f"set_follow_mode: no subscriber on {self._follow_mode_topic} "
                "— the FACTR node does not listen for follow-mode requests; "
                "the leader arm will NOT change mode. Add a subscriber in the "
                "FACTR node to enable this."
            )
        msg = Bool()
        msg.data = bool(enabled)
        self._follow_mode_publisher.publish(msg)
        logger.info(
            f"Requested FACTR leader follow_mode={msg.data} via "
            f"{self._follow_mode_topic}."
        )

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
