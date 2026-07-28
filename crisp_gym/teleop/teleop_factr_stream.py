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
      /factr_teleop/{name}/cmd_ur_pos      — 6-DOF arm joint positions (position[0:6])
      /factr_teleop/{name}/cmd_gripper_pos — gripper trigger position (position[0])

    This class additionally PUBLISHES:
      /factr_teleop/{name}/go_home    (std_msgs/Bool, data=True)
          — request that the FACTR leader arm moves to its home pose (sent by
            send_home(), e.g. between recorded episodes). The FACTR node must
            subscribe and execute the homing motion itself — this is only the
            trigger.
      /factr_teleop/{name}/home_pose  (sensor_msgs/JointState, position[0:6])
          — the TARGET home joint configuration, published just before the
            go_home trigger when send_home(home_config=...) is given one.
            Because the follower's home pose is randomized per episode
            (--home-config-noise), the leader cannot assume a fixed home; the
            FACTR node should store the latest home_pose and move there on the
            go_home trigger (fall back to its own default if none received).

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
        self._home_topic = f"/factr_teleop/{name}/go_home"
        self._home_pose_topic = f"/factr_teleop/{name}/home_pose"

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
        self._home_publisher = self.node.create_publisher(
            Bool,
            self._home_topic,
            qos_profile_system_default,
            callback_group=ReentrantCallbackGroup(),
        )
        self._home_pose_publisher = self.node.create_publisher(
            JointState,
            self._home_pose_topic,
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

    def send_home(self, home_config: "list[float] | None" = None) -> None:
        """Ask the FACTR leader node to move the leader arm to its home pose.

        Publishes:
          1. the target joint configuration on /factr_teleop/{name}/home_pose
             (sensor_msgs/JointState, position field) — only when home_config
             is given. With --home-config-noise the follower's home differs
             every episode; this tells the leader the EXACT pose to match.
          2. std_msgs/Bool(data=True) on /factr_teleop/{name}/go_home — the
             trigger to execute the motion.

        Fire-and-forget: the FACTR node owns the actual homing motion. If it
        does not subscribe, the messages are ignored (a warning is logged).

        Args:
            home_config: Joint values (radians) the leader should home to —
                typically the follower's randomized home for this episode.
                None publishes only the trigger (leader uses its own default).
        """
        if self._home_publisher.get_subscription_count() == 0:
            logger.warning(
                f"send_home: no subscriber on {self._home_topic} — the FACTR "
                "node does not listen for home requests; the leader arm will "
                "NOT move. Add a subscriber in the FACTR node to enable this."
            )
        if home_config is not None:
            pose_msg = JointState()
            pose_msg.header.stamp = self.node.get_clock().now().to_msg()
            pose_msg.position = [float(v) for v in home_config]
            self._home_pose_publisher.publish(pose_msg)
            logger.info(
                f"Published FACTR leader home pose ({len(pose_msg.position)} "
                f"joints) via {self._home_pose_topic}."
            )
        msg = Bool()
        msg.data = True
        self._home_publisher.publish(msg)
        logger.info(f"Requested FACTR leader home via {self._home_topic}.")

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
