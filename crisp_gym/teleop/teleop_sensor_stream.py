"""Class defining the teleoperation for a pose streamer."""

import logging
import threading
import time

import rclpy
import rclpy.executors
from crisp_py.utils.geometry import Pose
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32

logger = logging.getLogger(__name__)


class TeleopStreamedPose:
    """Class to handle teleoperation using streamed pose and gripper data potentially from a phone, VR device, etc."""

    def __init__(
        self,
        namespace: str = "",
        pose_topic: str | None = None,
        gripper_topic: str | None = None,
    ):
        """Initialize the TeleopStreamedPose class.

        Args:
            namespace: ROS2 namespace.
            pose_topic: Override the default pose topic (default: /{namespace}_phone_pose).
            gripper_topic: Override the default gripper topic (default: /{namespace}_phone_gripper).
        """
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("pose_streamer", namespace=namespace)

        self._prefix = f"{namespace}_" if namespace else ""

        self._last_pose: Pose | None = None
        self._last_gripper: float | None = None

        self._gripper_topic = gripper_topic or f"/{self._prefix}phone_gripper"
        self._pose_topic = pose_topic or f"/{self._prefix}phone_pose"
        
        logger.info(f"Creating subscriptions to: {self._pose_topic}, {self._gripper_topic}")

        self._pose_sub = self.node.create_subscription(
            PoseStamped,
            self._pose_topic,
            callback=self._callback_pose,
            callback_group=ReentrantCallbackGroup(),
            qos_profile=qos_profile_sensor_data,
        )
        
        self._gripper_sub = self.node.create_subscription(
            Float32,
            self._gripper_topic,
            callback=self._callback_gripper,
            callback_group=ReentrantCallbackGroup(),
            qos_profile=qos_profile_sensor_data,
        )
        
        logger.info("Subscriptions created, starting executor thread...")

        threading.Thread(target=self._spin_node, daemon=True).start()

    @property
    def last_pose(self) -> Pose:
        """Get the last received pose.

        Returns:
            Pose | None: The last received pose or None if no pose has been received yet.
        """
        if self._last_pose is None:
            raise RuntimeError(
                f"No pose received yet. Is the teleop device running? Check with 'ros2 topic echo {self._pose_topic}'"
            )
        return self._last_pose

    @property
    def last_gripper(self) -> float:
        """Get the last received gripper value.

        Returns:
            float | None: The last received gripper value or None if no gripper value has been received yet.
        """
        if self._last_gripper is None:
            raise RuntimeError("No gripper value received yet. Is the teleop device running?")
        return self._last_gripper

    def _spin_node(self):
        if not rclpy.ok():
            rclpy.init()
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
        executor.add_node(self.node)
        logger.info("Executor thread started, spinning...")
        try:
            while rclpy.ok():
                # Use 0.01s timeout for responsive callback processing (10ms instead of 100ms)
                executor.spin_once(timeout_sec=0.01)
        except Exception as e:
            logger.error(f"Executor error: {e}", exc_info=True)
        finally:
            logger.info("Executor thread stopping")

    def _callback_gripper(self, msg: Float32):
        self._last_gripper = msg.data
        # logger.debug(f"Gripper callback: {msg.data}")

    def _callback_pose(self, msg: PoseStamped):
        # logger.info(f"PoseStamped callback received: pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})")
        try:
            self._last_pose = Pose.from_ros_msg(msg)
        except Exception as e:
            logger.error(f"Error converting ROS pose to Pose: {e}", exc_info=True)

    def is_ready(self) -> bool:
        """Check if the leader robot and its gripper are ready.

        Returns:
            bool: True if both the leader robot and its gripper are ready, False otherwise.
        """
        return self._last_pose is not None

    def wait_until_ready(self, timeout: float = 5.0):
        """Wait until the leader robot and its gripper are ready."""
        start_time = time.time()
        logger.info("Waiting for first pose message...")
        while not self.is_ready() and rclpy.ok():
            # Don't spin directly - let the background executor thread handle it
            time.sleep(0.01)
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for the teleop streamer to be ready.")
        if not rclpy.ok():
            raise RuntimeError("ROS2 has been shutdown.")
        logger.info("✓ First pose received, ready to proceed")
