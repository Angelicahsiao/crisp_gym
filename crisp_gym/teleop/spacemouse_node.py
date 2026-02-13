"""SpaceMouse publisher node for teleoperation.

This module reads events from a 3Dconnexion SpaceMouse (via PySpaceMouse)
and publishes pose updates (PoseStamped) and gripper values (Float32) on the
same topics used by the teleop streamer.

Install dependency on Linux:

    sudo apt install libhidapi-dev
    pip install pyspacemouse

If `pyspacemouse` is not available the node will raise an ImportError with instructions.
"""

import argparse
import math
import threading
import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from rclpy.qos import qos_profile_sensor_data

try:
    import pyspacemouse
except ImportError:
    pyspacemouse = None


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).
    
    Uses ZYX convention (yaw-pitch-roll).
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)


class SpaceMouseStreamer:
    """ROS2 node publishing SpaceMouse as pose + gripper.

    Publishes:
      - PoseStamped on `/{prefix}phone_pose` (default prefix is empty)
      - Float32 on `/{prefix}phone_gripper`

    The node reads motion events and increments an internal pose. This is a
    pragmatic approach so the device behaves similarly to mobile/phone
    streamers that send absolute poses.
    """

    def __init__(
        self,
        namespace: str = "",
        trans_scale: float = 0.001,
        rot_scale: float = 0.002,
        publish_rate: float = 50.0,
        axis_signs: Optional[Tuple[float, float, float, float, float, float]] = None,
        button_map: Optional[dict] = None,
        device_name: Optional[str] = None,
    ):
        if pyspacemouse is None:
            raise ImportError(
                "pyspacemouse not available. Install with 'sudo apt install libhidapi-dev' and 'pip install pyspacemouse'."
            )

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("spacemouse_streamer", namespace=namespace)

        self._prefix = f"{namespace}_" if namespace else ""
        self._pose_topic = f"/{self._prefix}phone_pose"
        self._gripper_topic = f"/{self._prefix}phone_gripper"

        self._pub_pose = self.node.create_publisher(PoseStamped, self._pose_topic, qos_profile_sensor_data)
        self._pub_gripper = self.node.create_publisher(Float32, self._gripper_topic, qos_profile_sensor_data)

        self._trans_scale = trans_scale
        self._rot_scale = rot_scale
        self._publish_rate = publish_rate
        
        # axis_signs controls sign for x,y,z,roll,pitch,yaw respectively
        if axis_signs is None:
            self._axis_signs = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        else:
            self._axis_signs = tuple(float(s) for s in axis_signs)

        # button_map maps button number to action. Supported actions:
        #   'toggle_gripper', 'gripper_open', 'gripper_close', 'set:<value>'
        self._button_map = button_map or {0: "toggle_gripper"}
        self._device_name = device_name

        # internal state
        self._pos = [0.0, 0.0, 0.0]
        self._euler = [0.0, 0.0, 0.0]  # roll, pitch, yaw
        self._gripper = 0.0
        self._device = None
        self._last_button_state: list = []

        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _publish(self):
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "spacemouse"
        msg.pose.position.x = float(self._pos[0])
        msg.pose.position.y = float(self._pos[1])
        msg.pose.position.z = float(self._pos[2])
        
        # Convert Euler angles to quaternion
        w, x, y, z = _euler_to_quat(self._euler[0], self._euler[1], self._euler[2])
        msg.pose.orientation.w = float(w)
        msg.pose.orientation.x = float(x)
        msg.pose.orientation.y = float(y)
        msg.pose.orientation.z = float(z)
        self._pub_pose.publish(msg)
        self._pub_gripper.publish(Float32(data=float(self._gripper)))

    def _handle_button(self, button_state: list):
        """Handle button state changes."""
        for idx, pressed in enumerate(button_state):
            prev_pressed = self._last_button_state[idx] if idx < len(self._last_button_state) else 0
            # Only trigger on press (0->1 transition)
            if pressed and not prev_pressed:
                action = self._button_map.get(idx)
                if not action:
                    continue
                if action == "toggle_gripper":
                    self._gripper = 1.0 - self._gripper
                elif action == "gripper_open":
                    self._gripper = 1.0
                elif action == "gripper_close":
                    self._gripper = 0.0
                elif isinstance(action, str) and action.startswith("set:"):
                    try:
                        val = float(action.split(":", 1)[1])
                        self._gripper = max(0.0, min(1.0, val))
                    except Exception:
                        pass
        self._last_button_state = button_state.copy()

    def _run(self):
        try:
            # Open the SpaceMouse device
            if self._device_name:
                self._device = pyspacemouse.open(device=self._device_name)
            else:
                self._device = pyspacemouse.open()
        except Exception as e:
            self.node.get_logger().error(f"Failed to open SpaceMouse: {e}")
            return

        last_pub = 0.0
        try:
            while self._running and rclpy.ok():
                try:
                    state = self._device.read()
                    if state is not None:
                        # Update position (incremental)
                        dx = state.x * self._trans_scale * self._axis_signs[0]
                        dy = state.y * self._trans_scale * self._axis_signs[1]
                        dz = state.z * self._trans_scale * self._axis_signs[2]
                        
                        # Debug: log motion events
                        if abs(state.x) > 50 or abs(state.y) > 50 or abs(state.z) > 50 or \
                           abs(state.roll) > 50 or abs(state.pitch) > 50 or abs(state.yaw) > 50:
                            self.node.get_logger().info(
                                f"SpaceMouse motion: x={state.x}, y={state.y}, z={state.z}, "
                                f"roll={state.roll}, pitch={state.pitch}, yaw={state.yaw}"
                            )
                        
                        self._pos[0] += dx
                        self._pos[1] += dy
                        self._pos[2] += dz

                        # Update orientation (incremental, using Euler angles)
                        droll = state.roll * self._rot_scale * self._axis_signs[3]
                        dpitch = state.pitch * self._rot_scale * self._axis_signs[4]
                        dyaw = state.yaw * self._rot_scale * self._axis_signs[5]
                        self._euler[0] += droll
                        self._euler[1] += dpitch
                        self._euler[2] += dyaw

                        # Handle buttons
                        if state.buttons:
                            self._handle_button(state.buttons)

                        # Publish at desired rate
                        now = time.time()
                        if now - last_pub >= 1.0 / max(1.0, self._publish_rate):
                            self._publish()
                            last_pub = now
                    time.sleep(0.001)
                except Exception as e:
                    self.node.get_logger().warn(f"Error reading SpaceMouse: {e}")
                    time.sleep(0.1)
        finally:
            if self._device is not None:
                try:
                    self._device.close()
                except Exception:
                    pass

    def shutdown(self):
        self._running = False


if __name__ == "__main__":
    # CLI for running the streamer with customization
    parser = argparse.ArgumentParser(description="Run SpaceMouse streamer for teleoperation")
    parser.add_argument("--namespace", type=str, default="", help="ROS namespace/prefix for topics")
    parser.add_argument("--trans-scale", type=float, default=0.001, help="Translation scale (m per device unit)")
    parser.add_argument("--rot-scale", type=float, default=0.002, help="Rotation scale (rad per device unit)")
    parser.add_argument("--publish-rate", type=float, default=50.0, help="Publish rate in Hz")
    parser.add_argument(
        "--axis-signs",
        type=str,
        default="1,1,1,1,1,1",
        help="Comma-separated signs for x,y,z,roll,pitch,yaw (e.g. 1,-1,1,1,1,-1)",
    )
    parser.add_argument(
        "--button-map",
        type=str,
        default="",
        help="Button mapping list, e.g. '0:toggle_gripper,1:gripper_open,2:set:0.5'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device name (e.g. 'SpaceNavigator', 'SpaceMouse Pro'). If not set, auto-detects the first device.",
    )

    args = parser.parse_args()

    def parse_axis_signs(s: str):
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) != 6:
            raise SystemExit("axis-signs must have 6 comma-separated values")
        return tuple(float(p) for p in parts)

    def parse_button_map(s: str):
        if not s:
            return None
        result = {}
        for token in s.split(","):
            token = token.strip()
            if not token or ":" not in token:
                continue
            idx, act = token.split(":", 1)
            try:
                result[int(idx)] = act
            except Exception:
                continue
        return result

    axis_signs = parse_axis_signs(args.axis_signs)
    button_map = parse_button_map(args.button_map)

    try:
        streamer = SpaceMouseStreamer(
            namespace=args.namespace,
            trans_scale=args.trans_scale,
            rot_scale=args.rot_scale,
            publish_rate=args.publish_rate,
            axis_signs=axis_signs,
            button_map=button_map,
            device_name=args.device,
        )
        print("SpaceMouse streamer running — press Ctrl+C to exit")
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            streamer.shutdown()
        except Exception:
            pass
