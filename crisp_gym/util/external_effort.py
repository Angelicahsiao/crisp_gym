"""Estimate external joint effort by subtracting model gravity torque.

The UR reports current-derived *total* joint torque in /joint_states, which
includes the torque spent holding the arm against gravity. This module builds
a Pinocchio model from the robot's URDF (ideally the live /robot_description,
which already includes the Robotiq gripper masses) and subtracts the gravity
term g(q):

    tau_ext = tau_measured - (a * g(q) + b)

where a/b are optional per-joint calibration gains fitted from data recorded
while nothing touches the arm (compensates current-to-torque scale errors and
static friction offsets). Quasi-static assumption: inertial/Coriolis torques
are not subtracted, so readings during fast motion overestimate contact.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String


def fetch_robot_description(node: Node, timeout_sec: float = 5.0) -> str:
    """Fetch the URDF from the latched /robot_description topic.

    The node must already be spinning (crisp_py's Robot spins its node in a
    background thread by default) - this waits for the executor to deliver
    the latched message rather than spinning the node itself.
    """
    import threading

    urdf: list[str] = []
    received = threading.Event()

    def _on_msg(msg: String) -> None:
        urdf.append(msg.data)
        received.set()

    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    sub = node.create_subscription(String, "/robot_description", _on_msg, qos)
    try:
        if not received.wait(timeout=timeout_sec):
            raise TimeoutError(
                "No message on /robot_description - is the robot bringup "
                "running and the node spinning?"
            )
    finally:
        node.destroy_subscription(sub)
    return urdf[0]


class ExternalEffortEstimator:
    """Gravity-free joint effort from measured effort and a Pinocchio model."""

    def __init__(
        self,
        urdf: str,
        joint_names: list[str],
        scale: NDArray | None = None,
        offset: NDArray | None = None,
    ):
        """Build the estimator.

        Args:
            urdf: URDF as an XML string (e.g. from fetch_robot_description).
            joint_names: Actuated arm joints, in the order used by crisp_py
                (robot.config.joint_names). All other movable joints in the
                URDF (e.g. gripper fingers) are locked at their neutral
                configuration, so their mass still loads the wrist.
            scale: Optional per-joint calibration gain a (default 1).
            offset: Optional per-joint calibration offset b (default 0).
        """
        full_model = pin.buildModelFromXML(urdf)
        lock_ids = [
            full_model.getJointId(name)
            for name in full_model.names
            if name != "universe" and name not in joint_names
        ]
        self.model = pin.buildReducedModel(
            full_model, lock_ids, pin.neutral(full_model)
        )
        self.data = self.model.createData()

        missing = [n for n in joint_names if not self.model.existJointName(n)]
        if missing:
            raise ValueError(f"Joints not found in URDF: {missing}")
        # Map crisp_py joint order -> pinocchio q indices (1-DOF joints only).
        self._q_index = np.array(
            [self.model.joints[self.model.getJointId(n)].idx_q for n in joint_names]
        )
        n = len(joint_names)
        self.scale = np.ones(n) if scale is None else np.asarray(scale, dtype=float)
        self.offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)

    @classmethod
    def from_robot(cls, robot, **kwargs) -> "ExternalEffortEstimator":
        """Build from a connected crisp_py Robot (URDF via /robot_description)."""
        urdf = fetch_robot_description(robot.node)
        return cls(urdf, list(robot.config.joint_names), **kwargs)

    def gravity_effort(self, q: NDArray) -> NDArray:
        """Model gravity torque g(q) in crisp_py joint order."""
        q_pin = pin.neutral(self.model)
        q_pin[self._q_index] = q
        tau_g = pin.computeGeneralizedGravity(self.model, self.data, q_pin)
        # For a serial arm with 1-DOF joints idx_v == idx_q ordering holds.
        return tau_g[self._q_index]

    def external_effort(self, q: NDArray, tau_measured: NDArray) -> NDArray:
        """tau_ext = tau_measured - (scale * g(q) + offset)."""
        return np.asarray(tau_measured) - (
            self.scale * self.gravity_effort(q) + self.offset
        )

    def fit_calibration(
        self, qs: NDArray, taus_measured: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Fit per-joint scale/offset from contact-free samples.

        Record (q, tau_measured) pairs while the arm moves slowly with nothing
        touching it, then least-squares fit tau_measured ~ a * g(q) + b per
        joint. Stores and returns (scale, offset).
        """
        qs = np.asarray(qs)
        taus = np.asarray(taus_measured)
        gravity = np.stack([self.gravity_effort(q) for q in qs])
        for j in range(gravity.shape[1]):
            A = np.stack([gravity[:, j], np.ones(len(qs))], axis=1)
            (a, b), *_ = np.linalg.lstsq(A, taus[:, j], rcond=None)
            self.scale[j], self.offset[j] = a, b
        return self.scale, self.offset
