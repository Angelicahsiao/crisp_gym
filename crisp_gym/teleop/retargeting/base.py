"""Abstract base for glove → gripper joint retargeting strategies."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class GloveRetargeter(ABC):
    """Strategy interface: glove message → normalized gripper joint targets.

    Each concrete strategy declares the ROS2 message type it consumes and the
    number of output joints. ``retarget`` converts a single message into an
    array of normalized joint targets in [0, 1] sized for the target gripper.
    """

    @property
    @abstractmethod
    def topic_type(self) -> type:
        """ROS2 message type this retargeter subscribes to."""

    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Number of output gripper joints."""

    @abstractmethod
    def retarget(self, msg: Any) -> np.ndarray:
        """Convert a glove message to a (num_joints,) array in [0, 1]."""
