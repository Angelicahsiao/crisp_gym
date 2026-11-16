"""Policy module for crisp_gym.

Two machines import this package, and neither has everything installed:

* the GPU machine that runs `scripts/policy_server.py` has torch + lerobot but
  no ROS, and
* a robot machine that only talks to that server needs neither torch nor
  lerobot.

`RelativeLerobotPolicy` and `RemotePolicy` keep their module tops free of all
three (torch and lerobot are imported inside the inference worker only), so
the package always imports.

The older `lerobot_policy` and `async_lerobot_policy` were removed (2026-11,
owner's decision): `relative_lerobot_policy` deploys every checkpoint they did,
and its `async_inference: true` is the asynchronous mode.
"""

from crisp_gym.policy.policy import Policy, make_policy, register_policy
from crisp_gym.policy.relative_lerobot_policy import RelativeLerobotPolicy
from crisp_gym.policy.remote_policy import RemotePolicy

__all__ = [
    "RelativeLerobotPolicy",
    "RemotePolicy",
    "Policy",
    "register_policy",
    "make_policy",
]
