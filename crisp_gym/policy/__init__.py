"""Policy module for crisp_gym.

Two machines import this package, and neither has everything installed:

* the GPU machine that runs `scripts/policy_server.py` has torch + lerobot but
  no ROS, and
* a robot machine that only talks to that server needs neither torch nor
  lerobot.

`RelativeLerobotPolicy` and `RemotePolicy` keep their module tops free of all
three (torch and lerobot are imported inside the inference worker only), so
they always import. `LerobotPolicy` and `AsyncLerobotPolicy` import torch,
lerobot and the ROS-bound env at module load; when one of those is missing
they are left out of the registry instead of making the whole package — and
with it `remote_protocol` — unimportable.
"""

import logging

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

try:
    from crisp_gym.policy.async_lerobot_policy import AsyncLerobotPolicy
    from crisp_gym.policy.lerobot_policy import LerobotPolicy

    __all__ += ["LerobotPolicy", "AsyncLerobotPolicy"]
except ImportError as exc:  # torch / lerobot / ROS not installed on this machine
    logging.getLogger(__name__).info(
        "lerobot_policy and async_lerobot_policy are not available here (%s); "
        "relative_lerobot_policy and remote_policy still are.", exc,
    )
