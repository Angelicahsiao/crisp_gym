"""The policy registry: what `make_policy` says when a policy is not there.

No ROS, torch or lerobot: `crisp_gym.policy` imports without them.
"""

import pytest

from crisp_gym.policy.policy import make_policy, policy_registry


def test_unknown_policy_error_is_one_readable_message():
    with pytest.raises(ValueError) as caught:
        make_policy("no_such_policy")
    err = caught.value
    # one message, not a tuple of two (which prints as "('...', '...')")
    assert len(err.args) == 1
    text = str(err)
    assert "no_such_policy" in text
    assert "not registered" in text
    assert not text.startswith("(")


def test_unknown_policy_error_lists_the_registered_ones():
    with pytest.raises(ValueError) as caught:
        make_policy("no_such_policy")
    for name in policy_registry:
        assert name in str(caught.value)
