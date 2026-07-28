"""Contains some home configurations."""
# TODO: make the configs robot specific

from enum import Enum

home_close_to_table = [
    -1.73960110e-02,
    9.55319758e-02,
    8.09703053e-04,
    -1.94272034e00,
    -4.01435784e-03,
    2.06584183e00,
    7.97426445e-01,
]

home_front_up = [
    -0.02312892,
    -0.10664185,
    -0.0195703,
    -1.75644521,
    -0.00732298,
    1.68992915,
    0.8040582,
]

home_open_up = [
    0.425725623070977,
    -0.013800044320084788,
    -0.33286129072276527,
    -2.7729492382868126,
    0.10167396715537252,
    4.262024898082136,
    -0.021739227284989265,
]

class HomeConfig(Enum):
    """Enum for different home configurations."""

    CLOSE_TO_TABLE = home_close_to_table
    FRONT_UP = home_front_up
    OPEN_POSE = home_open_up

    def randomize(self, noise: float = 0.01) -> list:
        """Randomize the home configuration."""
        import numpy as np

        return (
            np.array(self.value) + np.random.uniform(-noise, noise, size=len(self.value))
        ).tolist()


def randomized_home_for(
    nq: int,
    fallback: list,
    preferred: "HomeConfig | None" = None,
    noise: float = 0.01,
) -> list:
    """Pick a home configuration matching the robot's joint count, randomized.

    The HomeConfig enum poses are FRANKA (7-joint) configurations. Sending one
    to a robot with a different joint count makes the controller silently
    reject the trajectory (the robot just never homes — the classic symptom on
    a 6-joint UR). This helper uses `preferred` only when its length matches
    `nq`, otherwise the robot's own `fallback` (e.g. robot.config.home_config).

    Args:
        nq: The robot's number of joints (robot.nq).
        fallback: Home configuration of the right length (robot.config.home_config).
        preferred: Optional HomeConfig to use when its joint count matches.
        noise: Uniform noise added per joint.
    """
    import numpy as np

    if preferred is not None and len(preferred.value) == nq:
        return preferred.randomize(noise=noise)
    if len(fallback) != nq:
        raise ValueError(
            f"fallback home config has {len(fallback)} joints, robot has {nq}."
        )
    return (
        np.array(fallback, dtype=float) + np.random.uniform(-noise, noise, size=nq)
    ).tolist()


def home_for_env(env, name: str, noise: float = 0.01) -> list:
    """Resolve a NAMED home pose for this env's robot, randomized.

    Resolution order (first hit wins):
      1. env.config.named_home_configs[name]      — per-robot pose from the env
         YAML (the consistent way to define e.g. "open_pose" for a UR).
      2. HomeConfig[NAME] enum pose               — legacy Franka poses, used
         ONLY when the joint count matches the robot.
      3. env.robot.config.home_config             — the robot's own home.

    Raises ValueError if a named entry exists but its length does not match
    the robot's joint count (never silently sends a wrong-size trajectory).
    """
    import numpy as np

    nq = env.robot.nq
    named = getattr(env.config, "named_home_configs", None) or {}
    if name in named:
        cfg = list(named[name])
        if len(cfg) != nq:
            raise ValueError(
                f"named_home_configs['{name}'] has {len(cfg)} joints but the "
                f"robot has {nq} — fix the env YAML."
            )
        return (np.array(cfg, dtype=float) + np.random.uniform(-noise, noise, size=nq)).tolist()

    enum_name = name.upper()
    preferred = HomeConfig[enum_name] if enum_name in HomeConfig.__members__ else None
    return randomized_home_for(nq, env.robot.config.home_config, preferred=preferred, noise=noise)
