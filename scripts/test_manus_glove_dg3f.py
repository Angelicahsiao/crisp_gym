#!/usr/bin/env python
"""Smoke test: Manus glove → DG3F gripper.

Subscribes to the Manus glove topic, retargets each message into 12 DG3F joint
targets, and either prints them (--dry-run) or commands the real gripper.

Prerequisites:
  * Manus glove driver running and publishing on --glove-topic (default
    /manus_glove_0). Verify with: `ros2 topic echo /manus_glove_0`.
  * For non-dry-run: DG3F driver running with joint state and command topics
    matching crisp_gym/config/grippers/gripper_dg3f.yaml.

Usage:
  python scripts/test_manus_glove_dg3f.py --dry-run
  python scripts/test_manus_glove_dg3f.py --duration 20 --rate 30
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import rclpy

import crisp_gym  # noqa: F401
from crisp_gym.teleop.retargeting.ergonomics_retargeter import ErgonomicsRetargeter
from crisp_gym.teleop.teleop_manus_glove import ManusGloveTeleop
from crisp_gym.util.setup_logger import setup_logging

from crisp_py.gripper import MultiDofGripper, MultiDofGripperConfig


def _config_path(rel: str) -> Path:
    base = Path(crisp_gym.__file__).parent / "config"
    p = (base / rel).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return p


def _report(failures):
    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Manus glove → DG3F mapping.")
    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Seconds to wait for glove / gripper readiness.")
    parser.add_argument("--gripper-config", type=str,
                        default="grippers/gripper_dg3f.yaml",
                        help="Path relative to crisp_gym/config/.")
    parser.add_argument("--retargeting-config", type=str,
                        default="retargeting/manus_ergonomics_dg3f.yaml",
                        help="Path relative to crisp_gym/config/.")
    parser.add_argument("--glove-topic", type=str, default="/manus_glove_0")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print retargeted joints only; do not command the gripper.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="How long to run the teleop loop (seconds).")
    parser.add_argument("--rate", type=float, default=30.0, help="Loop rate (Hz).")
    args = parser.parse_args()

    setup_logging(level="INFO")
    logger = logging.getLogger("test_manus_glove_dg3f")

    failures = []

    # 1. Load configs and build the retargeter.
    try:
        gripper_yaml = _config_path(args.gripper_config)
        retarget_yaml = _config_path(args.retargeting_config)
        gripper_config = MultiDofGripperConfig.from_yaml(gripper_yaml)
        logger.info(
            f"Loaded gripper config: {gripper_config.num_joints} joints, "
            f"command_topic={gripper_config.command_topic}"
        )
        retargeter = ErgonomicsRetargeter.from_yaml(
            retarget_yaml,
            min_values=gripper_config.min_values,
            max_values=gripper_config.max_values,
        )
        logger.info(
            f"Loaded retargeter: {retargeter.num_joints} joints, "
            f"glove fields={retargeter.field_names}"
        )
    except Exception as e:
        failures.append(f"Config / retargeter setup failed: {e}")
        _report(failures)
        return 1

    if retargeter.num_joints != gripper_config.num_joints:
        failures.append(
            f"Retargeter outputs {retargeter.num_joints} joints but gripper "
            f"expects {gripper_config.num_joints}"
        )
        _report(failures)
        return 1

    # 2. Connect to the glove.
    try:
        glove = ManusGloveTeleop(
            retargeter=retargeter,
            topic=args.glove_topic,
            namespace=args.namespace,
        )
        glove.wait_until_ready(timeout=args.timeout)
        logger.info("✓ Glove ready, first retargeted message received")
    except Exception as e:
        failures.append(f"Glove not ready: {e}")
        _report(failures)
        return 1

    # 3. Optionally connect to the DG3F gripper.
    gripper = None
    if not args.dry_run:
        try:
            gripper = MultiDofGripper(
                namespace=args.namespace, gripper_config=gripper_config
            )
            gripper.wait_until_ready(timeout=args.timeout)
            logger.info(f"✓ Gripper ready, value shape={gripper.value.shape}")
        except Exception as e:
            failures.append(f"Gripper not ready: {e}")
            _report(failures)
            return 1

    # 4. Stream glove → gripper for the requested duration.
    period = 1.0 / args.rate
    end = time.time() + args.duration
    n_steps = 0
    log_every = max(1, int(args.rate))  # ~1 Hz log
    logger.info(
        f"Running glove → DG3F loop for {args.duration:.1f}s at {args.rate:.1f} Hz "
        f"({'dry-run' if args.dry_run else 'commanding gripper'})"
    )
    try:
        while time.time() < end and rclpy.ok():
            joints = glove.last_gripper_joints
            n_steps += 1
            if n_steps % log_every == 0:
                logger.info(
                    "normalized joints: " + " ".join(f"{v:.2f}" for v in joints)
                )
            if gripper is not None:
                gripper.set_target(joints)
            time.sleep(period)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    logger.info(f"Loop ran {n_steps} iterations")
    _report(failures)
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(rc)
