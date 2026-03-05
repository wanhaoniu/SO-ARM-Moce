#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from soarmmoce_sdk import Robot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional config path")
    args = parser.parse_args()

    robot = Robot.from_config(args.config) if args.config else Robot()
    robot.connect()

    # Use current pose as a reachable target for demonstration
    current_pose = robot.get_end_effector_pose()
    target_xyz = current_pose.xyz
    target_rpy = current_pose.rpy

    q = robot.move_pose(target_xyz, target_rpy, q0=robot.get_joint_state().q, duration=2.0)
    print("IK q:", q)

    pose = robot.get_end_effector_pose(q)
    print("FK pose xyz:", pose.xyz)
    print("FK pose rpy:", pose.rpy)

    robot.disconnect()


if __name__ == "__main__":
    main()
