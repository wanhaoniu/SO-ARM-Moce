import numpy as np
import pytest

from soarmmoce_sdk import LimitError, Robot


def test_limits_violation():
    robot = Robot()
    robot.connect()

    q = robot.get_joint_state().q
    upper = np.array([u for _, u in robot.robot_model.joint_limits], dtype=float)
    q[0] = upper[0] + 0.1

    with pytest.raises(LimitError):
        robot.move_joints(q)

    robot.disconnect()
