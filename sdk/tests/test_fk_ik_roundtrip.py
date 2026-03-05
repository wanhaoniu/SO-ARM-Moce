import numpy as np

from soarmmoce_sdk import Robot
from soarmmoce_sdk.kinematics import fk, matrix_to_rpy, solve_ik


def test_fk_ik_roundtrip():
    robot = Robot()
    robot.connect()

    lower = np.array([l for l, _ in robot.robot_model.joint_limits], dtype=float)
    upper = np.array([u for _, u in robot.robot_model.joint_limits], dtype=float)
    rng = np.random.default_rng(0)
    q = lower + (upper - lower) * rng.random(robot.robot_model.dof)

    T = fk(robot.robot_model, q)
    rpy = matrix_to_rpy(T[:3, :3])
    res = solve_ik(robot.robot_model, T[:3, 3], rpy, q0=q)

    assert res.success
    assert np.allclose(res.q, q, atol=1e-2)

    robot.disconnect()
