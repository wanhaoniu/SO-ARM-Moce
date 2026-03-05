from __future__ import annotations

import os

from soarmmoce_openclaw_skill.dispatcher import dispatch, reset_robot


def setup_function() -> None:
    os.environ["SOARMMOCE_TRANSPORT"] = "mock"
    os.environ.pop("SOARMMOCE_CONFIG", None)
    reset_robot()


def teardown_function() -> None:
    reset_robot()


def test_get_robot_state_ok() -> None:
    resp = dispatch("get_robot_state", {})
    assert resp["ok"] is True
    assert "state" in resp["result"]
    assert resp["result"]["state"]["connected"] is True


def test_move_robot_arm_ok() -> None:
    state_resp = dispatch("get_robot_state", {})
    assert state_resp["ok"] is True
    xyz = state_resp["result"]["state"]["tcp_pose"]["xyz"]

    resp = dispatch(
        "move_robot_arm",
        {
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
            "frame": "base",
            "duration": 0.01,
            "wait": True,
            "timeout": 1.0,
        },
    )
    assert resp["ok"] is True
    assert resp["result"]["tool"] == "move_robot_arm"
    assert resp["result"]["target"]["frame"] == "base"


def test_set_gripper_ok() -> None:
    resp = dispatch("set_gripper", {"open_ratio": 0.5, "wait": True, "timeout": 1.0})
    assert resp["ok"] is True
    assert resp["result"]["tool"] == "set_gripper"


def test_missing_required_arg_error() -> None:
    resp = dispatch("move_robot_arm", {"x": 0.1, "y": 0.2})
    assert resp["ok"] is False
    assert resp["error"]["type"] == "ValidationError"
    assert "missing required field" in resp["error"]["message"]
