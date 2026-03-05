from __future__ import annotations

from soarmmoce_openclaw_skill.schemas import TOOL_SCHEMAS


def test_required_tools_exist() -> None:
    names = {item["function"]["name"] for item in TOOL_SCHEMAS}
    assert "move_robot_arm" in names
    assert "get_robot_state" in names
    assert "stop_robot" in names
    assert "set_gripper" in names
