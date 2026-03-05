from __future__ import annotations

from typing import Any, Dict, List


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move_robot_arm",
            "description": "Move robot TCP in Cartesian space by calling Robot.move_tcp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Target X in meters"},
                    "y": {"type": "number", "description": "Target Y in meters"},
                    "z": {"type": "number", "description": "Target Z in meters"},
                    "frame": {"type": "string", "enum": ["base", "tool"], "default": "base"},
                    "duration": {"type": "number", "minimum": 0.0, "default": 2.0},
                    "wait": {"type": "boolean", "default": True},
                    "timeout": {"type": ["number", "null"], "minimum": 0.0, "default": None},
                },
                "required": ["x", "y", "z"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_robot_state",
            "description": "Get structured robot state via Robot.get_state.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_robot",
            "description": "Stop robot motion via Robot.stop.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_gripper",
            "description": "Set gripper open ratio via Robot.set_gripper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "open_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "wait": {"type": "boolean", "default": True},
                    "timeout": {"type": ["number", "null"], "minimum": 0.0, "default": None},
                },
                "required": ["open_ratio"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "home",
            "description": "Move robot to home joint state via Robot.home.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {"type": "number", "minimum": 0.0, "default": 2.0},
                    "wait": {"type": "boolean", "default": True},
                    "timeout": {"type": ["number", "null"], "minimum": 0.0, "default": None},
                },
                "additionalProperties": False,
            },
        },
    },
]
