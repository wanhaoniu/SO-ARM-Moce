from __future__ import annotations

import dataclasses
import os
import threading
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from soarmmoce_sdk import Robot


class ValidationError(Exception):
    pass


_SUPPORTED_TRANSPORTS = {"mock", "tcp", "serial"}

_ROBOT_LOCK = threading.RLock()
_ROBOT_SINGLETON: Optional[Robot] = None
_ROBOT_OPTIONS: Optional[Tuple[Optional[str], Optional[str]]] = None


def _get_runtime_options() -> Tuple[Optional[str], Optional[str]]:
    config_path = str(os.getenv("SOARMMOCE_CONFIG", "")).strip() or None
    transport = str(os.getenv("SOARMMOCE_TRANSPORT", "")).strip().lower() or None
    if transport is not None and transport not in _SUPPORTED_TRANSPORTS:
        raise ValidationError(
            f"SOARMMOCE_TRANSPORT must be one of {sorted(_SUPPORTED_TRANSPORTS)}, got: {transport!r}"
        )
    return config_path, transport


def _build_robot(config_path: Optional[str], transport: Optional[str]) -> Robot:
    robot = Robot.from_config(config_path) if config_path else Robot()
    if transport:
        robot.config.setdefault("transport", {})["type"] = transport
    return robot


def _get_robot() -> Robot:
    global _ROBOT_SINGLETON, _ROBOT_OPTIONS

    options = _get_runtime_options()

    with _ROBOT_LOCK:
        if _ROBOT_SINGLETON is None or _ROBOT_OPTIONS != options:
            if _ROBOT_SINGLETON is not None:
                try:
                    _ROBOT_SINGLETON.disconnect()
                except Exception:
                    pass
            _ROBOT_SINGLETON = _build_robot(*options)
            _ROBOT_OPTIONS = options

        if not _ROBOT_SINGLETON.connected:
            _ROBOT_SINGLETON.connect()

        return _ROBOT_SINGLETON


def reset_robot() -> None:
    global _ROBOT_SINGLETON, _ROBOT_OPTIONS
    with _ROBOT_LOCK:
        if _ROBOT_SINGLETON is not None:
            try:
                _ROBOT_SINGLETON.disconnect()
            except Exception:
                pass
        _ROBOT_SINGLETON = None
        _ROBOT_OPTIONS = None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_object(args: Any) -> Dict[str, Any]:
    if not isinstance(args, dict):
        raise ValidationError("args must be an object/dict")
    return dict(args)


def _require_float(args: Dict[str, Any], key: str) -> float:
    if key not in args:
        raise ValidationError(f"missing required field: {key}")
    value = args.get(key)
    if not _is_number(value):
        raise ValidationError(f"{key} must be a number")
    return float(value)


def _optional_float(args: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    if key not in args:
        return default
    value = args.get(key)
    if value is None:
        return None
    if not _is_number(value):
        raise ValidationError(f"{key} must be a number or null")
    out = float(value)
    if out < 0.0:
        raise ValidationError(f"{key} must be >= 0")
    return out


def _optional_bool(args: Dict[str, Any], key: str, default: bool) -> bool:
    if key not in args:
        return bool(default)
    value = args.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
    raise ValidationError(f"{key} must be a boolean")


def _optional_str(args: Dict[str, Any], key: str, default: str) -> str:
    if key not in args:
        return default
    value = args.get(key)
    if not isinstance(value, str):
        raise ValidationError(f"{key} must be a string")
    out = value.strip()
    if not out:
        raise ValidationError(f"{key} must be non-empty")
    return out


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _make_ok(result: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "result": _to_jsonable(result), "error": None}


def _make_error(exc: Exception) -> Dict[str, Any]:
    return {
        "ok": False,
        "result": {},
        "error": {
            "type": exc.__class__.__name__,
            "message": str(exc) or "unknown error",
        },
    }


def _tool_get_robot_state(robot: Robot, args: Dict[str, Any]) -> Dict[str, Any]:
    if args:
        raise ValidationError("get_robot_state does not accept arguments")
    state = robot.get_state()
    return {"state": state}


def _tool_move_robot_arm(robot: Robot, args: Dict[str, Any]) -> Dict[str, Any]:
    x = _require_float(args, "x")
    y = _require_float(args, "y")
    z = _require_float(args, "z")
    frame = _optional_str(args, "frame", "base").lower()
    if frame not in {"base", "tool"}:
        raise ValidationError("frame must be 'base' or 'tool'")

    duration = _optional_float(args, "duration", 2.0)
    assert duration is not None
    wait = _optional_bool(args, "wait", True)
    timeout = _optional_float(args, "timeout", None)

    q = robot.move_tcp(
        x=x,
        y=y,
        z=z,
        frame=frame,
        duration=duration,
        wait=wait,
        timeout=timeout,
    )
    state = robot.get_state()
    return {
        "tool": "move_robot_arm",
        "target": {"x": x, "y": y, "z": z, "frame": frame},
        "duration": duration,
        "wait": wait,
        "timeout": timeout,
        "wait_semantics": "best_effort",
        "joint_solution": q,
        "state": state,
    }


def _tool_stop_robot(robot: Robot, args: Dict[str, Any]) -> Dict[str, Any]:
    if args:
        raise ValidationError("stop_robot does not accept arguments")
    robot.stop()
    return {
        "tool": "stop_robot",
        "stopped": True,
    }


def _tool_set_gripper(robot: Robot, args: Dict[str, Any]) -> Dict[str, Any]:
    open_ratio = _require_float(args, "open_ratio")
    if open_ratio < 0.0 or open_ratio > 1.0:
        raise ValidationError("open_ratio must be within [0.0, 1.0]")

    wait = _optional_bool(args, "wait", True)
    timeout = _optional_float(args, "timeout", None)

    robot.set_gripper(open_ratio=open_ratio, wait=wait, timeout=timeout)
    return {
        "tool": "set_gripper",
        "open_ratio": open_ratio,
        "wait": wait,
        "timeout": timeout,
        "wait_semantics": "best_effort",
    }


def _tool_home(robot: Robot, args: Dict[str, Any]) -> Dict[str, Any]:
    duration = _optional_float(args, "duration", 2.0)
    assert duration is not None
    wait = _optional_bool(args, "wait", True)
    timeout = _optional_float(args, "timeout", None)

    q_home = robot.home(duration=duration, wait=wait, timeout=timeout)
    state = robot.get_state()
    return {
        "tool": "home",
        "duration": duration,
        "wait": wait,
        "timeout": timeout,
        "wait_semantics": "best_effort",
        "q_home": q_home,
        "state": state,
    }


_HANDLERS: Dict[str, Callable[[Robot, Dict[str, Any]], Dict[str, Any]]] = {
    "move_robot_arm": _tool_move_robot_arm,
    "get_robot_state": _tool_get_robot_state,
    "stop_robot": _tool_stop_robot,
    "set_gripper": _tool_set_gripper,
    "home": _tool_home,
}


def dispatch(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch one OpenClaw tool call to soarmmoce-sdk Robot API.

    Returns structured result:
    - success: {"ok": True, "result": {...}, "error": None}
    - failure: {"ok": False, "result": {}, "error": {"type": ..., "message": ...}}
    """
    try:
        name = str(tool_name or "").strip()
        if not name:
            raise ValidationError("tool_name is required")
        if name not in _HANDLERS:
            raise ValidationError(f"unsupported tool: {name}")

        clean_args = _require_object(args)
        robot = _get_robot()
        result = _HANDLERS[name](robot, clean_args)
        return _make_ok(result)
    except Exception as exc:
        return _make_error(exc)
