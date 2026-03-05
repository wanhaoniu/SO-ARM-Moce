from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from .dispatcher import dispatch


def _parse_bool(raw: str) -> bool:
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {raw!r}")


def _add_common_wait_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wait", type=_parse_bool, default=True)
    parser.add_argument("--timeout", type=float, default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SOARMMOCE OpenClaw skill local debug CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("get_robot_state")
    sub.add_parser("stop_robot")

    move = sub.add_parser("move_robot_arm")
    move.add_argument("--x", type=float, required=True)
    move.add_argument("--y", type=float, required=True)
    move.add_argument("--z", type=float, required=True)
    move.add_argument("--frame", choices=["base", "tool"], default="base")
    move.add_argument("--duration", type=float, default=2.0)
    _add_common_wait_args(move)

    grip = sub.add_parser("set_gripper")
    grip.add_argument("--open-ratio", type=float, required=True)
    _add_common_wait_args(grip)

    home = sub.add_parser("home")
    home.add_argument("--duration", type=float, default=2.0)
    _add_common_wait_args(home)

    return parser


def _args_to_payload(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = args.command
    if cmd in {"get_robot_state", "stop_robot"}:
        return {}
    if cmd == "move_robot_arm":
        return {
            "x": args.x,
            "y": args.y,
            "z": args.z,
            "frame": args.frame,
            "duration": args.duration,
            "wait": args.wait,
            "timeout": args.timeout,
        }
    if cmd == "set_gripper":
        return {
            "open_ratio": args.open_ratio,
            "wait": args.wait,
            "timeout": args.timeout,
        }
    if cmd == "home":
        return {
            "duration": args.duration,
            "wait": args.wait,
            "timeout": args.timeout,
        }
    raise ValueError(f"unsupported command: {cmd}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    tool_name = str(args.command)
    payload = _args_to_payload(args)
    response = dispatch(tool_name, payload)

    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0 if bool(response.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
