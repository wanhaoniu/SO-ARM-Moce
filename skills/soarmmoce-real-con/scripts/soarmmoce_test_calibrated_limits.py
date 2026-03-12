#!/usr/bin/env python3
"""Exercise calibrated multi-turn joint limits with safety checks."""

from __future__ import annotations

import argparse
import time
from typing import Any

from soarmmoce_cli_common import cli_bool, print_error, print_success
from soarmmoce_sdk import (
    MULTI_TURN_JOINTS,
    RAW_COUNTS_PER_REV,
    SoArmMoceController,
    ValidationError,
    _load_calibration_payload,
    resolve_config,
)


def _parse_joints(raw: str) -> list[str]:
    joints = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    if not joints:
        raise ValidationError("At least one joint must be provided")
    unknown = [joint for joint in joints if joint not in MULTI_TURN_JOINTS]
    if unknown:
        raise ValidationError(
            "This limit test only supports multi-turn joints: "
            + ", ".join(sorted(MULTI_TURN_JOINTS))
            + f". Got: {', '.join(unknown)}"
        )
    return joints


def _relative_raw_to_joint_deg(arm: SoArmMoceController, joint: str, raw_value: int) -> float:
    motor_deg = float(raw_value) * 360.0 / float(RAW_COUNTS_PER_REV)
    return float(arm._motor_to_joint_deg(joint, motor_deg))

def _build_joint_plan(
    arm: SoArmMoceController,
    calibration: dict[str, dict[str, Any]],
    state: dict[str, Any],
    joint: str,
    *,
    margin_raw: int,
    allow_inconsistent_range: bool,
) -> dict[str, Any]:
    if joint not in calibration:
        raise ValidationError(f"Calibration entry missing for {joint}")

    multi_turn_state = state.get("multi_turn_state", {}).get(joint)
    if not isinstance(multi_turn_state, dict):
        raise ValidationError(f"Current multi-turn state missing for {joint}")

    cal = calibration[joint]
    try:
        range_min = int(cal["min_relative_raw"])
        range_max = int(cal["max_relative_raw"])
        home_wrapped_raw = int(cal["home_wrapped_raw"]) % int(RAW_COUNTS_PER_REV)
    except KeyError as exc:
        raise ValidationError(f"Calibration entry for {joint} is missing required field: {exc.args[0]}") from exc
    range_width = int(range_max - range_min)
    current_joint_deg = float(state["joint_state"][joint])
    current_continuous_raw = int(round(float(multi_turn_state["relative_raw"])))
    current_wrapped_raw = int(round(float(multi_turn_state["last_raw_mod"]))) % int(RAW_COUNTS_PER_REV)
    target_min_continuous_raw = int(range_min + margin_raw)
    target_max_continuous_raw = int(range_max - margin_raw)

    safety_issues: list[str] = []
    if range_min >= range_max:
        safety_issues.append("range_min must be smaller than range_max")
    if target_min_continuous_raw >= target_max_continuous_raw:
        safety_issues.append(
            f"range width {range_width} is too small for safety margin {int(margin_raw)}"
        )

    current_in_range = range_min <= current_continuous_raw <= range_max
    if not current_in_range and not allow_inconsistent_range:
        safety_issues.append(
            "current continuous raw is outside calibrated range; "
            "the range and current homing offset are likely inconsistent"
        )

    return {
        "joint": joint,
        "current_joint_deg": current_joint_deg,
        "current_continuous_raw": current_continuous_raw,
        "current_wrapped_raw": current_wrapped_raw,
        "home_wrapped_raw": home_wrapped_raw,
        "range_min_raw": range_min,
        "range_max_raw": range_max,
        "range_width_raw": range_width,
        "target_min_continuous_raw": target_min_continuous_raw,
        "target_max_continuous_raw": target_max_continuous_raw,
        "target_min_joint_deg": float(arm.config.home_joints.get(joint, 0.0))
        + _relative_raw_to_joint_deg(arm, joint, target_min_continuous_raw),
        "target_max_joint_deg": float(arm.config.home_joints.get(joint, 0.0))
        + _relative_raw_to_joint_deg(arm, joint, target_max_continuous_raw),
        "current_in_range": current_in_range,
        "safety_issues": safety_issues,
        "executable": not safety_issues,
    }


def _snapshot_joint_state(state: dict[str, Any], joint: str) -> dict[str, Any]:
    multi_turn_state = state.get("multi_turn_state", {}).get(joint, {})
    return {
        "joint_deg": float(state["joint_state"][joint]),
        "continuous_raw": int(round(float(multi_turn_state.get("relative_raw", 0.0)))),
        "wrapped_raw": int(round(float(multi_turn_state.get("last_raw_mod", 0.0)))) % int(RAW_COUNTS_PER_REV),
    }


def _move_and_record(
    arm: SoArmMoceController,
    *,
    joint: str,
    label: str,
    target_deg: float,
    target_raw: int,
    duration: float,
    settle_s: float,
    trace: bool,
) -> dict[str, Any]:
    result = arm.move_joint(
        joint=joint,
        target_deg=float(target_deg),
        duration=float(duration),
        wait=True,
        timeout=max(float(duration) + float(settle_s) + 1.0, 2.0),
        trace=trace,
    )
    time.sleep(max(0.0, float(settle_s)))
    snapshot = _snapshot_joint_state(result["state"], joint)
    payload = {
        "label": label,
        "target_joint_deg": float(target_deg),
        "target_continuous_raw": int(target_raw),
        "after": snapshot,
        "joint_error_deg": float(snapshot["joint_deg"] - float(target_deg)),
        "raw_error": int(snapshot["continuous_raw"] - int(target_raw)),
    }
    if trace and "trace" in result:
        payload["trace"] = result["trace"]
    return payload


def _exercise_joint(
    arm: SoArmMoceController,
    plan: dict[str, Any],
    *,
    duration: float,
    settle_s: float,
    return_to_start: bool,
    trace: bool,
) -> list[dict[str, Any]]:
    joint = str(plan["joint"])
    steps = [
        (
            "move_to_min",
            float(plan["target_min_joint_deg"]),
            int(plan["target_min_continuous_raw"]),
        ),
        (
            "return_to_start_after_min",
            float(plan["current_joint_deg"]),
            int(plan["current_continuous_raw"]),
        ),
        (
            "move_to_max",
            float(plan["target_max_joint_deg"]),
            int(plan["target_max_continuous_raw"]),
        ),
    ]
    if return_to_start:
        steps.append(
            (
                "return_to_start_after_max",
                float(plan["current_joint_deg"]),
                int(plan["current_continuous_raw"]),
            )
        )

    return [
        _move_and_record(
            arm,
            joint=joint,
            label=label,
            target_deg=target_deg,
            target_raw=target_raw,
            duration=duration,
            settle_s=settle_s,
            trace=trace,
        )
        for label, target_deg, target_raw in steps
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test calibrated raw limits for multi-turn joints. "
            "The script blocks execution when the calibration range looks unsafe."
        )
    )
    parser.add_argument("--joints", default="shoulder_lift,elbow_flex")
    parser.add_argument("--execute", type=cli_bool, default=False)
    parser.add_argument("--margin-raw", type=int, default=32)
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--trace", type=cli_bool, default=False)
    parser.add_argument("--return-to-start", type=cli_bool, default=True)
    parser.add_argument("--allow-inconsistent-range", type=cli_bool, default=False)
    args = parser.parse_args()

    try:
        joints = _parse_joints(args.joints)
        if args.margin_raw < 0:
            raise ValidationError("--margin-raw must be >= 0")

        with SoArmMoceController() as arm:
            initial_state = arm.get_state()
            config = resolve_config()
            calibration = _load_calibration_payload(config.robot_id, config.calib_dir)
            plans = [
                _build_joint_plan(
                    arm,
                    calibration,
                    initial_state,
                    joint,
                    margin_raw=int(args.margin_raw),
                    allow_inconsistent_range=bool(args.allow_inconsistent_range),
                )
                for joint in joints
            ]

            result: dict[str, Any] = {
                "execute": bool(args.execute),
                "config": {
                    "joints": joints,
                    "margin_raw": int(args.margin_raw),
                    "duration": float(args.duration),
                    "settle_s": float(args.settle_s),
                    "return_to_start": bool(args.return_to_start),
                    "trace": bool(args.trace),
                    "allow_inconsistent_range": bool(args.allow_inconsistent_range),
                },
                "initial_state": {
                    joint: _snapshot_joint_state(initial_state, joint)
                    for joint in joints
                },
                "plan": plans,
                "executed": [],
                "skipped": [],
            }

            if not args.execute:
                result["note"] = "Dry run only. Re-run with --execute=true after reviewing the plan."
                print_success(result)
                return

            for plan in plans:
                if not plan["executable"]:
                    result["skipped"].append(
                        {
                            "joint": plan["joint"],
                            "reason": list(plan["safety_issues"]),
                        }
                    )
                    continue

                steps = _exercise_joint(
                    arm,
                    plan,
                    duration=float(args.duration),
                    settle_s=float(args.settle_s),
                    return_to_start=bool(args.return_to_start),
                    trace=bool(args.trace),
                )
                result["executed"].append({"joint": plan["joint"], "steps": steps})

            if not result["executed"]:
                result["note"] = (
                    "Nothing moved. All requested joints were blocked by safety checks. "
                    "Review plan/skipped for the specific reasons."
                )

            print_success(result)
    except Exception as exc:
        print_error(exc)


if __name__ == "__main__":
    main()
