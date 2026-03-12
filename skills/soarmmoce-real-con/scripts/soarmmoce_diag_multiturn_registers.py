#!/usr/bin/env python3
"""Compare multi-turn register behavior across Feetech operating modes."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict

from soarmmoce_auto_calibrate import _parse_joints
from soarmmoce_sdk import MULTI_TURN_JOINTS, SoArmMoceController, resolve_config


REGISTERS = [
    "Present_Position",
    "Goal_Position",
    "Goal_Position_2",
    "Present_Velocity",
    "Moving",
    "Present_Current",
]


def _read_registers(bus, joint: str) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for name in REGISTERS:
        try:
            values[name] = int(bus.read(name, joint, normalize=False))
        except Exception as exc:  # pragma: no cover - hardware path
            values[name] = f"ERROR: {exc}"
    return values


def _blank_stats() -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "first": None,
            "last": None,
            "min": None,
            "max": None,
            "changed": False,
            "samples": 0,
        }
        for name in REGISTERS
    }


def _update_stats(stats: Dict[str, Dict[str, Any]], sample: Dict[str, Any]) -> None:
    for reg_name, value in sample.items():
        entry = stats[reg_name]
        if not isinstance(value, int):
            entry["last"] = value
            continue
        if entry["first"] is None:
            entry["first"] = value
            entry["min"] = value
            entry["max"] = value
            entry["last"] = value
            entry["samples"] = 1
            continue
        entry["changed"] = bool(entry["changed"] or value != entry["last"] or value != entry["first"])
        entry["last"] = value
        entry["min"] = min(int(entry["min"]), value)
        entry["max"] = max(int(entry["max"]), value)
        entry["samples"] = int(entry["samples"]) + 1


def _print_live_table(
    *,
    phase_label: str,
    elapsed_s: float,
    total_s: float,
    latest: Dict[str, Dict[str, Any]],
) -> None:
    print(
        f"\n[{phase_label}] t={elapsed_s:0.2f}s/{total_s:0.2f}s  "
        "Move the selected joints by hand during this window."
    )
    print(
        f"{'JOINT':<15} | {'POS':>8} | {'GOAL':>8} | {'GOAL2':>8} | "
        f"{'VEL':>8} | {'MOV':>5} | {'CUR':>8}"
    )
    for joint, sample in latest.items():
        print(
            f"{joint:<15} | "
            f"{str(sample.get('Present_Position', '')):>8} | "
            f"{str(sample.get('Goal_Position', '')):>8} | "
            f"{str(sample.get('Goal_Position_2', '')):>8} | "
            f"{str(sample.get('Present_Velocity', '')):>8} | "
            f"{str(sample.get('Moving', '')):>5} | "
            f"{str(sample.get('Present_Current', '')):>8}"
        )


def _sample_phase(
    *,
    bus,
    joints: list[str],
    phase_label: str,
    duration_s: float,
    poll_interval_s: float,
    display_values: bool,
) -> Dict[str, Any]:
    started = time.time()
    deadline = started + max(0.1, float(duration_s))
    latest: Dict[str, Dict[str, Any]] = {}
    stats = {joint: _blank_stats() for joint in joints}
    iterations = 0

    while True:
        now = time.time()
        if now > deadline and iterations > 0:
            break

        for joint in joints:
            latest[joint] = _read_registers(bus, joint)
            _update_stats(stats[joint], latest[joint])

        iterations += 1
        if display_values:
            _print_live_table(
                phase_label=phase_label,
                elapsed_s=max(0.0, now - started),
                total_s=max(0.1, float(duration_s)),
                latest=latest,
            )
            if now + max(0.01, float(poll_interval_s)) <= deadline:
                print("\x1b[{}A".format(len(joints) + 2), end="")

        time.sleep(max(0.01, float(poll_interval_s)))

    if display_values:
        print()

    return {
        "phase": phase_label,
        "duration_s": float(max(0.1, float(duration_s))),
        "iterations": int(iterations),
        "latest": latest,
        "stats": stats,
    }


def _restore_joint_registers(bus, saved: Dict[str, Dict[str, int]]) -> None:
    for joint, entry in saved.items():
        bus.write("Lock", joint, 0, normalize=False)
        bus.write("Min_Position_Limit", joint, int(entry["min_position_limit"]), normalize=False)
        bus.write("Max_Position_Limit", joint, int(entry["max_position_limit"]), normalize=False)
        bus.write("Operating_Mode", joint, int(entry["operating_mode"]), normalize=False)
        bus.write("Lock", joint, int(entry["lock"]), normalize=False)


def _configure_phase_mode(bus, joints: list[str], *, operating_mode: int, range_min: int, range_max: int) -> None:
    for joint in joints:
        bus.write("Lock", joint, 0, normalize=False)
        bus.write("Min_Position_Limit", joint, int(range_min), normalize=False)
        bus.write("Max_Position_Limit", joint, int(range_max), normalize=False)
        bus.write("Operating_Mode", joint, int(operating_mode), normalize=False)
        bus.write("Lock", joint, 1, normalize=False)


def _diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    config = resolve_config()
    joints = list(dict.fromkeys(_parse_joints(args.joints)))
    unsupported = [joint for joint in joints if joint not in MULTI_TURN_JOINTS]
    if unsupported:
        raise ValueError(f"This diagnostic only targets multi-turn joints: {unsupported}")

    with SoArmMoceController(config) as arm:
        bus = arm._ensure_bus()
        saved: Dict[str, Dict[str, int]] = {}
        for joint in joints:
            saved[joint] = {
                "lock": int(bus.read("Lock", joint, normalize=False)),
                "operating_mode": int(bus.read("Operating_Mode", joint, normalize=False)),
                "min_position_limit": int(bus.read("Min_Position_Limit", joint, normalize=False)),
                "max_position_limit": int(bus.read("Max_Position_Limit", joint, normalize=False)),
            }

        with bus.torque_disabled(joints):
            baseline = {joint: _read_registers(bus, joint) for joint in joints}
            try:
                _configure_phase_mode(bus, joints, operating_mode=0, range_min=0, range_max=4095)
                time.sleep(max(0.05, float(args.settle_s)))
                position_phase = _sample_phase(
                    bus=bus,
                    joints=joints,
                    phase_label="mode0_position",
                    duration_s=float(args.sample_seconds),
                    poll_interval_s=float(args.poll_interval_s),
                    display_values=bool(args.display_values),
                )

                _configure_phase_mode(bus, joints, operating_mode=3, range_min=0, range_max=0)
                time.sleep(max(0.05, float(args.settle_s)))
                step_phase = _sample_phase(
                    bus=bus,
                    joints=joints,
                    phase_label="mode3_step",
                    duration_s=float(args.sample_seconds),
                    poll_interval_s=float(args.poll_interval_s),
                    display_values=bool(args.display_values),
                )
            finally:
                _restore_joint_registers(bus, saved)

    return {
        "action": "diag_multiturn_registers",
        "port": str(config.port),
        "robot_id": str(config.robot_id),
        "joints": joints,
        "sample_seconds_per_phase": float(args.sample_seconds),
        "poll_interval_s": float(args.poll_interval_s),
        "saved_registers": saved,
        "baseline_before_mode_switch": baseline,
        "phases": [position_phase, step_phase],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multi-turn register behavior across Feetech mode 0 (position) and mode 3 (step). "
            "The script temporarily switches modes with torque disabled, samples raw registers, then restores "
            "the original registers."
        )
    )
    parser.add_argument(
        "--joints",
        default="shoulder_lift,elbow_flex",
        help="Comma-separated multi-turn joints to inspect",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=5.0,
        help="How long to sample each mode while you manually move the joints",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=0.05,
        help="Register polling interval during each phase",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.2,
        help="Short settle delay after each mode switch",
    )
    parser.add_argument(
        "--display-values",
        type=lambda raw: str(raw).strip().lower() not in {"0", "false", "no"},
        default=True,
        help="Whether to print live register values during sampling",
    )
    args = parser.parse_args()
    payload = _diagnose(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
