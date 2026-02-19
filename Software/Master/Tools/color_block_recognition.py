#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color block recognition tool with interactive color picking and HSV tuning.

Features:
1) Live camera view with detected color blocks and labels.
2) Click-to-pick color from frame.
3) HSV trackbar tuning and config save/load.

Run examples:
  python3 color_block_recognition.py
  python3 color_block_recognition.py --source realsense
  python3 color_block_recognition.py --camera-index 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


WINDOW_MAIN = "Color Blocks"
WINDOW_MASK = "Mask (selected color)"
WINDOW_CTRL = "HSV Picker / Tuner"


DEFAULT_PROFILES = {
    "red": {"h_low": 0, "h_high": 10, "s_low": 100, "s_high": 255, "v_low": 80, "v_high": 255, "min_area": 700, "kernel": 5, "draw_bgr": [0, 0, 255]},
    "green": {"h_low": 40, "h_high": 85, "s_low": 70, "s_high": 255, "v_low": 60, "v_high": 255, "min_area": 700, "kernel": 5, "draw_bgr": [0, 220, 0]},
    "blue": {"h_low": 90, "h_high": 130, "s_low": 80, "s_high": 255, "v_low": 60, "v_high": 255, "min_area": 700, "kernel": 5, "draw_bgr": [255, 120, 0]},
    "yellow": {"h_low": 18, "h_high": 38, "s_low": 90, "s_high": 255, "v_low": 90, "v_high": 255, "min_area": 700, "kernel": 5, "draw_bgr": [0, 255, 255]},
}


def _clamp_int(x: int, low: int, high: int) -> int:
    return int(max(low, min(high, int(x))))


def _sanitize_profile(p: Dict) -> Dict:
    out = {
        "h_low": _clamp_int(p.get("h_low", 0), 0, 179),
        "h_high": _clamp_int(p.get("h_high", 179), 0, 179),
        "s_low": _clamp_int(p.get("s_low", 0), 0, 255),
        "s_high": _clamp_int(p.get("s_high", 255), 0, 255),
        "v_low": _clamp_int(p.get("v_low", 0), 0, 255),
        "v_high": _clamp_int(p.get("v_high", 255), 0, 255),
        "min_area": _clamp_int(p.get("min_area", 700), 10, 200000),
        "kernel": _clamp_int(p.get("kernel", 5), 1, 31),
    }
    draw = p.get("draw_bgr", [255, 255, 255])
    if not isinstance(draw, (list, tuple)) or len(draw) != 3:
        draw = [255, 255, 255]
    out["draw_bgr"] = [_clamp_int(draw[0], 0, 255), _clamp_int(draw[1], 0, 255), _clamp_int(draw[2], 0, 255)]
    return out


def _random_bgr() -> Tuple[int, int, int]:
    return (random.randint(40, 255), random.randint(40, 255), random.randint(40, 255))


def load_profiles(config_path: Path) -> Tuple[Dict[str, Dict], str]:
    if not config_path.exists():
        profiles = {k: _sanitize_profile(v) for k, v in DEFAULT_PROFILES.items()}
        return profiles, next(iter(profiles))

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        profiles = {k: _sanitize_profile(v) for k, v in DEFAULT_PROFILES.items()}
        return profiles, next(iter(profiles))

    src_profiles = payload.get("profiles", {})
    if not isinstance(src_profiles, dict) or not src_profiles:
        profiles = {k: _sanitize_profile(v) for k, v in DEFAULT_PROFILES.items()}
        return profiles, next(iter(profiles))

    profiles = {}
    for name, prof in src_profiles.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(prof, dict):
            continue
        profiles[name.strip()] = _sanitize_profile(prof)

    if not profiles:
        profiles = {k: _sanitize_profile(v) for k, v in DEFAULT_PROFILES.items()}
    active = payload.get("active_color", next(iter(profiles)))
    if active not in profiles:
        active = next(iter(profiles))
    return profiles, active


def save_profiles(config_path: Path, profiles: Dict[str, Dict], active_color: str) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profiles": {k: _sanitize_profile(v) for k, v in profiles.items()},
        "active_color": active_color,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_mask(hsv: np.ndarray, prof: Dict) -> np.ndarray:
    h_low, h_high = int(prof["h_low"]), int(prof["h_high"])
    s_low, s_high = int(prof["s_low"]), int(prof["s_high"])
    v_low, v_high = int(prof["v_low"]), int(prof["v_high"])

    lower1 = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper1 = np.array([h_high, s_high, v_high], dtype=np.uint8)

    if h_low <= h_high:
        mask = cv2.inRange(hsv, lower1, upper1)
    else:
        # hue wrap-around (e.g. red around 0/179)
        mask_a = cv2.inRange(hsv, np.array([0, s_low, v_low], dtype=np.uint8), np.array([h_high, s_high, v_high], dtype=np.uint8))
        mask_b = cv2.inRange(hsv, np.array([h_low, s_low, v_low], dtype=np.uint8), np.array([179, s_high, v_high], dtype=np.uint8))
        mask = cv2.bitwise_or(mask_a, mask_b)

    k = int(prof["kernel"])
    if k % 2 == 0:
        k += 1
    k = max(1, min(31, k))
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def create_control_trackbars() -> None:
    cv2.namedWindow(WINDOW_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CTRL, 520, 380)
    cv2.createTrackbar("H low", WINDOW_CTRL, 0, 179, lambda _v: None)
    cv2.createTrackbar("H high", WINDOW_CTRL, 179, 179, lambda _v: None)
    cv2.createTrackbar("S low", WINDOW_CTRL, 0, 255, lambda _v: None)
    cv2.createTrackbar("S high", WINDOW_CTRL, 255, 255, lambda _v: None)
    cv2.createTrackbar("V low", WINDOW_CTRL, 0, 255, lambda _v: None)
    cv2.createTrackbar("V high", WINDOW_CTRL, 255, 255, lambda _v: None)
    cv2.createTrackbar("MinArea", WINDOW_CTRL, 700, 200000, lambda _v: None)
    cv2.createTrackbar("Kernel", WINDOW_CTRL, 5, 31, lambda _v: None)


def set_trackbars_from_profile(prof: Dict) -> None:
    cv2.setTrackbarPos("H low", WINDOW_CTRL, int(prof["h_low"]))
    cv2.setTrackbarPos("H high", WINDOW_CTRL, int(prof["h_high"]))
    cv2.setTrackbarPos("S low", WINDOW_CTRL, int(prof["s_low"]))
    cv2.setTrackbarPos("S high", WINDOW_CTRL, int(prof["s_high"]))
    cv2.setTrackbarPos("V low", WINDOW_CTRL, int(prof["v_low"]))
    cv2.setTrackbarPos("V high", WINDOW_CTRL, int(prof["v_high"]))
    cv2.setTrackbarPos("MinArea", WINDOW_CTRL, int(prof["min_area"]))
    cv2.setTrackbarPos("Kernel", WINDOW_CTRL, int(prof["kernel"]))


def read_profile_from_trackbars(old_prof: Dict) -> Dict:
    out = dict(old_prof)
    out["h_low"] = cv2.getTrackbarPos("H low", WINDOW_CTRL)
    out["h_high"] = cv2.getTrackbarPos("H high", WINDOW_CTRL)
    out["s_low"] = cv2.getTrackbarPos("S low", WINDOW_CTRL)
    out["s_high"] = cv2.getTrackbarPos("S high", WINDOW_CTRL)
    out["v_low"] = cv2.getTrackbarPos("V low", WINDOW_CTRL)
    out["v_high"] = cv2.getTrackbarPos("V high", WINDOW_CTRL)
    out["min_area"] = cv2.getTrackbarPos("MinArea", WINDOW_CTRL)
    out["kernel"] = max(1, cv2.getTrackbarPos("Kernel", WINDOW_CTRL))
    if out["kernel"] % 2 == 0:
        out["kernel"] += 1
    return _sanitize_profile(out)


def draw_palette(frame: np.ndarray, profiles: Dict[str, Dict], selected: str) -> None:
    x0 = 8
    y0 = 8
    row_h = 22
    for i, (name, prof) in enumerate(profiles.items()):
        y = y0 + i * row_h
        b, g, r = prof["draw_bgr"]
        cv2.rectangle(frame, (x0, y), (x0 + 14, y + 14), (int(b), int(g), int(r)), -1)
        txt = f"{name}"
        if name == selected:
            txt += " *"
        cv2.putText(frame, txt, (x0 + 20, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)


@dataclass
class OpenCVCam:
    cam_index: int
    width: int
    height: int

    def __post_init__(self) -> None:
        self.cap = cv2.VideoCapture(self.cam_index)
        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


@dataclass
class RealSenseCam:
    width: int
    height: int
    fps: int

    def __post_init__(self) -> None:
        import pyrealsense2 as rs  # type: ignore

        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.pipeline.start(cfg)

    def read(self) -> np.ndarray | None:
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


class MousePicker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.hsv_frame: np.ndarray | None = None
        self.selected_color: str = ""
        self.profiles: Dict[str, Dict] = {}

    def bind(self) -> None:
        cv2.setMouseCallback(WINDOW_MAIN, self._on_mouse)

    def _on_mouse(self, event, x, y, _flags, _userdata) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.hsv_frame is None:
            return
        h, w = self.hsv_frame.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        if self.selected_color not in self.profiles:
            return

        hsv = self.hsv_frame[y, x].astype(int)
        hh, ss, vv = int(hsv[0]), int(hsv[1]), int(hsv[2])
        h_delta = int(self.args.pick_h_delta)
        s_delta = int(self.args.pick_s_delta)
        v_delta = int(self.args.pick_v_delta)

        prof = dict(self.profiles[self.selected_color])
        prof["h_low"] = (hh - h_delta) % 180
        prof["h_high"] = (hh + h_delta) % 180
        prof["s_low"] = _clamp_int(ss - s_delta, 0, 255)
        prof["s_high"] = _clamp_int(ss + s_delta, 0, 255)
        prof["v_low"] = _clamp_int(vv - v_delta, 0, 255)
        prof["v_high"] = _clamp_int(vv + v_delta, 0, 255)
        self.profiles[self.selected_color] = _sanitize_profile(prof)
        set_trackbars_from_profile(self.profiles[self.selected_color])
        print(f"[PICK] {self.selected_color}: hsv=({hh},{ss},{vv})")


def cycle_selected(profiles: Dict[str, Dict], selected: str, step: int = 1) -> str:
    keys = list(profiles.keys())
    if not keys:
        return selected
    if selected not in keys:
        return keys[0]
    i = keys.index(selected)
    return keys[(i + step) % len(keys)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Color block detector with picker/tuner UI.")
    parser.add_argument("--source", choices=["opencv", "realsense"], default="opencv", help="Frame source type.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS for RealSense.")
    parser.add_argument("--flip", action="store_true", help="Mirror image horizontally.")
    parser.add_argument("--config", default="", help="Color profile JSON path.")
    parser.add_argument("--pick-h-delta", type=int, default=8, help="Hue half-range when click-picking.")
    parser.add_argument("--pick-s-delta", type=int, default=60, help="Saturation half-range when click-picking.")
    parser.add_argument("--pick-v-delta", type=int, default=60, help="Value half-range when click-picking.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve() if args.config else (Path(__file__).resolve().parent / "color_profiles.json")
    profiles, selected = load_profiles(config_path)

    cam = None
    try:
        if args.source == "opencv":
            cam = OpenCVCam(args.camera_index, args.width, args.height)
        else:
            try:
                cam = RealSenseCam(args.width, args.height, args.fps)
            except Exception as e:
                print(f"[ERROR] RealSense init failed: {e}")
                return 2

        cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_MAIN, 1280, 720)
        cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_MASK, 640, 480)
        create_control_trackbars()
        set_trackbars_from_profile(profiles[selected])

        picker = MousePicker(args)
        picker.bind()

        print("=" * 72)
        print("Color Block Recognition - Controls")
        print("  Left click: pick color from image for selected profile")
        print("  c / x:      next / previous profile")
        print("  n:          new color profile")
        print("  d:          delete selected profile")
        print("  s:          save profiles")
        print("  r:          reset to default profiles")
        print("  q / ESC:    quit")
        print(f"Config: {config_path}")
        print("=" * 72)

        while True:
            frame = cam.read()
            if frame is None:
                cv2.waitKey(10)
                continue

            if args.flip:
                frame = cv2.flip(frame, 1)

            annotated = frame.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Sync current trackbar values into selected profile in real time.
            profiles[selected] = read_profile_from_trackbars(profiles[selected])

            picker.hsv_frame = hsv
            picker.selected_color = selected
            picker.profiles = profiles

            detections_by_color: Dict[str, int] = {k: 0 for k in profiles.keys()}
            mask_selected = None

            for name, prof in profiles.items():
                mask = make_mask(hsv, prof)
                if name == selected:
                    mask_selected = mask

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < int(prof["min_area"]):
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    b, g, r = prof["draw_bgr"]
                    color_bgr = (int(b), int(g), int(r))
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color_bgr, 2)
                    cx, cy = x + w // 2, y + h // 2
                    cv2.circle(annotated, (cx, cy), 3, color_bgr, -1)
                    cv2.putText(
                        annotated,
                        f"{name} ({int(area)})",
                        (x, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color_bgr,
                        2,
                        cv2.LINE_AA,
                    )
                    detections_by_color[name] += 1

            draw_palette(annotated, profiles, selected)
            summary = " | ".join([f"{k}:{detections_by_color[k]}" for k in profiles.keys()])
            cv2.putText(annotated, f"Detected {summary}", (8, annotated.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(annotated, "Keys: c/x cycle  n new  d del  s save  r reset  q quit", (8, annotated.shape[0] - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_MAIN, annotated)
            if mask_selected is None:
                mask_selected = np.zeros((annotated.shape[0], annotated.shape[1]), dtype=np.uint8)
            cv2.imshow(WINDOW_MASK, mask_selected)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                selected = cycle_selected(profiles, selected, step=1)
                set_trackbars_from_profile(profiles[selected])
                print(f"[INFO] selected={selected}")
            elif key == ord("x"):
                selected = cycle_selected(profiles, selected, step=-1)
                set_trackbars_from_profile(profiles[selected])
                print(f"[INFO] selected={selected}")
            elif key == ord("s"):
                save_profiles(config_path, profiles, selected)
                print(f"[INFO] Saved: {config_path}")
            elif key == ord("r"):
                profiles = {k: _sanitize_profile(v) for k, v in DEFAULT_PROFILES.items()}
                selected = next(iter(profiles))
                set_trackbars_from_profile(profiles[selected])
                print("[INFO] reset profiles to defaults")
            elif key == ord("d"):
                if len(profiles) <= 1:
                    print("[WARN] keep at least one profile")
                else:
                    print(f"[INFO] delete profile: {selected}")
                    profiles.pop(selected, None)
                    selected = next(iter(profiles))
                    set_trackbars_from_profile(profiles[selected])
            elif key == ord("n"):
                name = input("New profile name: ").strip()
                if not name:
                    print("[WARN] empty name")
                    continue
                if name in profiles:
                    print(f"[WARN] profile exists: {name}")
                    continue
                new_prof = read_profile_from_trackbars(profiles[selected])
                new_prof["draw_bgr"] = list(_random_bgr())
                profiles[name] = _sanitize_profile(new_prof)
                selected = name
                set_trackbars_from_profile(profiles[selected])
                print(f"[INFO] added profile: {name}")

        save_profiles(config_path, profiles, selected)
        print(f"[INFO] Auto-saved profiles: {config_path}")
        return 0
    finally:
        if cam is not None:
            cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

