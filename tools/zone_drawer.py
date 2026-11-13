import argparse
import json
import math
import os
import sys
import cv2
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class InteractiveSurveillanceCalibrator:
    """
    Visual calibration utility to draw, label, and export both
    polygonal surveillance zones and virtual tripwire lines.
    """

    def __init__(self, frame: np.ndarray, output_zones_json: str, output_tripwires_json: str):
        self.orig_frame = frame
        self.h, self.w = frame.shape[:2]
        self.output_zones_json = output_zones_json
        self.output_tripwires_json = output_tripwires_json

        self.mode = "ZONE"  # "ZONE" or "TRIPWIRE"
        self.zones = []
        self.tripwires = []

        self.current_points = []
        self.colors = [
            (0, 220, 100),   # Emerald Green
            (255, 180, 0),   # Cyan/Sky
            (0, 165, 255),   # Orange
            (200, 100, 255), # Magenta
            (100, 255, 200), # Lime
        ]
        self.color_idx = 0
        self.window_name = "Surveillance Calibrator [Press T to switch Mode]"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "ZONE":
                self.current_points.append((x, y))
            elif self.mode == "TRIPWIRE":
                if len(self.current_points) < 2:
                    self.current_points.append((x, y))
                    if len(self.current_points) == 2:
                        self._complete_tripwire()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_points:
                self.current_points.pop()

    def _complete_tripwire(self):
        p1, p2 = self.current_points
        default_name = f"Tripwire {len(self.tripwires) + 1}"
        print(f"\nDefining tripwire line from {p1} to {p2}.")
        name = input(f"Enter Tripwire Name [{default_name}]: ").strip() or default_name

        norm_p1 = [round(p1[0] / float(self.w), 4), round(p1[1] / float(self.h), 4)]
        norm_p2 = [round(p2[0] / float(self.w), 4), round(p2[1] / float(self.h), 4)]

        tw_dict = {
            "name": name,
            "pt1": norm_p1,
            "pt2": norm_p2,
            "in_label": "IN",
            "out_label": "OUT",
            "color": [0, 165, 255],
            "is_normalized": True,
            "cooldown_sec": 1.5,
        }
        self.tripwires.append(tw_dict)
        print(f"✅ Added Tripwire: {name}")
        self.current_points = []

    def draw_state(self) -> np.ndarray:
        display = self.orig_frame.copy()

        # 1. Draw completed zones
        overlay = display.copy()
        for z in self.zones:
            poly_norm = np.array(z["polygon"])
            pts = (poly_norm * [self.w, self.h]).astype(np.int32)
            c = tuple(z["color"])
            cv2.fillPoly(overlay, [pts], c)
            cv2.polylines(display, [pts], isClosed=True, color=c, thickness=2)

            M = cv2.moments(pts)
            cx = int(M["m10"] / max(M["m00"], 1e-5))
            cy = int(M["m01"] / max(M["m00"], 1e-5))
            cv2.putText(
                display, f"{z['name']} ({z['max_capacity']})",
                (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )

        cv2.addWeighted(overlay, 0.22, display, 0.78, 0, display)

        # 2. Draw completed tripwires
        for tw in self.tripwires:
            p1 = (int(tw["pt1"][0] * self.w), int(tw["pt1"][1] * self.h))
            p2 = (int(tw["pt2"][0] * self.w), int(tw["pt2"][1] * self.h))
            c = tuple(tw["color"])

            cv2.line(display, p1, p2, c, 2, cv2.LINE_AA)
            cv2.circle(display, p1, 4, c, -1)
            cv2.circle(display, p2, 4, c, -1)

            # Arrow indicating normal direction
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.hypot(dx, dy)
            if dist > 1e-4:
                nx = -dy / dist
                ny = dx / dist
                arrow_end = (int(mid_x + nx * 20), int(mid_y + ny * 20))
                cv2.arrowedLine(display, (mid_x, mid_y), arrow_end, (0, 255, 0), 2, tipLength=0.35)

            cv2.putText(
                display, f"TRIPWIRE: {tw['name']}",
                (mid_x - 30, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1
            )

        # 3. Draw in-progress shapes
        if self.current_points:
            c = (0, 0, 255)
            for pt in self.current_points:
                cv2.circle(display, pt, 5, c, -1)

            if len(self.current_points) > 1 and self.mode == "ZONE":
                pts_arr = np.array(self.current_points, np.int32)
                cv2.polylines(display, [pts_arr], isClosed=False, color=(0, 255, 255), thickness=2)

        # 4. Instructions and Mode Banner
        bar_h = 36
        cv2.rectangle(display, (0, self.h - bar_h), (self.w, self.h), (18, 18, 22), -1)
        mode_badge = f"[MODE: {self.mode}]"
        badge_color = (0, 220, 100) if self.mode == "ZONE" else (0, 165, 255)
        cv2.putText(display, mode_badge, (12, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_color, 1)

        help_txt = "[T]: Toggle Mode | [L-Click]: Point | [R-Click]: Undo | [C]: Close Zone | [S]: Save | [Q]: Quit"
        cv2.putText(display, help_txt, (150, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1)

        return display

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n=== Surveillance Spatial Calibrator ===")
        print("  - Press 't' to switch between ZONE and TRIPWIRE drawing modes.")
        print("  - ZONE mode: Left-click vertices -> Press 'c' to complete.")
        print("  - TRIPWIRE mode: Left-click start, then left-click end of line.")
        print("  - Right-click: Undo last point.")
        print("  - Press 's': Save both zones and tripwires.")
        print("  - Press 'q': Exit without saving.")
        print("=======================================\n")

        while True:
            display = self.draw_state()
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('t'):
                self.mode = "TRIPWIRE" if self.mode == "ZONE" else "ZONE"
                self.current_points = []
                print(f"Switched mode to: {self.mode}")

            elif key == ord('c') or key == 13:  # Complete zone
                if self.mode == "ZONE" and len(self.current_points) >= 3:
                    default_name = f"Zone {len(self.zones) + 1}"
                    print(f"\nDefining polygon with {len(self.current_points)} vertices.")
                    name = input(f"Enter Zone Name [{default_name}]: ").strip() or default_name
                    cap_str = input("Enter Max Person Capacity Threshold [15]: ").strip()
                    try:
                        max_cap = int(cap_str) if cap_str else 15
                    except ValueError:
                        max_cap = 15

                    norm_polygon = [
                        [round(pt[0] / float(self.w), 4), round(pt[1] / float(self.h), 4)]
                        for pt in self.current_points
                    ]
                    self.zones.append({
                        "name": name,
                        "polygon": norm_polygon,
                        "max_capacity": max_cap,
                        "color": list(self.colors[self.color_idx % len(self.colors)]),
                        "alert_color": [0, 0, 240],
                        "is_normalized": True
                    })
                    self.color_idx += 1
                    self.current_points = []
                    print(f"✅ Added {name} (Capacity: {max_cap})")
                elif self.mode == "ZONE":
                    print("⚠️ A polygon needs at least 3 vertices!")

            elif key == ord('s'):
                # Save zones
                if self.zones:
                    os.makedirs(os.path.dirname(os.path.abspath(self.output_zones_json)), exist_ok=True)
                    with open(self.output_zones_json, "w", encoding="utf-8") as f:
                        json.dump({"version": "1.0", "zones": self.zones}, f, indent=2)
                    print(f"🎉 Saved {len(self.zones)} zones to: {self.output_zones_json}")

                # Save tripwires
                if self.tripwires:
                    os.makedirs(os.path.dirname(os.path.abspath(self.output_tripwires_json)), exist_ok=True)
                    with open(self.output_tripwires_json, "w", encoding="utf-8") as f:
                        json.dump({"version": "1.0", "tripwires": self.tripwires}, f, indent=2)
                    print(f"🎉 Saved {len(self.tripwires)} tripwires to: {self.output_tripwires_json}")

                break

            elif key == ord('q') or key == 27:
                print("Exiting without saving.")
                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive Spatial Calibrator (Zones + Tripwires)")
    parser.add_argument("--source", default="test_videos/mall_crowd_test.mp4",
                        help="Video file, RTSP URL, or webcam ID (0)")
    parser.add_argument("--zones", default="configs/zones_config.json",
                        help="Output path for zones JSON")
    parser.add_argument("--tripwires", default="configs/tripwires_config.json",
                        help="Output path for tripwires JSON")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: Could not open source: {args.source}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: Could not read sample frame from video source.")
        return

    calibrator = InteractiveSurveillanceCalibrator(frame, args.zones, args.tripwires)
    calibrator.run()


if __name__ == "__main__":
    main()
