import math
import os
import json
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

from .tracker import TrackedPerson


def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    """Check if three points are in counter-clockwise order."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float]
) -> bool:
    """Return True if line segment p1-p2 intersects line segment q1-q2."""
    return (
        ccw(p1, q1, q2) != ccw(p2, q1, q2)
        and ccw(p1, p2, q1) != ccw(p1, p2, q2)
    )


class VirtualTripwire:
    """
    Virtual Tripwire Line for directional In/Out footfall counting.
    """

    def __init__(
        self,
        name: str,
        pt1: List[float],
        pt2: List[float],
        in_label: str = "IN",
        out_label: str = "OUT",
        color: Tuple[int, int, int] = (0, 165, 255),  # BGR Orange
        is_normalized: Optional[bool] = None,
        cooldown_sec: float = 2.0,
        is_one_way: bool = False,
        allowed_direction: str = "IN",
    ):
        """
        Args:
            name: Human-readable line name (e.g., "Main Door", "Corridor Turnstile").
            pt1: [x1, y1] line start point.
            pt2: [x2, y2] line end point.
            in_label: Label for forward crossing.
            out_label: Label for backward crossing.
            color: Line color in BGR.
            is_normalized: True if coordinates in [0..1] range.
            cooldown_sec: Minimum seconds between repeated triggers for same track.
            is_one_way: If True, crossing in reverse triggers counterflow alerts.
            allowed_direction: 'IN' or 'OUT' if is_one_way is True.
        """
        self.name = name
        self.raw_pt1 = np.array(pt1, dtype=np.float32)
        self.raw_pt2 = np.array(pt2, dtype=np.float32)
        self.in_label = in_label
        self.out_label = out_label
        self.color = tuple(color)
        self.cooldown_sec = cooldown_sec
        self.is_one_way = is_one_way
        self.allowed_direction = allowed_direction

        if is_normalized is None:
            self.is_normalized = bool(
                np.all(self.raw_pt1 <= 1.05) and np.all(self.raw_pt2 <= 1.05)
            )
        else:
            self.is_normalized = is_normalized

        # Footfall counters
        self.in_count = 0
        self.out_count = 0

        # Track crossing states: track_id -> (last_side, last_timestamp)
        self.track_states: Dict[int, Tuple[int, float]] = {}

        # Visual pulse effect when crossing occurs
        self.last_cross_time = 0.0
        self.last_cross_direction = ""

    def get_pixel_endpoints(self, frame_width: int, frame_height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get integer pixel coordinates of line endpoints."""
        if self.is_normalized:
            p1 = (int(self.raw_pt1[0] * frame_width), int(self.raw_pt1[1] * frame_height))
            p2 = (int(self.raw_pt2[0] * frame_width), int(self.raw_pt2[1] * frame_height))
        else:
            p1 = (int(self.raw_pt1[0]), int(self.raw_pt1[1]))
            p2 = (int(self.raw_pt2[0]), int(self.raw_pt2[1]))
        return p1, p2

    def point_side(self, pt: Tuple[float, float], p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """
        Compute which side of directed line p1 -> p2 a point lies on.
        Returns:
            +1: Left side (IN)
            -1: Right side (OUT)
             0: Exactly on line
        """
        val = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
        if val > 1e-4:
            return 1
        elif val < -1e-4:
            return -1
        return 0

    def evaluate_tracks(
        self,
        tracks: List[TrackedPerson],
        frame_width: int,
        frame_height: int,
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all active tracks against the tripwire line.
        
        Returns:
            List of crossing event dictionaries.
        """
        p1, p2 = self.get_pixel_endpoints(frame_width, frame_height)
        events = []

        for track in tracks:
            if len(track.history) < 2:
                continue

            curr_pt = track.current_point
            prev_pt = track.previous_point

            curr_side = self.point_side(curr_pt, p1, p2)
            if curr_side == 0:
                continue

            # Check if trajectory segment intersected the tripwire segment
            has_intersected = segments_intersect(prev_pt, curr_pt, p1, p2)

            # Retrieve prior recorded side for this track
            last_recorded_side, last_time = self.track_states.get(
                track.track_id, (None, 0.0)
            )

            # If track was never recorded, register initial side
            if last_recorded_side is None:
                self.track_states[track.track_id] = (curr_side, timestamp)
                continue

            # Did a crossing happen?
            # 1. Segment intersection test, OR
            # 2. Side flip (e.g. from -1 to +1 or +1 to -1) while near the line
            side_flipped = (last_recorded_side != curr_side) and (last_recorded_side != 0)
            cooldown_passed = (timestamp - last_time) >= self.cooldown_sec

            if (has_intersected or side_flipped) and cooldown_passed:
                # Determine direction
                if last_recorded_side == -1 and curr_side == 1:
                    direction = "IN"
                    self.in_count += 1
                elif last_recorded_side == 1 and curr_side == -1:
                    direction = "OUT"
                    self.out_count += 1
                else:
                    direction = "CROSS"

                self.track_states[track.track_id] = (curr_side, timestamp)
                self.last_cross_time = timestamp
                self.last_cross_direction = direction

                event = {
                    "event_type": "TRIPWIRE_CROSSING",
                    "tripwire": self.name,
                    "direction": direction,
                    "track_id": track.track_id,
                    "timestamp": timestamp,
                    "in_count": self.in_count,
                    "out_count": self.out_count,
                    "net_count": self.in_count - self.out_count,
                }
                events.append(event)
            else:
                # Update current side
                if curr_side != 0:
                    self.track_states[track.track_id] = (curr_side, last_time)

        return events

    def draw(self, frame: np.ndarray, timestamp: float):
        """Render tripwire line, orientation indicator, and counts on frame."""
        h, w = frame.shape[:2]
        p1, p2 = self.get_pixel_endpoints(w, h)

        # Pulse highlight if crossed recently (within 0.8s)
        is_pulsing = (timestamp - self.last_cross_time) < 0.8
        line_color = (0, 255, 255) if is_pulsing else self.color
        line_thick = 4 if is_pulsing else 2

        # 1. Draw Tripwire Line
        cv2.line(frame, p1, p2, line_color, line_thick, cv2.LINE_AA)
        cv2.circle(frame, p1, 5, line_color, -1)
        cv2.circle(frame, p2, 5, line_color, -1)

        # 2. Draw Directional Orientation Arrow (pointing in "IN" direction)
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        line_len = math.hypot(dx, dy)
        if line_len > 1e-4:
            # Unit normal pointing to left (+1 side)
            nx = -dy / line_len
            ny = dx / line_len

            arrow_len = 22
            arrow_end = (int(mid_x + nx * arrow_len), int(mid_y + ny * arrow_len))
            cv2.arrowedLine(frame, (mid_x, mid_y), arrow_end, (0, 255, 0), 2, tipLength=0.35)

        # 3. Draw Badge at Midpoint
        net = self.in_count - self.out_count
        badge_text = f"{self.name} | IN: {self.in_count}  OUT: {self.out_count} | NET: {net:+d}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.44
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)

        bx = max(10, mid_x - tw // 2)
        by = max(25, mid_y - 12)

        # Background pill
        cv2.rectangle(frame, (bx - 5, by - th - 5), (bx + tw + 5, by + baseline + 3), (25, 25, 25), -1)
        cv2.rectangle(frame, (bx - 5, by - th - 5), (bx + tw + 5, by + baseline + 3), line_color, 1)
        cv2.putText(frame, badge_text, (bx, by), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pt1": self.raw_pt1.tolist(),
            "pt2": self.raw_pt2.tolist(),
            "in_label": self.in_label,
            "out_label": self.out_label,
            "color": list(self.color),
            "is_normalized": self.is_normalized,
            "cooldown_sec": self.cooldown_sec,
            "is_one_way": self.is_one_way,
            "allowed_direction": self.allowed_direction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VirtualTripwire":
        return cls(
            name=data["name"],
            pt1=data["pt1"],
            pt2=data["pt2"],
            in_label=data.get("in_label", "IN"),
            out_label=data.get("out_label", "OUT"),
            color=tuple(data.get("color", (0, 165, 255))),
            is_normalized=data.get("is_normalized", None),
            cooldown_sec=data.get("cooldown_sec", 2.0),
            is_one_way=data.get("is_one_way", False),
            allowed_direction=data.get("allowed_direction", "IN"),
        )


class TripwireManager:
    """Manages virtual tripwire lines, event dispatching, and visualization."""

    def __init__(self, tripwires: Optional[List[VirtualTripwire]] = None):
        self.tripwires: List[VirtualTripwire] = tripwires or []
        self.event_callbacks = []

    def add_tripwire(self, tripwire: VirtualTripwire):
        self.tripwires.append(tripwire)

    def register_callback(self, callback):
        self.event_callbacks.append(callback)

    def evaluate(
        self,
        tracks: List[TrackedPerson],
        frame_width: int,
        frame_height: int,
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """Evaluate all tripwires against active tracks."""
        all_events = []
        for tw in self.tripwires:
            events = tw.evaluate_tracks(tracks, frame_width, frame_height, timestamp)
            for event in events:
                all_events.append(event)
                for cb in self.event_callbacks:
                    try:
                        cb(event)
                    except Exception as e:
                        print(f"Error in tripwire callback: {e}")
        return all_events

    def render_overlay(self, frame: np.ndarray, timestamp: float):
        """Draw all tripwires onto the frame."""
        for tw in self.tripwires:
            tw.draw(frame, timestamp)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Return total In/Out/Net stats across all tripwires."""
        total_in = sum(tw.in_count for tw in self.tripwires)
        total_out = sum(tw.out_count for tw in self.tripwires)
        return {
            "total_in": total_in,
            "total_out": total_out,
            "net_flow": total_in - total_out,
            "tripwires": {
                tw.name: {
                    "in": tw.in_count,
                    "out": tw.out_count,
                    "net": tw.in_count - tw.out_count
                }
                for tw in self.tripwires
            }
        }

    def save_to_json(self, file_path: str):
        data = {"tripwires": [tw.to_dict() for tw in self.tripwires]}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: str) -> "TripwireManager":
        if not os.path.exists(file_path):
            return cls()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tripwires = [VirtualTripwire.from_dict(td) for td in data.get("tripwires", [])]
        return cls(tripwires=tripwires)
