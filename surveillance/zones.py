import json
import os
import time
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2


class SurveillanceZone:
    """
    Represents a polygonal Region of Interest (ROI) for crowd monitoring.
    """

    def __init__(
        self,
        name: str,
        polygon: List[List[float]],
        max_capacity: int = 20,
        color: Tuple[int, int, int] = (0, 200, 100),       # BGR (Emerald Green)
        alert_color: Tuple[int, int, int] = (0, 0, 240),   # BGR (Vibrant Red)
        is_normalized: Optional[bool] = None,
        min_alert_frames: int = 2,
        alert_cooldown: float = 4.0,
    ):
        """
        Args:
            name: Human-readable zone identifier (e.g. "Gate 1", "Escalator Queue").
            polygon: List of [x, y] coordinates defining the polygon vertices.
            max_capacity: Threshold above which an overcrowding alert is raised.
            color: BGR color for normal operation.
            alert_color: BGR color when capacity is exceeded.
            is_normalized: Whether coordinates are in [0, 1] range. Auto-detected if None.
            min_alert_frames: Number of consecutive frames threshold must be breached before alerting.
            alert_cooldown: Minimum seconds between repeated alert dispatches.
        """
        self.name = name
        self.raw_polygon = np.array(polygon, dtype=np.float32)
        self.max_capacity = max_capacity
        self.color = tuple(color)
        self.alert_color = tuple(alert_color)
        self.min_alert_frames = min_alert_frames
        self.alert_cooldown = alert_cooldown

        if is_normalized is None:
            # Check if all coordinates are <= 1.05
            self.is_normalized = bool(np.all(self.raw_polygon <= 1.05))
        else:
            self.is_normalized = is_normalized

        # Runtime state
        self.current_count = 0
        self.is_alert = False
        self.consecutive_alert_frames = 0
        self.last_alert_time = 0.0
        self.total_alerts_triggered = 0
        self.points_inside: List[List[float]] = []

    def get_pixel_polygon(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Convert polygon vertices to integer pixel coordinates for a specific resolution."""
        if self.is_normalized:
            scaled = self.raw_polygon.copy()
            scaled[:, 0] *= frame_width
            scaled[:, 1] *= frame_height
            return scaled.astype(np.int32)
        else:
            return self.raw_polygon.astype(np.int32)

    def evaluate_points(
        self,
        points: List[List[float]],
        frame_width: int,
        frame_height: int,
        timestamp: float = 0.0,
    ) -> Tuple[int, bool, List[List[float]], bool]:
        """
        Determine which detected points lie inside this zone polygon.

        Args:
            points: List of [x, y] pixel coordinates.
            frame_width: Width of the current frame.
            frame_height: Height of the current frame.
            timestamp: Frame timestamp in seconds.

        Returns:
            (count, is_alert, points_inside, should_dispatch_alert)
        """
        poly_pts = self.get_pixel_polygon(frame_width, frame_height)
        inside = []

        for pt in points:
            # cv2.pointPolygonTest returns > 0 if inside, 0 on edge, < 0 outside
            if cv2.pointPolygonTest(poly_pts, (float(pt[0]), float(pt[1])), False) >= 0:
                inside.append(pt)

        self.current_count = len(inside)
        self.points_inside = inside
        self.is_alert = self.current_count > self.max_capacity
        should_dispatch_alert = False

        if self.is_alert:
            self.consecutive_alert_frames += 1
            if (
                self.consecutive_alert_frames >= self.min_alert_frames
                and (timestamp - self.last_alert_time) >= self.alert_cooldown
            ):
                self.last_alert_time = timestamp
                self.total_alerts_triggered += 1
                should_dispatch_alert = True
        else:
            self.consecutive_alert_frames = 0

        return self.current_count, self.is_alert, self.points_inside, should_dispatch_alert

    def to_dict(self) -> Dict[str, Any]:
        """Serialize zone to dictionary."""
        return {
            "name": self.name,
            "polygon": self.raw_polygon.tolist(),
            "max_capacity": self.max_capacity,
            "color": list(self.color),
            "alert_color": list(self.alert_color),
            "is_normalized": self.is_normalized,
            "min_alert_frames": self.min_alert_frames,
            "alert_cooldown": self.alert_cooldown,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurveillanceZone":
        """Instantiate zone from dictionary."""
        return cls(
            name=data["name"],
            polygon=data["polygon"],
            max_capacity=data.get("max_capacity", 20),
            color=tuple(data.get("color", (0, 200, 100))),
            alert_color=tuple(data.get("alert_color", (0, 0, 240))),
            is_normalized=data.get("is_normalized", None),
            min_alert_frames=data.get("min_alert_frames", 2),
            alert_cooldown=data.get("alert_cooldown", 4.0),
        )


class ZoneManager:
    """
    Manages multiple surveillance zones, evaluators, alert listeners, and visual overlays.
    """

    def __init__(self, zones: Optional[List[SurveillanceZone]] = None):
        self.zones: List[SurveillanceZone] = zones or []
        self.alert_callbacks = []

    def add_zone(self, zone: SurveillanceZone):
        self.zones.append(zone)

    def register_alert_callback(self, callback):
        """Register a callback fn(zone_name, count, threshold, frame_timestamp)."""
        self.alert_callbacks.append(callback)

    def evaluate(
        self,
        points: List[List[float]],
        frame_width: int,
        frame_height: int,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate all zones against the current frame's detected head points.
        """
        now = timestamp or time.time()
        results = {}
        active_alerts = []

        for zone in self.zones:
            count, is_alert, inside_pts, should_dispatch = zone.evaluate_points(
                points, frame_width, frame_height, timestamp=now
            )
            results[zone.name] = {
                "count": count,
                "max_capacity": zone.max_capacity,
                "is_alert": is_alert,
                "points_inside": inside_pts,
                "consecutive_alert_frames": zone.consecutive_alert_frames,
            }

            if should_dispatch:
                active_alerts.append(zone.name)
                for cb in self.alert_callbacks:
                    try:
                        cb(zone.name, count, zone.max_capacity, now)
                    except Exception as e:
                        print(f"Error in alert callback: {e}")

        return {
            "total_scene_count": len(points),
            "zones": results,
            "has_active_alert": any(z["is_alert"] for z in results.values()),
            "alerting_zones": [name for name, z in results.items() if z["is_alert"]],
        }

    def render_overlay(
        self,
        frame: np.ndarray,
        points: List[List[float]],
        stats: Dict[str, Any],
        fps: float = 0.0,
        inference_ms: float = 0.0,
        camera_name: str = "CAM-01",
        tracks: Optional[List[Any]] = None,
        tripwire_manager: Optional[Any] = None,
        timestamp: float = 0.0,
        show_trajectories: bool = True,
        show_track_ids: bool = True,
    ) -> np.ndarray:
        """
        Render a clean, professional surveillance HUD over the video frame.
        Includes translucent zone polygons, tripwires, trajectories, badges, alert warnings, and metrics bar.
        """
        h, w = frame.shape[:2]
        output = frame.copy()

        # 1. Render Translucent Polygonal Zones
        overlay = output.copy()
        for zone in self.zones:
            poly_pts = zone.get_pixel_polygon(w, h)
            zone_color = zone.alert_color if zone.is_alert else zone.color
            alpha = 0.30 if zone.is_alert else 0.18

            # Fill translucent polygon
            cv2.fillPoly(overlay, [poly_pts], zone_color)
            cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, output)
            overlay = output.copy()

            # Crisp boundary line
            border_thick = 3 if zone.is_alert else 2
            cv2.polylines(output, [poly_pts], isClosed=True, color=zone_color, thickness=border_thick)

            # Zone Badge Label
            self._draw_zone_badge(output, zone, poly_pts)

        # 2. Render Virtual Tripwires (if available)
        if tripwire_manager is not None:
            tripwire_manager.render_overlay(output, timestamp)

        # 3. Render Motion Trails & Trajectories (if tracks provided)
        if tracks is not None and show_trajectories:
            for track in tracks:
                hist = list(track.history)
                if len(hist) >= 2:
                    # Draw fading trajectory tail
                    for k in range(1, len(hist)):
                        alpha_t = k / float(len(hist))
                        color_t = (
                            int(255 * alpha_t),
                            int(200 * alpha_t),
                            int(50 * alpha_t)
                        )
                        pt_start = (int(hist[k - 1][0]), int(hist[k - 1][1]))
                        pt_end = (int(hist[k][0]), int(hist[k][1]))
                        thick = 2 if alpha_t > 0.6 else 1
                        cv2.line(output, pt_start, pt_end, color_t, thick, cv2.LINE_AA)

        # 4. Render Head Centroid Points & Track IDs
        if tracks is not None:
            for track in tracks:
                x, y = int(track.current_point[0]), int(track.current_point[1])
                if 0 <= x < w and 0 <= y < h:
                    in_alert = any(
                        zone.is_alert and [track.current_point[0], track.current_point[1]] in zone.points_inside
                        for zone in self.zones
                    )
                    pt_color = (0, 0, 255) if in_alert else (0, 255, 255)
                    radius = 3 if w >= 1280 else 2
                    cv2.circle(output, (x, y), radius, pt_color, -1)

                    if show_track_ids:
                        id_str = f"#{track.track_id}"
                        cv2.putText(
                            output, id_str, (x + 4, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA
                        )
        else:
            # Fallback to drawing raw points if tracking not active
            for pt in points:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    in_alert = any(zone.is_alert and pt in zone.points_inside for zone in self.zones)
                    pt_color = (0, 0, 255) if in_alert else (0, 255, 255)
                    radius = 3 if w >= 1280 else 2
                    cv2.circle(output, (x, y), radius, pt_color, -1)

        # 5. Top HUD Banner
        tripwire_summary = tripwire_manager.get_summary_stats() if tripwire_manager else None
        self._draw_hud_header(output, stats, fps, inference_ms, camera_name, tripwire_summary)

        return output

    def _draw_zone_badge(self, frame: np.ndarray, zone: SurveillanceZone, poly_pts: np.ndarray):
        """Draw an informative card / badge near the centroid of the polygon."""
        # Find centroid
        M = cv2.moments(poly_pts)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = poly_pts[0]

        status_text = f"OVERFLOW {zone.current_count}/{zone.max_capacity}" if zone.is_alert else f"{zone.name}: {zone.current_count}/{zone.max_capacity}"
        bg_color = (0, 0, 180) if zone.is_alert else (20, 20, 20)
        border_color = (0, 0, 255) if zone.is_alert else zone.color

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(status_text, font, font_scale, thickness)

        bx = max(10, cx - tw // 2)
        by = max(30, cy - th // 2)

        # Draw badge background
        cv2.rectangle(frame, (bx - 6, by - th - 6), (bx + tw + 6, by + baseline + 4), bg_color, -1)
        cv2.rectangle(frame, (bx - 6, by - th - 6), (bx + tw + 6, by + baseline + 4), border_color, 1)
        # Draw badge text
        cv2.putText(frame, status_text, (bx, by), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_hud_header(
        self,
        frame: np.ndarray,
        stats: Dict[str, Any],
        fps: float,
        inference_ms: float,
        camera_name: str,
        tripwire_summary: Optional[Dict[str, Any]] = None,
    ):
        """Draw an enterprise-grade top metrics banner."""
        h, w = frame.shape[:2]
        banner_h = 42

        # Translucent dark header banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 18), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Top border accent line
        accent_color = (0, 0, 255) if stats.get("has_active_alert") else (0, 220, 100)
        cv2.line(frame, (0, banner_h), (w, banner_h), accent_color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S")

        # 1. Left: Camera Name + Live Status
        status_dot = (0, 255, 0)
        cv2.circle(frame, (18, 21), 6, status_dot, -1)
        cv2.putText(frame, f"LIVE | {camera_name} | {ts_str}", (32, 26), font, 0.44, (230, 230, 230), 1, cv2.LINE_AA)

        # 2. Middle: Global Scene Count & Tripwire Footfall
        total_count = stats.get("total_scene_count", 0)
        if tripwire_summary and (tripwire_summary.get("total_in", 0) > 0 or tripwire_summary.get("total_out", 0) > 0):
            tin = tripwire_summary.get("total_in", 0)
            tout = tripwire_summary.get("total_out", 0)
            net = tripwire_summary.get("net_flow", 0)
            count_str = f"COUNT: {total_count} | IN: {tin} OUT: {tout} (NET: {net:+d})"
        else:
            count_str = f"TOTAL DETECTED: {total_count}"

        (cw, _), _ = cv2.getTextSize(count_str, font, 0.50, 2)
        mid_x = (w // 2) - (cw // 2)
        cv2.putText(frame, count_str, (mid_x, 27), font, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

        # 3. Right: FPS / Latency / Alert Pill
        right_info = f"{fps:.1f} FPS | {inference_ms:.0f}ms"
        if stats.get("has_active_alert"):
            alerts = ", ".join(stats.get("alerting_zones", []))
            alert_text = f"[!] ALERT: {alerts}"
            (aw, _), _ = cv2.getTextSize(alert_text, font, 0.45, 1)
            cv2.rectangle(frame, (w - aw - 160, 8), (w - 145, 34), (0, 0, 220), -1)
            cv2.putText(frame, alert_text, (w - aw - 155, 25), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        (rw, _), _ = cv2.getTextSize(right_info, font, 0.44, 1)
        cv2.putText(frame, right_info, (w - rw - 15, 26), font, 0.44, (180, 200, 200), 1, cv2.LINE_AA)

    def save_to_json(self, file_path: str):
        """Save zone configurations to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = {
            "version": "1.0",
            "zones": [zone.to_dict() for zone in self.zones],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: str) -> "ZoneManager":
        """Load zone configurations from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Zones config not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        zones = [SurveillanceZone.from_dict(zd) for zd in data.get("zones", [])]
        return cls(zones=zones)
