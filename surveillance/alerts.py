import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from .database import SurveillanceDatabase
from .notifier import NotificationManager
from .tracker import TrackedPerson
from .zones import SurveillanceZone, ZoneManager

logger = logging.getLogger(__name__)


class AlertRuleEngine:
    """
    Multi-rule surveillance anomaly and alert engine.
    
    Evaluates:
      1. Overcrowding in monitored zones.
      2. Loitering / prolonged dwell time in sensitive areas.
      3. Wrong-way / counterflow crossing on one-way gates.
      4. Sudden crowd density surges.
    """

    def __init__(
        self,
        db: SurveillanceDatabase,
        notifier: NotificationManager,
        camera_id: str = "CAM-01",
        loitering_threshold_sec: float = 20.0,
        surge_threshold: int = 15,
        default_cooldown_sec: float = 5.0,
    ):
        self.db = db
        self.notifier = notifier
        self.camera_id = camera_id
        self.loitering_threshold_sec = loitering_threshold_sec
        self.surge_threshold = surge_threshold
        self.default_cooldown_sec = default_cooldown_sec

        # Cooldown state: alert_key -> last_alert_time
        self.cooldowns: Dict[str, float] = {}

        # History for sudden surge detection
        self.count_history: List[int] = []

    def _is_on_cooldown(self, key: str, now: float, cooldown: Optional[float] = None) -> bool:
        cd = cooldown if cooldown is not None else self.default_cooldown_sec
        last_time = self.cooldowns.get(key, 0.0)
        if (now - last_time) < cd:
            return True
        self.cooldowns[key] = now
        return False

    def evaluate_frame(
        self,
        frame: np.ndarray,
        zone_manager: ZoneManager,
        tracks: Optional[List[TrackedPerson]],
        total_count: int,
        timestamp: float
    ):
        """Evaluate all active surveillance rules on current frame."""
        now = timestamp or time.time()

        # 1. Overcrowding Rule
        for zone in zone_manager.zones:
            if zone.is_alert:
                key = f"OVERCROWD_{zone.name}"
                if not self._is_on_cooldown(key, now, cooldown=zone.alert_cooldown):
                    # Capture snapshot and log
                    snapshot_path = self.notifier.capture_snapshot(
                        frame=frame,
                        camera_id=self.camera_id,
                        event_type="OVERCROWDING",
                        location_name=zone.name
                    )

                    # Store in SQLite
                    self.db.log_alert(
                        camera_id=self.camera_id,
                        event_type="OVERCROWDING_ALERT",
                        location_name=zone.name,
                        detected_value=zone.current_count,
                        threshold_value=zone.max_capacity,
                        snapshot_path=snapshot_path,
                        details=f"Excess: +{zone.current_count - zone.max_capacity}",
                        timestamp=now
                    )

                    # Dispatch external notifications
                    self.notifier.dispatch(
                        camera_id=self.camera_id,
                        event_type="OVERCROWDING_ALERT",
                        location_name=zone.name,
                        detected_value=zone.current_count,
                        threshold_value=zone.max_capacity,
                        snapshot_path=snapshot_path,
                        details=f"Zone count {zone.current_count} exceeded limit of {zone.max_capacity}",
                        timestamp=now
                    )

        # 2. Loitering / Dwell Time Rule
        if tracks:
            for track in tracks:
                if track.dwell_time >= self.loitering_threshold_sec:
                    key = f"LOITERING_{track.track_id}"
                    if not self._is_on_cooldown(key, now, cooldown=30.0):
                        snapshot_path = self.notifier.capture_snapshot(
                            frame=frame,
                            camera_id=self.camera_id,
                            event_type="LOITERING",
                            location_name=f"Track #{track.track_id}"
                        )

                        self.db.log_alert(
                            camera_id=self.camera_id,
                            event_type="LOITERING_ALERT",
                            location_name=f"Person #{track.track_id}",
                            detected_value=int(track.dwell_time),
                            threshold_value=int(self.loitering_threshold_sec),
                            snapshot_path=snapshot_path,
                            details=f"Individual stationary for {track.dwell_time:.1f}s",
                            timestamp=now
                        )

                        self.notifier.dispatch(
                            camera_id=self.camera_id,
                            event_type="LOITERING_ALERT",
                            location_name=f"Person #{track.track_id}",
                            detected_value=int(track.dwell_time),
                            threshold_value=int(self.loitering_threshold_sec),
                            snapshot_path=snapshot_path,
                            details=f"Dwell time exceeded: {track.dwell_time:.1f}s",
                            timestamp=now
                        )

        # 3. Sudden Surge Rule (Sudden jump in scene crowd count)
        self.count_history.append(total_count)
        if len(self.count_history) > 20:
            self.count_history.pop(0)

        if len(self.count_history) >= 10:
            baseline = self.count_history[0]
            surge_diff = total_count - baseline
            if surge_diff >= self.surge_threshold:
                key = "CROWD_SURGE"
                if not self._is_on_cooldown(key, now, cooldown=20.0):
                    snapshot_path = self.notifier.capture_snapshot(
                        frame=frame,
                        camera_id=self.camera_id,
                        event_type="CROWD_SURGE",
                        location_name="Global Scene"
                    )

                    self.db.log_alert(
                        camera_id=self.camera_id,
                        event_type="CROWD_SURGE_ALERT",
                        location_name="Global Scene",
                        detected_value=total_count,
                        threshold_value=baseline + self.surge_threshold,
                        snapshot_path=snapshot_path,
                        details=f"Surge of +{surge_diff} persons within seconds",
                        timestamp=now
                    )

                    self.notifier.dispatch(
                        camera_id=self.camera_id,
                        event_type="CROWD_SURGE_ALERT",
                        location_name="Global Scene",
                        detected_value=total_count,
                        threshold_value=baseline + self.surge_threshold,
                        snapshot_path=snapshot_path,
                        details=f"Rapid crowd influx: +{surge_diff} persons",
                        timestamp=now
                    )

    def evaluate_tripwire_event(
        self,
        frame: np.ndarray,
        event: Dict[str, Any],
        tripwire_obj: Any
    ):
        """Evaluate crossing events for counterflow / wrong-way entry."""
        # Check if tripwire is one-way
        is_one_way = getattr(tripwire_obj, "is_one_way", False)
        allowed_direction = getattr(tripwire_obj, "allowed_direction", "IN")

        if is_one_way and event["direction"] != allowed_direction and event["direction"] != "CROSS":
            now = event["timestamp"]
            key = f"WRONG_WAY_{tripwire_obj.name}_{event['track_id']}"
            if not self._is_on_cooldown(key, now, cooldown=5.0):
                snapshot_path = self.notifier.capture_snapshot(
                    frame=frame,
                    camera_id=self.camera_id,
                    event_type="WRONG_WAY",
                    location_name=tripwire_obj.name
                )

                self.db.log_alert(
                    camera_id=self.camera_id,
                    event_type="COUNTERFLOW_ALERT",
                    location_name=tripwire_obj.name,
                    detected_value=1,
                    threshold_value=0,
                    snapshot_path=snapshot_path,
                    details=f"Person #{event['track_id']} moved in forbidden direction: {event['direction']}",
                    timestamp=now
                )

                self.notifier.dispatch(
                    camera_id=self.camera_id,
                    event_type="COUNTERFLOW_ALERT",
                    location_name=tripwire_obj.name,
                    detected_value=1,
                    threshold_value=0,
                    snapshot_path=snapshot_path,
                    details=f"Reverse entry detected: {event['direction']} on one-way gate",
                    timestamp=now
                )
