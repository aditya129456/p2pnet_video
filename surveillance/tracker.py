import time
import math
from collections import deque
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanPointFilter:
    """
    Constant-Velocity 2D Kalman Filter for tracking a single person's head centroid.
    State vector: [x, y, vx, vy]^T
    Measurement vector: [x, y]^T
    """

    def __init__(self, init_x: float, init_y: float):
        # State vector [x, y, vx, vy]
        self.x = np.array([init_x, init_y, 0.0, 0.0], dtype=np.float32)

        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Measurement matrix H
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        # State covariance matrix P
        self.P = np.diag([10.0, 10.0, 50.0, 50.0]).astype(np.float32)

        # Process noise covariance Q
        self.Q = np.diag([1.5, 1.5, 3.0, 3.0]).astype(np.float32)

        # Measurement noise covariance R
        self.R = np.diag([4.0, 4.0]).astype(np.float32)

    def predict(self) -> Tuple[float, float]:
        """Project state ahead and return predicted (x, y)."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, meas_x: float, meas_y: float):
        """Incorporate new measurement into state estimate."""
        z = np.array([meas_x, meas_y], dtype=np.float32)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(4, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))


class TrackedPerson:
    """Represents an active multi-frame tracked individual."""

    def __init__(
        self,
        track_id: int,
        initial_point: Tuple[float, float],
        timestamp: float,
        trajectory_len: int = 30
    ):
        self.track_id = track_id
        self.kf = KalmanPointFilter(initial_point[0], initial_point[1])

        self.hits = 1
        self.time_since_update = 0
        self.is_confirmed = False

        self.first_seen_time = timestamp
        self.last_seen_time = timestamp

        # Trajectory history of (x, y) coordinates
        self.history = deque(maxlen=trajectory_len)
        self.history.append((float(initial_point[0]), float(initial_point[1])))

        # Crossed tripwires tracking: {tripwire_id: last_side}
        self.tripwire_sides: Dict[str, int] = {}

    def predict(self) -> Tuple[float, float]:
        pos = self.kf.predict()
        self.time_since_update += 1
        return pos

    def update(self, point: Tuple[float, float], timestamp: float, min_hits: int = 3):
        self.kf.update(point[0], point[1])
        self.hits += 1
        self.time_since_update = 0
        self.last_seen_time = timestamp

        curr_pos = (float(self.kf.position[0]), float(self.kf.position[1]))
        self.history.append(curr_pos)

        if self.hits >= min_hits:
            self.is_confirmed = True

    @property
    def current_point(self) -> Tuple[float, float]:
        return self.history[-1] if self.history else (0.0, 0.0)

    @property
    def previous_point(self) -> Tuple[float, float]:
        return self.history[-2] if len(self.history) >= 2 else self.current_point

    @property
    def velocity(self) -> Tuple[float, float]:
        return self.kf.velocity

    @property
    def speed(self) -> float:
        return self.kf.speed

    @property
    def dwell_time(self) -> float:
        return max(0.0, self.last_seen_time - self.first_seen_time)


class KalmanPointTracker:
    """
    Real-Time Multi-Object Point Tracker.
    
    Associates 2D head centroids frame-to-frame using a Kalman Filter
    and Munkres (Hungarian) linear assignment algorithm.
    """

    def __init__(
        self,
        max_distance: float = 65.0,
        max_age: int = 12,
        min_hits: int = 2,
        trajectory_len: int = 30,
    ):
        """
        Args:
            max_distance: Maximum pixel distance to associate detection with existing track.
            max_age: Maximum consecutive missed frames before track deletion.
            min_hits: Frames required before track transitions to 'confirmed' status.
            trajectory_len: Number of historical points to store for motion trails.
        """
        self.max_distance = max_distance
        self.max_age = max_age
        self.min_hits = min_hits
        self.trajectory_len = trajectory_len

        self.next_track_id = 1
        self.tracks: List[TrackedPerson] = []

    def update(
        self,
        detections: List[List[float]],
        timestamp: Optional[float] = None
    ) -> List[TrackedPerson]:
        """
        Update tracker with new frame detections.

        Args:
            detections: List of [x, y] coordinates from crowd detector.
            timestamp: Frame timestamp in seconds.

        Returns:
            List of active confirmed TrackedPerson objects.
        """
        now = timestamp if timestamp is not None else time.time()

        # 1. Predict positions for all existing tracks
        for track in self.tracks:
            track.predict()

        num_tracks = len(self.tracks)
        num_dets = len(detections)

        # 2. Association step
        if num_tracks > 0 and num_dets > 0:
            # Build Euclidean cost matrix
            cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
            for i, track in enumerate(self.tracks):
                tx, ty = track.kf.position
                for j, det in enumerate(detections):
                    dx = tx - det[0]
                    dy = ty - det[1]
                    cost_matrix[i, j] = math.hypot(dx, dy)

            # Solve linear assignment problem
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_dets = set()

            for t_idx, d_idx in zip(track_indices, det_indices):
                dist = cost_matrix[t_idx, d_idx]
                if dist <= self.max_distance:
                    self.tracks[t_idx].update(
                        (detections[d_idx][0], detections[d_idx][1]),
                        now,
                        self.min_hits
                    )
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)

            # Create new tracks for unmatched detections
            for j in range(num_dets):
                if j not in matched_dets:
                    self._init_track(detections[j], now)

        elif num_dets > 0:
            # No tracks exist yet: initialize all detections as new tracks
            for det in detections:
                self._init_track(det, now)

        # 3. Clean up dead tracks (exceeded max_age)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return all confirmed tracks
        return [t for t in self.tracks if t.is_confirmed]

    def _init_track(self, point: List[float], timestamp: float):
        new_track = TrackedPerson(
            track_id=self.next_track_id,
            initial_point=(point[0], point[1]),
            timestamp=timestamp,
            trajectory_len=self.trajectory_len
        )
        self.next_track_id += 1
        self.tracks.append(new_track)
