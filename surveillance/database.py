import sqlite3
import os
import json
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SurveillanceDatabase:
    """
    Thread-safe SQLite persistent storage for crowd surveillance events,
    hourly footfall metrics, and real-time telemetry.
    """

    def __init__(self, db_path: str = "logs/surveillance.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")  # Write-Ahead Logging for high concurrency
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 1. Alerts Table (Overcrowding, Loitering, Counterflow, etc.)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        camera_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        location_name TEXT NOT NULL,
                        detected_value INTEGER,
                        threshold_value INTEGER,
                        snapshot_path TEXT,
                        details TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_cam ON alerts(camera_id);")

                # 2. Footfall / Tripwire Events Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS footfall_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        camera_id TEXT NOT NULL,
                        tripwire_name TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        track_id INTEGER,
                        in_count INTEGER,
                        out_count INTEGER,
                        net_count INTEGER,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_footfall_ts ON footfall_events(timestamp);")

                # 3. Periodic Telemetry Rollups (Crowd count trends over time)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS telemetry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        camera_id TEXT NOT NULL,
                        total_count INTEGER NOT NULL,
                        fps REAL,
                        latency_ms REAL,
                        zones_json TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry(timestamp);")

                conn.commit()
        logger.info(f"Surveillance SQLite database initialized at: {self.db_path}")

    def log_alert(
        self,
        camera_id: str,
        event_type: str,
        location_name: str,
        detected_value: int,
        threshold_value: int,
        snapshot_path: Optional[str] = None,
        details: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> int:
        """Insert an incident alert record."""
        ts_str = datetime.fromtimestamp(timestamp or datetime.now().timestamp()).isoformat()
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO alerts (
                        timestamp, camera_id, event_type, location_name,
                        detected_value, threshold_value, snapshot_path, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    ts_str, camera_id, event_type, location_name,
                    detected_value, threshold_value, snapshot_path or "", details or ""
                ))
                conn.commit()
                return cursor.lastrowid

    def log_footfall(
        self,
        camera_id: str,
        tripwire_name: str,
        direction: str,
        track_id: int,
        in_count: int,
        out_count: int,
        net_count: int,
        timestamp: Optional[float] = None
    ) -> int:
        """Insert a directional crossing event record."""
        ts_str = datetime.fromtimestamp(timestamp or datetime.now().timestamp()).isoformat()
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO footfall_events (
                        timestamp, camera_id, tripwire_name, direction,
                        track_id, in_count, out_count, net_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    ts_str, camera_id, tripwire_name, direction,
                    track_id, in_count, out_count, net_count
                ))
                conn.commit()
                return cursor.lastrowid

    def log_telemetry(
        self,
        camera_id: str,
        total_count: int,
        fps: float,
        latency_ms: float,
        zones_data: Dict[str, Any],
        timestamp: Optional[float] = None
    ):
        """Insert periodic crowd level telemetry."""
        ts_str = datetime.fromtimestamp(timestamp or datetime.now().timestamp()).isoformat()
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO telemetry (
                        timestamp, camera_id, total_count, fps, latency_ms, zones_json
                    ) VALUES (?, ?, ?, ?, ?, ?);
                """, (
                    ts_str, camera_id, total_count, round(fps, 1),
                    round(latency_ms, 1), json.dumps(zones_data)
                ))
                conn.commit()

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve the most recent incident alerts."""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM alerts ORDER BY id DESC LIMIT ?;", (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute aggregate footfall and alert statistics."""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) AS total_alerts FROM alerts;")
                total_alerts = cursor.fetchone()["total_alerts"]

                cursor.execute("SELECT COUNT(*) AS total_crossings FROM footfall_events;")
                total_crossings = cursor.fetchone()["total_crossings"]

                cursor.execute("""
                    SELECT MAX(total_count) AS peak_crowd, AVG(total_count) AS avg_crowd
                    FROM telemetry;
                """)
                crowd_row = cursor.fetchone()
                peak_crowd = crowd_row["peak_crowd"] or 0
                avg_crowd = round(crowd_row["avg_crowd"] or 0.0, 1)

                return {
                    "total_alerts": total_alerts,
                    "total_crossings": total_crossings,
                    "peak_crowd": peak_crowd,
                    "average_crowd": avg_crowd,
                }
