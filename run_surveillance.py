import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
import cv2

from surveillance.detector import CrowdDetector
from surveillance.stream import ThreadedCameraStream
from surveillance.zones import ZoneManager, SurveillanceZone
from surveillance.tracker import KalmanPointTracker
from surveillance.tripwire import TripwireManager, VirtualTripwire
from surveillance.database import SurveillanceDatabase
from surveillance.notifier import NotificationManager
from surveillance.alerts import AlertRuleEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Surveillance")


def get_args():
    parser = argparse.ArgumentParser(
        description="Enterprise Real-Time Crowd Surveillance & Spatial Analytics Engine"
    )
    parser.add_argument(
        "--source",
        default="test_videos/mall_crowd_test.mp4",
        help="Input video source: RTSP stream URL, camera index (0), or video file path."
    )
    parser.add_argument(
        "--weights",
        default="./pretrained_model/best_mae.pth",
        help="Path to trained P2PNet model checkpoint."
    )
    parser.add_argument(
        "--zones",
        default="configs/zones_config.json",
        help="Path to JSON file containing polygonal zone configurations."
    )
    parser.add_argument(
        "--tripwires",
        default="configs/tripwires_config.json",
        help="Path to JSON file containing virtual tripwire lines."
    )
    parser.add_argument(
        "--alerts-config",
        default="configs/alerts_config.json",
        help="Path to JSON file configuring alerts, snapshots, webhooks, and telegram."
    )
    parser.add_argument(
        "--db-path",
        default="logs/surveillance.db",
        help="Path to persistent SQLite database."
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Point confidence threshold (default: 0.5)."
    )
    parser.add_argument(
        "--camera-name",
        default="CAM-01-MALL",
        help="Camera identifier for logs, database, and HUD."
    )
    parser.add_argument(
        "--output",
        default="prediction/surveillance_output.mp4",
        help="Path to save annotated surveillance video (leave empty to disable saving)."
    )
    parser.add_argument(
        "--alert-log",
        default="logs/surveillance_alerts.jsonl",
        help="Path to write JSON Lines alert log."
    )
    parser.add_argument(
        "--event-log",
        default="logs/surveillance_events.jsonl",
        help="Path to write tripwire and crossing events log."
    )
    parser.add_argument(
        "--snapshot-dir",
        default="logs/snapshots",
        help="Directory to save incident snapshot images."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live GUI playback window (press 'q' to stop)."
    )
    parser.add_argument(
        "--max-frames",
        default=0,
        type=int,
        help="Maximum frames to process (0 = entire stream/file)."
    )
    parser.add_argument(
        "--downsample",
        default=1,
        type=int,
        help="Inference frame step (e.g. 1 = every frame, 2 = every 2nd frame)."
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable multi-object Kalman point tracking."
    )
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Hide motion trail lines."
    )
    parser.add_argument(
        "--max-track-dist",
        default=65.0,
        type=float,
        help="Max pixel distance for associating points to tracks."
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "amp", "fp32"],
        help="Inference precision (fp16 enables NVIDIA Ampere Tensor Cores for ~2x speedup)."
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA Graph capture and replay for ultra-low launch latency."
    )
    return parser.parse_args()


class EventLogger:
    """Thread-safe event and alert logging to JSONL files and SQLite."""

    def __init__(self, alert_log_path: str, event_log_path: str, camera_name: str, db: SurveillanceDatabase):
        self.alert_log_path = alert_log_path
        self.event_log_path = event_log_path
        self.camera_name = camera_name
        self.db = db

        if alert_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(alert_log_path)), exist_ok=True)
        if event_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(event_log_path)), exist_ok=True)

    def log_tripwire_crossing(self, event_data: dict):
        record = {
            "timestamp": datetime.fromtimestamp(event_data["timestamp"]).isoformat(),
            "camera": self.camera_name,
            "event_type": "TRIPWIRE_CROSSING",
            "tripwire": event_data["tripwire"],
            "direction": event_data["direction"],
            "track_id": event_data["track_id"],
            "in_count": event_data["in_count"],
            "out_count": event_data["out_count"],
            "net_count": event_data["net_count"],
        }
        logger.info(
            f"🚶 Cross Event! Person #{event_data['track_id']} crossed '{event_data['tripwire']}' "
            f"[{event_data['direction']}] | Totals: IN={event_data['in_count']} OUT={event_data['out_count']}"
        )
        if self.event_log_path:
            try:
                with open(self.event_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                logger.error(f"Failed to write event log: {e}")

        # Store in SQLite database
        self.db.log_footfall(
            camera_id=self.camera_name,
            tripwire_name=event_data["tripwire"],
            direction=event_data["direction"],
            track_id=event_data["track_id"],
            in_count=event_data["in_count"],
            out_count=event_data["out_count"],
            net_count=event_data["net_count"],
            timestamp=event_data["timestamp"]
        )


def main():
    args = get_args()

    # 1. Load Alerts & Notifications Configuration
    alerts_cfg = {}
    if os.path.exists(args.alerts_config):
        try:
            with open(args.alerts_config, "r", encoding="utf-8") as f:
                alerts_cfg = json.load(f)
            logger.info(f"Loaded alerts configuration from: {args.alerts_config}")
        except Exception as e:
            logger.warning(f"Failed to load alerts configuration: {e}")

    snapshot_dir = alerts_cfg.get("snapshots", {}).get("dir", args.snapshot_dir)

    # 2. Initialize Database & Asynchronous Notifier
    db = SurveillanceDatabase(db_path=args.db_path)
    notifier = NotificationManager(config=alerts_cfg, snapshot_dir=snapshot_dir)

    # 3. Initialize Anomaly & Alert Rule Engine
    rules_cfg = alerts_cfg.get("rules", {})
    alert_engine = AlertRuleEngine(
        db=db,
        notifier=notifier,
        camera_id=args.camera_name,
        loitering_threshold_sec=rules_cfg.get("loitering_threshold_sec", 15.0),
        surge_threshold=rules_cfg.get("surge_threshold", 12),
        default_cooldown_sec=rules_cfg.get("default_cooldown_sec", 5.0)
    )

    # 4. Initialize Event Logger
    event_logger = EventLogger(args.alert_log, args.event_log, args.camera_name, db)

    # 5. Initialize Zone Manager
    if os.path.exists(args.zones):
        logger.info(f"Loading zones from: {args.zones}")
        zone_mgr = ZoneManager.load_from_json(args.zones)
    else:
        logger.warning(f"Zones file '{args.zones}' not found. Running without zones.")
        zone_mgr = ZoneManager()

    # 6. Initialize Tripwire Manager
    if os.path.exists(args.tripwires):
        logger.info(f"Loading tripwires from: {args.tripwires}")
        tripwire_mgr = TripwireManager.load_from_json(args.tripwires)
    else:
        logger.warning(f"Tripwires file '{args.tripwires}' not found. Running without tripwires.")
        tripwire_mgr = TripwireManager()

    # Register tripwire event listener
    tripwire_mgr.register_callback(event_logger.log_tripwire_crossing)

    # 7. Initialize Multi-Object Point Tracker
    tracker = None
    if not args.no_tracking:
        logger.info(f"Initializing Kalman Point Tracker (max_dist={args.max_track_dist}px)")
        tracker = KalmanPointTracker(max_distance=args.max_track_dist)

    # 8. Initialize Deep Crowd Detector with Stage 4 Tensor Core Optimization
    detector = CrowdDetector(
        weight_path=args.weights,
        confidence_threshold=args.threshold,
        precision=args.precision,
        use_cuda_graph=args.cuda_graph,
    )

    # 9. Initialize Threaded Video Stream
    logger.info(f"Opening threaded camera stream from: {args.source}")
    stream = ThreadedCameraStream(
        source=args.source,
    ).start()

    # Allow stream to warm up
    time.sleep(0.5)

    # Video Writer
    writer = None
    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)

    window_name = f"Surveillance Feed - {args.camera_name}"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps = 0.0
    t_start = time.time()
    t_last_fps = time.time()
    fps_frame_count = 0

    cached_points = []
    cached_tracks = []
    cached_stats = {"total_scene_count": 0, "zones": {}}
    last_inference_ms = 0.0

    try:
        while True:
            ret, frame, frame_ts = stream.read()
            if not ret or frame is None:
                if stream.stopped:
                    logger.info("Video stream completed.")
                    break
                time.sleep(0.01)
                continue

            frame_count += 1
            fps_frame_count += 1

            # Compute FPS every 1 second
            now = time.time()
            if now - t_last_fps >= 1.0:
                fps = fps_frame_count / (now - t_last_fps)
                fps_frame_count = 0
                t_last_fps = now

            h, w = frame.shape[:2]

            # Lazy initialize video writer
            if args.output and writer is None:
                fourcc = (
                    cv2.VideoWriter_fourcc(*"mp4v")
                    if args.output.lower().endswith(".mp4")
                    else cv2.VideoWriter_fourcc(*"DIVX")
                )
                writer = cv2.VideoWriter(args.output, fourcc, 20.0, (w, h))
                logger.info(f"Video writer initialized ({w}x{h} @ 20 FPS) -> {args.output}")

            # 10. Deep Inference
            if frame_count % args.downsample == 0:
                points, scores, inference_ms = detector.detect(frame, threshold=args.threshold)
                last_inference_ms = inference_ms
                cached_points = points

                # 11. Tracking & Spatial Analytics
                if tracker is not None:
                    tracks = tracker.update(points, timestamp=frame_ts)
                    cached_tracks = tracks
                    # Tripwire line-crossing evaluation
                    crossing_events = tripwire_mgr.evaluate(tracks, w, h, timestamp=frame_ts)
                    for ev in crossing_events:
                        tw_obj = next((t for t in tripwire_mgr.tripwires if t.name == ev["tripwire"]), None)
                        if tw_obj:
                            alert_engine.evaluate_tripwire_event(frame, ev, tw_obj)

                    # Evaluate zones against tracked points
                    track_pts = [t.current_point for t in tracks]
                    stats = zone_mgr.evaluate(track_pts if track_pts else points, w, h, timestamp=frame_ts)
                else:
                    tracks = None
                    stats = zone_mgr.evaluate(points, w, h, timestamp=frame_ts)

                cached_stats = stats

                # 12. Evaluate Surveillance Alert Rules (Overcrowding, Loitering, Surge)
                alert_engine.evaluate_frame(
                    frame=frame,
                    zone_manager=zone_mgr,
                    tracks=tracks,
                    total_count=stats["total_scene_count"],
                    timestamp=frame_ts
                )

            else:
                points = cached_points
                tracks = cached_tracks
                stats = cached_stats
                inference_ms = last_inference_ms

            # 13. Render Modern Surveillance HUD
            display_frame = zone_mgr.render_overlay(
                frame=frame,
                points=points,
                stats=stats,
                fps=fps,
                inference_ms=inference_ms,
                camera_name=args.camera_name,
                tracks=tracks,
                tripwire_manager=tripwire_mgr,
                timestamp=frame_ts,
                show_trajectories=not args.no_trajectories,
                show_track_ids=True,
            )

            # 14. Save or Display
            if writer is not None:
                writer.write(display_frame)

            if args.show:
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("Termination signal received from GUI.")
                    break

            # 15. Periodic Telemetry Rollup (Every 30 frames)
            if frame_count % 30 == 0:
                zone_summary = ", ".join(
                    [f"{z}: {data['count']}/{data['max_capacity']}" for z, data in stats["zones"].items()]
                )
                tw_summary = tripwire_mgr.get_summary_stats()
                logger.info(
                    f"Frame {frame_count:04d} | Total: {stats['total_scene_count']:2d} | "
                    f"Zones: [{zone_summary}] | Tripwires: IN={tw_summary['total_in']} OUT={tw_summary['total_out']} | "
                    f"Latency: {inference_ms:.1f}ms | FPS: {fps:.1f}"
                )

                # Persist telemetry to SQLite database
                db.log_telemetry(
                    camera_id=args.camera_name,
                    total_count=stats["total_scene_count"],
                    fps=fps,
                    latency_ms=inference_ms,
                    zones_data={z: data["count"] for z, data in stats["zones"].items()},
                    timestamp=frame_ts
                )

            if args.max_frames > 0 and frame_count >= args.max_frames:
                logger.info(f"Reached maximum frame limit ({args.max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    finally:
        total_time = time.time() - t_start
        stream.stop()
        if writer is not None:
            writer.release()
            logger.info(f"Surveillance recording successfully saved to: {args.output}")
        if args.show:
            cv2.destroyAllWindows()

        # Shutdown notifier worker thread
        notifier.shutdown()

        # Database summary statistics
        summary = db.get_summary_statistics()
        tw_final = tripwire_mgr.get_summary_stats()

        logger.info(
            f"Surveillance session finished. Processed {frame_count} frames in {total_time:.2f}s "
            f"(Avg: {frame_count / max(total_time, 1e-5):.1f} FPS) | "
            f"Footfall: IN={tw_final['total_in']} OUT={tw_final['total_out']} (Net: {tw_final['net_flow']:+d}) | "
            f"DB Logged: {summary['total_alerts']} Alerts, {summary['total_crossings']} Crossings, Peak: {summary['peak_crowd']} persons."
        )


if __name__ == "__main__":
    main()
