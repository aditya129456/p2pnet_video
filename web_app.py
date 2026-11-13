import argparse
import asyncio
import json
import logging
import os
import sys
import time
import threading
from typing import List, Dict, Any, Optional
import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from surveillance.detector import CrowdDetector
from surveillance.stream import ThreadedCameraStream
from surveillance.zones import ZoneManager
from surveillance.tracker import KalmanPointTracker
from surveillance.tripwire import TripwireManager
from surveillance.database import SurveillanceDatabase
from surveillance.notifier import NotificationManager
from surveillance.alerts import AlertRuleEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SurveillanceWeb")

app = FastAPI(title="DeepCrowd Surveillance Command Center")


class ServerState:
    """Thread-safe state shared between the AI video loop and FastAPI endpoints."""
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_telemetry: Dict[str, Any] = {
            "total_scene_count": 0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "has_active_alert": False,
            "alerting_zones": [],
            "zones": {},
            "tripwires": {"total_in": 0, "total_out": 0, "net_flow": 0},
            "timestamp": time.time(),
        }
        self.active_websockets: List[WebSocket] = []
        self.running = True


state = ServerState()
global_args = None
db: Optional[SurveillanceDatabase] = None


def surveillance_worker(args):
    """
    Dedicated background worker thread running AI detection, tracking,
    zone/tripwire spatial analytics, and alert triggers.
    """
    global state, db

    logger.info("Initializing Surveillance Engine for Web Command Center...")

    # Load Database & Notifications
    db = SurveillanceDatabase(db_path=args.db_path)
    alerts_cfg = {}
    if os.path.exists(args.alerts_config):
        try:
            with open(args.alerts_config, "r", encoding="utf-8") as f:
                alerts_cfg = json.load(f)
        except Exception:
            pass

    notifier = NotificationManager(
        config=alerts_cfg,
        snapshot_dir=args.snapshot_dir
    )

    rules_cfg = alerts_cfg.get("rules", {})
    alert_engine = AlertRuleEngine(
        db=db,
        notifier=notifier,
        camera_id=args.camera_name,
        loitering_threshold_sec=rules_cfg.get("loitering_threshold_sec", 15.0),
        surge_threshold=rules_cfg.get("surge_threshold", 12),
        default_cooldown_sec=rules_cfg.get("default_cooldown_sec", 5.0)
    )

    # Zones & Tripwires
    zone_mgr = ZoneManager.load_from_json(args.zones) if os.path.exists(args.zones) else ZoneManager()
    tripwire_mgr = TripwireManager.load_from_json(args.tripwires) if os.path.exists(args.tripwires) else TripwireManager()

    def on_tripwire_cross(event_data):
        db.log_footfall(
            camera_id=args.camera_name,
            tripwire_name=event_data["tripwire"],
            direction=event_data["direction"],
            track_id=event_data["track_id"],
            in_count=event_data["in_count"],
            out_count=event_data["out_count"],
            net_count=event_data["net_count"],
            timestamp=event_data["timestamp"]
        )
    tripwire_mgr.register_callback(on_tripwire_cross)

    # Tracker & Detector
    tracker = KalmanPointTracker(max_distance=args.max_track_dist)
    detector = CrowdDetector(
        weight_path=args.weights,
        confidence_threshold=args.threshold,
        precision=args.precision,
        use_cuda_graph=args.cuda_graph,
    )

    # Loop video if source is a file
    is_file = os.path.isfile(args.source)
    
    while state.running:
        stream = ThreadedCameraStream(source=args.source).start()
        time.sleep(0.4)

        fps = 0.0
        fps_count = 0
        t_fps = time.time()
        frame_idx = 0

        cached_points = []
        cached_tracks = []
        cached_stats = {"total_scene_count": 0, "zones": {}}
        last_latency = 0.0

        try:
            while state.running:
                ret, frame, frame_ts = stream.read()
                if not ret or frame is None:
                    if stream.stopped:
                        break
                    time.sleep(0.01)
                    continue

                frame_idx += 1
                fps_count += 1
                now = time.time()
                if now - t_fps >= 1.0:
                    fps = fps_count / (now - t_fps)
                    fps_count = 0
                    t_fps = now

                h, w = frame.shape[:2]

                # Inference
                if frame_idx % args.downsample == 0:
                    points, scores, inference_ms = detector.detect(frame, threshold=args.threshold)
                    last_latency = inference_ms
                    cached_points = points

                    tracks = tracker.update(points, timestamp=frame_ts)
                    cached_tracks = tracks

                    # Tripwires
                    crossing_events = tripwire_mgr.evaluate(tracks, w, h, timestamp=frame_ts)
                    for ev in crossing_events:
                        tw_obj = next((t for t in tripwire_mgr.tripwires if t.name == ev["tripwire"]), None)
                        if tw_obj:
                            alert_engine.evaluate_tripwire_event(frame, ev, tw_obj)

                    track_pts = [t.current_point for t in tracks]
                    stats = zone_mgr.evaluate(track_pts if track_pts else points, w, h, timestamp=frame_ts)
                    cached_stats = stats

                    # Alert Engine
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
                    inference_ms = last_latency

                # Render HUD
                annotated = zone_mgr.render_overlay(
                    frame=frame,
                    points=points,
                    stats=stats,
                    fps=fps,
                    inference_ms=inference_ms,
                    camera_name=args.camera_name,
                    tracks=tracks,
                    tripwire_manager=tripwire_mgr,
                    timestamp=frame_ts,
                    show_trajectories=True,
                    show_track_ids=True
                )

                # Compress to JPEG for Web Stream (quality 80 for low bandwidth)
                ret_enc, jpeg_bytes = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret_enc:
                    with state.lock:
                        state.latest_frame_jpeg = jpeg_bytes.tobytes()
                        state.latest_telemetry = {
                            "camera_name": args.camera_name,
                            "total_scene_count": stats.get("total_scene_count", 0),
                            "fps": round(fps, 1),
                            "latency_ms": round(inference_ms, 1),
                            "has_active_alert": stats.get("has_active_alert", False),
                            "alerting_zones": stats.get("alerting_zones", []),
                            "zones": stats.get("zones", {}),
                            "tripwires": tripwire_mgr.get_summary_stats(),
                            "timestamp": frame_ts,
                        }

                # Periodic Telemetry to DB
                if frame_idx % 30 == 0:
                    db.log_telemetry(
                        camera_id=args.camera_name,
                        total_count=stats["total_scene_count"],
                        fps=fps,
                        latency_ms=inference_ms,
                        zones_data={z: data["count"] for z, data in stats["zones"].items()},
                        timestamp=frame_ts
                    )

                # Sleep slightly to match target FPS if offline file
                if is_file:
                    time.sleep(0.015)

        finally:
            stream.stop()

        if not is_file:
            # If live stream ends without file loop, stop
            break

    notifier.shutdown()
    logger.info("Surveillance worker terminated.")


# --------------------------------------------------------------------------
# FastAPI Endpoints
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the Command Center Web UI."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


def mjpeg_generator():
    """Generator yielding MJPEG multipart frames to the browser."""
    while state.running:
        with state.lock:
            frame_bytes = state.latest_frame_jpeg

        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(0.035)  # ~28 FPS cap for video feed


@app.get("/video_feed")
async def video_feed():
    """Live MJPEG video streaming route."""
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stats")
async def get_stats():
    """Get current real-time telemetry."""
    with state.lock:
        return state.latest_telemetry


@app.get("/api/alerts")
async def get_alerts(limit: int = 20):
    """Fetch recent incident alerts from SQLite."""
    global db
    if db is None:
        return []
    return db.get_recent_alerts(limit=limit)


@app.get("/api/snapshots/{filename}")
async def get_snapshot(filename: str):
    """Serve an incident evidence snapshot."""
    global global_args
    snapshot_dir = global_args.snapshot_dir if global_args else "logs/snapshots"
    file_path = os.path.join(snapshot_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(file_path, media_type="image/jpeg")


@app.get("/api/summary")
async def get_summary():
    """Get aggregate statistics."""
    global db
    if db is None:
        return {}
    return db.get_summary_statistics()


@app.websocket("/ws")
async def websocket_telemetry(websocket: WebSocket):
    """Real-time WebSocket endpoint pushing telemetry every 250ms."""
    await websocket.accept()
    state.active_websockets.append(websocket)
    try:
        while state.running:
            with state.lock:
                telemetry = state.latest_telemetry
            await websocket.send_text(json.dumps(telemetry))
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.active_websockets:
            state.active_websockets.remove(websocket)


def main():
    global global_args
    parser = argparse.ArgumentParser(description="DeepCrowd Surveillance Command Center Web Server")
    parser.add_argument("--source", default="test_videos/mall_crowd_test.mp4", help="Video source (file, RTSP, webcam)")
    parser.add_argument("--weights", default="weights/best_mae.pth", help="Model weights path")
    parser.add_argument("--zones", default="configs/zones_config.json", help="Zones JSON config")
    parser.add_argument("--tripwires", default="configs/tripwires_config.json", help="Tripwires JSON config")
    parser.add_argument("--alerts-config", default="configs/alerts_config.json", help="Alerts JSON config")
    parser.add_argument("--db-path", default="logs/surveillance.db", help="SQLite database path")
    parser.add_argument("--snapshot-dir", default="logs/snapshots", help="Snapshot images folder")
    parser.add_argument("--camera-name", default="CAM-01-MALL", help="Camera name")
    parser.add_argument("--threshold", default=0.5, type=float, help="Confidence threshold")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "amp", "fp32"], help="Precision")
    parser.add_argument("--cuda-graph", action="store_true", help="Enable CUDA Graph replay")
    parser.add_argument("--max-track-dist", default=65.0, type=float, help="Tracker max pixel distance")
    parser.add_argument("--downsample", default=1, type=int, help="Inference step downsample")
    parser.add_argument("--host", default="0.0.0.0", help="Web host")
    parser.add_argument("--port", default=8000, type=int, help="Web port")
    args = parser.parse_args()
    global_args = args

    # Start the background AI surveillance loop thread
    ai_thread = threading.Thread(target=surveillance_worker, args=(args,), daemon=True)
    ai_thread.start()

    logger.info("=" * 65)
    logger.info(f"🚀 Surveillance Command Center starting on http://localhost:{args.port}")
    logger.info("=" * 65)

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        state.running = False
        ai_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
