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
from pydantic import BaseModel
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

# Registry of available P2PNet crowd models
AVAILABLE_MODELS = {
    "generalist": {
        "id": "generalist",
        "name": "Generalist Fallback",
        "weights": "weights/best_mae.pth",
        "badge": "General Baseline",
        "desc": "Cross-domain baseline trained on diverse environments"
    },
    "large_area": {
        "id": "large_area",
        "name": "Large-Area Specialist",
        "weights": "weights/p2pnet_large_area.pth",
        "badge": "Malls & Stations",
        "desc": "Wide FOV, CCTV surveillance angles & distributed crowds"
    },
    "confined": {
        "id": "confined",
        "name": "Confined-Space Specialist",
        "weights": "weights/p2pnet_confined.pth",
        "badge": "Lifts & Rooms",
        "desc": "Elevators, close overhead views & strict false-alarm rejection"
    },
    "supercrowd": {
        "id": "supercrowd",
        "name": "Supercrowd Specialist",
        "weights": "weights/p2pnet_supercrowd.pth",
        "badge": "Stadiums & Rallies",
        "desc": "Mass gatherings, tiny occluded heads & extreme crowd density"
    }
}


class ModelSelectRequest(BaseModel):
    model_id: str


class FeatureToggleRequest(BaseModel):
    feature: str
    enabled: bool


class ServerState:
    """Thread-safe state shared between the AI video loop and FastAPI endpoints."""
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame_jpeg: Optional[bytes] = None
        self.active_model_id: str = "generalist"
        self.model_switch_requested: Optional[str] = None
        self.features: Dict[str, bool] = {
            "show_zones": True,
            "show_tracking": True,
            "show_trajectories": True,
            "show_tripwires": True,
            "show_points": True,
            "show_track_ids": True,
            "show_heatmap": False,
        }
        self.latest_telemetry: Dict[str, Any] = {
            "camera_name": "CAM-01-MALL",
            "active_model": "generalist",
            "active_model_name": "Generalist Fallback",
            "total_scene_count": 0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "has_active_alert": False,
            "alerting_zones": [],
            "zones": {},
            "tripwires": {"total_in": 0, "total_out": 0, "net_flow": 0},
            "timestamp": time.time(),
            "features": dict(self.features),
        }
        self.active_websockets: List[WebSocket] = []
        self.running = True


state = ServerState()
global_args = None
db: Optional[SurveillanceDatabase] = None


def surveillance_worker(args):
    """
    Dedicated background worker thread running AI detection, tracking,
    zone/tripwire spatial analytics, dynamic model switching, and alert triggers.
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

    # Determine initial model ID based on args.weights
    initial_model = "generalist"
    for mid, minfo in AVAILABLE_MODELS.items():
        if os.path.abspath(minfo["weights"]) == os.path.abspath(args.weights):
            initial_model = mid
            break
    with state.lock:
        state.active_model_id = initial_model

    # Tracker & Initial Detector
    tracker = KalmanPointTracker(max_distance=args.max_track_dist)
    logger.info(f"Loading initial model [{initial_model}]: {args.weights}")
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
                # 1. Check for requested model switch
                with state.lock:
                    switch_req = state.model_switch_requested
                    state.model_switch_requested = None

                if switch_req and switch_req in AVAILABLE_MODELS:
                    new_cfg = AVAILABLE_MODELS[switch_req]
                    target_w = new_cfg["weights"]
                    if os.path.exists(target_w):
                        logger.info(f"🔄 Switching model -> {switch_req} ({target_w})...")
                        try:
                            # Confined space benefits from higher confidence threshold; Supercrowd from lower
                            t_thresh = 0.65 if switch_req == "confined" else (0.40 if switch_req == "supercrowd" else args.threshold)
                            detector = CrowdDetector(
                                weight_path=target_w,
                                confidence_threshold=t_thresh,
                                precision=args.precision,
                                use_cuda_graph=False,
                            )
                            with state.lock:
                                state.active_model_id = switch_req
                            logger.info(f"✅ Successfully switched to {new_cfg['name']}")
                        except Exception as e:
                            logger.error(f"❌ Failed to switch model to {switch_req}: {e}")

                # 2. Snapshot current feature toggles and active model
                with state.lock:
                    feat = dict(state.features)
                    curr_mid = state.active_model_id

                show_tracking = feat.get("show_tracking", True)
                show_zones = feat.get("show_zones", True)
                show_trajectories = feat.get("show_trajectories", True)
                show_tripwires = feat.get("show_tripwires", True)
                show_points = feat.get("show_points", True)
                show_track_ids = feat.get("show_track_ids", True)
                show_heatmap = feat.get("show_heatmap", False)

                # 3. Read camera frame
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

                # 4. Deep Inference
                if frame_idx % args.downsample == 0:
                    points, scores, inference_ms = detector.detect(frame, threshold=detector.confidence_threshold)
                    last_latency = inference_ms
                    cached_points = points

                    # Tracking (if enabled)
                    if show_tracking:
                        tracks = tracker.update(points, timestamp=frame_ts)
                        cached_tracks = tracks
                    else:
                        tracks = []
                        cached_tracks = []

                    # Tripwires (evaluated if tripwires enabled)
                    if show_tripwires and tracks:
                        crossing_events = tripwire_mgr.evaluate(tracks, w, h, timestamp=frame_ts)
                        for ev in crossing_events:
                            tw_obj = next((t for t in tripwire_mgr.tripwires if t.name == ev["tripwire"]), None)
                            if tw_obj:
                                alert_engine.evaluate_tripwire_event(frame, ev, tw_obj)

                    track_pts = [t.current_point for t in tracks] if show_tracking else points
                    stats = zone_mgr.evaluate(track_pts if track_pts else points, w, h, timestamp=frame_ts)
                    cached_stats = stats

                    # Alert Engine
                    alert_engine.evaluate_frame(
                        frame=frame,
                        zone_manager=zone_mgr,
                        tracks=tracks if show_tracking else [],
                        total_count=stats["total_scene_count"],
                        timestamp=frame_ts
                    )
                else:
                    points = cached_points
                    tracks = cached_tracks
                    stats = cached_stats
                    inference_ms = last_latency

                # 5. Render HUD with active feature toggles
                annotated = zone_mgr.render_overlay(
                    frame=frame,
                    points=points,
                    stats=stats,
                    fps=fps,
                    inference_ms=inference_ms,
                    camera_name=args.camera_name,
                    tracks=tracks if show_tracking else None,
                    tripwire_manager=tripwire_mgr if show_tripwires else None,
                    timestamp=frame_ts,
                    show_trajectories=show_trajectories and show_tracking,
                    show_track_ids=show_track_ids and show_tracking,
                    show_zones=show_zones,
                    show_points=show_points,
                    show_heatmap=show_heatmap,
                )

                # 6. Compress to JPEG for Web Stream
                ret_enc, jpeg_bytes = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret_enc:
                    with state.lock:
                        state.latest_frame_jpeg = jpeg_bytes.tobytes()
                        state.latest_telemetry = {
                            "camera_name": args.camera_name,
                            "active_model": curr_mid,
                            "active_model_name": AVAILABLE_MODELS.get(curr_mid, {}).get("name", "Unknown"),
                            "total_scene_count": stats.get("total_scene_count", 0),
                            "fps": round(fps, 1),
                            "latency_ms": round(inference_ms, 1),
                            "has_active_alert": stats.get("has_active_alert", False),
                            "alerting_zones": stats.get("alerting_zones", []),
                            "zones": stats.get("zones", {}),
                            "tripwires": tripwire_mgr.get_summary_stats() if show_tripwires else {"total_in": 0, "total_out": 0, "net_flow": 0},
                            "timestamp": frame_ts,
                            "features": feat,
                        }

                # 7. Periodic Telemetry to DB
                if frame_idx % 30 == 0:
                    db.log_telemetry(
                        camera_id=args.camera_name,
                        total_count=stats["total_scene_count"],
                        fps=fps,
                        latency_ms=inference_ms,
                        zones_data={z: data["count"] for z, data in stats["zones"].items()},
                        timestamp=frame_ts
                    )

                if is_file:
                    time.sleep(0.015)

        finally:
            stream.stop()

        if not is_file:
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
        time.sleep(0.035)


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


@app.get("/api/models")
async def get_models():
    """Return available models and current active model."""
    with state.lock:
        curr_id = state.active_model_id

    models_list = []
    for mid, info in AVAILABLE_MODELS.items():
        exists = os.path.exists(info["weights"])
        models_list.append({
            "id": mid,
            "name": info["name"],
            "badge": info["badge"],
            "desc": info["desc"],
            "weights": info["weights"],
            "active": (mid == curr_id),
            "available": exists,
        })
    return {"current_model": curr_id, "models": models_list}


@app.post("/api/model/select")
async def select_model(req: ModelSelectRequest):
    """Dynamically switch the active AI crowd model."""
    if req.model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model ID: {req.model_id}")
    weights_path = AVAILABLE_MODELS[req.model_id]["weights"]
    if not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail=f"Model weights not found at: {weights_path}")

    with state.lock:
        state.model_switch_requested = req.model_id
    logger.info(f"Model switch requested -> {req.model_id} ({weights_path})")
    return {"status": "ok", "requested_model": req.model_id}


@app.get("/api/features")
async def get_features():
    """Return current feature toggles."""
    with state.lock:
        return dict(state.features)


@app.post("/api/features/toggle")
async def toggle_feature(req: FeatureToggleRequest):
    """Toggle a visualization or processing feature."""
    with state.lock:
        if req.feature in state.features:
            state.features[req.feature] = req.enabled
            logger.info(f"Feature toggled: {req.feature} -> {req.enabled}")
            return {"status": "ok", "feature": req.feature, "enabled": req.enabled, "features": state.features}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown feature toggle: {req.feature}")


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
    base_dir = os.path.abspath(snapshot_dir)
    file_path = os.path.abspath(os.path.join(base_dir, filename))
    if not file_path.startswith(base_dir) or not os.path.isfile(file_path):
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
    parser.add_argument("--source", default="samples/videos/mall_crowd_test.mp4", help="Video source (file, RTSP, webcam)")
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
