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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
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

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads", "videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="DeepCrowd Surveillance Command Center")

# Registry of available P2PNet crowd models with empirically verified optimal headcount thresholds
AVAILABLE_MODELS = {
    "generalist": {
        "id": "generalist",
        "name": "Generalist Fallback",
        "weights": "weights/best_mae.pth",
        "badge": "General Baseline",
        "desc": "Cross-domain baseline trained on diverse environments",
        "optimal_threshold": 0.50,
    },
    "large_area": {
        "id": "large_area",
        "name": "Large-Area Specialist",
        "weights": "weights/p2pnet_large_area.pth",
        "badge": "Malls & Stations",
        "desc": "Wide FOV, CCTV surveillance angles & distributed crowds",
        "optimal_threshold": 0.50,
    },
    "confined": {
        "id": "confined",
        "name": "Confined-Space Specialist",
        "weights": "weights/p2pnet_confined.pth",
        "badge": "Lifts & Rooms",
        "desc": "Elevators, close overhead views & strict false-alarm rejection",
        "optimal_threshold": 0.45,  # Empirically verified for most accurate headcount (Min MAE: 2.30)
    },
    "supercrowd": {
        "id": "supercrowd",
        "name": "Supercrowd Specialist",
        "weights": "weights/p2pnet_supercrowd.pth",
        "badge": "Stadiums & Rallies",
        "desc": "Mass gatherings, tiny occluded heads & extreme crowd density",
        "optimal_threshold": 0.60,  # Empirically verified for most accurate headcount (Min MAE: 90.84, Unbiased)
    }
}


class ModelSelectRequest(BaseModel):
    model_id: str


class WebcamSelectRequest(BaseModel):
    camera_index: int = 0


class FeatureToggleRequest(BaseModel):
    feature: str
    enabled: bool


class ServerState:
    """Thread-safe state shared between the AI video loop and FastAPI endpoints."""
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_heatmap_jpeg: Optional[bytes] = None
        self.active_model_id: str = "generalist"
        self.model_switch_requested: Optional[str] = None
        self.active_source: str = ""
        self.active_source_name: str = "Default Feed"
        self.default_source: Union[str, int] = "samples/videos/mall_crowd_test.mp4"
        self.default_source_name: str = "mall_crowd_test.mp4"
        self.source_switch_requested: Optional[Union[str, int]] = None
        self.features: Dict[str, bool] = {
            "show_zones": True,
            "show_tracking": True,
            "show_trajectories": True,
            "show_tripwires": True,
            "show_points": True,
            "show_track_ids": True,
        }
        self.latest_telemetry: Dict[str, Any] = {
            "camera_name": "CAM-01-MALL",
            "active_source_name": "Default Feed",
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
    is_init_digit = str(args.source).isdigit() or isinstance(args.source, int)
    init_name = f"Live Webcam (Device {args.source})" if is_init_digit else (os.path.basename(str(args.source)) if os.path.isfile(str(args.source)) else str(args.source))
    
    with state.lock:
        state.active_model_id = initial_model
        state.active_source = str(args.source)
        state.active_source_name = init_name
        state.default_source = args.source
        state.default_source_name = init_name

    # Tracker & Initial Detector
    tracker = KalmanPointTracker(max_distance=args.max_track_dist)
    initial_cfg = AVAILABLE_MODELS.get(initial_model, {})
    init_thresh = args.threshold if args.threshold is not None else initial_cfg.get("optimal_threshold", 0.50)
    logger.info(f"Loading initial model [{initial_model}]: {args.weights} (Threshold: {init_thresh})")
    detector = CrowdDetector(
        weight_path=args.weights,
        confidence_threshold=init_thresh,
        precision=args.precision,
        use_cuda_graph=args.cuda_graph,
    )

    # Loop video if source is a file
    is_file = os.path.isfile(str(args.source)) if not is_init_digit else False
    
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
                # 0. Check for requested video source switch (file, webcam, or stream)
                with state.lock:
                    src_req = state.source_switch_requested
                    state.source_switch_requested = None

                if src_req is not None:
                    is_webcam = False
                    if isinstance(src_req, int):
                        is_webcam = True
                        actual_src = src_req
                    elif isinstance(src_req, str) and (src_req.isdigit() or src_req.lower() in ("webcam", "camera")):
                        is_webcam = True
                        actual_src = int(src_req) if src_req.isdigit() else 0
                    elif os.path.exists(str(src_req)):
                        actual_src = str(src_req)
                    else:
                        actual_src = None

                    if actual_src is not None:
                        logger.info(f"🔄 Switching video feed source -> {actual_src} (Webcam={is_webcam})")
                        args.source = actual_src
                        is_file = os.path.isfile(str(args.source)) if not is_webcam else False
                        with state.lock:
                            state.active_source = str(actual_src)
                            if is_webcam:
                                state.active_source_name = f"Live Webcam (Device {actual_src})"
                            else:
                                state.active_source_name = os.path.basename(str(actual_src))
                        tracker = KalmanPointTracker(max_distance=args.max_track_dist)
                        stream.stop()
                        break

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
                            # Apply empirically verified optimal headcount threshold per model
                            t_thresh = args.threshold if args.threshold is not None else new_cfg.get("optimal_threshold", 0.50)
                            detector = CrowdDetector(
                                weight_path=target_w,
                                confidence_threshold=t_thresh,
                                precision=args.precision,
                                use_cuda_graph=False,
                            )
                            with state.lock:
                                state.active_model_id = switch_req
                            logger.info(f"✅ Successfully switched to {new_cfg['name']} (Headcount-optimal threshold: {t_thresh})")
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
                )

                # 6. Generate Standalone Side-Panel Heatmap Matrix
                heatmap_img = ZoneManager.generate_density_heatmap(points, w, h, out_w=400, out_h=225)
                ret_hm, hm_bytes = cv2.imencode(".jpg", heatmap_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret_hm:
                    with state.lock:
                        state.latest_heatmap_jpeg = hm_bytes.tobytes()

                # 7. Compress Main Annotated Video to JPEG for Web Stream
                ret_enc, jpeg_bytes = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret_enc:
                    with state.lock:
                        state.latest_frame_jpeg = jpeg_bytes.tobytes()
                        state.latest_telemetry = {
                            "camera_name": args.camera_name,
                            "active_source_name": state.active_source_name,
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

        if not state.running:
            break

        if not is_file:
            time.sleep(0.5)

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


def heatmap_mjpeg_generator():
    """Generator yielding standalone Heatmap MJPEG multipart frames to the browser."""
    while state.running:
        with state.lock:
            frame_bytes = state.latest_heatmap_jpeg

        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        time.sleep(0.040)


@app.get("/heatmap_feed")
async def heatmap_feed():
    """Live MJPEG crowd density heatmap streaming route for side panel."""
    return StreamingResponse(
        heatmap_mjpeg_generator(),
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
            "optimal_threshold": info.get("optimal_threshold", 0.50),
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


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file (.mp4, .avi, .mkv, .mov) and switch live surveillance processing to it."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    valid_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported video formats: {', '.join(sorted(valid_exts))}"
        )

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    clean_base = os.path.basename(file.filename).replace(" ", "_")
    safe_name = f"upload_{timestamp_str}_{clean_base}"
    dest_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        with open(dest_path, "wb") as f_out:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                f_out.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # Validate video with OpenCV
    cap = cv2.VideoCapture(dest_path)
    if not cap.isOpened():
        cap.release()
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid video or codec is unsupported.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Request dynamic stream switch
    with state.lock:
        state.source_switch_requested = dest_path
        state.active_source_name = file.filename

    logger.info(f"✅ Video uploaded -> {dest_path} ({w}x{h}, {fps:.1f} FPS, {total_frames} frames). Switched live stream!")

    return {
        "status": "ok",
        "filename": file.filename,
        "saved_path": dest_path,
        "resolution": f"{w}x{h}",
        "fps": round(fps, 1),
        "total_frames": total_frames,
    }


@app.get("/api/source/webcams")
async def list_available_webcams():
    """Scan and list all accessible webcam devices (indices 0 through 4)."""
    cameras = []
    for idx in range(5):
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                cap.release()
                label = f"Camera {idx}"
                if idx == 0:
                    label += " (Integrated / Default)"
                elif idx == 1:
                    label += " (External USB Camera)"
                else:
                    label += f" (USB Device {idx})"
                cameras.append({
                    "index": idx,
                    "name": label,
                    "resolution": f"{w}x{h}",
                    "working": bool(ret)
                })
        except Exception:
            pass

    with state.lock:
        curr_src = state.active_source
        is_webcam_active = str(curr_src).isdigit() or isinstance(curr_src, int)
        active_idx = int(curr_src) if is_webcam_active else None

    # Recommend camera 1 if present (standard external webcam index), otherwise camera 0
    recommended_idx = 1 if any(c["index"] == 1 for c in cameras) else (cameras[0]["index"] if cameras else 0)

    return {
        "cameras": cameras,
        "active_index": active_idx,
        "default_recommended_index": recommended_idx
    }


@app.post("/api/source/webcam")
async def select_webcam(req: WebcamSelectRequest = WebcamSelectRequest()):
    """Switch live surveillance processing to selected webcam by device index."""
    cam_idx = req.camera_index
    # Test availability of webcam using DirectShow (Windows) or default backend
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(cam_idx)
    if not cap or not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    if not cap or not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail=f"Unable to access Camera Device {cam_idx}. Please verify the external webcam is plugged in and recognized in Windows Device Manager."
        )
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cap.release()

    cam_name = f"External Webcam (Camera {cam_idx})" if cam_idx > 0 else f"Camera {cam_idx} (Integrated)"

    with state.lock:
        state.source_switch_requested = cam_idx
        state.active_source_name = cam_name

    logger.info(f"📷 Switched surveillance feed to {cam_name} ({w}x{h})")
    return {
        "status": "ok",
        "source": f"webcam_{cam_idx}",
        "device_index": cam_idx,
        "name": cam_name,
        "resolution": f"{w}x{h}"
    }


@app.post("/api/source/default")
async def select_default_video():
    """Switch back to default video feed."""
    with state.lock:
        target_src = state.default_source
        target_name = state.default_source_name
        state.source_switch_requested = target_src
        state.active_source_name = target_name

    logger.info(f"⏮️ Reset surveillance feed to default: {target_src}")
    return {
        "status": "ok",
        "source": str(target_src),
        "source_name": target_name
    }


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
    parser.add_argument("--source", default="samples/videos/mall_crowd_test.mp4", help="Video source (file path, RTSP, or webcam index like 0, 1)")
    parser.add_argument("--webcam", nargs="?", const=1, default=None, type=int, help="Start directly using webcam index (e.g. --webcam 1 for external, or --webcam for camera 1)")
    parser.add_argument("--weights", default="weights/best_mae.pth", help="Model weights path")
    parser.add_argument("--zones", default="configs/zones_config.json", help="Zones JSON config")
    parser.add_argument("--tripwires", default="configs/tripwires_config.json", help="Tripwires JSON config")
    parser.add_argument("--alerts-config", default="configs/alerts_config.json", help="Alerts JSON config")
    parser.add_argument("--db-path", default="logs/surveillance.db", help="SQLite database path")
    parser.add_argument("--snapshot-dir", default="logs/snapshots", help="Snapshot images folder")
    parser.add_argument("--camera-name", default="CAM-01-MALL", help="Camera name")
    parser.add_argument("--threshold", default=None, type=float, help="Confidence threshold (default: auto-fixed per model: 0.45 confined, 0.60 supercrowd, 0.50 others)")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "amp", "fp32"], help="Precision")
    parser.add_argument("--cuda-graph", action="store_true", help="Enable CUDA Graph replay")
    parser.add_argument("--max-track-dist", default=65.0, type=float, help="Tracker max pixel distance")
    parser.add_argument("--downsample", default=1, type=int, help="Inference step downsample")
    parser.add_argument("--host", default="0.0.0.0", help="Web host")
    parser.add_argument("--port", default=8000, type=int, help="Web port")
    args = parser.parse_args()
    if args.webcam is not None:
        args.source = args.webcam
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
