# Crowd Counting and Localization for Surveillance Videos

An end-to-end crowd counting, individual localization, and real-time spatial surveillance intelligence system built on a point-based architecture.

## Preprocessing

Images and ground truth annotations can be preprocessed into standard training formats:

```bash
python dataset_preparation.py --application='make_list' \
    --list_dir $DATA_DIR \
    --groundtruth_dir $GT_DIR \
    --images_path $IMG_PATH \
    --groundtruth_txt_path $GT_PATH
```

## Train

To train and fine-tune the model:

```bash
python train.py --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --epochs 100 \
    --lr_drop 3500 \
    --output_dir './logs' \
    --checkpoints_dir './logs/checkpoint' \
    --tensorboard_dir './logs' \
    --frozen_weights './weights/best_mae.pth' \
    --lr 0.00005 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 4 \
    --gpu_id 0
```

* **`data_root`**: Training dataset directory
* **`output_dir`**: Directory to save training logs and model checkpoints
* **`checkpoints_dir`**: Address to save checkpoints
* **`frozen_weights`**: Path to initial weights (if fine-tuning specific heads)

## Test & Evaluation

Trained weights are available in the `./weights` directory. To evaluate individual localization on test images:

```bash
python test.py --threshold 0.5 \
    --images_path ./test_images \
    --weight_path ./weights/best_mae.pth \
    --output_dir ./prediction/images/ \
    --predicts_txt_dir ./prediction/predict_txt.txt \
    --predicts_point_dir ./prediction/points
```

* **`images_path`**: Test image folder path
* **`weight_path`**: Path to model weights (`./weights/best_mae.pth`)
* **`output_dir`**: Directory to save predicted annotated images
* **`predicts_txt_dir`**: Destination path for detection coordinates

## Visualization of Predictions

Sample localization predictions on crowd scenes:

<p align="center">
  <img src="images/img2.jpg" width="1000" >
</p>

<p align="center">
  <img src="images/img3.jpg" width="1000">
</p>

## Density Heatmap Generation

To generate density heatmaps from point predictions:

```bash
python density_map.py --images_path ./test_images \
    --points_path ./prediction/predict_txt.txt \
    --method 'fixed' \
    --output_dir ./heatmaps
```

Sample generated density heatmap:

<p align="center">
  <img src="images/heatmap.png" width="1700">
</p>

## Video Inference

To run offline point localization on a video file:

```bash
python video.py --threshold 0.5 \
    --video_path $VIDEO_PATH \
    --weight_path ./weights/best_mae.pth \
    --output_dir prediction/video.mp4
```

<p align="center">
  <img src="images/video_cover.png" width="800">
</p>

---

## 🚨 Real-Time Surveillance & Spatial Analytics (Stages 1, 2, 3, 4 & 5)

Transforming the research P2PNet into an enterprise surveillance & crowd intelligence platform:

### Key Surveillance Features
* **Zero-Latency Ingestion (Stage 1):** Multi-threaded stream reader (`ThreadedCameraStream`) supporting RTSP IP cameras, USB webcams, and local video with hardware buffer dropping to guarantee live synchronization.
* **Polygonal ROI Zones (Stage 1):** Define arbitrary monitoring polygons (e.g., Gate Entrance, Queue Corridor, Hazard Zone) with independent capacity limits.
* **Debounced Alert Engine (Stage 1):** Detects overcrowding and logs structured events with hysteresis protection against alert fluttering.
* **Multi-Object Kalman Point Tracking (Stage 2):** Associates individual head centroid detections across frames using a 2D constant-velocity Kalman filter and Hungarian linear sum assignment (`KalmanPointTracker`).
* **Motion Trajectory Trails (Stage 2):** Renders dynamic, fading motion trails displaying movement paths for every individual.
* **Virtual Tripwires & Bidirectional Footfall (Stage 2):** Virtual line segments that detect line-crossing direction (`IN` vs `OUT`) using 2D geometric vector cross-products, maintaining real-time occupancy counts (`NET = IN - OUT`).
* **Unified Spatial Calibrator (Stage 1 & 2):** Interactive GUI utility (`tools/zone_drawer.py`) to draw both polygonal zones and tripwire lines with point-and-click ease.
* **Incident Snapshot Capturer (Stage 3):** Automatically saves timestamped, annotated JPEG incident photos into `logs/snapshots/` whenever a violation occurs.
* **Multi-Rule Surveillance Engine (Stage 3):** Real-time anomaly detection for **Overcrowding**, **Loitering / Prolonged Dwell Time**, **Counterflow / Wrong-Way Entry**, and **Sudden Crowd Surges**.
* **Asynchronous Multi-Channel Dispatcher (Stage 3):** Non-blocking background worker dispatching alerts across **Webhooks** (Slack/Discord/REST), **Telegram Bot** (photos with HTML captions), and **Email (SMTP)**.
* **Persistent SQLite Surveillance Database (Stage 3):** Stores incidents, directional footfall events, and periodic telemetry (`logs/surveillance.db`) with Write-Ahead Logging (WAL).
* **Hardware Acceleration & Tensor Cores (Stage 4):** FP16 Tensor Core execution, TensorFloat-32 (TF32), cuDNN auto-tuning, GPU anchor caching, and CUDA Graph replay achieving **44+ FPS** (up from 24 FPS baseline).
* **Command Center Web Dashboard (Stage 5):** FastAPI web application + WebSocket real-time telemetry + low-latency MJPEG live stream + interactive dark-mode surveillance control room with Chart.js occupancy trends and snapshot evidence modal viewer (`web_app.py`).

### Performance Benchmark (RTX 3050 GPU)

| Configuration | Batch | Latency / Frame | Throughput | Speedup | VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch FP32 (Baseline)** | 1 | 41.12 ms | 24.3 FPS | 1.00x (Ref) | 266 MB |
| **Ampere FP16 (Tensor Cores)** | 1 | **22.85 ms** | **43.8 FPS** | **1.80x** | **170 MB** |
| **FP16 + CUDA Graph** | 1 | **22.55 ms** | **44.4 FPS** | **1.82x** | 758 MB |
| **Batched Multi-Cam (B=2)** | 2 | 22.47 ms | 44.5 FPS | 1.83x | 287 MB |
| **Batched Multi-Cam (B=4)** | 4 | 23.35 ms | 42.8 FPS | 1.76x | 531 MB |

---

### 1. Launching the Web Command Center (Stage 5)
To launch the browser-based Surveillance Command Center:
```bash
python web_app.py --source test_videos/mall_crowd_test.mp4 --port 8000
```
Open **`http://localhost:8000`** in any web browser to view:
* **Live Video Player:** Real-time AI HUD video stream with translucent zones, tripwires, and motion trails.
* **Telemetry Counters:** Global scene head count, bidirectional footfall (In/Out/Net), and engine FPS.
* **Zone Capacities:** Real-time percentage bars turning red during capacity breaches.
* **Live Trend Chart:** Rolling time-series crowd occupancy curve.
* **Evidence Gallery:** Clickable incident snapshot thumbnails with modal view.

### 2. Calibrating Zones and Tripwires
To draw custom monitoring zones and virtual tripwires on any camera feed:
```bash
python tools/zone_drawer.py --source test_videos/mall_crowd_test.mp4 --zones configs/zones_config.json --tripwires configs/tripwires_config.json
```
* **`t`:** Toggle between **ZONE** mode and **TRIPWIRE** mode
* **ZONE Mode:** Left-click vertices -> Press **`c`** to close polygon and set capacity threshold
* **TRIPWIRE Mode:** Left-click Start Point -> Left-click End Point to define crossing line
* **Right-click:** Undo last point
* **`s`:** Save all zones and tripwires to JSON

### 3. Configuring Notifications (Webhooks, Telegram, Email)
Edit `configs/alerts_config.json` to enable desired notification channels:
```json
{
  "webhook": {
    "enabled": true,
    "url": "https://discord.com/api/webhooks/..."
  },
  "telegram": {
    "enabled": true,
    "bot_token": "YOUR_BOT_TOKEN",
    "chat_id": "YOUR_CHAT_ID"
  }
}
```

### 4. Running CLI Surveillance Pipeline
To run headless surveillance without the web server:

```bash
# Optimized FP16 execution
python run_surveillance.py --source test_videos/mall_crowd_test.mp4 \
    --precision fp16 \
    --zones configs/zones_config.json \
    --tripwires configs/tripwires_config.json \
    --alerts-config configs/alerts_config.json \
    --output prediction/surveillance_output_stage4.mp4

# On an RTSP IP Camera
python run_surveillance.py --source "rtsp://username:password@192.168.1.100:554/stream1" \
    --camera-name "ENTRANCE-GATE" \
    --show
```




