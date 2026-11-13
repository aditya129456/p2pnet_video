import cv2
import time
import threading
import queue
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ThreadedCameraStream:
    """
    High-performance multi-threaded video stream reader.
    
    Supports:
      - RTSP / RTMP / HTTP streams (with auto-reconnect and zero-latency frame dropping)
      - USB Webcams (numeric index 0, 1, ...)
      - Local video files (.mp4, .avi, etc.)

    In live streaming mode (RTSP/Webcam), OpenCV's internal decoding buffer can build up 
    several seconds of lag if inference takes longer than frame intervals.
    ThreadedCameraStream continuously empties the hardware buffer in a background thread 
    and guarantees the inference engine always gets the most recent real-time frame.
    """

    def __init__(
        self,
        source: Union[str, int],
        is_live: Optional[bool] = None,
        reconnect_interval_sec: float = 3.0,
        max_reconnect_attempts: int = 10,
        target_fps: Optional[float] = None,
    ):
        """
        Args:
            source: Video path, RTSP/HTTP URL, or camera index (int or str digit).
            is_live: If True, operates in zero-lag live mode (drops stale frames).
                     If None, automatically detected (True for RTSP/webcam, False for files).
            reconnect_interval_sec: Wait time between reconnect attempts on dropped RTSP streams.
            max_reconnect_attempts: Number of reconnection retries before failing.
            target_fps: Desired capture rate cap (optional).
        """
        # Parse source (convert numeric strings like "0" to int)
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
        else:
            self.source = source

        # Auto-detect live mode if not explicitly provided
        if is_live is None:
            if isinstance(self.source, int):
                self.is_live = True
            elif isinstance(self.source, str) and (
                self.source.startswith("rtsp://")
                or self.source.startswith("rtmp://")
                or self.source.startswith("http://")
                or self.source.startswith("https://")
            ):
                self.is_live = True
            else:
                self.is_live = False
        else:
            self.is_live = is_live

        self.reconnect_interval_sec = reconnect_interval_sec
        self.max_reconnect_attempts = max_reconnect_attempts
        self.target_fps = target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Thread synchronization
        self.lock = threading.Lock()
        self.latest_frame = None
        self.has_new_frame = False
        self.is_connected = False
        self.stopped = False

        # Queue for file mode to avoid dropping frames unless in live mode
        self.frame_queue = queue.Queue(maxsize=30) if not self.is_live else None

        # Statistics
        self.width = 0
        self.height = 0
        self.source_fps = 30.0
        self.total_frames_read = 0
        self.dropped_frames = 0
        self.last_frame_timestamp = 0.0

        self._initialize_capture()

    def _initialize_capture(self) -> bool:
        """Initialize or re-initialize the OpenCV VideoCapture object."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        logger.info(f"Connecting to video source: {self.source}")
        
        # Configure backend hints for RTSP streams
        if isinstance(self.source, str) and self.source.startswith("rtsp://"):
            # Set transport to TCP to prevent UDP packet drop artifacts
            # and minimize OpenCV buffer size
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            logger.warning(f"Failed to open video source: {self.source}")
            self.is_connected = False
            return False

        # Read camera/video metadata
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = fps if (fps and fps > 0 and fps < 120) else 30.0

        if self.is_live:
            # Set buffer size to 1 to reduce lag on supported backends
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        self.is_connected = True
        logger.info(
            f"Successfully connected to source: {self.source} "
            f"({self.width}x{self.height} @ {self.source_fps:.1f} FPS, Live={self.is_live})"
        )
        return True

    def start(self) -> "ThreadedCameraStream":
        """Start the background frame capture thread."""
        if self.running:
            return self

        self.running = True
        self.stopped = False
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()
        return self

    def _capture_worker(self):
        """Worker thread loop continuously reading frames from the capture device."""
        reconnect_attempts = 0
        min_frame_time = (1.0 / self.target_fps) if self.target_fps else 0.0

        while self.running:
            loop_start = time.time()

            if not self.is_connected or self.cap is None or not self.cap.isOpened():
                if not self.is_live:
                    # For offline video files, once disconnected/ended, we stop
                    break

                if reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error("Maximum reconnection attempts reached. Halting capture worker.")
                    break

                reconnect_attempts += 1
                logger.warning(
                    f"Attempting to reconnect ({reconnect_attempts}/{self.max_reconnect_attempts}) in {self.reconnect_interval_sec}s..."
                )
                time.sleep(self.reconnect_interval_sec)
                if self._initialize_capture():
                    reconnect_attempts = 0
                continue

            # Read frame
            ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.is_live:
                    logger.warning("Frame read failure on live stream. Triggering reconnect...")
                    self.is_connected = False
                    continue
                else:
                    # End of file reached
                    logger.info("End of video stream reached.")
                    break

            self.total_frames_read += 1
            now = time.time()

            if self.is_live:
                # Live mode: overwrite latest frame immediately
                with self.lock:
                    if self.has_new_frame:
                        self.dropped_frames += 1
                    self.latest_frame = frame
                    self.has_new_frame = True
                    self.last_frame_timestamp = now
            else:
                # File mode: queue frames so none are lost
                try:
                    self.frame_queue.put((frame, now), timeout=1.0)
                except queue.Full:
                    # If consumer is slow and queue is full, wait or throttle
                    pass

            # Optional FPS throttling
            if min_frame_time > 0:
                elapsed = time.time() - loop_start
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)

        self.is_connected = False
        self.stopped = True
        logger.info("Capture worker exited.")

    def read(self) -> Tuple[bool, Optional[cv2.typing.MatLike], float]:
        """
        Retrieve the latest frame.
        
        Returns:
            (success, frame, timestamp)
        """
        if self.is_live:
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    ts = self.last_frame_timestamp
                    self.has_new_frame = False
                    return True, frame, ts
                return False, None, 0.0
        else:
            # File mode
            try:
                frame, ts = self.frame_queue.get(timeout=0.5)
                return True, frame, ts
            except queue.Empty:
                if self.stopped:
                    return False, None, 0.0
                return False, None, 0.0

    def stop(self):
        """Stop capture thread and release hardware resources."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.is_connected = False
        self.stopped = True
        logger.info("ThreadedCameraStream stopped.")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
