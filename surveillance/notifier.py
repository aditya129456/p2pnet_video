import os
import time
import json
import base64
import queue
import logging
import threading
import urllib.request
import urllib.parse
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
from typing import Dict, Any, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Asynchronous, non-blocking notification dispatch engine.
    Dispatches alerts with incident snapshots across Webhooks, Telegram, and Email.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        snapshot_dir: str = "logs/snapshots"
    ):
        self.config = config or {}
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.queue = queue.Queue(maxsize=100)
        self.running = True
        self.worker_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.worker_thread.start()

    def capture_snapshot(
        self,
        frame: np.ndarray,
        camera_id: str,
        event_type: str,
        location_name: str
    ) -> str:
        """Save an annotated incident snapshot image to disk."""
        safe_loc = "".join(c if c.isalnum() else "_" for c in location_name)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"{camera_id}_{event_type}_{safe_loc}_{ts_str}.jpg"
        filepath = os.path.join(self.snapshot_dir, filename)

        # Draw red warning stamp on top-left of the snapshot
        snapshot = frame.copy()
        cv2.rectangle(snapshot, (10, 10), (380, 50), (0, 0, 200), -1)
        stamp_text = f"INCIDENT: {event_type} | {location_name}"
        cv2.putText(
            snapshot, stamp_text, (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

        cv2.imwrite(filepath, snapshot, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return filepath

    def dispatch(
        self,
        camera_id: str,
        event_type: str,
        location_name: str,
        detected_value: int,
        threshold_value: int,
        snapshot_path: Optional[str] = None,
        details: Optional[str] = None,
        timestamp: Optional[float] = None
    ):
        """Enqueue alert for non-blocking asynchronous dispatch."""
        payload = {
            "timestamp": datetime.fromtimestamp(timestamp or time.time()).isoformat(),
            "camera_id": camera_id,
            "event_type": event_type,
            "location_name": location_name,
            "detected_value": detected_value,
            "threshold_value": threshold_value,
            "excess": detected_value - threshold_value if threshold_value > 0 else 0,
            "snapshot_path": snapshot_path or "",
            "details": details or "",
        }

        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            logger.warning("Notification queue full. Dropping alert payload.")

    def _dispatch_loop(self):
        """Background thread consuming alert items from queue."""
        while self.running:
            try:
                alert = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._send_webhook(alert)
                self._send_telegram(alert)
                self._send_email(alert)
            except Exception as e:
                logger.error(f"Error dispatching notification: {e}")
            finally:
                self.queue.task_done()

    def _send_webhook(self, alert: Dict[str, Any]):
        """Send JSON POST request to configured Webhook (e.g. Slack/Discord/SOC API)."""
        webhook_cfg = self.config.get("webhook", {})
        if not webhook_cfg.get("enabled", False) or not webhook_cfg.get("url"):
            return

        url = webhook_cfg["url"]
        try:
            headers = {"Content-Type": "application/json"}
            body = {
                "text": (
                    f"🚨 *SURVEILLANCE ALERT* [{alert['event_type']}]\n"
                    f"• Camera: `{alert['camera_id']}`\n"
                    f"• Location: *{alert['location_name']}*\n"
                    f"• Count: *{alert['detected_value']}* (Threshold: {alert['threshold_value']})\n"
                    f"• Time: {alert['timestamp']}"
                ),
                "alert_data": alert
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.info(f"Webhook notification dispatched (HTTP {resp.status})")
        except Exception as e:
            logger.warning(f"Webhook dispatch failed: {e}")

    def _send_telegram(self, alert: Dict[str, Any]):
        """Send alert notification with photo attachment to a Telegram channel or bot."""
        tg_cfg = self.config.get("telegram", {})
        if not tg_cfg.get("enabled", False) or not tg_cfg.get("bot_token") or not tg_cfg.get("chat_id"):
            return

        token = tg_cfg["bot_token"]
        chat_id = tg_cfg["chat_id"]
        caption = (
            f"🚨 <b>SURVEILLANCE ALERT: {alert['event_type']}</b>\n"
            f"📹 <b>Camera:</b> {alert['camera_id']}\n"
            f"📍 <b>Location:</b> {alert['location_name']}\n"
            f"👥 <b>Detected:</b> {alert['detected_value']} (Max: {alert['threshold_value']})\n"
            f"⏰ <b>Timestamp:</b> {alert['timestamp']}"
        )

        snapshot_path = alert.get("snapshot_path")
        if snapshot_path and os.path.exists(snapshot_path):
            # Send Photo with Caption (multipart/form-data)
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            try:
                boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
                body = bytearray()

                # chat_id field
                body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{chat_id}\r\n".encode())
                # caption field
                body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode())
                # parse_mode
                body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"parse_mode\"\r\n\r\nHTML\r\n".encode())

                # photo file
                with open(snapshot_path, "rb") as f:
                    file_bytes = f.read()
                filename = os.path.basename(snapshot_path)
                body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"{filename}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode())
                body.extend(file_bytes)
                body.extend(f"\r\n--{boundary}--\r\n".encode())

                req = urllib.request.Request(
                    url,
                    data=bytes(body),
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    logger.info(f"Telegram photo alert sent successfully (HTTP {resp.status})")
            except Exception as e:
                logger.warning(f"Telegram photo alert failed: {e}")
        else:
            # Fallback to Text Message
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            try:
                data = urllib.parse.urlencode({
                    "chat_id": chat_id,
                    "text": caption,
                    "parse_mode": "HTML"
                }).encode("utf-8")
                req = urllib.request.Request(url, data=data, method="POST")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    logger.info(f"Telegram text alert sent (HTTP {resp.status})")
            except Exception as e:
                logger.warning(f"Telegram text alert failed: {e}")

    def _send_email(self, alert: Dict[str, Any]):
        """Send alert via SMTP email with snapshot attachment."""
        email_cfg = self.config.get("email", {})
        if not email_cfg.get("enabled", False) or not email_cfg.get("smtp_server"):
            return

        smtp_server = email_cfg["smtp_server"]
        smtp_port = email_cfg.get("smtp_port", 587)
        sender = email_cfg.get("sender")
        password = email_cfg.get("password")
        recipients = email_cfg.get("recipients", [])

        if not sender or not recipients:
            return

        msg = MIMEMultipart()
        msg["Subject"] = f"🚨 ALERT: {alert['event_type']} on {alert['camera_id']} ({alert['location_name']})"
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)

        body = (
            f"Surveillance Incident Alert Triggered:\n\n"
            f"Event Type: {alert['event_type']}\n"
            f"Camera: {alert['camera_id']}\n"
            f"Location: {alert['location_name']}\n"
            f"Detected Value: {alert['detected_value']}\n"
            f"Threshold Value: {alert['threshold_value']}\n"
            f"Timestamp: {alert['timestamp']}\n\n"
            f"Annotated incident snapshot attached."
        )
        msg.attach(MIMEText(body, "plain"))

        snapshot_path = alert.get("snapshot_path")
        if snapshot_path and os.path.exists(snapshot_path):
            try:
                with open(snapshot_path, "rb") as f:
                    img_data = f.read()
                image = MIMEImage(img_data, name=os.path.basename(snapshot_path))
                msg.attach(image)
            except Exception as e:
                logger.warning(f"Failed to attach image to email: {e}")

        try:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=8) as server:
                server.starttls()
                if password:
                    server.login(sender, password)
                server.send_message(msg)
                logger.info("Email alert dispatched successfully.")
        except Exception as e:
            logger.warning(f"Email alert failed: {e}")

    def shutdown(self):
        """Flush queue and cleanly stop background dispatcher."""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        logger.info("NotificationManager shutdown complete.")
