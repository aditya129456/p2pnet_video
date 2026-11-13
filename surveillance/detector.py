import time
import types
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image
import cv2

from models import build_model


class CrowdDetector:
    """
    Optimized Deep Inference Engine for P2PNet Crowd Counting & Localization.
    
    Supports:
      - NVIDIA Ampere Tensor Core FP16 execution (2x+ speedup)
      - TensorFloat-32 (TF32) and cuDNN auto-tuner benchmarks
      - CUDA Graph execution for zero CPU launch overhead
      - Batched multi-camera stream inference
    """

    def __init__(
        self,
        weight_path: str = "./pretrained_model/best_mae.pth",
        confidence_threshold: float = 0.5,
        backbone: str = "vgg16_bn",
        row: int = 2,
        line: int = 2,
        precision: str = "fp16",  # "fp16", "amp", or "fp32"
        use_cuda_graph: bool = False,
        device: Optional[str] = None,
        max_dimension: int = 2000,
    ):
        """
        Args:
            weight_path: Path to PyTorch model weights checkpoint.
            confidence_threshold: Confidence cutoff for head detections.
            backbone: Backbone name (default: vgg16_bn).
            row: Row number of anchor points.
            line: Line number of anchor points.
            precision: 'fp16' (fastest on Tensor Cores), 'amp', or 'fp32'.
            use_cuda_graph: Whether to capture and replay a CUDA Graph.
            device: 'cuda', 'cuda:0', 'cpu', or auto-detected if None.
            max_dimension: Maximum dimension cap before scaling down.
        """
        self.weight_path = weight_path
        self.confidence_threshold = confidence_threshold
        self.max_dimension = max_dimension
        self.precision = precision.lower()
        self.use_cuda_graph = use_cuda_graph

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Enable Tensor Core acceleration flags on NVIDIA Ampere+ GPUs
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Build model configuration namespace
        model_args = types.SimpleNamespace(
            backbone=backbone,
            row=row,
            line=line,
        )

        print(f"[CrowdDetector] Initializing P2PNet on device: {self.device} (Precision: {self.precision.upper()})")
        self.model = build_model(model_args, training=False).to(self.device)

        # Load weights
        checkpoint = torch.load(weight_path, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        # Configure precision
        if self.precision == "fp16" and self.device.type == "cuda":
            self.model.half()
            state_dict = {k: v.half() if torch.is_floating_point(v) else v for k, v in state_dict.items()}
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Image transforms
        self.transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.resample_mode = (
            Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
        )

        # CUDA Graph cache
        self.cuda_graph = None
        self.static_input = None
        self.static_outputs = None

        # Warmup
        if self.device.type == "cuda":
            with torch.inference_mode():
                dummy = torch.zeros((1, 3, 384, 640), device=self.device, dtype=self.dtype)
                _ = self.model(dummy)
                torch.cuda.synchronize()
            print(f"[CrowdDetector] Tensor Core warmup complete. Ready for real-time inference.")

    def _init_cuda_graph(self, input_tensor: torch.Tensor):
        """Capture CUDA Graph for fixed resolution input."""
        print(f"[CrowdDetector] Capturing CUDA Graph for input shape {tuple(input_tensor.shape)}...")
        self.static_input = input_tensor.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                _ = self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(s)

        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self.static_outputs = self.model(self.static_input)
        print("[CrowdDetector] CUDA Graph captured successfully.")

    def detect(
        self,
        frame: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[List[List[float]], List[float], float]:
        """
        Run optimized deep inference on a single video frame.

        Returns:
            (points, confidences, inference_time_ms)
        """
        conf_thresh = threshold if threshold is not None else self.confidence_threshold
        orig_h, orig_w = frame.shape[:2]

        t0 = time.perf_counter()

        # 1. Convert BGR to RGB and PIL Image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 2. Compute valid dimensions (multiples of 128)
        w, h = orig_w, orig_h
        if w > self.max_dimension or h > self.max_dimension:
            r = w / h
            if w > h:
                w = self.max_dimension
                h = int(self.max_dimension / r)
            else:
                h = self.max_dimension
                w = int(self.max_dimension * r)

        new_w = max(128, (w // 128) * 128)
        new_h = max(128, (h // 128) * 128)

        scale_x = orig_w / float(new_w)
        scale_y = orig_h / float(new_h)

        img_resized = img_pil.resize((new_w, new_h), self.resample_mode)

        # 3. Preprocess Tensor
        img_tensor = self.transform(img_resized).unsqueeze(0).to(self.device, dtype=self.dtype)

        # 4. Deep Inference
        with torch.inference_mode():
            if self.use_cuda_graph and self.device.type == "cuda":
                if self.cuda_graph is None or self.static_input.shape != img_tensor.shape:
                    self._init_cuda_graph(img_tensor)
                self.static_input.copy_(img_tensor)
                self.cuda_graph.replay()
                outputs = self.static_outputs
            elif self.precision == "amp" and self.device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(img_tensor)
            else:
                outputs = self.model(img_tensor)

            pred_scores = torch.nn.functional.softmax(outputs["pred_logits"], dim=-1)[:, :, 1][0]
            pred_points = outputs["pred_points"][0]

            mask = pred_scores > conf_thresh
            filtered_points = pred_points[mask].detach().cpu().float().numpy()
            filtered_scores = pred_scores[mask].detach().cpu().float().numpy().tolist()

        # 5. Map coordinates back to original frame resolution
        if len(filtered_points) > 0:
            rescaled_points = [
                [float(pt[0] * scale_x), float(pt[1] * scale_y)]
                for pt in filtered_points
            ]
        else:
            rescaled_points = []

        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return rescaled_points, filtered_scores, inference_time_ms

    def detect_batch(
        self,
        frames: List[np.ndarray],
        threshold: Optional[float] = None
    ) -> Tuple[List[List[List[float]]], List[List[float]], float]:
        """
        Run batched inference across multiple camera streams simultaneously.

        Args:
            frames: List of BGR frames (one per camera).

        Returns:
            (batch_points, batch_scores, total_latency_ms)
        """
        conf_thresh = threshold if threshold is not None else self.confidence_threshold
        batch_size = len(frames)
        if batch_size == 0:
            return [], [], 0.0

        t0 = time.perf_counter()

        tensors = []
        scales = []

        for frame in frames:
            orig_h, orig_w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            w, h = orig_w, orig_h
            if w > self.max_dimension or h > self.max_dimension:
                r = w / h
                w = self.max_dimension if w > h else int(self.max_dimension * r)
                h = int(self.max_dimension / r) if w > h else self.max_dimension

            new_w = max(128, (w // 128) * 128)
            new_h = max(128, (h // 128) * 128)

            scales.append((orig_w / float(new_w), orig_h / float(new_h)))
            img_resized = img_pil.resize((new_w, new_h), self.resample_mode)
            tensors.append(self.transform(img_resized))

        batch_tensor = torch.stack(tensors, dim=0).to(self.device, dtype=self.dtype)

        with torch.inference_mode():
            if self.precision == "amp" and self.device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(batch_tensor)
            else:
                outputs = self.model(batch_tensor)

            all_scores = torch.nn.functional.softmax(outputs["pred_logits"], dim=-1)[:, :, 1]
            all_points = outputs["pred_points"]

        batch_points = []
        batch_scores = []

        for i in range(batch_size):
            sx, sy = scales[i]
            scores_i = all_scores[i]
            points_i = all_points[i]

            mask = scores_i > conf_thresh
            filt_pts = points_i[mask].detach().cpu().float().numpy()
            filt_scs = scores_i[mask].detach().cpu().float().numpy().tolist()

            pts_rescaled = [[float(p[0] * sx), float(p[1] * sy)] for p in filt_pts]
            batch_points.append(pts_rescaled)
            batch_scores.append(filt_scs)

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        return batch_points, batch_scores, total_latency_ms
