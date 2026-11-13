import argparse
import os
import sys
import time
import types
import numpy as np
import torch

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model


def benchmark_engine(
    weights_path: str = "./pretrained_model/best_mae.pth",
    resolution: tuple = (384, 640),
    iterations: int = 100,
    warmup: int = 20
):
    if not torch.cuda.is_available():
        print("Error: CUDA device required for Tensor Core benchmarks.")
        return

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)

    print("=" * 75)
    print(f"SURVEILLANCE INFERENCE BENCHMARK | {gpu_name} (Compute {cc[0]}.{cc[1]})")
    print(f"Input Shape: (3, {resolution[0]}, {resolution[1]}) | Iterations: {iterations}")
    print("=" * 75)

    args = types.SimpleNamespace(backbone="vgg16_bn", row=2, line=2)
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    results = []

    # Benchmark Configurations
    configs = [
        {"name": "PyTorch FP32 (Baseline)", "precision": "fp32", "batch": 1, "cuda_graph": False},
        {"name": "Ampere FP16 (Tensor Cores)", "precision": "fp16", "batch": 1, "cuda_graph": False},
        {"name": "FP16 + CUDA Graph", "precision": "fp16", "batch": 1, "cuda_graph": True},
        {"name": "Batched Multi-Cam (B=2)", "precision": "fp16", "batch": 2, "cuda_graph": False},
        {"name": "Batched Multi-Cam (B=4)", "precision": "fp16", "batch": 4, "cuda_graph": False},
    ]

    baseline_latency = None

    for cfg in configs:
        b = cfg["batch"]
        prec = cfg["precision"]
        use_graph = cfg["cuda_graph"]

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = build_model(args, training=False).to(device)

        if prec == "fp16":
            model.half()
            half_state = {k: v.half() if torch.is_floating_point(v) else v for k, v in state_dict.items()}
            model.load_state_dict(half_state)
            dtype = torch.float16
        else:
            model.load_state_dict(state_dict)
            dtype = torch.float32

        model.eval()

        dummy_input = torch.randn(b, 3, resolution[0], resolution[1], device=device, dtype=dtype)

        # Warmup
        with torch.inference_mode():
            for _ in range(warmup):
                _ = model(dummy_input)
            torch.cuda.synchronize()

        if use_graph:
            # Capture Graph
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    _ = model(dummy_input)
            torch.cuda.current_stream().wait_stream(s)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _ = model(dummy_input)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iterations):
                graph.replay()
            torch.cuda.synchronize()
            total_elapsed = time.perf_counter() - t0
        else:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(iterations):
                    _ = model(dummy_input)
                    torch.cuda.synchronize()
            total_elapsed = time.perf_counter() - t0

        avg_latency_ms = (total_elapsed / iterations) * 1000.0
        per_frame_ms = avg_latency_ms / b
        throughput_fps = (b * iterations) / total_elapsed
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        if baseline_latency is None:
            baseline_latency = per_frame_ms
            speedup = "1.00x (Ref)"
        else:
            speedup = f"{baseline_latency / per_frame_ms:.2f}x"

        results.append({
            "Configuration": cfg["name"],
            "Batch": b,
            "Latency / Frame": f"{per_frame_ms:.2f} ms",
            "Throughput": f"{throughput_fps:.1f} FPS",
            "Speedup": speedup,
            "VRAM": f"{peak_vram_mb:.0f} MB",
        })

    # Print Table
    header = f"{'Configuration':<30} | {'Batch':<5} | {'Latency':<15} | {'Throughput':<12} | {'Speedup':<12} | {'VRAM':<8}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r['Configuration']:<30} | {r['Batch']:<5} | {r['Latency / Frame']:<15} | {r['Throughput']:<12} | {r['Speedup']:<12} | {r['VRAM']:<8}")
    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="P2PNet Deep Optimization & Latency Benchmark")
    parser.add_argument("--weights", default="pretrained_model/best_mae.pth", help="Model weights path")
    parser.add_argument("--iterations", default=100, type=int, help="Number of benchmark iterations")
    parser.add_argument("--width", default=640, type=int, help="Input width (multiple of 128)")
    parser.add_argument("--height", default=384, type=int, help="Input height (multiple of 128)")
    args = parser.parse_args()

    benchmark_engine(
        weights_path=args.weights,
        resolution=(args.height, args.width),
        iterations=args.iterations
    )


if __name__ == "__main__":
    main()
