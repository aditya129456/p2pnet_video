import argparse
import os
import sys
import types
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model


class P2PNetExportModel(nn.Module):
    """
    Export wrapper that strips internal dictionaries and outputs raw tensors
    compatible with ONNX and NVIDIA TensorRT runtimes.
    """

    def __init__(self, p2pnet_model: nn.Module):
        super().__init__()
        self.model = p2pnet_model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        logits = outputs["pred_logits"]
        points = outputs["pred_points"]
        scores = torch.softmax(logits, dim=-1)[:, :, 1]
        return scores, points


def export_model(
    weights_path: str = "pretrained_model/best_mae.pth",
    output_onnx: str = "prediction/p2pnet_optimized.onnx",
    height: int = 384,
    width: int = 640,
    precision: str = "fp16",
    opset: int = 14,
):
    print(f"Loading P2PNet model from: {weights_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = types.SimpleNamespace(backbone="vgg16_bn", row=2, line=2)
    model = build_model(args, training=False).to(device)

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    if precision == "fp16" and device.type == "cuda":
        model.half()
        state_dict = {k: v.half() if torch.is_floating_point(v) else v for k, v in state_dict.items()}
        dtype = torch.float16
    else:
        dtype = torch.float32

    model.load_state_dict(state_dict)
    model.eval()

    export_wrapper = P2PNetExportModel(model)
    dummy_input = torch.randn(1, 3, height, width, device=device, dtype=dtype)

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx)), exist_ok=True)

    print(f"Exporting to ONNX: {output_onnx} (Input: 1x3x{height}x{width}, Precision: {precision.upper()})")

    try:
        torch.onnx.export(
            export_wrapper,
            dummy_input,
            output_onnx,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input_images"],
            output_names=["confidence_scores", "point_coordinates"],
            dynamic_axes={
                "input_images": {0: "batch_size"},
                "confidence_scores": {0: "batch_size"},
                "point_coordinates": {0: "batch_size"},
            }
        )
        print(f"\n🎉 Successfully exported ONNX model to: {output_onnx}")
        print("\nTo compile with NVIDIA TensorRT for maximum performance:")
        print(f"  trtexec --onnx={output_onnx} --fp16 --saveEngine=p2pnet_{precision}.engine")
    except Exception as e:
        print(f"\n⚠️ ONNX export encountered: {e}")
        print("Note: To export ONNX graphs directly, ensure 'pip install onnx' is installed.")


def main():
    parser = argparse.ArgumentParser(description="Export P2PNet to Optimized ONNX / TensorRT format")
    parser.add_argument("--weights", default="pretrained_model/best_mae.pth", help="Model checkpoint path")
    parser.add_argument("--output", default="prediction/p2pnet_optimized.onnx", help="Output ONNX filename")
    parser.add_argument("--width", default=640, type=int, help="Input width")
    parser.add_argument("--height", default=384, type=int, help="Input height")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"], help="Export precision")
    args = parser.parse_args()

    export_model(
        weights_path=args.weights,
        output_onnx=args.output,
        height=args.height,
        width=args.width,
        precision=args.precision
    )


if __name__ == "__main__":
    main()
