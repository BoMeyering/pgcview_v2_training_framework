"""
scripts/export_to_onnx.py
Export a trained segmentation model checkpoint to ONNX format.
BoMeyering 2026
"""

import sys
import json
from pathlib import Path
import torch
import numpy as np
import onnx
import onnxruntime as ort
from omegaconf import OmegaConf
from argparse import ArgumentParser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models import create_smp_model
from src.utils.config import ModelArchitecture


def parse_args():
    parser = ArgumentParser(description="Export a trained segmentation model checkpoint to ONNX.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML training config used to train the model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pth checkpoint file.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs="+",
        default=None,
        metavar="DIM",
        help=(
            "Input image size as H W (e.g. --img-size 512 512) or a single value for "
            "a square (e.g. --img-size 512). Overrides the size stored in the config."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the exported ONNX file will be written.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Filename for the exported ONNX model (without extension).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to target (default: 17).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip OnnxRuntime validation after export.",
    )
    return parser.parse_args()


def resolve_img_size(args_img_size, conf):
    """Return (height, width) from CLI arg or config."""
    if args_img_size is not None:
        if len(args_img_size) == 1:
            return (args_img_size[0], args_img_size[0])
        if len(args_img_size) == 2:
            return tuple(args_img_size)
        raise ValueError("--img-size takes 1 or 2 integers.")
    try:
        h, w = conf.images.resize
        return (int(h), int(w))
    except Exception:
        raise ValueError(
            "Could not read image size from config. "
            "Please supply --img-size H W explicitly."
        )


def load_model(conf, checkpoint_path):
    """Build the model from config and load weights from checkpoint."""
    conf.model.architecture = ModelArchitecture[conf.model.architecture]

    model = create_smp_model(conf)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Unexpected checkpoint format — expected a dict with 'ema_state_dict' "
            "or 'model_state_dict' keys."
        )

    if "ema_state_dict" in checkpoint:
        state_dict = checkpoint["ema_state_dict"]
        print("  Using EMA state dict from checkpoint.")
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("  Using model state dict from checkpoint.")
    else:
        raise KeyError(
            "Checkpoint does not contain 'ema_state_dict' or 'model_state_dict'."
        )

    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(model, dummy_input, output_path, opset):
    """Run torch.onnx.export and return the path."""
    print(f"\nExporting to ONNX (opset {opset}) ...")
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"  Saved to: {output_path}")


def validate_provider(output_path, dummy_input_np, provider_name, provider_label):
    """Run a forward pass through OnnxRuntime with a specific provider."""
    try:
        sess = ort.InferenceSession(str(output_path), providers=[provider_name])
    except Exception as exc:
        print(f"  [{provider_label}] Could not create session: {exc}")
        return

    input_name = sess.get_inputs()[0].name
    try:
        outputs = sess.run(None, {input_name: dummy_input_np})
        out = outputs[0]
        print(
            f"  [{provider_label}] OK — output shape: {out.shape}, "
            f"dtype: {out.dtype}"
        )
    except Exception as exc:
        print(f"  [{provider_label}] Inference failed: {exc}")


def main():
    args = parse_args()

    # ------------------------------------------------------------------ setup
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = args.name if args.name.endswith(".onnx") else args.name + ".onnx"
    output_path = output_dir / name

    # ----------------------------------------------------------------- config
    print(f"\nLoading config from: {args.config}")
    conf = OmegaConf.load(args.config)

    img_h, img_w = resolve_img_size(args.img_size, conf)

    try:
        channels = int(conf.model.config.input_channels)
    except Exception:
        channels = 3
        print("  input_channels not found in config, defaulting to 3.")

    print(f"  Image size : {img_h} x {img_w}")
    print(f"  Channels   : {channels}")

    # ------------------------------------------------------------------ model
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    model = load_model(conf, checkpoint_path)

    # ----------------------------------------------------------- dummy tensor
    dummy_input = torch.zeros(1, channels, img_h, img_w, dtype=torch.float32)

    with torch.no_grad():
        torch_out = model(dummy_input)
    print(f"  PyTorch forward pass OK — output shape: {tuple(torch_out.shape)}")

    # ----------------------------------------------------------------- export
    export_onnx(model, dummy_input, output_path, args.opset)

    # --------------------------------------------------------------- validate
    if not args.no_validate:
        print("\nValidating exported model with OnnxRuntime ...")

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  ONNX model check passed.")

        dummy_np = dummy_input.numpy()

        validate_provider(output_path, dummy_np, "CPUExecutionProvider", "CPU")

        if torch.cuda.is_available():
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                validate_provider(
                    output_path, dummy_np, "CUDAExecutionProvider", "CUDA"
                )
            else:
                print(
                    "  [CUDA] torch.cuda is available but OnnxRuntime was built "
                    "without CUDA support — skipping."
                )
        else:
            print("  [CUDA] No CUDA device detected — skipping CUDA validation.")

    print(f"\nDone. Model exported to: {output_path}\n")


if __name__ == "__main__":
    main()
