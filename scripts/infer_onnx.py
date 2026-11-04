"""
scripts/infer_onnx.py
Run ONNX segmentation inference on a directory of images and save overlay visualizations.
BoMeyering 2026
"""

import sys
import site
import json
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# nvidia-*-cu12 packages install CUDA .so files into site-packages, not system lib paths.
# Pre-loading them via ctypes puts them in the process link map so onnxruntime's CUDA
# provider plugin finds them as already-loaded transitive dependencies — more reliable
# than setting LD_LIBRARY_PATH, which glibc caches at process startup.
def _preload_cuda_libs():
    import ctypes
    sp = Path(site.getsitepackages()[0])
    for lib in sorted(sp.glob("nvidia/*/lib/lib*.so.*")):
        try:
            ctypes.CDLL(str(lib))
        except OSError:
            pass

_preload_cuda_libs()

import onnxruntime as ort  # noqa: E402 — must come after CUDA libs are pre-loaded
sys.path.append(str(PROJECT_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args():
    parser = ArgumentParser(
        description="Run ONNX segmentation inference on a directory of images and save overlay visualizations."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory of input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where overlay images will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs="+",
        default=[512, 512],
        metavar="DIM",
        help="Model input size as H W, or a single value for square (default: 512 512).",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default=str(PROJECT_ROOT / "metadata" / "dataset_norm.json"),
        help="Path to normalization JSON with 'means' and 'std' keys.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default=str(PROJECT_ROOT / "metadata" / "target_mapping.json"),
        help="Path to target_mapping.json with per-class RGB values.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Mask overlay opacity (default: 0.6).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="OnnxRuntime execution provider (default: cpu).",
    )
    return parser.parse_args()


def resolve_img_size(img_size_arg):
    if len(img_size_arg) == 1:
        return (img_size_arg[0], img_size_arg[0])
    if len(img_size_arg) == 2:
        return tuple(img_size_arg)
    raise ValueError("--img-size takes 1 or 2 integers.")


def build_colormap(colormap_path):
    """Return a (num_classes, 3) uint8 RGB array indexed by class id."""
    with open(colormap_path, "r") as f:
        mapping = json.load(f)
    max_idx = max(v["class_idx"] for v in mapping.values())
    lut = np.zeros((max_idx + 1, 3), dtype=np.uint8)
    for cls_info in mapping.values():
        lut[cls_info["class_idx"]] = cls_info["rgb"]
    return lut


def preprocess(img_rgb, img_h, img_w, means, std):
    """Resize, normalize, and return a (1, 3, H, W) float32 array."""
    resized = cv2.resize(img_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32) / 255.0
    normalized = (resized - np.array(means, dtype=np.float32)) / np.array(std, dtype=np.float32)
    # HWC -> CHW -> NCHW
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)


def make_overlay(model_output, orig_h, orig_w, colormap, alpha, orig_rgb):
    """Argmax model output, upsample to original resolution, and blend onto the image."""
    # model_output: (1, num_classes, H, W)
    pred = np.argmax(model_output[0], axis=0).astype(np.uint8)  # (H, W)
    pred_full = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    mask_rgb = colormap[pred_full].astype(np.float32)
    blended = alpha * mask_rgb + (1.0 - alpha) * orig_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    img_h, img_w = resolve_img_size(args.img_size)

    with open(args.norm, "r") as f:
        norm = json.load(f)
    means, std = norm["means"], norm["std"]

    colormap = build_colormap(args.colormap)

    available_providers = ort.get_available_providers()
    if args.provider == "cuda":
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "CUDAExecutionProvider is not available in this OnnxRuntime installation.\n"
                f"Available providers: {available_providers}\n"
                "Install onnxruntime-gpu and ensure CUDA is configured, or pass --provider cpu."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(args.model, providers=providers)
    active_provider = sess.get_providers()[0]

    # Warn if ORT silently fell back despite CUDA being requested
    if args.provider == "cuda" and active_provider != "CUDAExecutionProvider":
        print(f"WARNING: requested CUDA but OnnxRuntime is running on {active_provider}.")

    input_name = sess.get_inputs()[0].name
    print(f"Model   : {args.model}")
    print(f"Input   : '{input_name}'  size: {img_h}x{img_w}")
    print(f"Provider: {active_provider}")

    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"\nFound {len(image_paths)} images. Running inference...\n")

    for i, img_path in enumerate(image_paths, 1):
        img_rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)
        if img_rgb is None:
            print(f"  [{i:>{len(str(len(image_paths)))}}/{len(image_paths)}] SKIP (unreadable): {img_path.name}")
            continue

        orig_h, orig_w = img_rgb.shape[:2]
        tensor = preprocess(img_rgb, img_h, img_w, means, std)
        outputs = sess.run(None, {input_name: tensor})
        overlay = make_overlay(outputs[0], orig_h, orig_w, colormap, args.alpha, img_rgb)

        out_path = output_dir / (img_path.stem + ".png")
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  [{i:>{len(str(len(image_paths)))}}/{len(image_paths)}] {img_path.name} -> {out_path.name}")

    print(f"\nDone. Overlays saved to: {output_dir}")


if __name__ == "__main__":
    main()
