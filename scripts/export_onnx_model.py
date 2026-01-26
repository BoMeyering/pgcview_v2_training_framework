"""
export_onnx_model.py
Export a trained segmentation model to ONNX format.
BoMeyering 2026
"""

import base64
import io
import sys
from pathlib import Path
import torch
import numpy as np
import cv2
import onnxruntime as ort
import json
import omegaconf
from omegaconf import OmegaConf
from argparse import ArgumentParser
import segmentation_models_pytorch as smp

# Append parent path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models import create_smp_model
from src.utils.config import TrainSupervisedConfig
from src.utils.config import ModelArchitecture
from src.parameters import EMA, apply_ema
from src.transforms import get_tensor_transforms

parser = ArgumentParser(description="Export a trained segmentation model to ONNX format.")
parser.add_argument("--config", type=str, required=True, help="Path to the trained configuration file.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")

args = parser.parse_args()

conf = OmegaConf.load(args.config)
conf.model.architecture = ModelArchitecture[conf.model.architecture] # Grab the model class from the enum class

# Create a dummy input tensor
img_height, img_width = conf.images.resize
img_channels = conf.images.input_channels
img_tensor = torch.randn(1, img_channels, img_height, img_width)

# Create base model
model = create_smp_model(conf)
checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['ema_state_dict'])
model.eval()

# Test out the model with a sample image
test_image = cv2.imread("data/toy_dataset/test/images/0a0c93b9-ccf6-48a0-ab2a-71299ff894eb.jpg", cv2.IMREAD_COLOR_RGB)
transforms = get_tensor_transforms(
    resize=(img_height, img_width),
    normalize=True,
    means=conf.metadata.norm.means,
    std=conf.metadata.norm.std
)

augmented = transforms(image=test_image)
test_image = augmented['image'].cpu().unsqueeze(0)

torch_out = model(test_image)
torch_out = torch_out.detach().squeeze(0).numpy().argmax(axis=0)
cv2.imwrite("pytorch_output_mask.png", torch_out.astype(np.uint8)*20)

# Export the model to ONNX
torch.onnx.export(
    model,
    (test_image,),
    "exported_model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamo=False,
    opset_version=17
)

# Test out onnx inference
sess = ort.InferenceSession("exported_model.onnx", providers=["CPUExecutionProvider"])
onnx_out = sess.run(['output'], {'input': test_image.numpy()})

onnx_out = onnx_out[0].squeeze(0)
out_map = np.argmax(onnx_out, axis=0)
cv2.imwrite("onnx_output_mask.png", out_map.astype(np.uint8)*20)
