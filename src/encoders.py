"""
src.encoders.py
Custom encoder backbones for segmentation_models_pytorch
BoMeyering 2025
"""
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from segmentation_models_pytorch.encoders._base import EncoderMixin


class DINOv3Encoder(nn.Module, EncoderMixin):
    """Wraps a HuggingFace DINOv3 ViT model as an smp-compatible encoder.

    Extracts intermediate patch-token feature maps at `depth` uniformly-sampled
    transformer block outputs. All feature maps are at the same spatial stride
    (equal to patch_size), so pair with a decoder that handles uniform-stride
    features (e.g. smp.UPerNet).

    Args:
        model_name: HuggingFace model ID, e.g. "facebook/dinov3-vitb16-pretrain-lvd1689m"
        depth: Number of intermediate feature stages to extract (default 4).
        output_indices: 0-based block indices to extract features from. If None,
            samples `depth` indices uniformly across all blocks.
        pretrained: If True, loads pretrained weights from HuggingFace Hub.
            If False, initialises with random weights using the model config.
    """

    def __init__(
        self,
        model_name: str,
        depth: int = 4,
        output_indices: Optional[List[int]] = None,
        pretrained: bool = True,
        frozen: bool = True
    ):
        nn.Module.__init__(self)
        EncoderMixin.__init__(self)

        if pretrained:
            self.dinov3 = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.dinov3 = AutoModel.from_config(config)

        if frozen:
            for param in self.dinov3.parameters():
                param.requires_grad = False

        cfg = self.dinov3.config
        self.patch_size: int = cfg.patch_size
        self.embed_dim: int = cfg.hidden_size
        num_layers: int = cfg.num_hidden_layers
        num_register_tokens: int = getattr(cfg, "num_register_tokens", 0)
        # prefix tokens = [CLS] + register tokens; strip these to get patch tokens
        self.num_prefix_tokens: int = 1 + num_register_tokens

        if output_indices is None:
            # Sample depth indices uniformly across num_layers (0-based block indices)
            output_indices = [
                int(num_layers / depth * i) - 1 for i in range(1, depth + 1)
            ]

        if len(output_indices) != depth:
            raise ValueError(
                f"len(output_indices) must equal depth, got {len(output_indices)} != {depth}"
            )
        for idx in output_indices:
            if not (0 <= idx < num_layers):
                raise ValueError(
                    f"output_indices must be in [0, {num_layers}), got {idx}"
                )

        self.output_indices: List[int] = output_indices
        # hidden_states[0] is the patch-embedding output (before any block),
        # so block i's output lives at hidden_states[i+1].
        self._hs_indices: List[int] = [idx + 1 for idx in output_indices]

        # EncoderMixin required attributes
        self._depth = depth
        self._in_channels = 3
        self._output_stride = self.patch_size
        # First element follows smp convention of returning the raw input as
        # the highest-resolution "feature"; remaining depth elements are
        # patch-token feature maps at stride patch_size.
        self._out_channels = [3] + [self.embed_dim] * depth

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return [raw_input, feat_stage_1, ..., feat_stage_depth].

        Each feat_stage_i has shape (B, embed_dim, H//patch_size, W//patch_size).
        """
        B, _C, H, W = x.shape
        pH = H // self.patch_size
        pW = W // self.patch_size

        outputs = self.dinov3(pixel_values=x, output_hidden_states=True)

        features: List[torch.Tensor] = [x]
        for hs_idx in self._hs_indices:
            hs = outputs.hidden_states[hs_idx]          # (B, num_tokens, embed_dim)
            patch_tokens = hs[:, self.num_prefix_tokens:]  # (B, pH*pW, embed_dim)
            feat = (
                patch_tokens
                .reshape(B, pH, pW, self.embed_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            features.append(feat)

        return features

    def set_in_channels(self, in_channels: int, pretrained: bool = True) -> None:
        if in_channels != 3:
            raise ValueError(
                f"DINOv3Encoder only supports in_channels=3, got {in_channels}"
            )
        self._in_channels = in_channels

    def make_dilated(self, output_stride: int) -> None:
        raise ValueError(
            "Dilated mode is not supported for DINOv3Encoder. "
            "ViT patch embeddings operate at a fixed stride."
        )
