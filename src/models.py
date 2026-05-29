"""
src.models.py
Model Instantiation
BoMeyering 2025
"""
import sys
import omegaconf
from omegaconf import OmegaConf
import segmentation_models_pytorch as smp
import argparse
import torch

from omegaconf.errors import ConfigAttributeError
from src.encoders import DINOv3Encoder

_DINOV3_HF_PREFIX = "facebook/dinov3"


def _is_dinov3(encoder_name: str) -> bool:
    return encoder_name.startswith(_DINOV3_HF_PREFIX)


def _build_dinov3_model(conf: omegaconf.dictconfig.DictConfig, model_config: omegaconf.dictconfig.DictConfig) -> torch.nn.Module:
    """Builds an smp model with a DINOv3Encoder backbone.

    Patches get_encoder in all loaded smp module namespaces so the normal smp
    model constructor receives the pre-built DINOv3Encoder without any changes
    to the config or the model class.
    """
    encoder_name = model_config.encoder_name
    depth = OmegaConf.select(model_config, "encoder_depth", default=4)
    pretrained = OmegaConf.select(model_config, "encoder_weights", default=None) is not None

    pre_built = DINOv3Encoder(model_name=encoder_name, depth=depth, pretrained=pretrained)

    def _patched_get_encoder(name, **kwargs):
        return pre_built

    # smp model modules bind get_encoder locally via `from ... import get_encoder`,
    # so we patch each loaded module's namespace directly.
    originals = {}
    for mod_name, mod in sys.modules.items():
        if mod_name.startswith("segmentation_models_pytorch") and hasattr(mod, "get_encoder"):
            originals[mod_name] = mod.get_encoder
            mod.get_encoder = _patched_get_encoder

    try:
        model_class = getattr(smp, conf.model.architecture.value)
        model = model_class(**model_config)
    finally:
        for mod_name, orig_fn in originals.items():
            sys.modules[mod_name].get_encoder = orig_fn

    return model


def create_smp_model(conf: omegaconf.dictconfig.DictConfig) -> torch.nn.Module:
    """Creates an smp Pytorch model

    conf:
        conf (omegaconf.dictconfig.DictConfig): The OmegaConf configuration dictionary

    Raises:
        ValueError: If conf.model.config.encoder_name is not listed in smp.encoders.get_encoder_names()
            and is not a supported DINOv3 HuggingFace model ID.
        ValueError: If conf.model.architecture does not match any of the specified architectures.

    Returns:
        torch.nn.Module: A model as a pytorch module
    """

    try:
        model_config = conf.model.config
    except Exception as e:
        raise Exception(e)

    if _is_dinov3(model_config.encoder_name):
        return _build_dinov3_model(conf, model_config)

    if model_config.encoder_name not in smp.encoders.get_encoder_names():
        raise ValueError(f"Encoder name {model_config.encoder_name} is not one of the accepted encoders. Please select an encoder from {smp.encoders.get_encoder_names()}")

    try:
        model_class = getattr(smp, conf.model.architecture.value)
        model = model_class(**model_config)

        return model
    except AttributeError as e:
        raise ValueError(f"Model architecture {conf.model.architecture} is not a valid SMP architecture.\nSelect one from 'smp._MODEL_ARCHITECTURES'")
