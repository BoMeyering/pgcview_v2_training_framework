"""
src.models.py
Model Instantiation
BoMeyering 2025
"""
import omegaconf
from omegaconf import OmegaConf
import segmentation_models_pytorch as smp
import argparse
import torch

from omegaconf.errors import ConfigAttributeError

def create_smp_model(conf: omegaconf.dictconfig.DictConfig) -> torch.nn.Module:
    """Creates an smp Pytorch model

    conf:
        conf (omegaconf.dictconfig.DictConfig): The OmegaConf configuration dictionary

    Raises:
        ValueError: If conf.model.config.encoder_name is not listed in smp.encoders.get_encoder_names().
        ValueError: If conf.model.architecture does not match any of the specified architectures.

    Returns:
        torch.nn.Module: A model as a pytorch module
    """

    # Select the model cofiguration
    try:
        model_config = conf.model.config
    except Exception as e:
        raise Exception(e)
    
    if model_config.encoder_name not in smp.encoders.get_encoder_names():
        raise ValueError(f"Encoder name {model_config.encoder_name} is not one of the accepted encoders. Please select an encoder from {smp.encoders.get_encoder_names()}")
    
    try:
        model_class = getattr(smp, conf.model.architecture)
        model = model_class(**model_config)

        return model
    except AttributeError as e:
        raise ValueError(f"Model architecture {conf.model.architecture} is not a valid SMP architecture.\nSelect one from 'smp._MODEL_ARCHITECTURES'")




if __name__ == '__main__':
    
    conf = OmegaConf.create(
        {
            "model": {
                "architecture": "DeepLabV3Plus",
                "config": {
                    "encoder_name": "resnet18",
                    "encoder_depth": 5,
                    "encoder_weights": None,
                    "in_channels": 3,
                    "classes": 4
                }
            }
        }
    )
    model = create_smp_model(conf)

    print(type(model))