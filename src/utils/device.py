"""
src.utils.device.py
Device Setter script
BoMeyering 2025
"""

import torch
import logging
import omegaconf
from omegaconf import OmegaConf

def set_torch_device(conf: OmegaConf) -> OmegaConf:
    """_summary_

    Args:
        conf (OmegaConf): _description_

    Returns:
        OmegaConf: _description_
    """

    logger = logging.getLogger()
    if not isinstance(conf, omegaconf.dictconfig.DictConfig):
        raise ValueError(f"Argument 'conf' should be of type 'omegaconf.dictconfig.DictConfig'.")
    
    if 'device' in conf:
        if conf.device == 'cuda':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device != conf.device:
                logger.info("CUDA Device is not available at this time. Falling back to CPU computation.")
            conf.device = device
        elif conf.device not in ['cpu', 'cuda']:
            print("Incorrect value set for 'conf.device'. Must be one of ['cpu', 'cuda']. Falling back to 'cpu' computation.")
            logger.info("Incorrect value set for 'conf.device'. Must be one of ['cpu', 'cuda']. Falling back to 'cpu' computation.")
            conf.device = 'cpu'
    else:
        logger.info("No value set for 'conf.training.device'. Setting to 'cpu'. If 'cuda' devices are available, please explicitly pass 'device: cuda' in the configuration YAML.")
        conf.device = 'cpu'