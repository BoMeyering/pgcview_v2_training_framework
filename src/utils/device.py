"""
src.utils.device.py
Device Setter script
BoMeyering 2025
"""

import torch
import logging
import omegaconf
from omegaconf import OmegaConf
from src.utils.loggers import rank_log
from src.distributed import is_main_process

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
        # If device is set to CUDA
        if conf.device == 'cuda':
            # Check if CUDA is available and corresponds to the set device
            device_str = ":".join(['cuda', str(conf.local_rank)]) if torch.cuda.is_available() else ":".join(['cpu', str(conf.local_rank)])
            if not conf.device in device_str:
                rank_log(conf.is_main, logger.info, "CUDA Device is not available at this time. Falling back to CPU computation.")
            conf.device = device_str
        elif conf.device not in ['cpu', 'cuda']:
            rank_log(conf.is_main, logger.info, "Incorrect value set for 'conf.device'. Must be one of ['cpu', 'cuda']. Falling back to 'cpu' computation.")
            conf.device = ":".join(['cpu', str(conf.local_rank)])
    else:
        rank_log(conf.is_main, logger.info, "No value set for 'conf.training.device'. Setting to 'cpu'. If 'cuda' devices are available, please explicitly pass 'device: cuda' in the configuration YAML.")
        conf.device = ":".join(['cpu', str(conf.local_rank)])