"""
src.utils.loggers.py
Logger Instantiation script
BoMeyering 2025
"""

import logging
import sys
import omegaconf
from pathlib import Path
from omegaconf import OmegaConf

def setup_loggers(conf):
    """
    Configures a simple logger to log outputs to the console and the output file.

    Args:
        args (argparse.Namespace): arguments object from the configuration file.
    """
    filename = conf.model_run + '.log'
    filepath = Path(conf.directories.log_dir) / filename

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.handlers.RotatingFileHandler(filepath, 'a', 1000000, 3)
    stream_handler = logging.StreamHandler(sys.stdout,)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(conf.logging_level)