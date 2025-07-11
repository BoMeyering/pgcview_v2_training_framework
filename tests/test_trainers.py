"""
tests.test_trainers.py
src.trainer.py unit tests
BoMeyering 2025
"""

import unittest
from math import ceil
from omegaconf import OmegaConf
from glob import glob
from wonderwords import RandomWord
from pathlib import Path
from typing import Tuple
from random import randint
from torch import Tensor
from torch.utils.data import DataLoader
from src.dataloaders import InfiniteSampler, DataLoaderBalancer
from src.datasets import LabeledDataset, UnlabeledDataset
from src.trainer import SupervisedTrainer, FlexMatchTrainer