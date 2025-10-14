"""
src/distributed.py
Torch distributed functions
BoMeyering 2025
"""
from omegaconf import OmegaConf
import os
import torch.distributed as dist

def setup_ddp(backend: str):
    """ Set up DDP """
    dist.init_process_group(backend=backend, init_method="env://")

def set_env_ranks(conf: OmegaConf):
    """ Get ranks and world size for a process """

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    conf.rank = rank
    conf.local_rank = local_rank
    conf.world_size = world_size