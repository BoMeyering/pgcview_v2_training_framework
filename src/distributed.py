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

def shutdown_ddp():
    """ Shutdown the distributed processes """
    dist.barrier()
    dist.destroy_process_group()

def set_env_ranks(conf: OmegaConf):
    """ Get ranks and world size for a process """

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    conf.rank = rank
    conf.local_rank = local_rank
    conf.world_size = world_size
    conf.is_main = is_main_process()

def is_main_process() -> bool:
    """ Returns True if process is the main process """
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0