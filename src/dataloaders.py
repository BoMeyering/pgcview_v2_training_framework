"""
src.dataloaders.py
Dataloader and data sampler functions
BoMeyering 2025
####### Script needs to be updated ############
"""

import math
import torch
import random
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Iterable, Optional, List, Tuple, Optional, Sequence
import torch.distributed as dist

class InfiniteSampler(Sampler):
    """ Infinite Sampler for a torch.utils.data.Dataloader """
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.indices = list(range(len(dataset)))
        self.dataset = dataset

    def __iter__(self):
        """ Iterate through the dataloader by wrapping the shuffled indices """
        while True:
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            for idx in indices:
                yield idx

    def __len__(self):
        """ Return the length of the data """
        return len(self.dataset)
    

class DataLoaderBalancer:
    """
    Takes an arbitrary number of datasets with corresponding batch sizes
    Balances them all to the length of the longest dataset using an InfiniteSampler
    
    Methods:
        balance_loaders: Returns a list of dataloaders, and the length of the longest dataloader
    """
    def __init__(self, *datasets, batch_sizes: Iterable[int, ], drop_last: bool=True):
        """
        Initialize the dataloader balancer

        Args:
            *datasets (Iterable[Dataset, ]): Any iterable containing elements of type torch.utils.data.Dataset.
            batch_sizes (Iterable[int, ]): Any iterable containing elements of type int.
            drop_last (bool, optional): Whether to drop the last batch if not full. Defaults to True.

        Raises:
            ValueError: If the number of datasets is not the same as the number of batch_size elements.
            ValueError: If a dataset has fewer elements than its corresponding batch size.
            ValueError: If there is one or more negative elements in 'batch_sizes'.
            ValueError: If there is one or more elements in 'batch_sizes' that is not an integer.
        """
        
        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.drop_last = drop_last
        
        # Create empty dataloader and length lists
        self.dl_lengths = []
        self.dataloaders = []

        # Quality checks for batch sizes
        if any([type(i) != int for i in batch_sizes]):
            raise ValueError(f"One or more of the batch sizes is not an integer. Please ensure that all batch sizes are positive integers.")
        elif any([i < 0 for i in batch_sizes]):
            raise ValueError(f"One of the batch size elements is negative. Please ensure that all batch sizes are positive integers.")
        
        # Quality checks for datasets
        if len(self.datasets) != len(self.batch_sizes):
            raise ValueError("The number of datasets does not equal the number of batch sizes. Please ammend appropriately")
        for i, (ds, bs) in enumerate(zip(self.datasets, self.batch_sizes)):
            if len(ds) < bs:
                raise ValueError(f"Dataset {i+1} has fewer elements than its specified batch size. Please select a batch size smaller than {bs} and try again.")

        # Enforce correct dataloader lengths
        if drop_last:
            for i, ds in enumerate(self.datasets):
                self.dl_lengths.append(len(ds) // self.batch_sizes[i])
        else:
            for i, ds in enumerate(self.datasets):
                self.dl_lengths.append(math.ceil(len(ds) / self.batch_sizes[i]))

        # Grab the index for the longest dataloader
        self.max_idx = np.argmax(self.dl_lengths)
                
    def balance_loaders(self):
        # Get the index of the longest dataloader
        for i, ds in enumerate(self.datasets):
            # For the longest dataloader, create a loader that iterates over everything once
            if i == self.max_idx:
                self.dataloaders.append(DataLoader(ds, batch_size=self.batch_sizes[i], shuffle=True, drop_last=self.drop_last))
            else: # Wrap the rest of the dataloaders with InfiniteSampler
                self.dataloaders.append(DataLoader(ds, batch_size=self.batch_sizes[i], sampler=InfiniteSampler(ds), drop_last=self.drop_last))
        
        return self.dataloaders, self.dl_lengths[self.max_idx]
    
class DistributedInfiniteSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Only require dist init if we need to infer rank/world_size
        if num_replicas is None or rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            if not dist.is_initialized():
                raise RuntimeError("Distributed package is not initialized")
            if num_replicas is None:
                num_replicas = dist.get_world_size()
            if rank is None:
                rank = dist.get_rank()

        if num_replicas is None or rank is None:
            raise ValueError("num_replicas and rank must be set (either passed in or via torch.distributed)")

        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank}, expected 0 <= rank < {num_replicas}")

        self.num_replicas = num_replicas
        self.rank = rank

        n = len(self.dataset)
        if self.drop_last and n < self.num_replicas:
            raise ValueError(
                f"drop_last=True but dataset length ({n}) < num_replicas ({self.num_replicas}); "
                "this would yield zero samples per rank."
            )

        if self.drop_last:
            self.num_samples = n // self.num_replicas
        else:
            self.num_samples = int(math.ceil(n / self.num_replicas))

        if self.num_samples == 0:
            raise ValueError("DistributedInfiniteSampler computed num_samples=0; check dataset length/drop_last.")

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        n = len(self.dataset)

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if not self.drop_last:
            if len(indices) < self.total_size:
                indices += indices[: (self.total_size - len(indices))]
            else:
                indices = indices[: self.total_size]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(f"Expected {self.num_samples} indices for rank {self.rank}, got {len(indices)}")

        while True:
            yield from indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


@dataclass
class BalancedLoaders:
    dataloaders: List[DataLoader]
    samplers: List[object]
    driver_index: int
    steps_per_epoch: int

    def set_epoch(self, epoch: int):
        """ Set epoch for all samplers """
        for sampler in self.samplers:
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)


class DistributedDataLoaderBalancer:
    """ Class to balance multiple dataloaders in distributed training """
    def __init__(
        self,
        datasets: Sequence[Dataset],
        batch_sizes: Sequence[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool=True,
        drop_last: bool=True,
        **dataloader_kwargs,
    ):
        if len(datasets) != len(batch_sizes):
            raise ValueError("The number of datasets does not equal the number of batch sizes. Please ammend appropriately")
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Torch distributediver package must be available and initialized")

        self.datasets = datasets
        self.batch_sizes = list(batch_sizes)
        self.num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
        self.rank = dist.get_rank() if rank is None else rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataloader_kwargs = dataloader_kwargs

        for i, (ds, bs) in enumerate(zip(self.datasets, self.batch_sizes)):
            if len(ds) < bs:
                raise ValueError(
                    f"Dataset {i} has fewer elements than its specified batch size ({bs})."
                    f" Please select a batch size smaller than {bs} and try again."
                )
        
    def _per_rank_samples(self, n: int) -> int:
        if self.drop_last:
            if n < self.num_replicas:
                return 0
            return n // self.num_replicas
        else:
            return int(math.ceil(n / self.num_replicas))
        
    def balance_loaders(self) -> BalancedLoaders:
        """ Balance dataloaders across datasets """
        per_rank_steps = []

        for ds, bs in zip(self.datasets, self.batch_sizes):
            prs = self._per_rank_samples(len(ds))
            if prs == 0:
                per_rank_steps.append(0)
            else:
                if self.drop_last:
                    per_rank_steps.append(prs // bs)
                else:
                    per_rank_steps.append(int(math.ceil(prs / bs)))
        driver_index = int(max(range(len(per_rank_steps)), key=lambda i: per_rank_steps[i])) # Grab the "driver" dataset index
        steps_per_epoch = per_rank_steps[driver_index]
        if steps_per_epoch == 0:
            raise ValueError("All datasets yielded 0 steps per epoch under current settings.")
        
        dataloaders: List[DataLoader] = []
        samplers: List[object] = []

        for i, (ds, bs) in enumerate(zip(self.datasets, self.batch_sizes)):
            if i == driver_index:
                sampler = DistributedSampler(
                    dataset=ds,
                    num_replicas=self.num_replicas,
                    rank=self.rank,
                    shuffle=self.shuffle,
                    drop_last=self.drop_last,
                )
            else:
                sampler = DistributedInfiniteSampler(
                    dataset=ds,
                    num_replicas=self.num_replicas,
                    rank=self.rank,
                    shuffle=self.shuffle,
                    drop_last=self.drop_last,
                )

            dl = DataLoader(
                dataset=ds,
                batch_size=bs,
                sampler=sampler,
                shuffle=False,
                drop_last=self.drop_last,
                num_workers=6,
                **self.dataloader_kwargs,
            )

            samplers.append(sampler)
            dataloaders.append(dl)

        return BalancedLoaders(
            dataloaders=dataloaders,
            samplers=samplers,
            driver_index=driver_index,
            steps_per_epoch=steps_per_epoch,
        )


class RotatingSampler(Sampler):
    def __init__(self, dataset_size: int, train: bool=True, train_size: float=0.8, seed: int=42, device: str='cpu'):
        self.dataset_size = dataset_size
        self.train = train
        self.train_size = train_size
        self.seed = seed
        self.device = device
        self.split_idx = int(self.dataset_size * self.train_size)

    def set_epoch(self, epoch):
        """ Set the internal epoch state """
        self.epoch = epoch

    def __iter__(self):
        """ Define Rotating Sampler iteration """
        # Define a torch generator
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.epoch + self.seed)

        indices = torch.randperm(self.dataset_size, generator=gen).tolist()
        if self.train:
            return iter(indices[:self.split_idx])
        else:
            return iter(indices[self.split_idx:])

    def __len__(self):
        """ Return the length of the sampler """
        return self.split_idx if self.train else (self.dataset_size - self.split_idx)

