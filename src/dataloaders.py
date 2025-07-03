"""
src.dataloaders.py
Dataloader and data sampler functions
BoMeyering 2025
"""

import math
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Iterable, Optional
import torch.distributed as dist

class InfiniteSampler(Sampler):
    """ Infinite Sampler for a torch.utils.data.Dataloader """
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.indices = list(range(len(dataset)))
        self.dataset = dataset

    def __iter__(self):
        """ Iterate through the dataloader by wrapping the shuffled indices """
        random.shuffle(self.indices)
        while True:
            for i in self.indices:
                yield i % len(self.indices)

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

        # Quality checks for datasets
        if len(self.datasets) != len(self.batch_sizes):
            raise ValueError("The number of datasets does not equal the number of batch sizes. Please ammend appropriately")
        for i, (ds, bs) in enumerate(zip(self.datasets, self.batch_sizes)):
            if len(ds) < bs:
                raise ValueError(f"Dataset {i+1} has fewer elements than its specified batch size. Please select a batch size smaller than {bs} and try again.")

        # Quality checks for batch sizes
        if any([i < 0 for i in batch_sizes]):
            raise ValueError(f"One of the batch sizes has a negative element. Please ensure that all batch sizes are positive integers.")
        elif any([type(i) != int for i in batch_sizes]):
            raise ValueError(f"One or more of the batch sizes is not an integer. Please ensure that all batch sizes are positive integers.")

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
    
class DistributedInfiniteSampler(Sampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int]=None, shuffle: bool=True, seed: int=0):
        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Requires distribbuted package to be available")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Requiires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)) # Make sure that everything is evenly divisible by the number of replicas
        self.total_size = self.num_samples * self.num_replicas
        self.indices = list(range(len(self.dataset)))
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # Shuffle the indices before handling
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # Remove tail of the indices to ensure it is evenly divisible
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = self.indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        while True:
            yield from indices

    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
    
class DistributedDataloaderBalancer:
    def __init__(self, *datasets, batch_sizes: Iterable[int, ], num_replicas=None, rank=None):
        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()
        self.dl_lengths = []
        self.dataloaders = []

        if len(self.datasets) != len(self.batch_sizes):
            raise ValueError("The number of datasets does not equal the number of batch sizes. Please ammend appropriately")
        for i, (ds, bs) in enumerate(zip(self.datasets, self.batch_sizes)):
            if len(ds) < bs:
                raise ValueError(f"Dataset {i+1} has fewer elements than its specified batch size. Please select a batch size smaller than {bs} and try again.")
            self.dl_lengths.append(len(ds) // bs)

        self.max_idx = np.argmax(self.dl_lengths)

    def balance_loaders(self):
        for i, (ds, bs) in enumerate(zip(self.datasets, self.batch_sizes)):
            # For the longest dataloader, create a loader that iterates over everything once
            if i == self.max_idx:
                # Create a distributed sampler with DistributedSampler
                self.sampler = DistributedSampler(
                    dataset=ds,
                    num_replicas=self.num_replicas,
                    rank=self.rank,
                    shuffle=True,
                    drop_last=True
                )
                # Instantiate the dataloader
                dataloader = DataLoader(
                    dataset=ds,
                    batch_size=bs,
                    shuffle=False, 
                    sampler=self.sampler,
                    drop_last=True
                )
                # Append the dataloader
                self.dataloaders.append(dataloader)

            else: # Wrap the second dataloader with the DistributedInfiniteSampler
                self.infinite_sampler = DistributedInfiniteSampler(
                    dataset=ds,
                    num_replicas=self.num_replicas,
                    rank=self.rank,
                    shuffle=True
                )
                # Instantiate the dataloader
                dataloader = DataLoader(
                    dataset=ds,
                    batch_size=bs,
                    shuffle=False,
                    sampler=self.infinite_sampler,
                    drop_last=True
                )
                # Append the dataloader
                self.dataloaders.append(dataloader)
         
        return self.dataloaders, self.sampler.num_samples

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