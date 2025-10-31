"""
src.metrics.py
Torchmetrics for image prediction
BoMeyering 2025
"""

import random
from collections import deque
import numpy as np
import logging
import torch
from torchmetrics import MetricCollection
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore, HausdorffDistance
from typing import Union, List, Optional, Sequence, Iterable
from numbers import Real
from abc import ABC, abstractmethod

logger = logging.getLogger()

class Meter(ABC):
    """Abstract base class for metric accumulators."""

    def __init__(self) -> None:
        self._values = []

    @abstractmethod
    def update(self, val: Real, *args, **kwargs) -> None:
        """Update the meter with a new value (and optional count)."""
        ...

    def reset(self) -> None:
        """ Reset internal state """
        self._values.clear()

    @property
    def mean(self) -> Optional[Real]:
        """ Mean of tracked values (or None if empty) """
        if not self._values:
            return None
        return float(sum(self._values) / len(self._values))

    @property
    def min(self) -> Optional[Real]:
        """Minimum of tracked values (or None if empty)."""
        if not self._values:
            return None
        return min(self._values)

    @property
    def max(self) -> Optional[Real]:
        """Maximum of tracked values (or None if empty)."""
        if not self._values:
            return None
        return max(self._values)

    @property
    def values(self) -> Sequence[Real]:
        """All tracked values."""
        return self._values

class ValueMeter(Meter):
    """ A class to handle any numerical values """

    def __init__(self):
        super().__init__()

    def update(self, val: Real, n: int=1):
        """Append `val` to the list `n` times."""
        if not isinstance(val, (float, int)):
            raise ValueError(
                f"Argument 'val' must be a numeric data type; got {type(val)} instead."
            )
        if not isinstance(n, int) or n < 1:
            raise ValueError(
                f"Argument 'n' must be a positive integer; got {type(n)} instead."
            )
        self._values.extend([val]*n)
    
    def __str__(self):
        """ Implement str format """
        return f"Mean: {self.mean} - Min: {self.min} - Max: {self.max}"
    
    def __repr__(self):
        """ Implement object representation """
        if len(self._values) <= 10:
            return f"ValueMeter(values={self._values}, len={len(self._values)})"
        else:
            first = ", ".join(map(str, self._values[:3]))
            last = ", ".join(map(str, self._values[-3:]))
            return f"ValueMeter(values=[{first}, ..., {last}], len={len(self._values)})"
    
class RunningAvgMeter(Meter):
    def __init__(self, window_length: int=10):
        """Initialize the RunningAvgMeter

        parameters:
        -----------
            window_length : int
                The number of numeric elements to include in the running average. Defaults to 10.        
        """
        super().__init__()
        if not isinstance(window_length, (float, int)):
            raise ValueError(
                f"Argument 'window_length' must be a scalar numeric data type"
            )
        # Clip window_length between 2 and 100
        wl = int(window_length)
        wl = max(2, min(wl, 100))
        # Create the deque
        self._values = deque(maxlen=wl)

    def update(self, val: Real, *args):
        """ Update the deque with a new value """
        if not isinstance(val, (float, int)):
            raise ValueError(
                f"Argument 'val' must be a numeric data type; got {type(val)} instead."
            )
        self._values.append(val)

    def __str__(self):
        """ Implement str format """
        return f"Mean: {self.mean} - Min: {self.min} - Max: {self.max}"
    
    def __repr__(self):
        """ Implement object representation """
        if len(self._values) <= 10:
            return f"RunningAvgMeter(values={self._values}, len={len(self._values)})"
        else:
            first = ", ".join(map(str, list(self._values)[:3]))
            last = ", ".join(map(str, list(self._values)[-3:]))
            return f"RunningAvgMeter(values=[{first}, ..., {last}], len={len(self._values)})"

class MeterSet:
    """ MeterSet manages a group of abstract Meter instances """
    def __init__(self, meter_dict: dict[str, Meter]):
        
        if not isinstance(meter_dict, dict):
            raise ValueError(
                f"Parameter 'meter_dict' must be a dictionary with meter names as keys and values of type abstract class Meter"
            )
        
        for k, v in meter_dict.items():
            if not isinstance(v, (Meter, ValueMeter, RunningAvgMeter)):
                raise ValueError(
                    f"All values in parameter 'meter_dict' must be of base type Meter"
                )
        
        self.meters = meter_dict
    
    def __getitem__(self, name: str):
        try:
            return self.meters[name]
        except KeyError:
            raise KeyError(f"No meter named '{name}'. Existing: {list(self.meters)}")
        
    def _add_one_meter(self, name: str, meter_type: str, **kwargs):
        """ Add another meter to the MeterSet """
        if name in self.meters.keys():
            raise KeyError(
                f"A meter named {name} is already present in the MeterSet. Please choose a different key."
            )
        wl = kwargs['window_length'] if 'window_length' in kwargs else 10

        if meter_type == 'value':
            meter = ValueMeter()
        elif meter_type == 'running_avg':
            meter = RunningAvgMeter(window_length=wl)
        else:
            raise ValueError(
                f"Parameter 'meter_type' must be one of ['value', 'running_avg']. Got {type(meter_type)} instead."
            )
        
        self.meters[name] = meter

    def _update_one_meter(self, name: str, val: float, n: int = 1):
        # assumes ValueMeter.update(value, n=1) exists
        self.meters[name].update(val, n)
    
    def _delete_one_meter(self, name: str):
        """ Delete one meter from the set """

        if name in self.meters.keys():
            del self.meters[name]
        else:
            logger.warning(f"Key {name} not found in self.meters.keys().")

    def update(self, val_dict: dict):
        if not isinstance(val_dict, dict):
            raise ValueError("'val_dict' must be a valid dictionary")
        for k, v in val_dict.items():
            self._update_one_meter(name=k, val=v.get('val'), n=v.get('n', 1))

    def reset(self, name: Optional[str] = None):
        """ Reset all Meters in the set """
        if name is not None:
            self[name].reset()
        else:
            for meter in self.meters.values():
                meter.reset()

    def clear(self):
        """ Remove all Meters from the MeterSet """
        self.meters = {}

    def values(self, name: Optional[str] = None, postfix: str = ""):
        if name is not None:
            return {f"{name}_values{('_' + postfix) if postfix else ''}": self[name].values}
        return {f"{n}_values{('_' + postfix) if postfix else ''}": m.values for n, m in self.meters.items()}

    def mins(self, name: Optional[str] = None, postfix: str = ""):
        if name is not None:
            return {f"{name}_min{('_' + postfix) if postfix else ''}": self[name].min}
        return {f"{n}_min{('_' + postfix) if postfix else ''}": m.min for n, m in self.meters.items()}

    def maxs(self, name: Optional[str] = None, postfix: str = ""):
        if name is not None:
            return {f"{name}_max{('_' + postfix) if postfix else ''}": self[name].max}
        return {f"{n}_max{('_' + postfix) if postfix else ''}": m.max for n, m in self.meters.items()}

    def means(self, name: Optional[str] = None, postfix: str = ""):
        if name is not None:
            return {f"{name}_mean{('_' + postfix) if postfix else ''}": self[name].mean}
        return {f"{n}_mean{('_' + postfix) if postfix else ''}": m.mean for n, m in self.meters.items()}

    def __str__(self):
        lines = [f"{name}: {vm}" for name, vm in sorted(self.meters.items())]
        return "\n".join(lines)

    def __repr__(self):
        pairs = ", ".join(f"{name}: {repr(vm)}" for name, vm in sorted(self.meters.items()))
        return f"ValueMeterSet(meters={{ {pairs} }})"

class MetricLogger:
    """ Wrapper for torchmetrics.MetricCollection """
    def __init__(self, name: str, num_classes: int, device: str):
        """
        Initialize the MetricLogger

        Parameters:
        -----------
            num_classes : int
                The total number of classes to track
            device : torch.device
                The computational device for metric calculation
        """
        self.name = name
        self.avg_metrics = MetricCollection(
            [
                MeanIoU(num_classes=num_classes, per_class=False, input_format='index').to(device),
                GeneralizedDiceScore(num_classes=num_classes, per_class=False, input_format='index').to(device),
                # HausdorffDistance(num_classes=num_classes, input_format='index').to(device)
            ]
        )

        self.mc_metrics = MetricCollection(
            [
                MeanIoU(num_classes=num_classes, per_class=True, input_format='index').to(device),
                GeneralizedDiceScore(num_classes=num_classes, per_class=True, input_format='index').to(device),
            ]
        )

        self.results = {
            'avg': {},
            'mc': {}
        }
    
    def update(self, preds: torch.tensor, targets: torch.tensor, verbose: bool=False):
        # update avg metrics
        self.avg_metrics.update(preds, targets)
        
        # update multiclass metrics
        self.mc_metrics.update(preds, targets)
        
    def compute(self):
        try:
            self.results['avg'] = self.avg_metrics.compute()
            self.results['mc'] = self.mc_metrics.compute()
        except Exception as e:
            logger.error(f"Encountered error when computing metrics. Error: {e}")
            self.results['avg'], self.results['mc'] = None, None
    
    def __str__(self) -> str:
        """
        Nicely format the metric results for printing.
        """
        lines = [self.name + ":"]

        def _format_dict(name: str, d: dict):
            if d is None or len(d) == 0:
                return f"   {name}: None"
            out = [f"   {name}:"]
            for k, v in d.items():
                # handle tensor outputs from torchmetrics
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        v = f"{v.item():.5f}"
                    else:
                        v = [round(x, 5) for x in v.flatten().tolist()]
                if isinstance(v, float):
                    out.append(f"       {k:<20s}: {v:.4f}")
                else:
                    out.append(f"       {k:<20s}: {v}")
            return "\n".join(out)
        
        lines.append(_format_dict("Average", self.results.get("avg")))
        lines.append(_format_dict("Multi-class", self.results.get("mc")))

        return "\n" + "\n".join(lines)

    def reset(self):
        """ Rest Metric Collections """
        self.avg_metrics.reset()
        self.mc_metrics.reset()
