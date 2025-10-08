"""
src.metrics.py
Torchmetrics for image prediction
BoMeyering 2025
"""

import random
import numpy as np
import logging
import torch
from torchmetrics import MetricCollection
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore, HausdorffDistance
from typing import Union, List, Optional

logger = logging.getLogger()


class ValueMeter:
    """ A class to handle any numerical values """

    def __init__(self):
        self._values = []

    def update(self, val: float, n: int=1):
        """ Update the value list with new item(s) """
        self._values.extend([val]*n)
    
    def reset(self):
        """ Reset the value meter """
        self._values.clear()
    
    @property
    def mean(self):
        """ Retrieve the mean of the values"""
        if len(self._values) == 0:
            return
        return sum(self._values) / len(self._values)
    
    @property
    def min(self):
        """ Retrieve the minimum value """
        if len(self._values) == 0:
            return
        return min(self._values)
    
    @property
    def max(self):
        """ Retrieve the maximum value """
        if len(self._values) == 0:
            return
        return max(self._values)

    @property
    def values(self):
        """ Retrieve the entire list of values """
        return self._values
    
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
    
class ValueMeterSet:
    """ ValueMeterSet manages a group of ValueMeter instances """
    def __init__(self):
        self.meters = {}

    def __getitem__(self, name: str):
        try:
            return self.meters[name]
        except KeyError:
            raise KeyError(f"No meter named '{name}'. Existing: {list(self.meters)}")

    def _update_one_meter(self, name: str, val: float, n: int = 1):
        if name not in self.meters:
            self.meters[name] = ValueMeter()
        # assumes ValueMeter.update(value, n=1) exists
        self.meters[name].update(val, n)

    def update(self, val_dict: dict):
        if not isinstance(val_dict, dict):
            raise ValueError("'val_dict' must be a valid dictionary")
        for k, v in val_dict.items():
            self._update_one_meter(k, v.get('val'), v.get('n', 1))

    def reset(self, name: Optional[str] = None):
        if name is not None:
            self[name].reset()
        else:
            for meter in self.meters.values():
                meter.reset()

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
    def __init__(self, num_classes: int, device: str):
        """
        Initialize the MetricLogger

        Parameters:
        -----------
            num_classes : int
                The total number of classes to track
            device : torch.device
                The computational device for metric calculation
        """
        self.avg_metrics = MetricCollection(
            [
                MeanIoU(num_classes=num_classes, per_class=False, input_format='mixed').to(device),
                GeneralizedDiceScore(num_classes=num_classes, per_class=False, input_format='mixed').to(device),
                HausdorffDistance(num_classes=num_classes, input_format='mixed').to(device)
            ]
        )

        self.mc_metrics = MetricCollection(
            [
                MeanIoU(num_classes=num_classes, per_class=True, input_format='mixed').to(device),
                GeneralizedDiceScore(num_classes=num_classes, per_class=True, input_format='mixed').to(device),
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
        return self.results
    
    def __str__(self, type: str=None):
        
        if type=='avg':
            print(self.results['avg'])
        elif type=='mc':
            print(self.results['mc'])
        elif type=='both':
            print(self.results)

    def reset(self):
        """ Rest Metric Collections """
        self.avg_metrics.reset()
        self.mc_metrrics.reset()

if __name__ == '__main__':

    logits = torch.randn(10, 4, 10, 10)
    targets = torch.randint(0, 4, (10, 10, 10))

    ml = MetricLogger(num_classes=4, device='cpu')

    ml.update(preds=logits, targets=targets)

    ml.compute()

    print(ml.results)

