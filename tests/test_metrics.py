"""
tests.test_metrics.py
src.metrics.py unit tests
BoMeyering 2025
"""

import unittest
import random
import torch
import numpy as np
from copy import deepcopy
from collections import deque, defaultdict
from omegaconf import OmegaConf
from glob import glob
from wonderwords import RandomWord
from pathlib import Path
from random import randint
from torch import Tensor
from src.metrics import ValueMeter, MeterSet, RunningAvgMeter, Meter, MetricLogger

class TestValueMeter(unittest.TestCase):
    """ Unit Tests for ValueMeter """
    def setUp(self):
        """ Setup the test case """
        self.vm = ValueMeter()

    def test_init(self):
        """ Test initialization """
        self.assertIsInstance(self.vm, ValueMeter)
        self.assertEqual(len(self.vm._values), 0)

    def test_update(self):
        """ Test the update method """
        # Create empty list for comparison
        shadow = []
        
        # Create values
        val1 = random.random()
        val2 = random.random()
        val3 = random.random()
        # Set integer values
        n2 = random.randint(1, 5)
        n3 = random.randint(1, 5)

        # Update with val1 and default n
        self.vm.update(val1)
        shadow.append(val1)
        self.assertEqual(self.vm._values, shadow)
        self.assertEqual(len(self.vm._values), len(shadow))

        # Update with val2, n2
        self.vm.update(val2, n2)
        shadow.extend([val2]*n2)
        self.assertEqual(self.vm._values, shadow)
        self.assertEqual(len(self.vm._values), len(shadow))

        # Update with val3, n3
        self.vm.update(val3, n3)
        shadow.extend([val3]*n3)
        self.assertEqual(self.vm._values, shadow)
        self.assertEqual(len(self.vm._values), len(shadow))
        self.assertEqual(len(self.vm._values), 1+n2+n3)

        # Test for ValueError
        self.vm.reset()
        with self.assertRaises(ValueError):
            self.vm.update('34')

        with self.assertRaises(ValueError):
            self.vm.update(34, '5')

    def test_reset(self):
        """ Test the reset method """
        val1 = random.random()
        n1 = random.randint(3, 7)

        self.vm.update(val1, n1)
        self.vm.reset()

        self.assertEqual(len(self.vm._values), 0)

    def test_mean_property(self):
        """ Test the mean property """
        shadow = [1, 5.4, 7, 8.2]

        for val in shadow:
            self.vm.update(val)

        self.assertEqual(self.vm.mean, 5.4) # Hardcoded test
        
        # Dynamic test
        shadow = []
        self.vm.reset()
        for i in range(20):
            val = random.random()
            n = random.randint(1, 4)

            self.vm.update(val, n)
            shadow.extend([val]*n)
        
        shadow_mean = sum(shadow) / len(shadow)
        self.assertEqual(self.vm.mean, shadow_mean)

        # Null test
        self.vm.reset()
        self.assertEqual(self.vm.mean, None)

    def test_min_max_properties(self):
        """ Test the min-max properties """

        shadow = [0.34, 9, 8, 2.1]

        for val in shadow:
            self.vm.update(val)

        self.assertEqual(self.vm.min, 0.34)
        self.assertEqual(self.vm.max, 9)

    def test_values_property(self):
        """ Test that the values property returns self._values """

        for _ in range(20):
            val = random.random()
            n = random.randint(1, 4)

            self.vm.update(val, n)

        self.assertEqual(self.vm.values, self.vm._values)

class TestRunningAvgMeter(unittest.TestCase):
    """ Unit Tests for RunningAvgMeter """
    def setUp(self):
        """ Setup the test case """
        self.ram = RunningAvgMeter()

    def test_init(self):
        """ Test initialization """
        self.assertIsInstance(self.ram, RunningAvgMeter)
        self.assertEqual(len(self.ram._values), 0)
        self.assertEqual(self.ram._values.maxlen, 10)

    def test_update(self):
        """ Test the update() method """
        # Create an empty deque for comparison
        shadow = deque(maxlen=10)

        for _ in range(20):
            val = random.random()
            self.ram.update(val)
            shadow.append(val)
        
        self.assertEqual(self.ram._values, shadow)

        # Test for ValueError
        self.ram.reset()
        with self.assertRaises(ValueError):
            self.ram.update(val='random_string')

    def test_reset(self):
        """ Test the reset method """
        val = random.random()
        self.ram.update(val)
        self.ram.reset()

        self.assertEqual(len(self.ram._values), 0)

    def test_window_length(self):
        """ Test window_length argument """
        wl = random.randint(2, 100)
        self.ram = RunningAvgMeter(window_length=wl)

        # Update wl + 10 values
        for _ in range(wl+10):
            val = random.random()
            self.ram.update(val)
        
        self.assertEqual(self.ram._values.maxlen, wl)
        self.assertEqual(len(self.ram._values), wl)

    def test_mean_property(self):
        """ Test the mean property """
        # Hardcoded test
        vals = [1, 5.4, 6, 8.1, 3.2, 9.0, 2.0, 7.4] # Length 8
        self.ram = RunningAvgMeter(window_length=6) # Set window_length=6

        for val in vals:
            self.ram.update(val)
        
        self.assertEqual(self.ram.mean, 5.95) # Hardcoded test

        # Dynamic test
        shadow = deque(maxlen=6)
        self.ram.reset()

        for i in range(20):
            val = random.random()
            self.ram.update(val)
            shadow.append(val)

        shadow_mean = sum(shadow) / len(shadow)
        self.assertEqual(self.ram.mean, shadow_mean)

        # Null test
        self.ram.reset()
        self.assertEqual(self.ram.mean, None)

    def test_min_max_property(self):
        """ Test the min-max properties """
        # Reset
        self.ram.reset()

        vals = [0.34, 9, 8, 2.1]

        for val in vals:
            self.ram.update(val)

        self.assertEqual(self.ram.min, 0.34)
        self.assertEqual(self.ram.max, 9)

    def test_values_property(self):
        """ Test that the values propery returns self._values """

        # Reset the meter
        self.ram = RunningAvgMeter(window_length=20)

        for _ in range(20):
            val = random.random()
            self.ram.update(val)
        
        self.assertEqual(self.ram.values, self.ram._values)
    
    def test_wl_clipping(self):
        """ Test that window_length gets clipped """

        # Reset the meter
        for _ in range(20):
            threshold = random.uniform(0, 1)
            if threshold > 0.5:
                wl = random.randint(101, 200)
                self.ram = RunningAvgMeter(window_length=wl)
                self.assertEqual(self.ram._values.maxlen, 100)
            elif threshold < 0.5:
                wl = random.randint(-100, 1)
                self.ram = RunningAvgMeter(window_length=wl)
                self.assertEqual(self.ram._values.maxlen, 2)

class TestMeterSet(unittest.TestCase):
    """ Unit tests for MeterSet """
    def setUp(self):
        """ Setup the test case """
        meter_dict = {
            'vm1': ValueMeter(),
            'vm2': ValueMeter(),
            'ram1': RunningAvgMeter(window_length=20),
            'ram2': RunningAvgMeter()
        }
        self.ms = MeterSet(meter_dict)

    def test_init(self):
        """ Test the MeterSet __init__ method """
        self.assertIsInstance(self.ms, MeterSet)
        
        for key in ['vm1', 'vm2', 'ram1', 'ram2']:
            self.assertIn(key, self.ms.meters.keys())

            meter = self.ms.meters.get(key, None)
            self.assertIsInstance(meter, Meter)
            self.assertEqual(len(meter.values), 0)

    def test_update(self):
        """ Test the update() method """
        # Clear out any previous states
        self.ms.reset()
        
        for _ in range(30):

            # Create dummy values
            val1 = random.random()+10
            val2 = random.random()+10
            n1 = random.randint(1, 4)
            n2 = random.randint(1, 4)

            # Create an update dictionary
            update_dict = {
                'vm1': {'val': val1, 'n': n1},
                'vm2': {'val': val2, 'n': n2},
                'ram1': {'val': val1},
                'ram2': {'val': val2}
            }

            # Update state
            self.ms.update(update_dict)

        # Check non-null properties
        for name, meter in self.ms.meters.items():
            self.assertGreater(meter.mean, 0)
            self.assertGreater(meter.min, 0)
            self.assertGreater(meter.max, 0)
            self.assertIsNotNone(meter.values)

    def test_update_one_meter(self):
        """ Test the _update_one_meter() method """

        # Clear states
        self.ms.reset()

        # Create dummy values
        val1 = random.random()
        val2 = random.random()
        n1 = random.randint(1, 4)

        self.ms._update_one_meter('vm1', val1, n1)
        self.ms._update_one_meter('ram1', val2)

        vm1 = self.ms.meters.get('vm1')
        ram1 = self.ms.meters.get('ram1')

        self.assertAlmostEqual(vm1.mean, val1, places=12)
        self.assertEqual(len(vm1.values), n1)

        self.assertAlmostEqual(ram1.mean, val2, places=12)
        self.assertEqual(len(ram1.values), 1)

    def test_getitem(self):
        """ Test the indexed __getitem__() method """
        
        # Clear out any previous states
        self.ms.reset()

        for key in ['vm1', 'vm2', 'ram1', 'ram2']:
            meter = self.ms[key]
            self.assertIsInstance(meter, Meter)

        with self.assertRaises(KeyError):
            self.ms['nonexistent']

    def test_crud_ops(self):
        """
        Tests
         _add_one_meter(), 
         _delete_one_meter(), 
         reset(),
         clear() methods
         """
        # Reset state and make a deep copy
        self.ms.reset()
        self.ms_copy = deepcopy(self.ms)

        # Test _delete_one_meter()
        for name in self.ms_copy.meters.keys():
            self.ms._delete_one_meter(name)
            self.assertNotIn(name, self.ms.meters.keys())

        # Test _add_one_meter()
        names = [RandomWord().word for _ in range(4)]
        types = ['value', 'running_avg']

        for name in names:
            idx = random.choice([0, 1])
            self.ms._add_one_meter(name, meter_type=types[idx])
            val = random.random()
            n = random.randint(1, 4)
            self.ms._update_one_meter(name, val=val, n=n)

        # Test soft reset - clears out all values but keeps the meters
        self.ms.reset()

        for name, meter in self.ms.meters.items():
            self.assertTrue(len(meter.values)==0)
            self.assertIsNone(meter.mean)
            self.assertIsNone(meter.min)
            self.assertIsNone(meter.max)

        # Test hard reset clear() method - delete all meters
        self.ms.clear()
        self.assertDictEqual(self.ms.meters, {})

        # Restore original MeterSet
        self.ms = self.ms_copy

    def test_properties(self):
        """ Test the properties """

        # Reset the state
        self.ms.reset()

        meter_names = list(self.ms.meters.keys())
        
        for _ in range(20):
            update_dict = defaultdict()
            val = random.random()+10
            n = random.randint(1, 4)
            for name in meter_names:
                update_dict[name] = {'val': val, 'n': n}
            
            self.ms.update(update_dict)

        means = self.ms.means()
        mins = self.ms.mins()
        maxs = self.ms.maxs()
        values = self.ms.values()

        for k, v in means.items():
            self.assertIsNotNone(v)
            self.assertGreaterEqual(v, 10.0)
        
        for k, v in mins.items():
            self.assertIsNotNone(v)
            self.assertGreaterEqual(v, 10.0)

        for k, v in maxs.items():
            self.assertIsNotNone(v)
            self.assertGreaterEqual(v, 10.0)

        for k, v in values.items():
            self.assertIsNotNone(v)
            self.assertGreaterEqual(len(v), 10)


if __name__ == "__main__":
    pass