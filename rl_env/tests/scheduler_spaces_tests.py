import unittest
import os
import sys
import numpy as np
from gymnasium import spaces
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from environment import scheduler_spaces

class SchedulerSpacesTests(unittest.TestCase):
    def test_discrete(self):
        for space in [spaces.Discrete(1), spaces.Discrete(10), spaces.Discrete(10, seed=42)]:
            with self.subTest(space=space):
                zeroed_sample = scheduler_spaces.generate_zeroed_sample(space)
                self.assertEqual(zeroed_sample, 0)
    
    def test_box(self):
        space = spaces.Box(low=0, high=7, shape=(1,), dtype=np.float32)
        zeroed_sample = scheduler_spaces.generate_zeroed_sample(space)
        np.testing.assert_array_equal(zeroed_sample, np.zeros((1,), np.float32))
        
    def test_tuple(self):
        test_cases = [
            {
                "space": spaces.Tuple((
                    spaces.Discrete(42),
                    spaces.Discrete(2)
                )),
                "expected": (
                    0,
                    0
                )
            },
            {
                "space": spaces.Tuple((
                    spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    spaces.Discrete(10)
                )),
                "expected": (
                    np.zeros((1,), dtype=np.float32),
                    0
                )
            }
        ]

        for case in test_cases:
            with self.subTest(case=case):
                space = case["space"]
                expected = case["expected"]
                result = scheduler_spaces.generate_zeroed_sample(space)
                for res, exp in zip(result, expected):
                    if isinstance(res, np.ndarray):
                        np.testing.assert_array_equal(res, exp)
                    else:
                        self.assertEqual(res, exp)

    def test_nested_dicts(self):
        test_cases = [
            {
                "space": spaces.Dict({
                    "sensor": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                    "discrete": spaces.Discrete(2)
                }),
                "expected": {
                    "sensor": np.zeros((1,), dtype=np.float32),
                    "discrete": 0
                }
            },
            {
                "space": spaces.Dict({
                    "position": spaces.Dict({
                        "x": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                        "y": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                    }),
                    "velocity_x": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                }),
                "expected": {
                    "position": {
                        "x": np.zeros((1,), dtype=np.float32),
                        "y": np.zeros((1,), dtype=np.float32)
                    },
                    "velocity_x": np.zeros((1,), dtype=np.float32)
                }
            },
            {
                "space": spaces.Dict({
                    "list": spaces.Tuple((
                        spaces.Discrete(3),
                        spaces.Box(low=-5, high=5, shape=(1,), dtype=np.float32)
                    ))
                }),
                "expected": {
                    "list": (
                        0,
                        np.zeros((1,), dtype=np.float32)
                    )
                }
            }
        ]

        for case in test_cases:
            with self.subTest(case=case):
                space = case["space"]
                expected = case["expected"]
                result = scheduler_spaces.generate_zeroed_sample(space)
                self.compare_dicts(result, expected)

    def compare_dicts(self, result, expected):
        for key in expected:
            res = result[key]
            exp = expected[key]
            if isinstance(exp, dict):
                self.compare_dicts(res, exp)
            elif isinstance(exp, tuple):
                for res_item, exp_item in zip(res, exp):
                    if isinstance(exp_item, np.ndarray):
                        np.testing.assert_array_equal(res_item, exp_item)
                    else:
                        self.assertEqual(res_item, exp_item)
            elif isinstance(exp, np.ndarray):
                np.testing.assert_array_equal(res, exp)
            else:
                self.assertEqual(res, exp)
    

if __name__ == "__main__":
    unittest.main()