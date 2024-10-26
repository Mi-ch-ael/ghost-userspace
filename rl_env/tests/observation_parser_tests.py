import unittest
import os
import sys
import numpy as np
from gymnasium import spaces
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from environment.observation_parser import LnCapObservationParser

class LnCapObservationParserTests(unittest.TestCase):
    maxDiff = None

    def test_reshape_runqueue_length_equal_to_cutoff(self):
        time_ln_cap = 1
        vsize_ln_cap = 1
        runqueue_cutoff_length = 3
        parser = LnCapObservationParser(spaces.Dict({
            "callback_type": spaces.Discrete(8),
            "task_metrics": spaces.Dict({
                "run_state": spaces.Discrete(4),
                "cpu_num": spaces.Discrete(6),
                "preempted": spaces.Discrete(2),
                "utime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "stime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "guest_time": spaces.Box(0, time_ln_cap, shape=(1,)),
                "vsize": spaces.Box(0, vsize_ln_cap, shape=(1,)),
            }),
            "runqueue": spaces.Tuple([spaces.Dict({
                "run_state": spaces.Discrete(4),
                "cpu_num": spaces.Discrete(6),
                "preempted": spaces.Discrete(2),
                "utime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "stime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "guest_time": spaces.Box(0, time_ln_cap, shape=(1,)),
                "vsize": spaces.Box(0, vsize_ln_cap, shape=(1,)),
            })] * runqueue_cutoff_length)
        }), runqueue_cutoff_length, time_ln_cap, vsize_ln_cap)
        metrics = [
            1, 
            0, 5, 0, 1515215, 0, 0, 31512800,
            0, 5, 0, 1515215, 0, 0, 31512800,
            2, 1, 1, 778502144, 226203100, 112112112, 595121352,
            0, 1, 0, 0, 0, 0, 828858368, 
        ]

        reshaped_metrics = parser._reshape(metrics)
        self.assertEqual(
            reshaped_metrics,
            {
                "callback_type": 1,
                "task_metrics": {
                    "run_state": 0,
                    "cpu_num": 5,
                    "preempted": 0,
                    "utime": 1515215,
                    "stime": 0,
                    "guest_time": 0,
                    "vsize": 31512800,
                },
                "runqueue": (
                    {
                        "run_state": 0,
                        "cpu_num": 5,
                        "preempted": 0,
                        "utime": 1515215,
                        "stime": 0,
                        "guest_time": 0,
                        "vsize": 31512800,
                    },
                    {
                        "run_state": 2,
                        "cpu_num": 1,
                        "preempted": 1,
                        "utime": 778502144,
                        "stime": 226203100,
                        "guest_time": 112112112,
                        "vsize": 595121352,
                    },
                    {
                        "run_state": 0,
                        "cpu_num": 1,
                        "preempted": 0,
                        "utime": 0,
                        "stime": 0,
                        "guest_time": 0,
                        "vsize": 828858368,
                    },
                )
            }
        )

    def test_reshape_runqueue_length_more_than_cutoff(self):
        time_ln_cap = 1
        vsize_ln_cap = 1
        runqueue_cutoff_length = 1
        space = spaces.Dict({
            "callback_type": spaces.Discrete(8),
            "task_metrics": spaces.Dict({
                "run_state": spaces.Discrete(4),
                "cpu_num": spaces.Discrete(6),
                "preempted": spaces.Discrete(2),
                "utime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "stime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "guest_time": spaces.Box(0, time_ln_cap, shape=(1,)),
                "vsize": spaces.Box(0, vsize_ln_cap, shape=(1,)),
            }),
            "runqueue": spaces.Tuple([spaces.Dict({
                "run_state": spaces.Discrete(4),
                "cpu_num": spaces.Discrete(6),
                "preempted": spaces.Discrete(2),
                "utime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "stime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "guest_time": spaces.Box(0, time_ln_cap, shape=(1,)),
                "vsize": spaces.Box(0, vsize_ln_cap, shape=(1,)),
            })] * runqueue_cutoff_length)
        })
        parser = LnCapObservationParser(space, runqueue_cutoff_length, time_ln_cap, vsize_ln_cap)
        metrics = [
            1, 
            0, 1, 2, 3, 4, 5, 6,
            0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13,
        ]

        reshaped_metrics = parser._reshape(metrics)
        self.assertEqual(
            reshaped_metrics,
            {
                "callback_type": 1,
                "task_metrics": {
                    "run_state": 0,
                    "cpu_num": 1,
                    "preempted": 2,
                    "utime": 3,
                    "stime": 4,
                    "guest_time": 5,
                    "vsize": 6,
                },
                "runqueue": (
                    {
                        "run_state": 0,
                        "cpu_num": 1,
                        "preempted": 2,
                        "utime": 3,
                        "stime": 4,
                        "guest_time": 5,
                        "vsize": 6,
                    },
                )
            }
        )

    def test_reshape_runqueue_length_less_than_cutoff(self):
        time_ln_cap = 1
        vsize_ln_cap = 1
        runqueue_cutoff_length = 2
        space = spaces.Dict({
            "callback_type": spaces.Discrete(8),
            "task_metrics": spaces.Dict({
                "run_state": spaces.Discrete(4),
                "cpu_num": spaces.Discrete(6),
                "preempted": spaces.Discrete(2),
                "utime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "stime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "guest_time": spaces.Box(0, time_ln_cap, shape=(1,)),
                "vsize": spaces.Box(0, vsize_ln_cap, shape=(1,)),
            }),
            "runqueue": spaces.Tuple([spaces.Dict({
                "run_state": spaces.Discrete(4),
                "cpu_num": spaces.Discrete(6),
                "preempted": spaces.Discrete(2),
                "utime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "stime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "guest_time": spaces.Box(0, time_ln_cap, shape=(1,)),
                "vsize": spaces.Box(0, vsize_ln_cap, shape=(1,)),
            })] * runqueue_cutoff_length)
        })
        parser = LnCapObservationParser(
            space=space,
            runqueue_cutoff_length=runqueue_cutoff_length, 
            time_cap=time_ln_cap,
            vsize_cap=vsize_ln_cap,
        )
        metrics = [
            1, 
            0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13,
        ]

        reshaped_metrics = parser._reshape(metrics)
        self.assertEqual(
            reshaped_metrics,
            {
                "callback_type": 1,
                "task_metrics": {
                    "run_state": 0,
                    "cpu_num": 1,
                    "preempted": 2,
                    "utime": 3,
                    "stime": 4,
                    "guest_time": 5,
                    "vsize": 6,
                },
                "runqueue": (
                    {
                        "run_state": 7,
                        "cpu_num": 8,
                        "preempted": 9,
                        "utime": 10,
                        "stime": 11,
                        "guest_time": 12,
                        "vsize": 13,
                    },
                    {
                        "run_state": 0,
                        "cpu_num": 0,
                        "preempted": 0,
                        "utime": 0,
                        "stime": 0,
                        "guest_time": 0,
                        "vsize": 0,
                    },
                )
            }
        )

    def test_transform(self):
        time_ln_cap = 4
        vsize_ln_cap = 3
        runqueue_cutoff_length = 1
        space = spaces.Dict({
            "callback_type": spaces.Discrete(8),
            "task_metrics": spaces.Dict({
                "run_state": spaces.Discrete(4),
                "cpu_num": spaces.Discrete(6),
                "preempted": spaces.Discrete(2),
                "utime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "stime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "guest_time": spaces.Box(0, time_ln_cap, shape=(1,)),
                "vsize": spaces.Box(0, vsize_ln_cap, shape=(1,)),
            }),
            "runqueue": spaces.Tuple([spaces.Dict({
                "run_state": spaces.Discrete(4),
                "cpu_num": spaces.Discrete(6),
                "preempted": spaces.Discrete(2),
                "utime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "stime": spaces.Box(0, time_ln_cap, shape=(1,)),
                "guest_time": spaces.Box(0, time_ln_cap, shape=(1,)),
                "vsize": spaces.Box(0, vsize_ln_cap, shape=(1,)),
            })] * runqueue_cutoff_length)
        })
        parser = LnCapObservationParser(
            space=space,
            runqueue_cutoff_length=runqueue_cutoff_length,
            time_cap=time_ln_cap,
            vsize_cap=vsize_ln_cap,
        )
        reshaped_metrics = {
            "callback_type": np.int64(2),
            "task_metrics": {
                "run_state": np.int64(11),
                "cpu_num": np.int64(22),
                "preempted": np.int64(33),
                "utime": np.int64(44),
                "stime": np.int64(55),
                "guest_time": np.int64(66),
                "vsize": np.int64(77),
            },
            "runqueue": (
                {
                    "run_state": np.int64(77),
                    "cpu_num": np.int64(66),
                    "preempted": np.int64(55),
                    "utime": np.int64(55),
                    "stime": np.int64(0),
                    "guest_time": np.int64(33),
                    "vsize": np.int64(11),
                },
                {
                    "run_state": np.int64(0),
                    "cpu_num": np.int64(5),
                    "preempted": np.int64(1),
                    "utime": np.int64(1),
                    "stime": np.int64(100),
                    "guest_time": np.int64(3),
                    "vsize": np.int64(0),
                }
            )
        }

        transformed_metrics = parser._transform(reshaped_metrics)
        expected_result = {
            "callback_type": np.int64(2),
            "task_metrics": {
                "run_state": np.int64(11),
                "cpu_num": np.int64(22),
                "preempted": np.int64(33),
                "utime": np.array([3.7841896], dtype=np.float32),
                "stime": np.array([4.0], dtype=np.float32),
                "guest_time": np.array([4.0], dtype=np.float32),
                "vsize": np.array([3.0], dtype=np.float32),
            },
            "runqueue": (
                {
                    "run_state": np.int64(77),
                    "cpu_num": np.int64(66),
                    "preempted": np.int64(55),
                    "utime": np.array([4.0], dtype=np.float32),
                    "stime": np.array([0.0], dtype=np.float32),
                    "guest_time": np.array([3.4965076], dtype=np.float32),
                    "vsize": np.array([2.3978953], dtype=np.float32),
                },
                {
                    "run_state": np.int64(0),
                    "cpu_num": np.int64(5),
                    "preempted": np.int64(1),
                    "utime": np.array([0.0], dtype=np.float32),
                    "stime": np.array([4.0], dtype=np.float32),
                    "guest_time": np.array([1.0986123], dtype=np.float32),
                    "vsize": np.array([0.0], dtype=np.float32),
                }
            )
        }
        self.assertEqual(transformed_metrics["callback_type"], expected_result["callback_type"])
        self.assertEqual(transformed_metrics["callback_type"], reshaped_metrics["callback_type"])
        self.assertEqual(
            transformed_metrics["task_metrics"]["run_state"],
            expected_result["task_metrics"]["run_state"],
        )
        self.assertEqual(
            transformed_metrics["task_metrics"]["cpu_num"],
            expected_result["task_metrics"]["cpu_num"],
        )
        self.assertEqual(
            transformed_metrics["task_metrics"]["preempted"],
            expected_result["task_metrics"]["preempted"],
        )
        self.assertAlmostEqual(
            transformed_metrics["task_metrics"]["utime"][0],
            expected_result["task_metrics"]["utime"][0],
        )
        self.assertAlmostEqual(
            transformed_metrics["task_metrics"]["stime"][0],
            expected_result["task_metrics"]["stime"][0],
        )
        self.assertAlmostEqual(
            transformed_metrics["task_metrics"]["guest_time"][0],
            expected_result["task_metrics"]["guest_time"][0],
        )
        self.assertAlmostEqual(
            transformed_metrics["task_metrics"]["vsize"][0],
            expected_result["task_metrics"]["vsize"][0],
        )
        self.assertEqual(len(transformed_metrics["runqueue"]), len(expected_result["runqueue"]))
        for i in range(len(expected_result["runqueue"])):
            self.assertEqual(
                transformed_metrics["runqueue"][i]["run_state"],
                expected_result["runqueue"][i]["run_state"],
            )
            self.assertEqual(
                transformed_metrics["runqueue"][i]["cpu_num"],
                expected_result["runqueue"][i]["cpu_num"],
            )
            self.assertEqual(
                transformed_metrics["runqueue"][i]["preempted"],
                expected_result["runqueue"][i]["preempted"],
            )
            self.assertAlmostEqual(
                transformed_metrics["runqueue"][i]["utime"][0],
                expected_result["runqueue"][i]["utime"][0],
            )
            self.assertAlmostEqual(
                transformed_metrics["runqueue"][i]["stime"][0],
                expected_result["runqueue"][i]["stime"][0],
            )
            self.assertAlmostEqual(
                transformed_metrics["runqueue"][i]["guest_time"][0],
                expected_result["runqueue"][i]["guest_time"][0],
            )
            self.assertAlmostEqual(
                transformed_metrics["runqueue"][i]["vsize"][0],
                expected_result["runqueue"][i]["vsize"][0],
            )


if __name__ == "__main__":
    unittest.main()
