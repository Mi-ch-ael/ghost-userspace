import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from environment.observation_parser import LnCapObservationParser

class LnCapObservationParserTests(unittest.TestCase):
    def test_reshape_runqueue_length_equal_to_cutoff(self):
        parser = LnCapObservationParser(3, 1, 1)
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
                "runqueue": [
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
                ]
            }
        )

    def test_reshape_runqueue_length_more_than_cutoff(self):
        parser = LnCapObservationParser(1, 1, 1)
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
                "runqueue": [
                    {
                        "run_state": 0,
                        "cpu_num": 1,
                        "preempted": 2,
                        "utime": 3,
                        "stime": 4,
                        "guest_time": 5,
                        "vsize": 6,
                    },
                ]
            }
        )

    def test_reshape_runqueue_length_less_than_cutoff(self):
        parser = LnCapObservationParser(2, 1, 1)
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
                "runqueue": [
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
                ]
            }
        )

    def test_transform(self):
        parser = LnCapObservationParser(1, 4, 3)
        reshaped_metrics = {
            "callback_type": 2,
            "task_metrics": {
                "run_state": 11,
                "cpu_num": 22,
                "preempted": 33,
                "utime": 44,
                "stime": 55,
                "guest_time": 66,
                "vsize": 77,
            },
            "runqueue": [
                {
                    "run_state": 77,
                    "cpu_num": 66,
                    "preempted": 55,
                    "utime": 55,
                    "stime": 0,
                    "guest_time": 33,
                    "vsize": 11,
                },
                {
                    "run_state": 0,
                    "cpu_num": 5,
                    "preempted": 1,
                    "utime": 1,
                    "stime": 100,
                    "guest_time": 3,
                    "vsize": 0,
                }
            ]
        }

        transformed_metrics = parser._transform(reshaped_metrics)
        expected_result = {
            "callback_type": 2,
            "task_metrics": {
                "run_state": 11,
                "cpu_num": 22,
                "preempted": 33,
                "utime": 3.7841896,
                "stime": 4.0,
                "guest_time": 4.0,
                "vsize": 3.0,
            },
            "runqueue": [
                {
                    "run_state": 77,
                    "cpu_num": 66,
                    "preempted": 55,
                    "utime": 4.0,
                    "stime": 0.0,
                    "guest_time": 3.4965076,
                    "vsize": 2.3978953,
                },
                {
                    "run_state": 0,
                    "cpu_num": 5,
                    "preempted": 1,
                    "utime": 0.0,
                    "stime": 4.0,
                    "guest_time": 1.0986123,
                    "vsize": 0.0,
                }
            ]
        }
        self.assertEqual(transformed_metrics["callback_type"], expected_result["callback_type"])
        self.assertEqual(transformed_metrics["callback_type"], reshaped_metrics["callback_type"])
        for key in expected_result["task_metrics"].keys():
            self.assertAlmostEqual(
                transformed_metrics["task_metrics"][key], 
                expected_result["task_metrics"][key],
                delta=1e-6,
            )
            self.assertAlmostEqual(
                transformed_metrics["task_metrics"][key], 
                reshaped_metrics["task_metrics"][key],
                delta=1e-6,
            )
        for i in range(len(expected_result["runqueue"])):
            for key in expected_result["runqueue"][i].keys():
                self.assertAlmostEqual(
                transformed_metrics["runqueue"][i][key], 
                expected_result["runqueue"][i][key],
                delta=1e-6,
            )
            self.assertAlmostEqual(
                transformed_metrics["runqueue"][i][key], 
                reshaped_metrics["runqueue"][i][key],
                delta=1e-6,
            )
        




if __name__ == "__main__":
    unittest.main()
