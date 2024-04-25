import unittest
import numpy as np
import os
import sys
import unittest.mock
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import gymnasium
import environment
from environment.scheduler_env import SchedulerEnv

class SchedulerEnvTests(unittest.TestCase):
    maxDiff = None
    def test_environment_creation_default_parameters(self):
        sched_environment = gymnasium.make('rl_env/SchedulerEnv-v0')
        self.assertIs(sched_environment.render_mode, None)
        self.assertEqual(sched_environment.unwrapped.share_counter, 0)
        self.assertEqual(sched_environment.unwrapped.actual_runqueue_length, 0)
        self.assertEqual(sched_environment.unwrapped.parser.runqueue_cutoff_length, 8)
        self.assertEqual(sched_environment.unwrapped.parser.time_cap, 16)
        self.assertEqual(sched_environment.unwrapped.parser.vsize_cap, 16)
        self.assertEqual(sched_environment.observation_space, gymnasium.spaces.Dict({
            "callback_type": gymnasium.spaces.Discrete(8),
            "task_metrics": gymnasium.spaces.Dict({
                "run_state": gymnasium.spaces.Discrete(4),
                "cpu_num": gymnasium.spaces.Discrete(1),
                "preempted": gymnasium.spaces.Discrete(2),
                "utime": gymnasium.spaces.Box(0, 16, shape=(1,)),
                "stime": gymnasium.spaces.Box(0, 16, shape=(1,)),
                "guest_time": gymnasium.spaces.Box(0, 16, shape=(1,)),
                "vsize": gymnasium.spaces.Box(0, 16, shape=(1,)),
            }),
            "runqueue": gymnasium.spaces.Tuple([gymnasium.spaces.Dict({
                "run_state": gymnasium.spaces.Discrete(4),
                "cpu_num": gymnasium.spaces.Discrete(1),
                "preempted": gymnasium.spaces.Discrete(2),
                "utime": gymnasium.spaces.Box(0, 16, shape=(1,)),
                "stime": gymnasium.spaces.Box(0, 16, shape=(1,)),
                "guest_time": gymnasium.spaces.Box(0, 16, shape=(1,)),
                "vsize": gymnasium.spaces.Box(0, 16, shape=(1,)),
            })] * 8)
        }))
        self.assertEqual(sched_environment.action_space, gymnasium.spaces.Discrete(8))

    def test_environment_creation_custom_parameters(self):
        sched_environment = gymnasium.make(
            'rl_env/SchedulerEnv-v0', 
            cpu_num=4,
            runqueue_cutoff_length=5,
            time_ln_cap=14,
            vsize_ln_cap=17,
        )
        self.assertIs(sched_environment.render_mode, None)
        self.assertEqual(sched_environment.unwrapped.share_counter, 0)
        self.assertEqual(sched_environment.unwrapped.actual_runqueue_length, 0)
        self.assertEqual(sched_environment.unwrapped.parser.runqueue_cutoff_length, 5)
        self.assertEqual(sched_environment.unwrapped.parser.time_cap, 14)
        self.assertEqual(sched_environment.unwrapped.parser.vsize_cap, 17)
        self.assertEqual(sched_environment.observation_space, gymnasium.spaces.Dict({
            "callback_type": gymnasium.spaces.Discrete(8),
            "task_metrics": gymnasium.spaces.Dict({
                "run_state": gymnasium.spaces.Discrete(4),
                "cpu_num": gymnasium.spaces.Discrete(4),
                "preempted": gymnasium.spaces.Discrete(2),
                "utime": gymnasium.spaces.Box(0, 14, shape=(1,)),
                "stime": gymnasium.spaces.Box(0, 14, shape=(1,)),
                "guest_time": gymnasium.spaces.Box(0, 14, shape=(1,)),
                "vsize": gymnasium.spaces.Box(0, 17, shape=(1,)),
            }),
            "runqueue": gymnasium.spaces.Tuple([gymnasium.spaces.Dict({
                "run_state": gymnasium.spaces.Discrete(4),
                "cpu_num": gymnasium.spaces.Discrete(4),
                "preempted": gymnasium.spaces.Discrete(2),
                "utime": gymnasium.spaces.Box(0, 14, shape=(1,)),
                "stime": gymnasium.spaces.Box(0, 14, shape=(1,)),
                "guest_time": gymnasium.spaces.Box(0, 14, shape=(1,)),
                "vsize": gymnasium.spaces.Box(0, 17, shape=(1,)),
            })] * 5)
        }))
        self.assertEqual(sched_environment.action_space, gymnasium.spaces.Discrete(5))

    def test_step_invalid_action(self):
        sched_environment = SchedulerEnv()
        sched_environment._send_action = unittest.mock.Mock(return_value=None)
        sched_environment.actual_runqueue_length = 2
        sched_environment._get_raw_metrics = unittest.mock.Mock(return_value=[
                7,
                1, 0, 0, 45152452, 5152452, 0, 89102602,
                1, 0, 0, 45152452, 5152452, 0, 89102602,
                0, 0, 1, 980551253, 1251213, 1515332, 60000000,
            ])

        observation, reward, terminated, truncated, _ = sched_environment.step(2)
        self.assertDictEqual(observation, {
            "callback_type": 0,
            "task_metrics": {
                "run_state": 0,
                "cpu_num": 0,
                "preempted": 0,
                "utime": 0.0,
                "stime": 0.0,
                "guest_time": 0.0,
                "vsize": 0.0,
            },
            "runqueue": ({
                "run_state": 0,
                "cpu_num": 0,
                "preempted": 0,
                "utime": 0.0,
                "stime": 0.0,
                "guest_time": 0.0,
                "vsize": 0.0,
            },) * 8
        })
        self.assertEqual(reward, -1)
        self.assertEqual(terminated, True)
        self.assertEqual(truncated, False)

    def test_step_valid_action(self):
        sched_environment = SchedulerEnv()
        sched_environment._send_action = unittest.mock.Mock(return_value=None)
        sched_environment.actual_runqueue_length = 2
        sched_environment._get_raw_metrics = unittest.mock.Mock(return_value=[
                7,
                1, 0, 0, 45152452, 45152452, 0, 89102602,
                1, 0, 0, 45152452, 45152452, 0, 89102602,
                0, 0, 1, 980551253, 1251213, 1515332, 60000000,
            ])

        observation, reward, terminated, truncated, _ = sched_environment.step(1)
        self.assertEqual(reward, 0)
        self.assertEqual(observation["callback_type"], 7)
        self.assertDictEqual(observation["task_metrics"], {
                "run_state": 1,
                "cpu_num": 0,
                "preempted": 0,
                "utime": 16.0,
                "stime": 16.0,
                "guest_time": 0.0,
                "vsize": 16.0,
            })
        self.assertDictEqual(observation["runqueue"][0], {
                "run_state": 1,
                "cpu_num": 0,
                "preempted": 0,
                "utime": 16.0,
                "stime": 16.0,
                "guest_time": 0.0,
                "vsize": 16.0,
            })
        self.assertEqual(observation["runqueue"][1]["run_state"], 0)
        self.assertEqual(observation["runqueue"][1]["cpu_num"], 0)
        self.assertEqual(observation["runqueue"][1]["preempted"], 1)
        self.assertEqual(observation["runqueue"][1]["utime"], 16.0)
        self.assertAlmostEqual(observation["runqueue"][1]["stime"], 14.039624, delta=1e-6)
        self.assertAlmostEqual(observation["runqueue"][1]["guest_time"], 14.231145, delta=1e-6)
        self.assertEqual(observation["runqueue"][1]["vsize"], 16.0)
        self.assertEqual(terminated, False)
        self.assertEqual(truncated, False)

    def test_reset_no_restart(self):
        sched_environment = SchedulerEnv()
        sched_environment.actual_runqueue_length = 2
        sched_environment._get_raw_metrics = unittest.mock.Mock(return_value=[
                7,
                1, 0, 0, 45152452, 45152452, 0, 89102602,
                1, 0, 0, 45152452, 45152452, 0, 89102602,
                0, 0, 1, 980551253, 1251213, 1515332, 60000000,
            ])

        observation, _ = sched_environment.reset()
        self.assertEqual(observation["callback_type"], 7)
        self.assertDictEqual(observation["task_metrics"], {
                "run_state": 1,
                "cpu_num": 0,
                "preempted": 0,
                "utime": 16.0,
                "stime": 16.0,
                "guest_time": 0.0,
                "vsize": 16.0,
            })
        self.assertDictEqual(observation["runqueue"][0], {
                "run_state": 1,
                "cpu_num": 0,
                "preempted": 0,
                "utime": 16.0,
                "stime": 16.0,
                "guest_time": 0.0,
                "vsize": 16.0,
            })
        self.assertEqual(observation["runqueue"][1]["run_state"], 0)
        self.assertEqual(observation["runqueue"][1]["cpu_num"], 0)
        self.assertEqual(observation["runqueue"][1]["preempted"], 1)
        self.assertEqual(observation["runqueue"][1]["utime"], 16.0)
        self.assertAlmostEqual(observation["runqueue"][1]["stime"], 14.039624, delta=1e-6)
        self.assertAlmostEqual(observation["runqueue"][1]["guest_time"], 14.231145, delta=1e-6)
        self.assertEqual(observation["runqueue"][1]["vsize"], 16.0)

    def test_reset_restart(self):
        sched_environment = SchedulerEnv()
        sched_environment.actual_runqueue_length = 2
        sched_environment._get_raw_metrics = unittest.mock.Mock(return_value=[
                7,
                1, 0, 0, 45152452, 45152452, 0, 89102602,
                1, 0, 0, 45152452, 45152452, 0, 89102602,
                0, 0, 1, 980551253, 1251213, 1515332, 60000000,
            ])
        
        with self.assertRaises(NotImplementedError):
            sched_environment.reset(seed=None, options={"restart": True})


if __name__ == "__main__":
    unittest.main()