import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import gymnasium
import environment
from environment.scheduler_env import SchedulerEnv

class SchedulerEnvTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()