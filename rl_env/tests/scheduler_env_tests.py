import unittest
import numpy as np
import os
import sys
import unittest.mock
import struct
import socket
import time

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import gymnasium
from gymnasium.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
    env_render_passive_checker,
    env_reset_passive_checker,
    env_step_passive_checker,
)
from gymnasium.utils.env_checker import check_reset_return_type
import environment
from threads.stoppable_thread import StoppableThread
from environment.scheduler_env import SchedulerEnv
from environment.scheduler_spaces import generate_zeroed_sample

def get_raw_metrics_mock_non_actionable():
    time.sleep(0.01)
    return [
        0,
        7,
        1, 0, 0, 45152452, 45152452, 0, 89102602,
        1, 0, 0, 45152452, 45152452, 0, 89102602,
        0, 0, 1, 980551253, 1251213, 1515332, 60000000,
    ]

def get_raw_metrics_mock_actionable():
    time.sleep(0.01)
    return [
        1,
        7,
        1, 0, 0, 45152452, 45152452, 0, 89102602,
        1, 0, 0, 45152452, 45152452, 0, 89102602,
        0, 0, 1, 980551253, 1251213, 1515332, 60000000,
    ]

class SchedulerEnvTests(unittest.TestCase):
    maxDiff = None

    def test_environment_creation_default_parameters(self):
        sched_environment = gymnasium.make('rl_env/SchedulerEnv-v0')
        self.assertIs(sched_environment.render_mode, None)
        self.assertEqual(sched_environment.unwrapped.share_counter, 0)
        self.assertEqual(sched_environment.unwrapped.actual_runqueue_length, 0)
        self.assertEqual(sched_environment.unwrapped.max_prev_events_stored, 2)
        self.assertEqual(sched_environment.unwrapped.socket_port, 14014)
        self.assertEqual(sched_environment.unwrapped.scheduler_port, 17213)
        self.assertEqual(sched_environment.unwrapped.parser.runqueue_cutoff_length, 8)
        self.assertEqual(sched_environment.unwrapped.parser.time_cap, 16)
        self.assertEqual(sched_environment.unwrapped.parser.vsize_cap, 16)
        self.assertEqual(sched_environment.unwrapped.accumulated_metrics, [
            {
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
            }
        ] * 2)
        self.assertEqual(sched_environment.unwrapped.accumulated_metrics_lock.locked(), False)
        self.assertEqual(sched_environment.unwrapped.observations_ready, False)
        self.assertIs(type(sched_environment.unwrapped.background_collector_thread), StoppableThread)
        self.assertEqual(sched_environment.observation_space, gymnasium.spaces.Tuple([
            gymnasium.spaces.Dict({
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
            })
        ] * 3))
        self.assertEqual(sched_environment.action_space, gymnasium.spaces.Discrete(8))
        self.assertEqual(sched_environment.unwrapped.background_collector_thread.is_alive(), False)

    def test_environment_creation_custom_parameters(self):
        sched_environment = gymnasium.make(
            'rl_env/SchedulerEnv-v0', 
            cpu_num=4,
            runqueue_cutoff_length=5,
            max_prev_events_stored=4,
            time_ln_cap=14,
            vsize_ln_cap=17,
            socket_port=9090,
            scheduler_port=12345,
        )
        self.assertIs(sched_environment.render_mode, None)
        self.assertEqual(sched_environment.unwrapped.share_counter, 0)
        self.assertEqual(sched_environment.unwrapped.actual_runqueue_length, 0)
        self.assertEqual(sched_environment.unwrapped.max_prev_events_stored, 4)
        self.assertEqual(sched_environment.unwrapped.socket_port, 9090)
        self.assertEqual(sched_environment.unwrapped.scheduler_port, 12345)
        self.assertEqual(sched_environment.unwrapped.parser.runqueue_cutoff_length, 5)
        self.assertEqual(sched_environment.unwrapped.parser.time_cap, 14)
        self.assertEqual(sched_environment.unwrapped.parser.vsize_cap, 17)
        self.assertEqual(sched_environment.unwrapped.accumulated_metrics, [
            {
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
                },) * 5
            }
        ] * 4)
        self.assertEqual(sched_environment.unwrapped.accumulated_metrics_lock.locked(), False)
        self.assertEqual(sched_environment.unwrapped.observations_ready, False)
        self.assertIs(type(sched_environment.unwrapped.background_collector_thread), StoppableThread)
        self.assertEqual(sched_environment.observation_space, gymnasium.spaces.Tuple([
            gymnasium.spaces.Dict({
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
            })
        ] * 5))
        self.assertEqual(sched_environment.action_space, gymnasium.spaces.Discrete(5))
        self.assertEqual(sched_environment.unwrapped.background_collector_thread.is_alive(), False)

    @unittest.mock.patch.object(SchedulerEnv, '_listen_for_updates')
    def test_start_collector(self, mock_listen_for_updates):
        mock_listen_for_updates.side_effect = lambda : time.sleep(0.01)
        sched_environment = SchedulerEnv()
        sched_environment._start_collector()
        time.sleep(0.02)
        self.assertEqual(sched_environment.background_collector_thread.is_alive(), True)
        sched_environment.background_collector_thread.stop()
        time.sleep(0.02)
        self.assertEqual(sched_environment.background_collector_thread.is_alive(), False)

    @unittest.mock.patch.object(SchedulerEnv, '_listen_for_updates')
    def test_finalize(self, mock_listen_for_updates):
        mock_listen_for_updates.side_effect = lambda : time.sleep(0.01)
        sched_environment = SchedulerEnv()
        sched_environment._start_collector()
        time.sleep(0.02)
        sched_environment.finalize()
        time.sleep(0.02)
        self.assertEqual(sched_environment.background_collector_thread.is_alive(), False)

    def test_listen_for_actionable_update(self):
        sched_environment = SchedulerEnv()
        sched_environment._get_raw_metrics = unittest.mock.Mock(return_value=[
            1,
            4,
            1, 0, 0, 45152452, 51524520, 0, 89102602,
            1, 0, 0, 45152452, 51524520, 0, 89102602,
            0, 0, 1, 980551253, 1251213, 1515332, 60000000,
        ])
        sched_environment._listen_for_updates()

        sched_environment._get_raw_metrics.assert_called_once()
        self.assertEqual(sched_environment.accumulated_metrics_lock.locked(), False)
        self.assertEqual(
            sched_environment.accumulated_metrics, 
            [generate_zeroed_sample(sched_environment.observation_space[0])] * 2,
            "Expect `accumulated_metrics` to be empty (i.e. zeroed out) after an actionable callback"
        )
        self.assertEqual(sched_environment.observations_ready, True)
        self.assertEqual(sched_environment.actionable_event_metrics["callback_type"], 4)
        self.assertDictEqual(sched_environment.actionable_event_metrics["task_metrics"], {
                "run_state": 1,
                "cpu_num": 0,
                "preempted": 0,
                "utime": 16.0,
                "stime": 16.0,
                "guest_time": 0.0,
                "vsize": 16.0,
            })
        self.assertDictEqual(sched_environment.actionable_event_metrics["runqueue"][0], {
                "run_state": 1,
                "cpu_num": 0,
                "preempted": 0,
                "utime": 16.0,
                "stime": 16.0,
                "guest_time": 0.0,
                "vsize": 16.0,
            })
        self.assertEqual(sched_environment.actionable_event_metrics["runqueue"][1]["run_state"], 0)
        self.assertEqual(sched_environment.actionable_event_metrics["runqueue"][1]["cpu_num"], 0)
        self.assertEqual(sched_environment.actionable_event_metrics["runqueue"][1]["preempted"], 1)
        self.assertEqual(sched_environment.actionable_event_metrics["runqueue"][1]["utime"], 16.0)
        self.assertAlmostEqual(sched_environment.actionable_event_metrics["runqueue"][1]["stime"], 14.039624, delta=1e-6)
        self.assertAlmostEqual(sched_environment.actionable_event_metrics["runqueue"][1]["guest_time"], 14.231145, delta=1e-6)
        self.assertEqual(sched_environment.actionable_event_metrics["runqueue"][1]["vsize"], 16.0)
        self.assertEqual(sched_environment.actionable_event_metrics["runqueue"][2:], [
            {
                "run_state": 0,
                "cpu_num": 0,
                "preempted": 0,
                "utime": 0.0,
                "stime": 0.0,
                "guest_time": 0.0,
                "vsize": 0.0,
            },
        ] * 6)

    def test_listen_for_non_actionable_update(self):
        sched_environment = SchedulerEnv()
        sched_environment._get_raw_metrics = unittest.mock.Mock(return_value=[
            0,
            4,
            1, 0, 0, 45152452, 51524520, 0, 89102602,
            1, 0, 0, 45152452, 51524520, 0, 89102602,
            0, 0, 1, 980551253, 1251213, 1515332, 60000000,
        ])
        sched_environment._listen_for_updates()

        sched_environment._get_raw_metrics.assert_called_once()
        self.assertEqual(sched_environment.accumulated_metrics_lock.locked(), False)
        self.assertEqual(len(sched_environment.accumulated_metrics), sched_environment.max_prev_events_stored)
        self.assertEqual(
            sched_environment.accumulated_metrics[0]["task_metrics"]["vsize"], 
            0.0,
            "Expect first record to be zeroed out"
        )
        self.assertEqual(
            sched_environment.accumulated_metrics[1]["task_metrics"]["vsize"], 
            16.0,
            "Expect last record not to be zeroed out"
        )
        self.assertEqual(sched_environment.observations_ready, False)
        self.assertEqual(sched_environment.actionable_event_metrics, None)

    # Need to remake observation to have new form: `Tuple` of `Dict`s
    @unittest.expectedFailure
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

    # Need to remake observation to have new form: `Tuple` of `Dict`s
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
        sched_environment._get_raw_metrics = unittest.mock.Mock(side_effect=get_raw_metrics_mock_actionable)

        try:
            observation, _ = sched_environment.reset()
            self.assertEqual(sched_environment.background_collector_thread.is_alive(), True)

            self.assertEqual(observation[-1]["callback_type"], 7)
            self.assertDictEqual(observation[-1]["task_metrics"], {
                    "run_state": 1,
                    "cpu_num": 0,
                    "preempted": 0,
                    "utime": 16.0,
                    "stime": 16.0,
                    "guest_time": 0.0,
                    "vsize": 16.0,
                })
            self.assertDictEqual(observation[-1]["runqueue"][0], {
                    "run_state": 1,
                    "cpu_num": 0,
                    "preempted": 0,
                    "utime": 16.0,
                    "stime": 16.0,
                    "guest_time": 0.0,
                    "vsize": 16.0,
                })
            self.assertEqual(observation[-1]["runqueue"][1]["run_state"], 0)
            self.assertEqual(observation[-1]["runqueue"][1]["cpu_num"], 0)
            self.assertEqual(observation[-1]["runqueue"][1]["preempted"], 1)
            self.assertEqual(observation[-1]["runqueue"][1]["utime"], 16.0)
            self.assertAlmostEqual(observation[-1]["runqueue"][1]["stime"], 14.039624, delta=1e-6)
            self.assertAlmostEqual(observation[-1]["runqueue"][1]["guest_time"], 14.231145, delta=1e-6)
            self.assertEqual(observation[-1]["runqueue"][1]["vsize"], 16.0)
            
            self.assertEqual(
                observation[0], 
                generate_zeroed_sample(sched_environment.observation_space[0])
            )
            self.assertEqual(len(observation), sched_environment.max_prev_events_stored + 1)
        finally:
            sched_environment.finalize()
            time.sleep(0.02)

    def test_check_reset_return_type(self):
        sched_environment = SchedulerEnv()
        sched_environment._get_raw_metrics = unittest.mock.Mock(side_effect=get_raw_metrics_mock_actionable)
        check_reset_return_type(sched_environment)

    def test_check_action_space(self):
        sched_environment = gymnasium.make('rl_env/SchedulerEnv-v0')
        check_action_space(sched_environment.action_space)

    def test_check_observation_space(self):
        sched_environment = gymnasium.make('rl_env/SchedulerEnv-v0')
        check_observation_space(sched_environment.observation_space)

    @unittest.mock.patch('socket.socket')
    def test_get_raw_metrics_positive(self, mock_socket):
        mock_socket_instance = unittest.mock.Mock()
        mock_socket.return_value = mock_socket_instance     
        mock_conn_instance = unittest.mock.Mock()
        mock_conn_instance.recv.side_effect = [
            struct.pack('!I', 15),  # The length of the sequence (15 integers)
            struct.pack('!15Q', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) # Sequence
        ]
        mock_socket_instance.accept.return_value = (mock_conn_instance, ('127.0.0.1', 14014))
        
        sched_environment = SchedulerEnv()
        result = sched_environment._get_raw_metrics()

        self.assertEqual(result, [i for i in range(1, 16)])
        self.assertEqual(sched_environment.actual_runqueue_length, 1)
        
        mock_socket_instance.bind.assert_called_with(('localhost', 14014))
        mock_socket_instance.listen.assert_called_once()
        mock_socket_instance.accept.assert_called_once()
        mock_conn_instance.recv.assert_any_call(4)
        mock_conn_instance.recv.assert_any_call(15 * 8)
        mock_conn_instance.close.assert_called_once()
        mock_socket_instance.close.assert_called_once()

    @unittest.mock.patch('socket.socket')
    def test_get_raw_metrics_positive_non_default_port(self, mock_socket):
        mock_socket_instance = unittest.mock.Mock()
        mock_socket.return_value = mock_socket_instance     
        mock_conn_instance = unittest.mock.Mock()
        mock_conn_instance.recv.side_effect = [
            struct.pack('!I', 15),
            struct.pack('!15Q', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        ]
        mock_socket_instance.accept.return_value = (mock_conn_instance, ('127.0.0.1', 9898))
        
        sched_environment = SchedulerEnv(socket_port=9898)
        result = sched_environment._get_raw_metrics()

        self.assertEqual(result, [i for i in range(1, 16)])
        self.assertEqual(sched_environment.actual_runqueue_length, 1)
        
        mock_socket_instance.bind.assert_called_with(('localhost', 9898))
        mock_socket_instance.listen.assert_called_once()
        mock_socket_instance.accept.assert_called_once()
        mock_conn_instance.recv.assert_any_call(4)
        mock_conn_instance.recv.assert_any_call(15 * 8)
        mock_conn_instance.close.assert_called_once()
        mock_socket_instance.close.assert_called_once()

    @unittest.mock.patch('socket.socket')
    def test_get_raw_metrics_non_whole_number_of_tasks(self, mock_socket):
        mock_socket_instance = unittest.mock.Mock()
        mock_socket.return_value = mock_socket_instance     
        mock_conn_instance = unittest.mock.Mock()
        mock_conn_instance.recv.side_effect = [
            struct.pack('!I', 18),
            struct.pack('!18Q', *[i for i in range(18)])
        ]
        mock_socket_instance.accept.return_value = (mock_conn_instance, ('127.0.0.1', 14014))
        
        sched_environment = SchedulerEnv()
        with self.assertRaises(AssertionError):
            sched_environment._get_raw_metrics()
        
        mock_socket_instance.bind.assert_called_with(('localhost', 14014))
        mock_socket_instance.listen.assert_called_once()
        mock_socket_instance.accept.assert_called_once()
        mock_conn_instance.recv.assert_called_once_with(4)
        mock_conn_instance.close.assert_called_once()
        mock_socket_instance.close.assert_called_once()

    @unittest.mock.patch('socket.socket')
    def test_get_raw_metrics_bind_error(self, mock_socket):
        mock_socket_instance = unittest.mock.Mock()
        mock_socket.return_value = mock_socket_instance        
        mock_socket_instance.bind.side_effect = socket.error("Socket bind error")
        sched_environment = SchedulerEnv()

        with self.assertRaises(socket.error):
            sched_environment._get_raw_metrics()
        
        mock_socket_instance.close.assert_called_once()

    @unittest.mock.patch('socket.socket')
    def test_get_raw_metrics_connection_error(self, mock_socket):
        mock_socket_instance = unittest.mock.Mock()
        mock_socket.return_value = mock_socket_instance        
        mock_socket_instance.accept.side_effect = socket.error("Socket accept error")
        sched_environment = SchedulerEnv()

        with self.assertRaises(socket.error):
            sched_environment._get_raw_metrics()
        
        mock_socket_instance.close.assert_called_once()

    @unittest.mock.patch('socket.socket')
    def test_receive_sequence_data_reception_error(self, mock_socket):
        mock_socket_instance = unittest.mock.Mock()
        mock_socket.return_value = mock_socket_instance        
        mock_conn_instance = unittest.mock.Mock()
        mock_socket_instance.accept.return_value = (mock_conn_instance, ('127.0.0.1', 14014))        
        mock_conn_instance.recv.side_effect = socket.error("Socket recv error")
        sched_environment = SchedulerEnv()

        with self.assertRaises(socket.error):
            sched_environment._get_raw_metrics()
        
        mock_conn_instance.close.assert_called_once()
        mock_socket_instance.close.assert_called_once()
    
    @unittest.mock.patch('socket.socket')
    def test_receive_sequence_unpack_error(self, mock_socket):
        mock_socket_instance = unittest.mock.Mock()
        mock_socket.return_value = mock_socket_instance        
        mock_conn_instance = unittest.mock.Mock()
        mock_socket_instance.accept.return_value = (mock_conn_instance, ('127.0.0.1', 14014))
        # Simulate an incorrect data length received
        mock_conn_instance.recv.side_effect = [struct.pack('!I', 15), b'']
        sched_environment = SchedulerEnv()
        
        with self.assertRaises(struct.error):
            sched_environment._get_raw_metrics()
        
        mock_conn_instance.close.assert_called_once()
        mock_socket_instance.close.assert_called_once()

    @unittest.mock.patch('socket.socket')
    def test_send_action_positive(self, mock_socket_class):
        mock_socket_instance = unittest.mock.Mock()
        mock_socket_class.return_value = mock_socket_instance
        value = 42
        port = 12345
        sched_environment = SchedulerEnv(scheduler_port=port)

        sched_environment._send_action(value)

        mock_socket_instance.connect.assert_called_once_with(("localhost", port))
        packed_data = struct.pack('!I', value)
        mock_socket_instance.sendall.assert_called_once_with(packed_data)
        mock_socket_instance.close.assert_called_once()

    @unittest.mock.patch('socket.socket')
    def test_send_action_positive_default_port(self, mock_socket_class):
        mock_socket_instance = unittest.mock.Mock()
        mock_socket_class.return_value = mock_socket_instance
        value = 42
        port = 17213
        sched_environment = SchedulerEnv()

        sched_environment._send_action(value)

        mock_socket_instance.connect.assert_called_once_with(("localhost", port))
        packed_data = struct.pack('!I', value)
        mock_socket_instance.sendall.assert_called_once_with(packed_data)
        mock_socket_instance.close.assert_called_once()

    @unittest.mock.patch('socket.socket')
    def test_send_connection_error(self, mock_socket_class):
        mock_socket_instance = unittest.mock.Mock()
        mock_socket_instance.connect.side_effect = socket.error
        mock_socket_class.return_value = mock_socket_instance
        value = 42
        port = 12345
        sched_environment = SchedulerEnv(scheduler_port=port)

        with self.assertRaises(socket.error):
            sched_environment._send_action(value)

        mock_socket_instance.connect.assert_called_once_with(("localhost", port))
        mock_socket_instance.sendall.assert_not_called()
        mock_socket_instance.close.assert_called_once()

    @unittest.mock.patch('socket.socket')
    def test_send_integer_send_failure(self, mock_socket_class):
        mock_socket_instance = unittest.mock.Mock()
        mock_socket_instance.sendall.side_effect = socket.error
        mock_socket_class.return_value = mock_socket_instance
        value = 42
        port = 12345
        sched_environment = SchedulerEnv(scheduler_port=port)

        with self.assertRaises(socket.error):
            sched_environment._send_action(value)

        mock_socket_instance.connect.assert_called_once_with(("localhost", port))
        mock_socket_instance.sendall.assert_called_once_with(struct.pack('!I', value))
        mock_socket_instance.close.assert_called_once()

if __name__ == "__main__":
    unittest.main()