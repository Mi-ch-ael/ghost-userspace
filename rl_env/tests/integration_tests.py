import threading
import time
import queue
import unittest
import socket
import struct
import unittest.mock
import gymnasium
import os
import sys
import time
import numpy as np
from contextlib import contextmanager
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from environment.scheduler_env import SchedulerEnv
from threads.stoppable_thread import StoppableThread

def send_sequence(sequence, sleep_before=0, host='localhost', port=14014):
    sleep_before and time.sleep(sleep_before)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        
        length = struct.pack('!I', len(sequence))
        s.sendall(length)
        message = struct.pack(f'!{len(sequence)}Q', *sequence)
        s.sendall(message)

def get_raw_metrics_mock_non_actionable():
    time.sleep(1)
    return [
        0,
        4,
        1, 0, 0, 45152452, 51524520, 0, 89102602,
        1, 0, 0, 45152452, 51524520, 0, 89102602,
        0, 0, 1, 980551253, 1251213, 1515332, 60000000,
    ]

def get_raw_metrics_mock_actionable():
    time.sleep(1)
    return [
        1,
        4,
        1, 0, 0, 45152452, 51524520, 0, 89102602,
        1, 0, 0, 45152452, 51524520, 0, 89102602,
        0, 0, 1, 980551253, 1251213, 1515332, 60000000,
    ]

@contextmanager
def thread_context_manager(stoppable_thread: StoppableThread, timeout_seconds=0.02):
    try:
        yield
    finally:
        if stoppable_thread.is_alive():
            stoppable_thread.stop()
            time.sleep(timeout_seconds)
            assert not stoppable_thread.is_alive()


class IntegrationTests(unittest.TestCase):
    def test_example_usage_of_stoppable_thread(self):
        def payload():
            time.sleep(0.0001)
        stoppable_thread = StoppableThread(target=payload, args=(), name="test-thread")
        stoppable_thread.start()
        time.sleep(0.0002)
        self.assertEqual(stoppable_thread.is_alive(), True)
        stoppable_thread.stop()
        time.sleep(0.0002)
        self.assertEqual(stoppable_thread.is_alive(), False)

    def test_thread_communication_on_non_actionable_callback(self):
        sched_environment = SchedulerEnv()
        with(thread_context_manager(sched_environment.background_collector_thread, 1)):
            sched_environment._get_raw_metrics = unittest.mock.Mock(side_effect=get_raw_metrics_mock_non_actionable)
            sched_environment._start_collector()
            time.sleep(0.5)
            sched_environment._get_raw_metrics.assert_called_once()
            time.sleep(0.6)
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

    def test_thread_communication_on_actionable_callback(self):
        sched_environment = SchedulerEnv()
        with(thread_context_manager(sched_environment.background_collector_thread, 1)):
            sched_environment._get_raw_metrics = unittest.mock.Mock(side_effect=get_raw_metrics_mock_actionable)
            sched_environment._start_collector()
            time.sleep(0.5)
            sched_environment._get_raw_metrics.assert_called_once()
            time.sleep(0.6)
            self.assertEqual(sched_environment.accumulated_metrics_lock.locked(), False)
            self.assertEqual(len(sched_environment.accumulated_metrics), sched_environment.max_prev_events_stored)
            self.assertEqual(
                sched_environment.accumulated_metrics[0]["task_metrics"]["vsize"], 
                0.0,
                "Expect first record to be zeroed out"
            )
            self.assertEqual(
                sched_environment.accumulated_metrics[1]["task_metrics"]["vsize"], 
                0.0,
                "Expect last record to be zeroed out"
            )
            self.assertEqual(sched_environment.observations_ready, True)
            self.assertEqual(sched_environment.actionable_event_metrics["callback_type"], 4)

    def test_get_raw_metrics_positive(self):
        sequence_to_send = [i for i in range(15)]
        port = 14014
        sched_environment = SchedulerEnv()
        sched_environment.receive_timeout_seconds = 10
        result_queue = queue.Queue()
        def run_server(result_queue):
            try:
                received_sequence = sched_environment._get_raw_metrics()
                self.assertEqual(received_sequence, sequence_to_send)
                result_queue.put(None)
            except Exception as ex:
                result_queue.put(ex)

        server_thread = threading.Thread(target=run_server, args=(result_queue,))
        server_thread.start()
        
        send_sequence(sequence_to_send, port=port)        
        server_thread.join()
        result = result_queue.get()
        if isinstance(result, Exception):
            self.fail(f"SchedulerEnv raised an exception: {result}")

    def test_listen_for_non_actionable_update(self):
        sequence_to_send = [0, *[i + 1 for i in range(15)]]
        port = 14014
        sched_environment = SchedulerEnv()
        with thread_context_manager(sched_environment.background_collector_thread, 0.5):
            sched_environment._start_collector()
            time.sleep(0.1)
            send_sequence(sequence_to_send, port=port)
            time.sleep(0.1)
            self.assertIs(sched_environment.actionable_event_metrics, None)
            self.assertEqual(sched_environment.observations_ready, False)
            self.assertEqual(sched_environment.accumulated_metrics[0]["callback_type"], 0)
            self.assertEqual(sched_environment.accumulated_metrics[-1]["callback_type"], 1)
            self.assertEqual(sched_environment.accumulated_metrics_lock.locked(), False)

    def test_reset_first_callback_actionable(self):
        sched_environment = gymnasium.make('rl_env/SchedulerEnv-v0')
        metrics_to_send = [
            1,
            7,
            1, 0, 0, 45152452, 51524520, 0, 89102602,
            1, 0, 0, 45152452, 51524520, 0, 89102602,
            0, 0, 1, 980551253, 1251213, 1515332, 60000000,
        ]
        with thread_context_manager(sched_environment.unwrapped.background_collector_thread):
            sending_thread = threading.Thread(target=send_sequence, args=(metrics_to_send, 0.001))
            sending_thread.start()
            observation, _ = sched_environment.reset()
            sending_thread.join()
            self.assertEqual(len(observation), 3)
            self.assertEqual(observation[0]["callback_type"], 0)
            self.assertEqual(observation[1]["callback_type"], 0)
            self.assertEqual(observation[2]["callback_type"], 7)

    def test_reset_non_actionable_then_actionable(self):
        metrics_to_send_1 = [
            0,
            7,
            3, 0, 0, 45152452, 51524520, 0, 89102602,
            3, 0, 0, 45152452, 51524520, 0, 89102602,
            1, 0, 1, 980551253, 1251213, 1515332, 60000000,
        ]
        metrics_to_send_2 = [
            1,
            4,
            1, 0, 0, 45152452, 51524520, 0, 89102602,
            1, 0, 0, 45152452, 51524520, 0, 89102602,
            3, 0, 1, 980551253, 1251213, 1515332, 60000000,
        ]
        def send_metrics_payload():
            success = False
            while not success:
                try:
                    send_sequence(metrics_to_send_1, 0.0001)
                    success = True
                except ConnectionRefusedError:
                    print("Connection refused, waiting")
            success = False
            while not success:
                try:
                    send_sequence(metrics_to_send_2, 0.0001)
                    success = True
                except ConnectionRefusedError:
                    print("Connection refused, waiting")

        sched_environment = gymnasium.make('rl_env/SchedulerEnv-v0')
        with thread_context_manager(sched_environment.unwrapped.background_collector_thread):
            sending_thread = threading.Thread(target=send_metrics_payload, args=())
            sending_thread.start()
            observation, _ = sched_environment.reset()
            sending_thread.join()
            self.assertEqual(len(observation), 3)
            self.assertEqual(observation[0]["callback_type"], 0)
            self.assertEqual(observation[1]["callback_type"], 7)
            self.assertEqual(observation[2]["callback_type"], 4)
            self.assertDictEqual(observation[-1]["task_metrics"], {
                "run_state": 1,
                "cpu_num": 0,
                "preempted": 0,
                "utime": np.array([16.0], dtype=np.float32),
                "stime": np.array([16.0], dtype=np.float32),
                "guest_time": np.array([0.0], dtype=np.float32),
                "vsize": np.array([16.0], dtype=np.float32),
            })
            self.assertDictEqual(observation[-1]["runqueue"][0], {
                "run_state": 1,
                "cpu_num": 0,
                "preempted": 0,
                "utime": np.array([16.0], dtype=np.float32),
                "stime": np.array([16.0], dtype=np.float32),
                "guest_time": np.array([0.0], dtype=np.float32),
                "vsize": np.array([16.0], dtype=np.float32),
            })

class TestIntegrationSendAction(unittest.TestCase):
    def setUp(self):
        self.server_host = '127.0.0.1'
        self.server_port = 17213
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.server_host, self.server_port))
        self.server_socket.listen(1)
        self.server_thread = threading.Thread(target=self.server_handler)
        self.server_thread.start()

    def tearDown(self):
        self.server_socket.close()
        self.server_thread.join()

    def server_handler(self):
        client_socket, _ = self.server_socket.accept()
        try:
            data = client_socket.recv(4)
            self.received_value = struct.unpack('!I', data)[0]
        finally:
            client_socket.close()

    def test_send_action_positive(self):
        sched_environment = SchedulerEnv()
        value_to_send = 42
        sched_environment._send_action(value_to_send)
        self.server_thread.join()

        self.assertEqual(self.received_value, value_to_send)
    

if __name__ == "__main__":
     unittest.main()