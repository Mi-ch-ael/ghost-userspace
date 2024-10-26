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
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from environment.scheduler_env import SchedulerEnv
from threads.stoppable_thread import StoppableThread

def send_sequence(sequence, host='localhost', port=14014):
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


class IntegrationTests(unittest.TestCase):
    def test_example_usage_of_stoppable_thread(self):
        def payload():
            time.sleep(0.1)
        stoppable_thread = StoppableThread(target=payload, args=(), name="test-thread")
        stoppable_thread.start()
        time.sleep(0.2)
        self.assertEqual(stoppable_thread.is_alive(), True)
        stoppable_thread.stop()
        time.sleep(0.2)
        self.assertEqual(stoppable_thread.is_alive(), False)

    def test_thread_communication_on_non_actionable_callback(self):
        sched_environment = SchedulerEnv()
        try:
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
        finally:
            sched_environment._finalize()
            time.sleep(1)
            self.assertEqual(sched_environment.background_collector_thread.is_alive(), False)

    def test_thread_communication_on_actionable_callback(self):
        sched_environment = SchedulerEnv()
        try:
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
        finally:
            sched_environment._finalize()
            time.sleep(1)
            self.assertEqual(sched_environment.background_collector_thread.is_alive(), False)

    def test_get_raw_metrics_positive(self):
        sequence_to_send = [i for i in range(15)]
        port = 14014
        sched_environment = SchedulerEnv()
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
        time.sleep(1)
        
        send_sequence(sequence_to_send, port=port)        
        server_thread.join()
        result = result_queue.get()
        if isinstance(result, Exception):
            self.fail(f"SchedulerEnv raised an exception: {result}")

    def test_reset_rl_environment(self):
        result_queue = queue.Queue()
        sched_environment = gymnasium.make('rl_env/SchedulerEnv-v0')
        metrics_to_send = [
            7,
            1, 0, 0, 45152452, 51524520, 0, 89102602,
            1, 0, 0, 45152452, 51524520, 0, 89102602,
            0, 0, 1, 980551253, 1251213, 1515332, 60000000,
        ]

        def reset_environment(result_queue):
            try:
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
                result_queue.put(None)
            except Exception as ex:
                result_queue.put(ex)

        server_thread = threading.Thread(target=reset_environment, args=(result_queue,))
        server_thread.start()
        time.sleep(0.5)
        
        send_sequence(metrics_to_send, port=14014)        
        server_thread.join()
        result = result_queue.get()
        if isinstance(result, Exception):
            self.fail(f"SchedulerEnv raised an exception: {result}")


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