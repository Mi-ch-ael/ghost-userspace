import threading
import time
import queue
import unittest
import socket
import struct
import gymnasium
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from environment.scheduler_env import SchedulerEnv

def send_sequence(sequence, host='localhost', port=14014):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        
        length = struct.pack('!I', len(sequence))
        s.sendall(length)
        message = struct.pack(f'!{len(sequence)}Q', *sequence)
        s.sendall(message)


class IntegrationTests(unittest.TestCase):
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
    

if __name__ == "__main__":
     unittest.main()