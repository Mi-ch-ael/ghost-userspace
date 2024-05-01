import gymnasium
from gymnasium import spaces
from threading import Lock
import time
import socket
import struct
from threads.stoppable_thread import StoppableThread
from environment.observation_parser import LnCapObservationParser
from environment.scheduler_spaces import generate_zeroed_sample


class SchedulerEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(
            self, 
            render_mode=None, 
            cpu_num=1, 
            runqueue_cutoff_length=8,
            max_prev_events_stored=2,
            time_ln_cap=16, 
            vsize_ln_cap=16,
            socket_port=14014,
            scheduler_port=17213,
        ):
        self.observation_space = spaces.Tuple([
            spaces.Dict({
                # Number of ghOSt scheduler callback (i.e. event) that triggered data transfer
                "callback_type": spaces.Discrete(8),
                # Information about the task with which the event occurred
                "task_metrics": spaces.Dict({
                    "run_state": spaces.Discrete(4),
                    "cpu_num": spaces.Discrete(cpu_num),
                    "preempted": spaces.Discrete(2),
                    "utime": spaces.Box(0, time_ln_cap, shape=(1,)),
                    "stime": spaces.Box(0, time_ln_cap, shape=(1,)),
                    "guest_time": spaces.Box(0, time_ln_cap, shape=(1,)),
                    "vsize": spaces.Box(0, vsize_ln_cap, shape=(1,)),
                }),
                # Information about tasks in the run queue
                "runqueue": spaces.Tuple([spaces.Dict({
                    "run_state": spaces.Discrete(4),
                    "cpu_num": spaces.Discrete(cpu_num),
                    "preempted": spaces.Discrete(2),
                    "utime": spaces.Box(0, time_ln_cap, shape=(1,)),
                    "stime": spaces.Box(0, time_ln_cap, shape=(1,)),
                    "guest_time": spaces.Box(0, time_ln_cap, shape=(1,)),
                    "vsize": spaces.Box(0, vsize_ln_cap, shape=(1,)),
                })] * runqueue_cutoff_length)
            })
        ] * (max_prev_events_stored + 1))
        self.action_space: spaces.Discrete = spaces.Discrete(runqueue_cutoff_length)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.socket_host = "localhost"
        self.socket_port = socket_port
        self.receive_timeout_seconds = 0.01
        self.scheduler_port = scheduler_port
        self.share_counter = 0
        self.parser = LnCapObservationParser(self.observation_space[0], runqueue_cutoff_length, time_ln_cap, vsize_ln_cap)
        self.actual_runqueue_length = 0
        self.max_prev_events_stored = max_prev_events_stored
        self.accumulated_metrics = [
            generate_zeroed_sample(self.observation_space[0]) for _ in range(self.max_prev_events_stored)
        ]
        self.actionable_event_metrics = None
        self.accumulated_metrics_lock = Lock()
        self.observations_ready = False
        self.background_collector_thread = StoppableThread(target=self._listen_for_updates, args=())

    def _start_collector(self):
        """Start background collector that waits for metrics."""
        self.background_collector_thread.start()

    def close(self):
        """Release external resources and perform other cleanup. 
        For now, just signal collector thread to stop."""
        self.background_collector_thread.stop()

    def _listen_for_updates(self):
        metrics_per_task = len(self.parser.task_metrics)
        metrics = self._get_raw_metrics()
        if metrics is None:
            # Timeout exceeded when getting new metrics. No worries, we'll retry in a bit.
            # For now, check if we need to exit (this runs in a checker loop, so just return).
            return
        assert (len(metrics) - 2) % metrics_per_task == 0
        action_required = (metrics[0] == 1)
        parsed_metrics = self.parser.parse(metrics[1:])
        try:
            self.accumulated_metrics_lock.acquire()
            if not action_required:
                self.accumulated_metrics.pop(0)
                self.accumulated_metrics.append(parsed_metrics)
            if action_required:
                self.actionable_event_metrics = parsed_metrics
                self.observations_ready = True
                self.actual_runqueue_length = (len(metrics) - 2 - metrics_per_task) // metrics_per_task
        finally:
            self.accumulated_metrics_lock.release()

    def _recv_n(self, sock: socket.socket, size: int):
        data = b""
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                raise ConnectionResetError("Scheduler has disconnected")
            data += packet
        return data

    def _get_raw_metrics(self):
        connection = None
        metrics_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            metrics_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            linger_struct = struct.pack('ii', 1, 0)
            metrics_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger_struct)
            metrics_socket.settimeout(self.receive_timeout_seconds)
            metrics_socket.bind((self.socket_host, self.socket_port))
            metrics_socket.listen()
            connection, address = metrics_socket.accept()
            print(f"SchedulerEnv._get_raw_metrics: Connected by address: {address}")
            length_data = self._recv_n(connection, struct.calcsize('!I'))
            length = struct.unpack('!I', length_data)[0]
            print(f"SchedulerEnv._get_raw_metrics: unpacked length: {length}")
            sequence_data = self._recv_n(connection, length * struct.calcsize('!Q'))
            sequence = struct.unpack(f'!{length}Q', sequence_data)
            return list(sequence)
        except OSError:
            return None
        finally:
            metrics_socket.close()
            connection and connection.close()

    def _send_action(self, action):
        action_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            action_socket.connect((self.socket_host, self.scheduler_port))
            packed_number = struct.pack('!I', action)
            action_socket.sendall(packed_number)
        finally:
            action_socket.close()

    def _is_action_valid(self, action: int):
        return action <= self.actual_runqueue_length and action < self.action_space.n

    def step(self, action):
        if not self._is_action_valid(action):
            default_observation = generate_zeroed_sample(self.observation_space)
            return default_observation, \
                -1, True, False, \
                    {"error": f"Actual run queue is too short to place this task on position {action}"}
        self._send_action(action)
        while not self.observations_ready:
            time.sleep(0.0001)
        self.accumulated_metrics_lock.acquire()
        observation = (*self.accumulated_metrics, self.actionable_event_metrics)
        self.observations_ready = False
        self.accumulated_metrics_lock.release()
        return observation, 0, False, False, {}

    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        `options` may have a boolean `restart` option (default is `False`).
        If set to `True`, will attempt to restart the scheduler, else will start a new
        learning episode right away.
        """
        super().reset(seed=seed)
        if options and options["restart"]:
            """try to restart the scheduler (it is running as root, so this need to run as root too!)
            Maybe getting SchedulerEnv to control scheduler's lifetime entirely with another
            constructor parameter is better. Would need `close()` method then."""
            raise NotImplementedError("Restart behavior is yet to be implemented")
        if not self.background_collector_thread.is_alive():
            self._start_collector()
        while not self.observations_ready:
            time.sleep(0.0001)
        self.accumulated_metrics_lock.acquire()
        observation = (*self.accumulated_metrics, self.actionable_event_metrics)
        self.observations_ready = False
        self.accumulated_metrics_lock.release()
        return observation, {}