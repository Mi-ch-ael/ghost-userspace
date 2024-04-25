import gymnasium
from gymnasium import spaces
from environment.observation_parser import LnCapObservationParser
from environment.scheduler_spaces import generate_zeroed_sample

class SchedulerEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(
            self, 
            render_mode=None, 
            cpu_num=1, 
            runqueue_cutoff_length=8, 
            time_ln_cap=16, 
            vsize_ln_cap=16,
        ):
        self.observation_space = spaces.Dict({
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
        self.action_space = spaces.Discrete(runqueue_cutoff_length)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.share_counter = 0
        self.parser = LnCapObservationParser(runqueue_cutoff_length, time_ln_cap, vsize_ln_cap)
        self.actual_runqueue_length = 0

    def _get_raw_metrics(self):
        raise NotImplementedError("Methods requiring actual communication are not implemented yet")

    def _send_action(self, action):
        raise NotImplementedError("Methods requiring actual communication are not implemented yet")

    def step(self, action):
        if action >= self.actual_runqueue_length:
            default_observation = generate_zeroed_sample(self.observation_space)
            return default_observation, \
                -1, True, False, {"error": f"No task number {action} in run queue"}
        self._send_action(action)
        raw_metrics = self._get_raw_metrics()
        observation = self.parser.parse(raw_metrics)
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
        raw_metrics = self._get_raw_metrics()
        observation = self.parser.parse(raw_metrics)
        return observation, {}