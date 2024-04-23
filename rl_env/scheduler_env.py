import gymnasium
from gymnasium import spaces

class SchedulerEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None, runqueue_cutoff_length=8):
        self.runqueue_cutoff_length = runqueue_cutoff_length
        self.observation_space = spaces.Tuple([
            spaces.Discrete(8),
            # `TaskInfo`,
            # array of self.runqueue_cutoff_length `TaskInfo`s
        ])