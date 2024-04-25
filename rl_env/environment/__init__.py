from gymnasium.envs.registration import register

register(
     id="rl_env/SchedulerEnv-v0",
     entry_point="environment.scheduler_env:SchedulerEnv",
)