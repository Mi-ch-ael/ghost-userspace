import multiprocessing
import os
import signal
import subprocess
import time

import gymnasium

from environment.scheduler_env import SchedulerEnv


def check_process_running(process):
    return process.poll() is None


def print_process_output(process, human_readable_name):
    stdout, stderr = process.communicate()
    print(f"---stdout of {human_readable_name}---")
    print(stdout.decode())
    print(f"---end stdout of {human_readable_name}---")
    print(f"---stderr of {human_readable_name}---")
    print(stderr.decode())
    print(f"---end stderr of {human_readable_name}---")


def use_environment(env, verbose = True):
    observation, _ = env.reset()
    assert observation[-1]["callback_type"] == 0
    if verbose:
        print(f"Observation from reset (task metrics): {observation[-1]['task_metrics']}")
    observation, reward, terminated, truncated, _ = env.step(0)
    assert reward == 0.0
    assert terminated == False
    assert truncated == False
    assert observation[-1]["callback_type"] == 0
    if verbose:
        print(f"Observation from step (task metrics): {observation[-1]['task_metrics']}")


def main():
    # Global params
    scheduler_to_environment_port = 14014
    environment_to_scheduler_port = 17213
    scheduler_verbosity = 2
    cpu_count = multiprocessing.cpu_count()
    path_to_scheduler_binary = os.path.join(
        os.path.dirname(__file__), 
        os.pardir,
        "bazel-bin",
        "rl_scheduler_agent",
    )
    path_to_experiment_binary = os.path.join(
        os.path.dirname(__file__), 
        os.pardir,
        "bazel-bin",
        "single_exp",
    )
    single_exp_process = None
    scheduler_process = None
    try:
        env = gymnasium.make(
            'rl_env/SchedulerEnv-v0', 
            cpu_num=cpu_count,
            socket_port=scheduler_to_environment_port,
            scheduler_port=environment_to_scheduler_port,
        )
        scheduler_process = subprocess.Popen(
            ["sudo", path_to_scheduler_binary, f"--ghost_cpus=0-{cpu_count - 1}", f"--verbose={scheduler_verbosity}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(1)
        if not check_process_running(scheduler_process):
            raise Exception("Scheduler has exited unexpectedly")
        # If agent has not exited yet, it's likely fine, so run single_exp from bazel-bin
        single_exp_process = subprocess.Popen(
            [path_to_experiment_binary],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        use_environment(env)
        single_exp_process.wait()  
    finally:
        env.close()
        if scheduler_process and check_process_running(scheduler_process):
            original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                os.killpg(os.getpgid(scheduler_process.pid), signal.SIGINT)
            finally:
                signal.signal(signal.SIGINT, original_handler)
            scheduler_process.wait()
        if scheduler_process:
            print_process_output(scheduler_process, "scheduler")
        if single_exp_process:
            print_process_output(single_exp_process, "single_exp")


if __name__ == "__main__":
    main()