import multiprocessing
import subprocess
import time
import os
import signal
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

def finalize(scheduler_process, experiments_process):
    original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        os.killpg(os.getpgid(scheduler_process.pid), signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, original_handler)
    scheduler_process.wait()
    if check_process_running(experiments_process):
        experiments_process.terminate()
        experiments_process.wait()

def main():
    # Global configuration parameters
    scheduler_to_environment_port = 14014
    environment_to_scheduler_port = 17213
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
    try:
        sched_environment = gymnasium.make(
            'rl_env/SchedulerEnv-v0', 
            cpu_num=cpu_count,
            socket_port=scheduler_to_environment_port,
            scheduler_port=environment_to_scheduler_port,
        )
        scheduler_process = subprocess.Popen(
            ["sudo", path_to_scheduler_binary, f"--ghost_cpus=0-{cpu_count - 1}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(2) # Startup time
        if not check_process_running(scheduler_process):
            raise Exception("Scheduler has exited unexpectedly")
        # If agent has not exited yet, it's likely fine, so run single_exp from bazel-bin
        single_exp_process = subprocess.Popen(
            [path_to_experiment_binary],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            observation, _ = sched_environment.reset()
            print(f"Reset environment and got initial observation: {observation}")
            print(f"Actual runqueue length: {sched_environment.unwrapped.actual_runqueue_length}")
            observation, reward, terminated, truncated, _ = sched_environment.step(0)
            print(f"Made a step and got observation: {observation}")
        except Exception as ex:
            finalize(scheduler_process, single_exp_process)
            raise ex
        single_exp_process.wait()
        finalize(scheduler_process, single_exp_process)
    finally:
        print_process_output(scheduler_process, "scheduler")
        print_process_output(single_exp_process, "single_exp")


if __name__ == "__main__":
    main()