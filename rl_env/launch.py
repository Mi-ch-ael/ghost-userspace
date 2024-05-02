from argparse import ArgumentParser, ArgumentTypeError
import multiprocessing
import os
import signal
import subprocess
import time
from typing import Union

import gymnasium

import environment.scheduler_env 


def check_process_running(process):
    return process.poll() is None


def print_scheduler_output(scheduler_process):
    stdout, stderr = scheduler_process.communicate()
    print(f"---scheduler stdout---")
    print(stdout.decode())
    print(f"---end scheduler stdout---")
    print(f"---scheduler stderr---")
    print(stderr.decode())
    print(f"---end scheduler stderr---")


def positive_integer(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise ArgumentTypeError(f"Expected positive integer, got {value}")
    return ivalue


def positive_float(value):
    fvalue = float(value)
    if fvalue <= 0.0:
        raise ArgumentTypeError(f"Expected positive real number, got {value}")
    return fvalue


def get_default_path_to_scheduler() -> str:
    return os.path.join(
        os.path.dirname(__file__), 
        os.pardir,
        "bazel-bin",
        "rl_scheduler_agent",
    )


def setup_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--cpu_count", type=positive_integer, default=multiprocessing.cpu_count(),
                        help="number of CPUs to use; actual CPUs used are the ones with the lowest indices (default: run on all CPUs)")
    parser.add_argument("--scheduler_verbosity", type=positive_integer, default=2,
                        help="scheduler verbosity level")
    parser.add_argument("--scheduler_binary", type=str, default=get_default_path_to_scheduler(),
                        help="path to `rl_scheduler_agent` binary (default: look in bazel-bin build dir)")
    parser.add_argument("--runqueue_length", type=positive_integer, default=5,
                        help="a fixed length to which run queue information will be either trimmed or zero-padded for RL agent")
    parser.add_argument("--max_prev_events_stored", type=positive_integer, default=2,
                        help="a number of recent non-actionable events, for which metrics are stored; must be positive")
    parser.add_argument("--time_ln_cap", type=positive_float, default=16.0,
                        help="cap for all time metrics to keep observation space from growing infinitely")
    parser.add_argument("--vsize_ln_cap", type=positive_float, default=16.0,
                        help="cap for `vsize` metric to keep observation space from growing infinitely")
    return parser


def use_env(env):
    # Your code goes here
    observation, _ = env.reset()
    while True:
        runqueue = observation[-1]["runqueue"]
        target_index = len(runqueue) - 1
        try:
            target_index = next(i for i, value in enumerate(runqueue) if value["vsize"] == 0.0)
        except StopIteration:
            pass
        observation, reward, terminated, truncated, _ = env.step(target_index)
        if terminated or truncated:
            break


def terminate_scheduler(scheduler_process: Union[subprocess.Popen, None], timeout_sigint=0.1, timeout_sigterm=1):
    if not scheduler_process or not check_process_running(scheduler_process):
        return
    try:
        os.killpg(os.getpgid(scheduler_process.pid), signal.SIGINT)
        scheduler_process.wait(timeout=timeout_sigint)
        return
    except subprocess.TimeoutExpired:
        print("\nScheduler does not respond to SIGINT; switching to SIGTERM.")
    try:
        os.killpg(os.getpgid(scheduler_process.pid), signal.SIGTERM)
        scheduler_process.wait(timeout=timeout_sigterm)
    except subprocess.TimeoutExpired:
        print("Scheduler does not respond to SIGTERM; switching to SIGKILL.")
        os.killpg(os.getpgid(scheduler_process.pid), signal.SIGKILL)
        scheduler_process.wait()


def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    env = gymnasium.make(
        'rl_env/SchedulerEnv-v0', 
        cpu_num=args.cpu_count,
        runqueue_cutoff_length=args.runqueue_length,
        max_prev_events_stored=args.max_prev_events_stored,
        time_ln_cap=args.time_ln_cap,
        vsize_ln_cap=args.vsize_ln_cap,
    )
    scheduler_process = None
    try:
        scheduler_process = subprocess.Popen(
            [
                "sudo", args.scheduler_binary, 
                f"--ghost_cpus=0-{args.cpu_count - 1}", 
                f"--verbose={args.scheduler_verbosity}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        time.sleep(1)
        if not check_process_running(scheduler_process):
            raise Exception("Scheduler has exited unexpectedly")
        use_env(env)
    finally:
        env.close()
        terminate_scheduler(scheduler_process)
        if scheduler_process:
            print_scheduler_output(scheduler_process)


if __name__ == "__main__":
    main()