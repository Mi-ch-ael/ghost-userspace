import signal
import unittest
import unittest.mock
import multiprocessing
import subprocess
import time
import os
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


class TestSchedulerPlusEnv(unittest.TestCase):
    @unittest.mock.patch("environment.scheduler_env.SchedulerEnv._send_action")
    def test_create_all(self, mock_send_action):
        mock_send_action.return_value = None
        # Global configuration parameters
        scheduler_to_environment_port = 14014
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
            sched_environment = SchedulerEnv(socket_port=scheduler_to_environment_port, cpu_num=cpu_count)
            scheduler_process = subprocess.Popen(
                ["sudo", path_to_scheduler_binary, f"--ghost_cpus=0-{cpu_count - 1}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(2) # Startup time
            if not check_process_running(scheduler_process):
                self.fail("Scheduler has exited unexpectedly")
            # If agent has not exited yet, it's likely fine, so run single_exp from bazel-bin
            single_exp_process = subprocess.Popen(
                [path_to_experiment_binary],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                observation, _ = sched_environment.reset()
                # First callback would be TaskNew (0) => run_state == 2 (RlTaskState::kQueued)
                self.assertEqual(observation["callback_type"], 0)
                self.assertEqual(observation["task_metrics"]["run_state"], 2)
                self.assertEqual(observation["runqueue"][0]["run_state"], 2)
                observation, reward, terminated, truncated, _ = sched_environment.step(0)
                self.assertLess(observation["callback_type"], 8)
                self.assertLess(observation["task_metrics"]["run_state"], 4)
                self.assertLess(observation["runqueue"][0]["run_state"], 4)
            except Exception as ex:
                finalize(scheduler_process, single_exp_process)
                self.fail(f"SchedulerEnv error: {ex}")
            single_exp_process.wait()
            finalize(scheduler_process, single_exp_process)
        finally:
            print_process_output(scheduler_process, "scheduler")
            print_process_output(single_exp_process, "single_exp")


if __name__ == "__main__":
    unittest.main()