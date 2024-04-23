from gymnasium import spaces

class ObservationParser:
    task_metrics = (
        "run_state",
        "cpu_num",
        "preempted",
        "utime",
        "stime",
        "guest_time",
        "vsize",
    )
    def parse(self, metrics):
        raise NotImplementedError(f"Don't use base {self.__class__.__name__} to parse metrics")
    

class LnCapObservationParser(ObservationParser):
    """
    Transforms time metrics and memory size by taking the logarithm and capping them if necessary.
    ln(0) is defined as 0. Leaves other metrics unchanged.
    Run queue information is trimmed or zero-padded to a fixed length.
    """
    def __init__(self, runqueue_cutoff_length, time_cap, vsize_cap):
        """
        * `runqueue_cutoff_length` — fixed length to zero-pad/trim run queue information.
        * `time_cap` — a cap for all time metrics. If logarithm of a time metric is larger, cap value
        will be returned instead.
        * `vsize_cap` — a cap for vsize. If logarithm of vsize is larger, cap value will be returned
        instead.
        """
        self.runqueue_cutoff_length = runqueue_cutoff_length
        self.time_cap = time_cap
        self.vsize_cap = vsize_cap

    def parse(self, metrics):
        result = {
            "callback_type": metrics[0],
            "task_metrics": {},
            "runqueue": [{} for i in range(self.runqueue_cutoff_length)]
        }
        for elem in result["runqueue"]:
            for metric_name in self.task_metrics:
                elem[metric_name] = 0.0 # int vs double ??
        metric_index = 1
        metrics_per_task = len(self.task_metrics)
        for metric_name in self.task_metrics:
            result["task_metrics"][metric_name] = metrics[metric_index]
            metric_index += 1
        while metric_index < len(metrics):
            runqueue_index = (metric_index - (metrics_per_task + 1)) // metrics_per_task
            for metric_name in self.task_metrics:
                result["runqueue"][runqueue_index][metric_name] = metrics[metric_index]
                metric_index += 1
        return result
