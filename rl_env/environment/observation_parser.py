import math

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

def ln_capped(num, cap):
        if num == 0:
            return 0.0
        ln_value = math.log(num)
        return ln_value if ln_value <= cap else cap    

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
        self.runqueue_cutoff_length = int(runqueue_cutoff_length)
        self.time_cap = float(time_cap)
        self.vsize_cap = float(vsize_cap)


    def _reshape(self, metrics):
        """
        Reshapes metrics into dictionaries while zero-padding or trimming variable-length data. 
        Reshaping into dictionaries in not strictly necessary and should be avoided
        in high-performance environments, but this makes metrics easier to read by adding names
        to them. However, turning variable-length data into fixed-length data is required anyway.
        """
        result = {
            "callback_type": metrics[0],
            "task_metrics": {},
            "runqueue": [{} for i in range(self.runqueue_cutoff_length)]
        }
        for elem in result["runqueue"]:
            for metric_name in self.task_metrics:
                elem[metric_name] = 0.0
        metric_index = 1
        metrics_per_task = len(self.task_metrics)
        for metric_name in self.task_metrics:
            result["task_metrics"][metric_name] = metrics[metric_index]
            metric_index += 1
        while metric_index < len(metrics):
            runqueue_index = (metric_index - (metrics_per_task + 1)) // metrics_per_task
            if runqueue_index >= len(result["runqueue"]):
                break
            for metric_name in self.task_metrics:
                result["runqueue"][runqueue_index][metric_name] = metrics[metric_index]
                metric_index += 1
        return result
    

    def _cap(self, task_metrics_dict):
        task_metrics_dict["utime"] = ln_capped(task_metrics_dict["utime"], self.time_cap)
        task_metrics_dict["stime"] = ln_capped(task_metrics_dict["stime"], self.time_cap)
        task_metrics_dict["guest_time"] = ln_capped(task_metrics_dict["guest_time"], self.time_cap)
        task_metrics_dict["vsize"] = ln_capped(task_metrics_dict["vsize"], self.vsize_cap)


    def _transform(self, reshaped_metrics):
        """
        Transforms individual metrics (in this case — by taking a logarithm and capping the result).
        """
        self._cap(reshaped_metrics["task_metrics"])
        for task_metrics_dict in reshaped_metrics["runqueue"]:
            self._cap(task_metrics_dict)
        return reshaped_metrics
    

    def parse(self, metrics):
        reshaped_metrics = self._reshape(metrics)
        transformed_metrics = self._transform(reshaped_metrics)
        return transformed_metrics

