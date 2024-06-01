"""Hamilton-runhouse integration"""
from typing import Any

from hamilton.execution import executors
from hamilton.execution.executors import MultiThreadingExecutor, TaskFuture, TaskFutureWrappingPythonFuture
from hamilton.execution.grouping import TaskImplementation

import runhouse as rh


class RunhouseExecutor(MultiThreadingExecutor):
    """Simple runhouse executor -- this doesn't handle state super well,
    but it does the trick nicely for most cases."""

    def __init__(self, max_tasks: int, cluster: rh.Cluster, env: rh.Env):
        """Initializes a RunhouseExecutor -- pass int he cluster + env to use.

        :param max_tasks: Maximum number of tasks to run concurrently
        :param cluster: Cluster to run on -- use runhouse primitives
        :param env: Environment to use -- use runhouse primitives
        """
        super(RunhouseExecutor, self).__init__(max_tasks=max_tasks)
        self.runhouse_function = rh.function(executors.base_execute_task).to(cluster, env)
        self.num_executions = 0

    def submit_task(self, task: TaskImplementation) -> TaskFuture:
        """Submits task using multithreading + runhouse execution"""
        future = self.pool.submit(self.runhouse_function, task)
        self.active_futures.append(future)
        return TaskFutureWrappingPythonFuture(future)
