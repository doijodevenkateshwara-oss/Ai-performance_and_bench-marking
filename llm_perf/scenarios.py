"""Build Locust User classes from config scenarios."""

from __future__ import annotations

import random

from locust import TaskSet, User, between, task

from .config import PerfConfig, ScenarioConfig
from .tasks import RequestContext

# Registry populated by build_users(); consumed by generated locustfile.
USERS: list[type[User]] = []
CTX: RequestContext | None = None


def _taskset_for(ctx: RequestContext, scenario: ScenarioConfig, wait):
    class ScenarioTaskSet(TaskSet):
        wait_time = wait

        @task
        def execute(self):
            from .metrics import set_environment

            set_environment(self.user.environment)
            ctx.run_scenario(scenario, random.Random())

    ScenarioTaskSet.__name__ = f"TaskSet_{scenario.name}"
    return ScenarioTaskSet


def build_users(config: PerfConfig) -> list[type[User]]:
    """Create one User class per scenario."""
    global CTX, USERS
    CTX = RequestContext(config)
    users = []
    for sc in config.scenarios:
        think = sc.think_time
        wait = between(think[0], think[1]) if think else between(0.5, 2.0)

        class ScenarioUser(User):
            wait_time = wait
            weight = sc.weight

            def on_start(self):
                from .metrics import record_users

                record_users(1)

            def on_stop(self):
                from .metrics import record_users

                record_users(-1)

        ScenarioUser.tasks = [_taskset_for(CTX, sc, wait)]
        ScenarioUser.__name__ = f"User_{sc.name}"
        users.append(ScenarioUser)
    USERS = users
    return users


def get_context() -> RequestContext | None:
    return CTX
