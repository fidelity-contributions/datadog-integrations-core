# (C) Datadog, Inc. 2024-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
import os
import time
from contextlib import ExitStack

import pytest

from datadog_checks.dev import get_here, run_command
from datadog_checks.dev.kind import kind_run
from datadog_checks.dev.kube_port_forward import port_forward

HERE = get_here()


def setup_tekton():
    run_command(["kubectl", "create", "namespace", "tekton-operator"])
    run_command(["kubectl", "apply", "-f", os.path.join(HERE, "kind", "tekton-operator.yaml"), "-n", "tekton-operator"])
    print("Waiting for tekton-operator to be ready...")
    time.sleep(30)
    run_command(
        ["kubectl", "wait", "pods", "--all", "--for=condition=Ready", "--timeout=300s", "-n", "tekton-operator"]
    )
    time.sleep(60)
    print("Waiting for tekton-pipelines-controller to be ready...")
    run_command(
        [
            "kubectl",
            "wait",
            "pods",
            "-l",
            "app=tekton-pipelines-controller",
            "--for=condition=Ready",
            "--timeout=300s",
            "-n",
            "tekton-pipelines",
        ]
    )
    print("Waiting for tekton-triggers-controller to be ready...")
    run_command(
        [
            "kubectl",
            "wait",
            "pods",
            "-l",
            "app=tekton-triggers-controller",
            "--for=condition=Ready",
            "--timeout=300s",
            "-n",
            "tekton-pipelines",
        ]
    )

    configs_to_apply = ["pipeline", "pipelinerun", "task", "taskrun"]

    for config in configs_to_apply:
        for action in ["hello", "sleep"]:
            run_command(
                [
                    "kubectl",
                    "apply",
                    "-f",
                    os.path.join(HERE, "kind", f"tekton-{config}-{action}.yaml"),
                    "-n",
                    "tekton-pipelines",
                ]
            )


@pytest.fixture(scope='session')
def dd_environment():
    with kind_run(conditions=[setup_tekton], sleep=60) as kubeconfig, ExitStack() as stack:
        instances = {}

        pipeline_host, pipeline_port = stack.enter_context(
            port_forward(kubeconfig, 'tekton-pipelines', 9090, 'service', 'tekton-pipelines-controller')
        )
        instances['pipelines_controller_endpoint'] = f'http://{pipeline_host}:{pipeline_port}/metrics'

        trigger_host, trigger_port = stack.enter_context(
            port_forward(kubeconfig, 'tekton-pipelines', 9000, 'service', 'tekton-triggers-controller')
        )
        instances['triggers_controller_endpoint'] = f'http://{trigger_host}:{trigger_port}/metrics'

        yield {'instances': [instances]}


@pytest.fixture
def pipelines_instance():
    return {
        "pipelines_controller_endpoint": "http://tekton-pipelines:9090",
    }


@pytest.fixture
def triggers_instance():
    return {
        "triggers_controller_endpoint": "http://tekton-triggers:9000",
    }
