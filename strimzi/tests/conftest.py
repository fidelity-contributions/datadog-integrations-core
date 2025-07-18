# (C) Datadog, Inc. 2023-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
import os
import os.path
import tempfile
from contextlib import ExitStack

import pytest

from datadog_checks.dev import run_command
from datadog_checks.dev.http import MockResponse
from datadog_checks.dev.kind import kind_run
from datadog_checks.dev.kube_port_forward import port_forward
from datadog_checks.strimzi import StrimziCheck

from .common import HERE, KUBERNETES_VERSION, STRIMZI_VERSION


def setup_strimzi():
    run_command(["kubectl", "create", "namespace", "kafka"])
    run_command(
        ["kubectl", "create", "-f", os.path.join(HERE, "kind", STRIMZI_VERSION, "strimzi_install.yaml"), "-n", "kafka"]
    )
    run_command(
        [
            "kubectl",
            "apply",
            "-f",
            os.path.join(HERE, "kind", STRIMZI_VERSION, "kafka.yaml"),
            "-n",
            "kafka",
        ]
    )
    run_command(["kubectl", "wait", "kafka/my-cluster", "--for=condition=Ready", "--timeout=600s", "-n", "kafka"])

    for file in ("topic.yaml", "user.yaml", "connect.yaml", "connectors.yaml"):
        run_command(
            [
                "kubectl",
                "apply",
                "-f",
                os.path.join(HERE, "kind", STRIMZI_VERSION, file),
                "-n",
                "kafka",
            ]
        )


def render_kind_config(kubernetes_version):
    template_config_path = os.path.join(HERE, 'kind', 'kind-config.yaml')
    with open(template_config_path, "r") as f:
        kind_config_content = f.read().replace('%%KUBERNETES_VERSION%%', kubernetes_version)
    return kind_config_content


@pytest.fixture(scope='session')
def dd_environment(dd_save_state):
    if not KUBERNETES_VERSION:
        pytest.fail("KUBERNETES_VERSION is not set")
    if not STRIMZI_VERSION:
        pytest.fail("STRIMZI_VERSION is not set")

    kind_config_content = render_kind_config(KUBERNETES_VERSION)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as kind_config:
        kind_config.write(kind_config_content)
        kind_config.flush()

        with kind_run(conditions=[setup_strimzi], kind_config=kind_config.name) as kubeconfig:
            with ExitStack() as stack:
                cluster_operator_ip, cluster_operator_port = stack.enter_context(
                    port_forward(kubeconfig, 'kafka', 8080, 'deployment', 'strimzi-cluster-operator')
                )
                topic_operator_ip, topic_operator_port = stack.enter_context(
                    port_forward(kubeconfig, 'kafka', 8080, 'deployment', 'my-cluster-entity-operator')
                )
                user_operator_ip, user_operator_port = stack.enter_context(
                    port_forward(kubeconfig, 'kafka', 8081, 'deployment', 'my-cluster-entity-operator')
                )

                yield {
                    "cluster_operator_endpoint": f"http://{cluster_operator_ip}:{cluster_operator_port}/metrics",
                    "topic_operator_endpoint": f"http://{topic_operator_ip}:{topic_operator_port}/metrics",
                    "user_operator_endpoint": f"http://{user_operator_ip}:{user_operator_port}/metrics",
                }


@pytest.fixture()
def check():
    return lambda instance: StrimziCheck('strimzi', {}, [instance])


def mock_http_responses(url, **_params):
    mapping = {
        'http://cluster-operator:8080/metrics': 'cluster_operator_metrics.txt',
        'http://entity-operator:8080/metrics': 'topic_operator_metrics.txt',
        'http://entity-operator:8081/metrics': 'user_operator_metrics.txt',
    }

    metrics_file = mapping.get(url)

    if not metrics_file:
        pytest.fail(f"url `{url}` not registered")

    with open(os.path.join(HERE, 'fixtures', STRIMZI_VERSION, metrics_file)) as f:
        return MockResponse(content=f.read())
