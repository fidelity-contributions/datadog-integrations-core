# (C) Datadog, Inc. 2018-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)

import os

import pytest

from datadog_checks.dev import get_docker_hostname
from datadog_checks.dev.ci import running_on_ci
from datadog_checks.dev.utils import ON_WINDOWS

# Ignore missing library to not require it for e2e
try:
    from datadog_checks.ibm_mq.metrics import COUNT, GAUGE
except ImportError:
    COUNT = GAUGE = ''

RUNNING_ON_WINDOWS_CI = ON_WINDOWS and running_on_ci()
skip_windows_ci = pytest.mark.skipif(RUNNING_ON_WINDOWS_CI, reason='MQ server cannot be setup on Windows VMs in CI')


HERE = os.path.dirname(os.path.abspath(__file__))
COMPOSE_DIR = os.path.join(HERE, 'compose')

HOST = get_docker_hostname()
PORT = '11414'

USERNAME = 'admin'
PASSWORD = 'passw0rd'

QUEUE_MANAGER = 'QM1'
CHANNEL = 'DEV.ADMIN.SVRCONN'
CHANNEL_SSL = 'PYMQI.SSL.SVRCONN'
SSL_CLIENT_LABEL = 'client'
SSL_CYPHER_SPEC = 'TLS_RSA_WITH_AES_256_CBC_SHA256'

QUEUE = 'DEV.QUEUE.1'

BAD_CHANNEL = 'DEV.NOTHERE.SVRCONN'

MQ_VERSION = int(os.environ.get('IBM_MQ_VERSION', '9'))
MQ_COMPOSE_VERSION = os.environ.get('IBM_MQ_COMPOSE_VERSION', '')
MQ_VERSION_RAW = os.environ.get('IBM_MQ_VERSION_RAW', '9.1.1.0')

IS_CLUSTER = 'cluster' in MQ_COMPOSE_VERSION

COMPOSE_FILE_NAME = 'docker-compose-v{}.yml'.format(MQ_COMPOSE_VERSION)

COMPOSE_FILE_PATH = os.path.join(COMPOSE_DIR, COMPOSE_FILE_NAME)

INSTANCE = {
    'channel': CHANNEL,
    'queue_manager': QUEUE_MANAGER,
    'host': HOST,
    'port': PORT,
    'username': USERNAME,
    'password': PASSWORD,
    'queues': [QUEUE],
    'channels': [CHANNEL, BAD_CHANNEL],
    'tags': ['foo:bar'],
    'collect_statistics_metrics': True,
}

INSTANCE_SSL = {
    'channel': CHANNEL_SSL,
    'queue_manager': QUEUE_MANAGER,
    'host': HOST,
    'port': PORT,
    'username': USERNAME,
    'password': PASSWORD,
    'auto_discover_queues': True,
    'collect_statistics_metrics': True,
    'channels': [CHANNEL, BAD_CHANNEL],
    'ssl_auth': 'yes',
    'ssl_cipher_spec': SSL_CYPHER_SPEC,
    'ssl_key_repository_location': '/opt/pki/keys/client',
    'ssl_certificate_label': SSL_CLIENT_LABEL,
}

INSTANCE_METADATA = {
    'channel': CHANNEL,
    'queue_manager': QUEUE_MANAGER,
    'host': HOST,
    'port': PORT,
    'username': USERNAME,
    'password': PASSWORD,
    'queues': [QUEUE],
    'channels': [CHANNEL, BAD_CHANNEL],
    'tags': ['foo:bar'],
}

INSTANCE_WITH_CONNECTION_NAME = {
    'channel': CHANNEL,
    'queue_manager': QUEUE_MANAGER,
    'connection_name': "{}({})".format(HOST, PORT),
    'username': USERNAME,
    'password': PASSWORD,
    'queues': [QUEUE],
    'channels': [CHANNEL, BAD_CHANNEL],
}

INSTANCE_QUEUE_PATTERN = {
    'channel': CHANNEL,
    'queue_manager': QUEUE_MANAGER,
    'host': HOST,
    'port': PORT,
    'username': USERNAME,
    'password': PASSWORD,
    'queue_patterns': ['DEV.*', 'SYSTEM.*'],
    'channels': [CHANNEL, BAD_CHANNEL],
}

INSTANCE_QUEUE_REGEX = {
    'channel': CHANNEL,
    'queue_manager': QUEUE_MANAGER,
    'host': HOST,
    'port': PORT,
    'username': USERNAME,
    'password': PASSWORD,
    'queue_regex': [r'^DEV\..*$', r'^SYSTEM\..*$'],
    'channels': [CHANNEL, BAD_CHANNEL],
}

INSTANCE_COLLECT_ALL = {
    'channel': CHANNEL,
    'queue_manager': QUEUE_MANAGER,
    'host': HOST,
    'port': PORT,
    'username': USERNAME,
    'password': PASSWORD,
    'auto_discover_queues': True,
    'collect_statistics_metrics': True,
    'channels': [CHANNEL, BAD_CHANNEL],
}

INSTANCE_QUEUE_REGEX_TAG = {
    'channel': CHANNEL,
    'queue_manager': QUEUE_MANAGER,
    'host': HOST,
    'port': PORT,
    'username': USERNAME,
    'password': PASSWORD,
    'queues': [QUEUE],
    'queue_tag_re': {'DEV.QUEUE.*': "foo:bar"},
}

E2E_METADATA = {
    'docker_volumes': ['{}/agent_scripts/start_commands.sh:/tmp/start_commands.sh'.format(HERE)],
    'start_commands': ['bash /tmp/start_commands.sh'],
    'env_vars': {'LD_LIBRARY_PATH': '/opt/mqm/lib64:/opt/mqm/lib', 'C_INCLUDE_PATH': '/opt/mqm/inc'},
}

DEFAULT_QUEUE_METRICS = [
    ('ibm_mq.queue.service_interval', GAUGE),
    ('ibm_mq.queue.inhibit_put', GAUGE),
    ('ibm_mq.queue.depth_low_limit', GAUGE),
    ('ibm_mq.queue.inhibit_get', GAUGE),
    ('ibm_mq.queue.harden_get_backout', GAUGE),
    ('ibm_mq.queue.service_interval_event', GAUGE),
    ('ibm_mq.queue.trigger_control', GAUGE),
    ('ibm_mq.queue.usage', GAUGE),
    ('ibm_mq.queue.scope', GAUGE),
    ('ibm_mq.queue.type', GAUGE),
    ('ibm_mq.queue.depth_max', GAUGE),
    ('ibm_mq.queue.backout_threshold', GAUGE),
    ('ibm_mq.queue.depth_high_event', GAUGE),
    ('ibm_mq.queue.depth_low_event', GAUGE),
    ('ibm_mq.queue.trigger_message_priority', GAUGE),
    ('ibm_mq.queue.depth_current', GAUGE),
    ('ibm_mq.queue.depth_max_event', GAUGE),
    ('ibm_mq.queue.open_input_count', GAUGE),
    ('ibm_mq.queue.persistence', GAUGE),
    ('ibm_mq.queue.trigger_depth', GAUGE),
    ('ibm_mq.queue.max_message_length', GAUGE),
    ('ibm_mq.queue.depth_high_limit', GAUGE),
    ('ibm_mq.queue.priority', GAUGE),
    ('ibm_mq.queue.input_open_option', GAUGE),
    ('ibm_mq.queue.message_delivery_sequence', GAUGE),
    ('ibm_mq.queue.retention_interval', GAUGE),
    ('ibm_mq.queue.open_output_count', GAUGE),
    ('ibm_mq.queue.trigger_type', GAUGE),
    ('ibm_mq.queue.depth_percent', GAUGE),
]

RESET_QUEUE_METRICS = [
    ('ibm_mq.queue.high_q_depth', GAUGE),
    ('ibm_mq.queue.msg_deq_count', COUNT),
    ('ibm_mq.queue.msg_enq_count', COUNT),
    ('ibm_mq.queue.time_since_reset', COUNT),
]

QUEUE_METRICS = DEFAULT_QUEUE_METRICS + RESET_QUEUE_METRICS

QUEUE_STATUS_METRICS = [
    ('ibm_mq.queue.oldest_message_age', GAUGE),
    ('ibm_mq.queue.uncommitted_msgs', GAUGE),
    ('ibm_mq.queue.last_get_time', GAUGE),
    ('ibm_mq.queue.last_put_time', GAUGE),
]

CHANNEL_METRICS = [
    ('ibm_mq.channel.batch_size', GAUGE),
    ('ibm_mq.channel.batch_interval', GAUGE),
    ('ibm_mq.channel.long_retry', GAUGE),
    ('ibm_mq.channel.long_timer', GAUGE),
    ('ibm_mq.channel.max_message_length', GAUGE),
    ('ibm_mq.channel.short_retry', GAUGE),
    ('ibm_mq.channel.disc_interval', GAUGE),
    ('ibm_mq.channel.hb_interval', GAUGE),
    ('ibm_mq.channel.keep_alive_interval', GAUGE),
    ('ibm_mq.channel.mr_count', GAUGE),
    ('ibm_mq.channel.mr_interval', GAUGE),
    ('ibm_mq.channel.network_priority', GAUGE),
    ('ibm_mq.channel.npm_speed', GAUGE),
    ('ibm_mq.channel.sharing_conversations', GAUGE),
    ('ibm_mq.channel.short_timer', GAUGE),
]

CHANNEL_STATUS_METRICS = [
    ('ibm_mq.channel.buffers_rcvd', GAUGE),
    ('ibm_mq.channel.buffers_sent', GAUGE),
    ('ibm_mq.channel.bytes_rcvd', GAUGE),
    ('ibm_mq.channel.bytes_sent', GAUGE),
    ('ibm_mq.channel.channel_status', GAUGE),
    ('ibm_mq.channel.mca_status', GAUGE),
    ('ibm_mq.channel.msgs', GAUGE),
    ('ibm_mq.channel.ssl_key_resets', GAUGE),
    ('ibm_mq.channel.conn_status', GAUGE),
    ('ibm_mq.channel.connections_active', GAUGE),
]

CHANNEL_STATS_METRICS = [
    ('ibm_mq.stats.channel.msgs', COUNT),
    ('ibm_mq.stats.channel.bytes', COUNT),
    ('ibm_mq.stats.channel.put_retries', COUNT),
]

QUEUE_STATS_METRICS = [
    ('ibm_mq.stats.queue.q_min_depth', GAUGE),
    ('ibm_mq.stats.queue.q_max_depth', GAUGE),
    ('ibm_mq.stats.queue.put_fail_count', COUNT),
    ('ibm_mq.stats.queue.get_fail_count', COUNT),
    ('ibm_mq.stats.queue.put1_fail_count', COUNT),
    ('ibm_mq.stats.queue.browse_fail_count', COUNT),
    ('ibm_mq.stats.queue.non_queued_msg_count', COUNT),
    ('ibm_mq.stats.queue.expired_msg_count', COUNT),
    ('ibm_mq.stats.queue.purge_count', COUNT),
]

# These are Queue Stat metrics that return a list containing both persistent and non-persistent metrics
# These metrics have an extra tag for `persistent`.
QUEUE_STATS_LIST_METRICS = [
    ('ibm_mq.stats.queue.avg_q_time', GAUGE),
    ('ibm_mq.stats.queue.put_count', COUNT),
    ('ibm_mq.stats.queue.get_count', COUNT),
    ('ibm_mq.stats.queue.browse_bytes', GAUGE),
    ('ibm_mq.stats.queue.browse_count', COUNT),
    ('ibm_mq.stats.queue.get_bytes', COUNT),
    ('ibm_mq.stats.queue.put_bytes', COUNT),
    ('ibm_mq.stats.queue.put1_count', COUNT),
]

if IS_CLUSTER:
    CHANNEL_STATUS_METRICS.extend(
        [
            ('ibm_mq.channel.batches', GAUGE),
            ('ibm_mq.channel.current_msgs', GAUGE),
            ('ibm_mq.channel.indoubt_status', GAUGE),
        ]
    )

METRICS = (
    [
        ('ibm_mq.queue_manager.dist_lists', GAUGE),
        ('ibm_mq.queue_manager.max_msg_list', GAUGE),
        ('ibm_mq.channel.channels', GAUGE),
        ('ibm_mq.channel.count', GAUGE),
    ]
    + QUEUE_METRICS
    + QUEUE_STATUS_METRICS
    + CHANNEL_METRICS
    + CHANNEL_STATUS_METRICS
)

OPTIONAL_METRICS = [
    ('ibm_mq.queue.max_channels', GAUGE),
    ('ibm_mq.stats.channel.full_batches', COUNT),
    ('ibm_mq.stats.channel.incomplete_batches', COUNT),
    ('ibm_mq.stats.channel.avg_batch_size', GAUGE),
]

# stats metrics are not always present at each check run
OPTIONAL_METRICS.extend(CHANNEL_STATS_METRICS)
OPTIONAL_METRICS.extend(QUEUE_STATS_METRICS)
OPTIONAL_METRICS.extend(QUEUE_STATS_LIST_METRICS)


def assert_all_metrics(aggregator, minimum_tags=None, hostname=None):
    for metric, metric_type in METRICS:
        aggregator.assert_metric(metric, metric_type=getattr(aggregator, metric_type.upper()), hostname=hostname)
        minimum_tags = minimum_tags or []
        for tag in minimum_tags:
            aggregator.assert_metric_has_tag(metric, tag)

    for metric, metric_type in OPTIONAL_METRICS:
        aggregator.assert_metric(
            metric, metric_type=getattr(aggregator, metric_type.upper()), hostname=hostname, at_least=0
        )

    aggregator.assert_all_metrics_covered()
