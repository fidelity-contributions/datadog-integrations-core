# (C) Datadog, Inc. 2020-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
import pytest

from datadog_checks.dev.jmx import JVM_E2E_METRICS_NEW

pytestmark = [pytest.mark.e2e]

METRICS = [
    'hivemq.cluster.name_request.retry.count',
    'hivemq.cpu_cores.licensed',
    'hivemq.cpu_cores.used',
    'hivemq.keep_alive.disconnect.count',
    'hivemq.messages.dropped.count',
    'hivemq.messages.dropped.internal_error.count',
    'hivemq.messages.dropped.message_too_large.count',
    'hivemq.messages.dropped.mqtt_packet_too_large.count',
    'hivemq.messages.dropped.not_writable.count',
    'hivemq.messages.dropped.qos_0_memory_exceeded.count',
    'hivemq.messages.dropped.queue_full.count',
    'hivemq.messages.expired_messages',
    'hivemq.messages.incoming.auth.count',
    'hivemq.messages.incoming.connect.count',
    'hivemq.messages.incoming.connect.mqtt3.count',
    'hivemq.messages.incoming.connect.mqtt5.count',
    'hivemq.messages.incoming.disconnect.count',
    'hivemq.messages.incoming.pingreq.count',
    'hivemq.messages.incoming.puback.count',
    'hivemq.messages.incoming.pubcomp.count',
    'hivemq.messages.incoming.publish.bytes.50th_percentile',
    'hivemq.messages.incoming.publish.bytes.75th_percentile',
    'hivemq.messages.incoming.publish.bytes.95th_percentile',
    'hivemq.messages.incoming.publish.bytes.98th_percentile',
    'hivemq.messages.incoming.publish.bytes.999th_percentile',
    'hivemq.messages.incoming.publish.bytes.99th_percentile',
    'hivemq.messages.incoming.publish.bytes.count',
    'hivemq.messages.incoming.publish.bytes.max',
    'hivemq.messages.incoming.publish.bytes.mean',
    'hivemq.messages.incoming.publish.bytes.min',
    'hivemq.messages.incoming.publish.bytes.snapshot_size',
    'hivemq.messages.incoming.publish.bytes.std_dev',
    'hivemq.messages.incoming.publish.count',
    'hivemq.messages.incoming.pubrec.count',
    'hivemq.messages.incoming.pubrel.count',
    'hivemq.messages.incoming.subscribe.count',
    'hivemq.messages.incoming.total.bytes.50th_percentile',
    'hivemq.messages.incoming.total.bytes.75th_percentile',
    'hivemq.messages.incoming.total.bytes.95th_percentile',
    'hivemq.messages.incoming.total.bytes.98th_percentile',
    'hivemq.messages.incoming.total.bytes.999th_percentile',
    'hivemq.messages.incoming.total.bytes.99th_percentile',
    'hivemq.messages.incoming.total.bytes.count',
    'hivemq.messages.incoming.total.bytes.max',
    'hivemq.messages.incoming.total.bytes.mean',
    'hivemq.messages.incoming.total.bytes.min',
    'hivemq.messages.incoming.total.bytes.snapshot_size',
    'hivemq.messages.incoming.total.bytes.std_dev',
    'hivemq.messages.incoming.total.count',
    'hivemq.messages.incoming.unsubscribe.count',
    'hivemq.messages.outgoing.auth.count',
    'hivemq.messages.outgoing.connack.count',
    'hivemq.messages.outgoing.disconnect.count',
    'hivemq.messages.outgoing.pingresp.count',
    'hivemq.messages.outgoing.puback.count',
    'hivemq.messages.outgoing.pubcomp.count',
    'hivemq.messages.outgoing.publish.bytes.50th_percentile',
    'hivemq.messages.outgoing.publish.bytes.75th_percentile',
    'hivemq.messages.outgoing.publish.bytes.95th_percentile',
    'hivemq.messages.outgoing.publish.bytes.98th_percentile',
    'hivemq.messages.outgoing.publish.bytes.999th_percentile',
    'hivemq.messages.outgoing.publish.bytes.99th_percentile',
    'hivemq.messages.outgoing.publish.bytes.count',
    'hivemq.messages.outgoing.publish.bytes.max',
    'hivemq.messages.outgoing.publish.bytes.mean',
    'hivemq.messages.outgoing.publish.bytes.min',
    'hivemq.messages.outgoing.publish.bytes.snapshot_size',
    'hivemq.messages.outgoing.publish.bytes.std_dev',
    'hivemq.messages.outgoing.publish.count',
    'hivemq.messages.outgoing.pubrec.count',
    'hivemq.messages.outgoing.pubrel.count',
    'hivemq.messages.outgoing.suback.count',
    'hivemq.messages.outgoing.total.bytes.50th_percentile',
    'hivemq.messages.outgoing.total.bytes.75th_percentile',
    'hivemq.messages.outgoing.total.bytes.95th_percentile',
    'hivemq.messages.outgoing.total.bytes.98th_percentile',
    'hivemq.messages.outgoing.total.bytes.999th_percentile',
    'hivemq.messages.outgoing.total.bytes.99th_percentile',
    'hivemq.messages.outgoing.total.bytes.count',
    'hivemq.messages.outgoing.total.bytes.max',
    'hivemq.messages.outgoing.total.bytes.mean',
    'hivemq.messages.outgoing.total.bytes.min',
    'hivemq.messages.outgoing.total.bytes.snapshot_size',
    'hivemq.messages.outgoing.total.bytes.std_dev',
    'hivemq.messages.outgoing.total.count',
    'hivemq.messages.outgoing.unsuback.count',
    'hivemq.messages.pending.qos_0.count',
    'hivemq.messages.pending.total.count',
    'hivemq.messages.queued.count',
    'hivemq.messages.retained.current',
    'hivemq.messages.retained.mean.50th_percentile',
    'hivemq.messages.retained.mean.75th_percentile',
    'hivemq.messages.retained.mean.95th_percentile',
    'hivemq.messages.retained.mean.98th_percentile',
    'hivemq.messages.retained.mean.999th_percentile',
    'hivemq.messages.retained.mean.99th_percentile',
    'hivemq.messages.retained.mean.count',
    'hivemq.messages.retained.mean.max',
    'hivemq.messages.retained.mean.mean',
    'hivemq.messages.retained.mean.min',
    'hivemq.messages.retained.mean.snapshot_size',
    'hivemq.messages.retained.mean.std_dev',
    'hivemq.messages.retained.pending.total.count',
    'hivemq.messages.retained.queued.count',
    'hivemq.networking.bytes.read.current',
    'hivemq.networking.bytes.read.total',
    'hivemq.networking.bytes.write.current',
    'hivemq.networking.bytes.write.total',
    'hivemq.networking.connections.current',
    'hivemq.networking.connections.mean.50th_percentile',
    'hivemq.networking.connections.mean.75th_percentile',
    'hivemq.networking.connections.mean.95th_percentile',
    'hivemq.networking.connections.mean.98th_percentile',
    'hivemq.networking.connections.mean.999th_percentile',
    'hivemq.networking.connections.mean.99th_percentile',
    'hivemq.networking.connections.mean.count',
    'hivemq.networking.connections.mean.max',
    'hivemq.networking.connections.mean.mean',
    'hivemq.networking.connections.mean.min',
    'hivemq.networking.connections.mean.snapshot_size',
    'hivemq.networking.connections.mean.std_dev',
    'hivemq.networking.connections_closed.graceful.count',
    'hivemq.networking.connections_closed.total.count',
    'hivemq.networking.connections_closed.ungraceful.count',
    'hivemq.overload_protection.clients.average_credits',
    'hivemq.overload_protection.clients.backpressure_active',
    'hivemq.overload_protection.clients.using_credits',
    'hivemq.overload_protection.credits.per_tick',
    'hivemq.overload_protection.level',
    'hivemq.payload_persistence.cleanup_executor.running',
    'hivemq.payload_persistence.cleanup_executor.scheduled.overrun',
    'hivemq.payload_persistence.cleanup_executor.scheduled.percent_of_period.50th_percentile',
    'hivemq.payload_persistence.cleanup_executor.scheduled.percent_of_period.75th_percentile',
    'hivemq.payload_persistence.cleanup_executor.scheduled.percent_of_period.95th_percentile',
    'hivemq.payload_persistence.cleanup_executor.scheduled.percent_of_period.98th_percentile',
    'hivemq.payload_persistence.cleanup_executor.scheduled.percent_of_period.999th_percentile',
    'hivemq.payload_persistence.cleanup_executor.scheduled.percent_of_period.99th_percentile',
    'hivemq.payload_persistence.cleanup_executor.scheduled.percent_of_period.count',
    'hivemq.payload_persistence.cleanup_executor.scheduled.percent_of_period.max',
    'hivemq.payload_persistence.cleanup_executor.scheduled.percent_of_period.mean',
    'hivemq.payload_persistence.cleanup_executor.scheduled.percent_of_period.min',
    'hivemq.payload_persistence.cleanup_executor.scheduled.percent_of_period.snapshot_size',
    'hivemq.payload_persistence.cleanup_executor.scheduled.percent_of_period.std_dev',
    'hivemq.persistence.executor.client_session.tasks',
    'hivemq.persistence.executor.queue_misses',
    'hivemq.persistence.executor.queued_messages.tasks',
    'hivemq.persistence.executor.request_event_bus.tasks',
    'hivemq.persistence.executor.retained_messages.tasks',
    'hivemq.persistence.executor.running.threads',
    'hivemq.persistence.executor.subscription.tasks',
    'hivemq.persistence.executor.total.tasks',
    'hivemq.persistence.payload_entries.count',
    'hivemq.persistence.removable_entries.count',
    'hivemq.persistence_scheduled_executor.running',
    'hivemq.persistence_scheduled_executor.scheduled.overrun',
    'hivemq.persistence_scheduled_executor.scheduled.percent_of_period.50th_percentile',
    'hivemq.persistence_scheduled_executor.scheduled.percent_of_period.75th_percentile',
    'hivemq.persistence_scheduled_executor.scheduled.percent_of_period.95th_percentile',
    'hivemq.persistence_scheduled_executor.scheduled.percent_of_period.98th_percentile',
    'hivemq.persistence_scheduled_executor.scheduled.percent_of_period.999th_percentile',
    'hivemq.persistence_scheduled_executor.scheduled.percent_of_period.99th_percentile',
    'hivemq.persistence_scheduled_executor.scheduled.percent_of_period.count',
    'hivemq.persistence_scheduled_executor.scheduled.percent_of_period.max',
    'hivemq.persistence_scheduled_executor.scheduled.percent_of_period.mean',
    'hivemq.persistence_scheduled_executor.scheduled.percent_of_period.min',
    'hivemq.persistence_scheduled_executor.scheduled.percent_of_period.snapshot_size',
    'hivemq.persistence_scheduled_executor.scheduled.percent_of_period.std_dev',
    'hivemq.publish.without_matching_subscribers',
    'hivemq.qos_0_memory.exceeded.per_client',
    'hivemq.qos_0_memory.max',
    'hivemq.qos_0_memory.used',
    'hivemq.sessions.overall.current',
    'hivemq.sessions.persistent.active',
    'hivemq.subscriptions.overall.current',
    'hivemq.system.max_file_descriptor',
    'hivemq.system.open_file_descriptor',
    'hivemq.system.os.file_descriptors.max',
    'hivemq.system.os.file_descriptors.open',
    'hivemq.system.os.global.memory.available',
    'hivemq.system.os.global.memory.swap.total',
    'hivemq.system.os.global.memory.swap.used',
    'hivemq.system.os.global.memory.total',
    'hivemq.system.os.global.uptime',
    'hivemq.system.os.process.disk.bytes_read',
    'hivemq.system.os.process.disk.bytes_written',
    'hivemq.system.os.process.memory.resident_set_size',
    'hivemq.system.os.process.memory.virtual',
    'hivemq.system.os.process.threads.count',
    'hivemq.system.os.process.time_spent.kernel',
    'hivemq.system.os.process.time_spent.user',
    'hivemq.system.physical_memory.free',
    'hivemq.system.physical_memory.total',
    'hivemq.system.process_cpu.load',
    'hivemq.system.process_cpu.time',
    'hivemq.system.swap_space.free',
    'hivemq.system.swap_space.total',
    'hivemq.system.system_cpu.load',
    'hivemq.topic_alias.count.total',
    'hivemq.topic_alias.memory.usage',
]
METRICS.extend(JVM_E2E_METRICS_NEW)


def test(dd_agent_check):
    aggregator = dd_agent_check(rate=True)

    for metric in METRICS:
        aggregator.assert_metric(metric)

    aggregator.assert_all_metrics_covered()
