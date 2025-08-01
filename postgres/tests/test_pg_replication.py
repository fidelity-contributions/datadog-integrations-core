# (C) Datadog, Inc. 2023-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
import time

import pytest

from .common import (
    DB_NAME,
    _get_expected_replication_tags,
    _get_expected_tags,
    assert_metric_at_least,
    check_bgw_metrics,
    check_common_metrics,
    check_conflict_metrics,
    check_connection_metrics,
    check_control_metrics,
    check_db_count,
    check_file_wal_metrics,
    check_metrics_metadata,
    check_performance_metrics,
    check_recovery_prefetch_metrics,
    check_replication_delay,
    check_slru_metrics,
    check_snapshot_txid_metrics,
    check_stat_wal_metrics,
    check_uptime_metrics,
    check_wait_event_metrics,
    check_wal_receiver_metrics,
)
from .utils import _get_superconn, _wait_for_value, requires_over_10

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures('dd_environment')]


@requires_over_10
def test_common_replica_metrics(aggregator, integration_check, metrics_cache_replica, pg_replica_instance):
    check = integration_check(pg_replica_instance)
    check._connect()

    # Use check.run() to go through initilization queries
    check.run()

    expected_tags = _get_expected_replication_tags(check, pg_replica_instance)
    check_common_metrics(aggregator, expected_tags=expected_tags)
    check_bgw_metrics(aggregator, expected_tags)
    check_connection_metrics(aggregator, expected_tags=expected_tags)
    check_control_metrics(aggregator, expected_tags=expected_tags)
    check_db_count(aggregator, expected_tags=expected_tags)
    check_slru_metrics(aggregator, expected_tags=expected_tags)
    check_replication_delay(aggregator, metrics_cache_replica, expected_tags=expected_tags)
    check_wal_receiver_metrics(aggregator, expected_tags=expected_tags + ['status:streaming'])
    check_conflict_metrics(aggregator, expected_tags=expected_tags)
    check_uptime_metrics(aggregator, expected_tags=expected_tags)
    check_snapshot_txid_metrics(aggregator, expected_tags=expected_tags)
    check_stat_wal_metrics(aggregator, expected_tags=expected_tags)
    check_file_wal_metrics(aggregator, expected_tags=expected_tags)
    check_recovery_prefetch_metrics(aggregator, expected_tags=expected_tags)
    expected_wait_event_tags = expected_tags + [
        'app:datadog-agent',
        'user:datadog',
        'db:datadog_test',
        'backend_type:client backend',
        'wait_event:NoWaitEvent',
    ]
    check_wait_event_metrics(aggregator, expected_tags=expected_wait_event_tags)

    check_performance_metrics(aggregator, expected_tags=check.debug_stats_kwargs()['tags'])

    aggregator.assert_all_metrics_covered()
    check_metrics_metadata(aggregator)


@requires_over_10
def test_replica_initialization_tags(integration_check, pg_replica_instance, pg_replica_instance2):
    check = integration_check(pg_replica_instance)
    check.run()
    assert check.cluster_name == ''
    assert check.system_identifier is not None

    check2 = integration_check(pg_replica_instance2)
    check2.run()
    # After run, initialization queries should have set system identifier and cluster_name tags
    assert check2.cluster_name == 'replica2'
    assert check2.system_identifier is not None


@requires_over_10
def test_wal_receiver_metrics(aggregator, integration_check, pg_instance, pg_replica_instance):
    check = integration_check(pg_replica_instance)
    check._connect()
    check.initialize_is_aurora()
    with _get_superconn(pg_instance) as conn:
        with conn.cursor() as cur:
            # Ask for a new txid to force a WAL change
            cur.execute('select txid_current();')
            cur.fetchall()
    # Wait for 200ms for WAL sender to send message
    time.sleep(0.2)

    check.check(pg_replica_instance)
    expected_tags = _get_expected_replication_tags(check, pg_replica_instance, status='streaming')
    aggregator.assert_metric('postgresql.wal_receiver.last_msg_send_age', count=1, tags=expected_tags)
    aggregator.assert_metric('postgresql.wal_receiver.last_msg_receipt_age', count=1, tags=expected_tags)
    aggregator.assert_metric('postgresql.wal_receiver.latest_end_age', count=1, tags=expected_tags)
    # All ages should have been updated within the last second
    assert_metric_at_least(
        aggregator,
        'postgresql.wal_receiver.last_msg_send_age',
        higher_bound=1,
        tags=expected_tags,
        count=1,
    )
    assert_metric_at_least(
        aggregator,
        'postgresql.wal_receiver.last_msg_receipt_age',
        higher_bound=1,
        tags=expected_tags,
        count=1,
    )
    assert_metric_at_least(
        aggregator,
        'postgresql.wal_receiver.latest_end_age',
        higher_bound=1,
        tags=expected_tags,
        count=1,
    )


@requires_over_10
def test_conflicts_lock(aggregator, integration_check, pg_instance, pg_replica_instance2):
    check = integration_check(pg_replica_instance2)

    replica_con = _get_superconn(pg_replica_instance2)
    replica_con.set_session(autocommit=False)
    replica_cur = replica_con.cursor()
    replica_cur.execute('BEGIN;')
    replica_cur.execute('select * from persons;')

    conn = _get_superconn(pg_instance)
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    cur.execute('update persons SET personid = 1 where personid = 1;')
    cur.execute('vacuum full persons;')
    time.sleep(0.2)
    conn.close()

    _wait_for_value(
        pg_replica_instance2,
        lower_threshold=0,
        query="select confl_lock from pg_stat_database_conflicts where datname='datadog_test';",
    )

    check.check(pg_replica_instance2)
    expected_tags = _get_expected_replication_tags(check, pg_replica_instance2, db=DB_NAME)
    aggregator.assert_metric('postgresql.conflicts.lock', value=1, tags=expected_tags)

    replica_con.close()


@requires_over_10
@pytest.mark.flaky(max_runs=5)
def test_conflicts_snapshot(aggregator, integration_check, pg_instance, pg_replica_instance2):
    check = integration_check(pg_replica_instance2)

    replica2_con = _get_superconn(pg_replica_instance2)
    replica2_con.set_session(autocommit=False)
    replica2_cur = replica2_con.cursor()
    replica2_cur.execute('BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;')
    replica2_cur.execute('select * from persons;')

    conn = _get_superconn(pg_instance)
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    cur.execute('update persons SET personid = 1 where personid = 1;')
    time.sleep(1.2)
    cur.execute('vacuum verbose persons;')
    conn.close()
    time.sleep(0.2)

    _wait_for_value(
        pg_replica_instance2,
        lower_threshold=0,
        query="select confl_snapshot from pg_stat_database_conflicts where datname='datadog_test';",
    )
    check.check(pg_replica_instance2)
    expected_tags = _get_expected_replication_tags(check, pg_replica_instance2, db=DB_NAME)
    aggregator.assert_metric('postgresql.conflicts.snapshot', value=1, tags=expected_tags)

    replica2_con.close()


@pytest.mark.skip(reason="Failing on master")
@requires_over_10
def test_conflicts_bufferpin(aggregator, integration_check, pg_instance, pg_replica_instance2):
    check = integration_check(pg_replica_instance2)

    with _get_superconn(pg_instance) as conn:
        with conn.cursor() as cur:
            cur.execute('BEGIN;')
            cur.execute("INSERT INTO persons VALUES (3,'t','t','t');")
            cur.execute('ROLLBACK;')

    replica2_con = _get_superconn(pg_replica_instance2)
    replica2_cur = replica2_con.cursor()
    replica2_cur.execute('BEGIN;')
    replica2_cur.execute('DECLARE cursor1 CURSOR FOR SELECT * FROM persons')
    replica2_cur.execute('FETCH FORWARD FROM cursor1')
    replica2_cur.fetchall()

    with _get_superconn(pg_instance) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('vacuum verbose persons;')

    _wait_for_value(
        pg_replica_instance2,
        lower_threshold=0,
        query="select confl_bufferpin from pg_stat_database_conflicts where datname='datadog_test';",
    )

    check.check(pg_replica_instance2)
    expected_tags = _get_expected_replication_tags(check, pg_replica_instance2, db=DB_NAME)
    aggregator.assert_metric('postgresql.conflicts.bufferpin', value=1, tags=expected_tags)


@requires_over_10
def test_pg_control_replication(aggregator, integration_check, pg_instance, pg_replica_instance):
    check = integration_check(pg_replica_instance)
    check.run()

    dd_agent_tags = _get_expected_tags(check, pg_replica_instance, role='standby')
    aggregator.assert_metric('postgresql.control.timeline_id', count=1, value=1, tags=dd_agent_tags)

    # Also checkpoint on primary to generate changes
    master_conn = _get_superconn(pg_instance)
    with master_conn.cursor() as cur:
        cur.execute("CHECKPOINT;")

    postgres_conn = _get_superconn(pg_replica_instance)
    with postgres_conn.cursor() as cur:
        cur.execute("CHECKPOINT;")

    aggregator.reset()
    check.run()
    # checkpoint should be less than 2s old
    assert_metric_at_least(
        aggregator, 'postgresql.control.checkpoint_delay', count=1, higher_bound=2.0, tags=dd_agent_tags
    )
    # After a checkpoint, we have the CHECKPOINT_ONLINE record (114 bytes) and also
    # likely receive RUNNING_XACTS (50 bytes) record
    assert_metric_at_least(
        aggregator, 'postgresql.control.checkpoint_delay_bytes', count=1, higher_bound=250, tags=dd_agent_tags
    )
    # And restart should be slightly more than checkpoint delay
    assert_metric_at_least(
        aggregator, 'postgresql.control.redo_delay_bytes', count=1, higher_bound=300, tags=dd_agent_tags
    )
