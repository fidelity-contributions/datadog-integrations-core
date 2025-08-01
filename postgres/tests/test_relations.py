# (C) Datadog, Inc. 2010-present
# All rights reserved
# Licensed under Simplified BSD License (see LICENSE)

import threading

import psycopg2
import pytest

from datadog_checks.base import ConfigurationError
from datadog_checks.postgres.relationsmanager import (
    QUERY_PG_CLASS,
    QUERY_PG_CLASS_SIZE,
    STATIO_METRICS,
    RelationsManager,
)

from .common import DB_NAME, HOST, POSTGRES_LOCALE, _get_expected_tags, _iterate_metric_name, assert_metric_at_least
from .utils import _get_superconn, _wait_for_value, requires_over_11

RELATION_INDEX_METRICS = [
    "postgresql.index_scans",
    "postgresql.index_rows_fetched",  # deprecated
    "postgresql.index_rel_rows_fetched",
    "postgresql.index_rel_scans",
    "postgresql.individual_index_size",
    # From STATIO_METRICS
    "postgresql.index_blocks_read",
    "postgresql.index_blocks_hit",
]

MAINTENANCE_METRICS = [
    "postgresql.last_vacuum_age",
    "postgresql.last_analyze_age",
    "postgresql.last_autovacuum_age",
    "postgresql.last_autoanalyze_age",
]
if POSTGRES_LOCALE != "C":
    # The C locale seems to prevent the creation of toast tables, so we don't collect these metrics
    MAINTENANCE_METRICS += [
        "postgresql.toast.last_vacuum_age",
        "postgresql.toast.last_autovacuum_age",
    ]

IDX_METRICS = ["postgresql.index_scans", "postgresql.index_rows_read", "postgresql.index_rows_fetched"]


def _check_metrics_for_relation_wo_index(aggregator, expected_tags):
    for name in _iterate_metric_name(QUERY_PG_CLASS):
        # Skip toast metrics if the locale is C
        if "toast" in name and POSTGRES_LOCALE == "C":
            continue

        # Skip vacuum metrics since autovacuum can make the state unpredictable
        if name in MAINTENANCE_METRICS:
            continue
        # 'persons' db doesn't have any indexes so no index metrics should be generated
        if name in RELATION_INDEX_METRICS:
            aggregator.assert_metric(name, count=0, tags=expected_tags)
        else:
            aggregator.assert_metric(name, count=1, tags=expected_tags)

    # Check metrics from QUERY_PG_CLASS_SIZE
    for name in _iterate_metric_name(QUERY_PG_CLASS_SIZE):
        if "toast" in name and POSTGRES_LOCALE == "C":
            continue
        aggregator.assert_metric(name, count=1, tags=expected_tags)

    # And metrics from STATIO_METRICS
    for name in _iterate_metric_name(STATIO_METRICS):
        if "toast" in name and POSTGRES_LOCALE == "C":
            continue
        if name in RELATION_INDEX_METRICS:
            aggregator.assert_metric(name, count=0, tags=expected_tags)
        else:
            aggregator.assert_metric(name, count=1, tags=expected_tags)


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
def test_relations_metrics(aggregator, integration_check, pg_instance):
    pg_instance["relations"] = ["persons"]
    check = integration_check(pg_instance)
    check.check(pg_instance)

    expected_tags = _get_expected_tags(check, pg_instance, db=pg_instance["dbname"], table="persons", schema="public")
    _check_metrics_for_relation_wo_index(aggregator, expected_tags)


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
def test_relations_metrics_access_exclusive_lock(aggregator, integration_check, pg_instance):
    """
    This test is an edge case where we want to make sure that relations metrics query does
    not timeout when a relation is locked with an AccessExclusiveLock.
    To do so, we lock the `persons` table with an AccessExclusiveLock and then run the check.
    """
    pg_instance["relations"] = ["persons"]
    check = integration_check(pg_instance)

    conn = _get_superconn(pg_instance)
    cursor = conn.cursor()
    # Lock the persons table with an AccessExclusiveLock
    cursor.execute("BEGIN")  # must be in a transaction to lock a table
    cursor.execute("LOCK persons IN ACCESS EXCLUSIVE MODE")

    # verify that the lock is in place
    cursor.execute(
        """
        SELECT mode, locktype, granted FROM pg_locks
        WHERE relation = (SELECT oid FROM pg_class WHERE relname = 'persons')
        """
    )
    row = cursor.fetchone()
    assert row is not None
    assert row == ("AccessExclusiveLock", "relation", True)

    check.check(pg_instance)
    expected_tags = _get_expected_tags(check, pg_instance, db=pg_instance["dbname"], table="persons", schema="public")

    for name in _iterate_metric_name(QUERY_PG_CLASS):
        # Expect no relation metrics to be collected for persons table
        # because locked relations are skipped in the query
        aggregator.assert_metric(name, count=0, tags=expected_tags)

    # Release the lock
    cursor.execute("COMMIT")
    cursor.close()
    conn.close()


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
def test_relations_metrics_share_lock(aggregator, integration_check, pg_instance):
    """
    This test is to verify the PG_CLASS query does not reporte duplicate metrics
    when a relation has multiple locks present in PG_LOCKS.
    """
    pg_instance["relations"] = ["persons"]
    check = integration_check(pg_instance)

    def access_share_lock():
        conn = _get_superconn(pg_instance)
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.execute("LOCK persons IN SHARE MODE;")  # Acquires SHARE LOCK
        finally:
            cursor.execute("COMMIT")
            cursor.close()
            conn.close()

    def row_share_lock():
        conn = _get_superconn(pg_instance)
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.execute("SELECT * FROM persons FOR SHARE;")  # Acquires ROW SHARE LOCK
        finally:
            cursor.execute("COMMIT")
            cursor.close()
            conn.close()
            print("ROW SHARE LOCK released.")

    t1 = threading.Thread(target=access_share_lock)
    t2 = threading.Thread(target=row_share_lock)

    # Start threads
    t1.start()
    t2.start()

    check.check(pg_instance)
    expected_tags = _get_expected_tags(check, pg_instance, db=pg_instance["dbname"], table="persons", schema="public")

    for name in ("postgresql.rows_inserted", "postgresql.rows_updated", "postgresql.rows_deleted"):
        # Expect no relation metrics to be collected for persons table
        # because locked relations are skipped in the query
        aggregator.assert_metric(name, count=1, tags=expected_tags)

    # Wait for threads to complete
    t1.join()
    t2.join()


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
@requires_over_11
def test_partition_relation(aggregator, integration_check, pg_instance):
    pg_instance["relations"] = [
        {"relation_regex": "test_.*"},
    ]

    check = integration_check(pg_instance)
    check.check(pg_instance)

    part_1_tags = _get_expected_tags(
        check, pg_instance, db=pg_instance["dbname"], table="test_part1", partition_of="test_part", schema="public"
    )
    aggregator.assert_metric("postgresql.relation.pages", value=3, count=1, tags=part_1_tags)
    aggregator.assert_metric("postgresql.relation.tuples", value=499, count=1, tags=part_1_tags)
    aggregator.assert_metric("postgresql.relation.all_visible", value=3, count=1, tags=part_1_tags)
    aggregator.assert_metric("postgresql.table_size", value=24576, count=1, tags=part_1_tags)
    aggregator.assert_metric("postgresql.relation_size", value=24576, count=1, tags=part_1_tags)
    aggregator.assert_metric("postgresql.index_size", value=65536, count=1, tags=part_1_tags)
    if POSTGRES_LOCALE != "C":
        aggregator.assert_metric("postgresql.toast_size", value=0, count=1, tags=part_1_tags)
        aggregator.assert_metric("postgresql.total_size", value=90112, count=1, tags=part_1_tags)

    part_2_tags = _get_expected_tags(
        check, pg_instance, db=pg_instance["dbname"], table="test_part2", partition_of="test_part", schema="public"
    )
    aggregator.assert_metric("postgresql.relation.pages", value=8, count=1, tags=part_2_tags)
    aggregator.assert_metric("postgresql.relation.tuples", value=1502, count=1, tags=part_2_tags)
    aggregator.assert_metric("postgresql.relation.all_visible", value=8, count=1, tags=part_2_tags)
    aggregator.assert_metric("postgresql.table_size", value=73728, count=1, tags=part_2_tags)
    aggregator.assert_metric("postgresql.relation_size", value=65536, count=1, tags=part_2_tags)
    aggregator.assert_metric("postgresql.index_size", value=98304, count=1, tags=part_2_tags)
    if POSTGRES_LOCALE != "C":
        aggregator.assert_metric("postgresql.toast_size", value=8192, count=1, tags=part_2_tags)
        aggregator.assert_metric("postgresql.total_size", value=172032, count=1, tags=part_2_tags)


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
@pytest.mark.parametrize(
    "collect_bloat_metrics, expected_count",
    [
        pytest.param(True, 1, id="bloat enabled"),
        pytest.param(False, 0, id="bloat disabled"),
    ],
)
def test_bloat_metrics(aggregator, collect_bloat_metrics, expected_count, integration_check, pg_instance):
    pg_instance["relations"] = ["pg_index"]
    pg_instance["collect_bloat_metrics"] = collect_bloat_metrics

    check = integration_check(pg_instance)
    check.check(pg_instance)

    base_tags = _get_expected_tags(check, pg_instance, db=pg_instance["dbname"], table="pg_index", schema="pg_catalog")
    aggregator.assert_metric("postgresql.table_bloat", count=expected_count, tags=base_tags)

    indices = ["pg_index_indrelid_index", "pg_index_indexrelid_index"]
    for index in indices:
        expected_tags = base_tags + ["index:{}".format(index)]
        aggregator.assert_metric("postgresql.index_bloat", count=expected_count, tags=expected_tags)


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
def test_relations_metrics_regex(aggregator, integration_check, pg_instance):
    pg_instance["relations"] = [
        {"relation_regex": ".*", "schemas": ["hello", "hello2"]},
        # Empty schemas means all schemas, even though the first relation matches first.
        {"relation_regex": r"[pP]ersons[-_]?(dup\d)?"},
    ]
    relations = ["persons", "personsdup1", "Personsdup2"]
    check = integration_check(pg_instance)
    check.check(pg_instance)

    expected_tags = {}
    for relation in relations:
        expected_tags[relation] = _get_expected_tags(
            check, pg_instance, db=pg_instance["dbname"], table=relation.lower(), schema="public"
        )
    for relation in relations:
        _check_metrics_for_relation_wo_index(aggregator, expected_tags[relation])


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
def test_relations_xmin(aggregator, integration_check, pg_instance):
    pg_instance["relations"] = ["persons"]

    conn = _get_superconn(pg_instance)
    cursor = conn.cursor()
    cursor.execute("SELECT xmin FROM pg_class WHERE relname='persons'")
    start_xmin = float(cursor.fetchone()[0])

    # Check that initial xmin metric match
    check = integration_check(pg_instance)
    check.check(pg_instance)
    expected_tags = _get_expected_tags(check, pg_instance, db=pg_instance["dbname"], table="persons", schema="public")
    aggregator.assert_metric("postgresql.relation.xmin", count=1, value=start_xmin, tags=expected_tags)
    aggregator.reset()

    # Run multiple DDL modifying the persons relation which will increase persons' xmin in pg_class
    cursor.execute("ALTER TABLE persons REPLICA IDENTITY FULL;")
    cursor.execute("ALTER TABLE persons REPLICA IDENTITY DEFAULT;")
    cursor.close()
    conn.close()

    check.check(pg_instance)

    # xmin metric should be greater than initial xmin
    assert_metric_at_least(aggregator, "postgresql.relation.xmin", lower_bound=start_xmin + 1, tags=expected_tags)


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
def test_max_relations(aggregator, integration_check, pg_instance):
    pg_instance.update({"relations": [{"relation_regex": ".*"}], "max_relations": 1})
    check = integration_check(pg_instance)
    check.check(pg_instance)

    def _metric_name_to_relation_list(metric_name):
        relation_metrics = []
        for m in aggregator._metrics[metric_name]:
            if any("table:" in tag for tag in m.tags):
                relation_metrics.append(m)
        return relation_metrics

    for name in _iterate_metric_name(QUERY_PG_CLASS):
        if name in RELATION_INDEX_METRICS + MAINTENANCE_METRICS:
            continue
        if "toast" in name and POSTGRES_LOCALE == "C":
            continue
        relation_metrics = _metric_name_to_relation_list(name)
        assert len(relation_metrics) == 1, f"Expected 1 results for {name}"

    # Also check PG_CLASS_SIZE
    for name in _iterate_metric_name(QUERY_PG_CLASS_SIZE):
        relation_metrics = _metric_name_to_relation_list(name)
        assert len(relation_metrics) == 1, f"Expected 1 results for {name}"

    # And STATIO metrics
    for name in _iterate_metric_name(STATIO_METRICS):
        if name in RELATION_INDEX_METRICS:
            # pg_statio_user_tables returns NULL if the relation doesn't have an index. Skip the index metrics
            continue
        if "toast" in name and POSTGRES_LOCALE == "C":
            continue
        relation_metrics = _metric_name_to_relation_list(name)
        assert len(relation_metrics) == 1, f"Expected 1 results for {name}"


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
def test_index_metrics(aggregator, integration_check, pg_instance):
    pg_instance["relations"] = ["breed"]
    pg_instance["dbname"] = "dogs"

    check = integration_check(pg_instance)
    check.check(pg_instance)

    expected_tags = _get_expected_tags(
        check, pg_instance, db="dogs", table="breed", index="breed_names", schema="public", valid="true"
    )
    for name in IDX_METRICS:
        aggregator.assert_metric(name, count=1, tags=expected_tags)


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
@pytest.mark.flaky(max_runs=5)
def test_vacuum_age(aggregator, integration_check, pg_instance):
    pg_instance["relations"] = ["persons"]
    pg_instance["dbname"] = "datadog_test"

    conn = _get_superconn(pg_instance)
    with conn.cursor() as cur:
        cur.execute("select pg_stat_reset()")
        cur.execute("VACUUM ANALYZE persons")
    conn.close()

    _wait_for_value(
        pg_instance,
        lower_threshold=0,
        query="select count(*) from pg_stat_user_tables where relname='persons' and vacuum_count > 0;",
    )

    check = integration_check(pg_instance)
    check.check(pg_instance)

    expected_tags = _get_expected_tags(check, pg_instance, db="datadog_test", table="persons", schema="public")
    metrics = ["postgresql.last_vacuum_age", "postgresql.last_analyze_age"]
    if POSTGRES_LOCALE != "C":
        metrics += ["postgresql.toast.last_vacuum_age"]
    for name in metrics:
        assert_metric_at_least(
            aggregator,
            name,
            lower_bound=0,
            higher_bound=100,
            tags=expected_tags,
            count=1,
        )


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
@pytest.mark.parametrize(
    "relations, lock_count, lock_table_name, tags",
    [
        pytest.param(
            ["persons"],
            1,
            "persons",
            [
                "db:datadog_test",
                "lock_mode:AccessExclusiveLock",
                "lock_type:relation",
                "granted:True",
                "fastpath:False",
                "table:persons",
                "schema:public",
            ],
            id="test with single table lock should return 1",
        ),
        pytest.param(
            [{"relation_regex": "perso.*", "relkind": ["r"]}],
            1,
            "persons",
            None,
            id="test with matching relkind should return 1",
        ),
        pytest.param(
            [{"relation_regex": "perso.*", "relkind": ["i"]}],
            0,
            "persons",
            None,
            id="test without matching relkind should return 0",
        ),
        pytest.param(
            ["pgtable"],
            1,
            "pgtable",
            None,
            id="pgtable should be included in lock metrics",
        ),
        pytest.param(
            ["pg_newtable"],
            0,
            "pg_newtable",
            None,
            id="pg_newtable should be excluded from query since it starts with `pg_`",
        ),
    ],
)
def test_locks_metrics(aggregator, integration_check, pg_instance, relations, lock_count, lock_table_name, tags):
    pg_instance["relations"] = relations
    pg_instance["query_timeout"] = 1000  # One of the relation queries waits for the table to not be locked

    check = integration_check(pg_instance)
    check_with_lock(check, pg_instance, lock_table_name)

    if tags is not None:
        expected_tags = _get_expected_tags(check, pg_instance) + tags
        aggregator.assert_metric("postgresql.locks", count=lock_count, tags=expected_tags)
    else:
        aggregator.assert_metric("postgresql.locks", count=lock_count)


@pytest.mark.integration
@pytest.mark.usefixtures("dd_environment")
def check_with_lock(check, instance, lock_table=None):
    lock_statement = "LOCK persons"
    if lock_table is not None:
        lock_statement = "LOCK {}".format(lock_table)
    with psycopg2.connect(host=HOST, dbname=DB_NAME, user="postgres", password="datad0g") as conn:
        with conn.cursor() as cur:
            cur.execute(lock_statement)
            check.check(instance)


@pytest.mark.unit
def test_relations_validation_accepts_list_of_str_and_dict():
    RelationsManager.validate_relations_config(
        [
            "alert_cycle_keys_aggregate",
            "api_keys",
            {"relation_regex": "perso.*", "relkind": ["i"]},
            {"relation_name": "person", "relkind": ["i"]},
            {"relation_name": "person", "schemas": ["foo"]},
        ]
    )


@pytest.mark.unit
def test_relations_validation_fails_if_no_relname_or_regex():
    with pytest.raises(ConfigurationError):
        RelationsManager.validate_relations_config([{"relkind": ["i"]}])


@pytest.mark.unit
def test_relations_validation_fails_if_schemas_is_wrong_type():
    with pytest.raises(ConfigurationError):
        RelationsManager.validate_relations_config([{"relation_name": "person", "schemas": "foo"}])


@pytest.mark.unit
def test_relations_validation_fails_if_relkind_is_wrong_type():
    with pytest.raises(ConfigurationError):
        RelationsManager.validate_relations_config([{"relation_name": "person", "relkind": "foo"}])
