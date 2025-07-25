# (C) Datadog, Inc. 2022-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)

import json

import fdb

from datadog_checks.base import AgentCheck

fdb.api_version(600)


class FoundationdbCheck(AgentCheck):
    __NAMESPACE__ = 'foundationdb'

    def __init__(self, name, init_config, instances):
        super(FoundationdbCheck, self).__init__(name, init_config, instances)
        self._tags = self.instance.get('tags', [])
        self._db = None

    def construct_database(self):
        if self._db is not None:
            return self._db

        # TLS options. Each option has a different function name, so we cannot be smart with it without ugly code
        if 'tls_ca_file' in self.instance:
            fdb.options.set_tls_ca_path(self.instance.get('tls_ca_file'))
        if 'tls_certificate_file' in self.instance:
            fdb.options.set_tls_cert_path(self.instance.get('tls_certificate_file'))
        if 'tls_password' in self.instance:
            fdb.options.set_tls_password(self.instance.get('tls_password'))
        if 'tls_key_file' in self.instance:
            fdb.options.set_tls_key_path(self.instance.get('tls_key_file'))
        if 'tls_verify_peers' in self.instance:
            fdb.options.set_tls_verify_peers(self.instance.get('tls_verify_peers').encode('latin-1'))

        if 'cluster_file' in self.instance:
            self._db = fdb.open(cluster_file=self.instance.get('cluster_file'))
        else:
            self._db = fdb.open()

    def fdb_status_data(self):
        self.construct_database()
        return self._db['\xff\xff/status/json'.encode('latin-1')]

    def check(self, _):
        status_data = self.fdb_status_data()
        try:
            data = json.loads(status_data)
        except Exception:
            self.service_check("can_connect", AgentCheck.CRITICAL, message="Could not parse `status json`")
            raise

        self.check_metrics(data)
        self.check_custom_queries()

    def check_custom_queries(self):
        custom_queries = self.instance.get('custom_queries')
        if not custom_queries:
            return
        for query in custom_queries:
            metric_prefix = query['metric_prefix']
            if not metric_prefix:
                self.log.error("custom query field `metric_prefix` is required")
                continue
            query_key = query['query_key']
            if not query_key:
                self.log.error("custom query field `query_key` is required for metric_prefix `%s`", metric_prefix)
                continue
            query_type = query['query_type']
            if not query_type:
                self.log.error("custom query field `query_type` is required for metric_prefix `%s`", metric_prefix)
                continue
            if not query['tags']:
                self.log.error("custom query field `tags` is required for metric_prefix `%s`", metric_prefix)
                continue
            query_tags = self._tags + query['tags']
            result = self._db[query_key.encode('UTF-8')]
            if not result:
                raise ValueError("No result for " + query_key)
            if not hasattr(self, query_type):
                self.log.error(
                    "invalid submission method `%s` for query key `%s` of metric_prefix `%s`",
                    query_type,
                    query_key,
                    metric_prefix,
                )
                break
            else:
                getattr(self, query_type)(
                    metric_prefix + '.' + query_key, float(result), tags=set(query_tags), raw=True
                )

    def report_process(self, process):
        if "address" not in process:
            return
        tags = self._tags + ["fdb_process:" + process["address"]]

        if "class_type" in process:
            tags.append("fdb_process_class:" + process["class_type"])

        if "roles" in process:
            # First, report role-specific metrics before we add role tags to the list…
            for role in process["roles"]:
                self.report_role(role, tags)

            # …then add role tags so we can associate process metrics with roles.
            for role in process["roles"]:
                if "role" in role:
                    tags.append("fdb_role:" + role["role"])

        if "cpu" in process:
            self.maybe_gauge("process.cpu.usage_cores", process["cpu"], "usage_cores", tags)
        if "disk" in process:
            disk = process["disk"]
            self.maybe_gauge("process.disk.busy", disk, "busy", tags)
            self.maybe_gauge("process.disk.free_bytes", disk, "free_bytes", tags)
            self.maybe_gauge("process.disk.total_bytes", disk, "total_bytes", tags)
            if "reads" in disk:
                self.maybe_gauge("process.disk.reads.hz", disk["reads"], "hz", tags)
                self.maybe_count("process.disk.reads.count", disk["reads"], "count", tags)
            if "writes" in disk:
                self.maybe_gauge("process.disk.writes.hz", disk["writes"], "hz", tags)
                self.maybe_count("process.disk.writes.count", disk["writes"], "count", tags)
        if "memory" in process:
            memory = process["memory"]
            self.maybe_gauge("process.memory.available_bytes", memory, "available_bytes", tags)
            self.maybe_gauge("process.memory.limit_bytes", memory, "limit_bytes", tags)
            self.maybe_gauge("process.memory.unused_allocated_memory", memory, "unused_allocated_memory", tags)
            self.maybe_gauge("process.memory.used_bytes", memory, "used_bytes", tags)
        if "network" in process:
            network = process["network"]
            self.maybe_gauge("process.network.current_connections", network, "current_connections", tags)
            self.maybe_hz_counter("process.network.connection_errors", network, "connection_errors", tags)
            self.maybe_hz_counter("process.network.connections_closed", network, "connections_closed", tags)
            self.maybe_hz_counter("process.network.connections_established", network, "connections_established", tags)
            self.maybe_hz_counter("process.network.megabits_received", network, "megabits_received", tags)
            self.maybe_hz_counter("process.network.megabits_sent", network, "megabits_sent", tags)
            self.maybe_hz_counter("process.network.tls_policy_failures", network, "tls_policy_failures", tags)

    def report_role(self, role, process_tags):
        if "role" not in role:
            return
        tags = self._tags + process_tags + ["fdb_role:" + role["role"]]

        self.maybe_hz_counter("process.role.input_bytes", role, "input_bytes", tags)
        self.maybe_hz_counter("process.role.durable_bytes", role, "durable_bytes", tags)
        self.maybe_diff_counter("process.role.queue_length", role, "input_bytes", "durable_bytes", tags)
        self.maybe_hz_counter("process.role.total_queries", role, "total_queries", tags)
        self.maybe_hz_counter("process.role.bytes_queried", role, "bytes_queried", tags)
        self.maybe_hz_counter("process.role.durable_bytes", role, "durable_bytes", tags)
        self.maybe_hz_counter("process.role.finished_queries", role, "finished_queries", tags)
        self.maybe_hz_counter("process.role.keys_queried", role, "keys_queried", tags)
        self.maybe_hz_counter("process.role.low_priority_queries", role, "low_priority_queries", tags)
        self.maybe_hz_counter("process.role.mutation_bytes", role, "mutation_bytes", tags)
        self.maybe_hz_counter("process.role.mutations", role, "mutations", tags)
        self.maybe_gauge("process.role.stored_bytes", role, "stored_bytes", tags)
        self.maybe_gauge("process.role.query_queue_max", role, "query_queue_max", tags)
        self.maybe_gauge("process.role.local_rate", role, "local_rate", tags)
        self.maybe_gauge("process.role.kvstore_available_bytes", role, "kvstore_available_bytes", tags)
        self.maybe_gauge("process.role.kvstore_free_bytes", role, "kvstore_free_bytes", tags)
        self.maybe_gauge("process.role.kvstore_inline_keys", role, "kvstore_inline_keys", tags)
        self.maybe_gauge("process.role.kvstore_total_bytes", role, "kvstore_total_bytes", tags)
        self.maybe_gauge("process.role.kvstore_total_nodes", role, "kvstore_total_nodes", tags)
        self.maybe_gauge("process.role.kvstore_total_size", role, "kvstore_total_size", tags)
        self.maybe_gauge("process.role.kvstore_used_bytes", role, "kvstore_used_bytes", tags)
        self.maybe_gauge("process.role.queue_disk_available_bytes", role, "queue_disk_available_bytes", tags)
        self.maybe_gauge("process.role.queue_disk_total_bytes", role, "queue_disk_total_bytes", tags)

        if "data_lag" in role:
            self.maybe_gauge("process.role.data_lag.seconds", role["data_lag"], "seconds", tags)
        if "durability_lag" in role:
            self.maybe_gauge("process.role.durability_lag.seconds", role["durability_lag"], "seconds", tags)

        if "grv_latency_statistics" in role:
            self.report_statistics(
                "process.role.grv_latency_statistics.default",
                role["grv_latency_statistics"],
                "default",
                tags,
            )

        self.report_statistics("process.role.read_latency_statistics", role, "read_latency_statistics", tags)
        self.report_statistics("process.role.commit_latency_statistics", role, "commit_latency_statistics", tags)

    def report_statistics(self, metric, obj, key, tags=None):
        if key in obj:
            statistics = obj[key]
            self.maybe_count(metric + ".count", statistics, "count", tags=tags)
            self.maybe_gauge(metric + ".min", statistics, "min", tags=tags)
            self.maybe_gauge(metric + ".max", statistics, "max", tags=tags)
            self.maybe_gauge(metric + ".p25", statistics, "p25", tags=tags)
            self.maybe_gauge(metric + ".p50", statistics, "p50", tags=tags)
            self.maybe_gauge(metric + ".p90", statistics, "p90", tags=tags)
            self.maybe_gauge(metric + ".p99", statistics, "p99", tags=tags)

    def check_metrics(self, status):
        if "cluster" not in status:
            raise ValueError("JSON Status data doesn't include cluster data")

        tags = self._tags

        if "client" in status:
            client = status["client"]

            if "coordinators" in client:
                if "coordinators" in client["coordinators"]:
                    coordinators = client["coordinators"]["coordinators"]

                    reachable_coordinators = sum([1 for coordinator in coordinators if coordinator["reachable"]])
                    unreachable_coordinators = len(coordinators) - reachable_coordinators

                    self.gauge("coordinators", reachable_coordinators, tags + ["reachable:true"])
                    self.gauge("coordinators", unreachable_coordinators, tags + ["reachable:false"])

        cluster = status["cluster"]
        if "machines" in cluster:
            machines = cluster["machines"]

            self.gauge("machines", len(machines), tags)
            self.gauge(
                "excluded_machines", sum([1 for machine_key in machines if machines[machine_key]["excluded"]]), tags
            )
        if "processes" in cluster:
            processes = cluster["processes"]

            self.gauge("processes", len(processes), tags)
            self.gauge(
                "excluded_processes", sum([1 for process_key in processes if processes[process_key]["excluded"]]), tags
            )

            self.count(
                "instances",
                sum((len(p["roles"]) if "roles" in p else 0 for p in processes.values())),
                tags,
            )

            role_counts = {}
            process_counts_by_tag_set = {}

            for process_key in processes:
                process = processes[process_key]

                self.report_process(process)

                process_tags = tags.copy()

                if "class_type" in process:
                    process_tags.append("fdb_process_class:" + process["class_type"])

                if "roles" in process:
                    for role in process["roles"]:
                        if "role" in role:
                            rolename = role["role"]
                            process_tags.append("fdb_role:" + rolename)
                            if rolename in role_counts:
                                role_counts[rolename] += 1
                            else:
                                role_counts[rolename] = 1

                process_tags.sort()

                # Packing and unpacking tags into sorted, comma-separated strings is a little hacky, but allows us to
                # keep a map of role sets to counts; lists/sets aren't hashable and won't work as dictionary keys.
                tags_string = ",".join(process_tags)

                if tags_string in process_counts_by_tag_set:
                    process_counts_by_tag_set[tags_string] += 1
                else:
                    process_counts_by_tag_set[tags_string] = 1

            for tags_string in process_counts_by_tag_set:
                process_tags = tags_string.split(",")
                self.gauge(
                    "processes_per_role",
                    process_counts_by_tag_set[tags_string],
                    process_tags if process_tags else None,
                )

            for role in role_counts:
                self.gauge("processes_per_role." + role, role_counts[role], tags)

        if "data" in cluster:
            data = cluster["data"]
            self.maybe_gauge("data.system_kv_size_bytes", data, "system_kv_size_bytes", tags)
            self.maybe_gauge("data.total_disk_used_bytes", data, "total_disk_used_bytes", tags)
            self.maybe_gauge("data.total_kv_size_bytes", data, "total_kv_size_bytes", tags)
            self.maybe_gauge(
                "data.least_operating_space_bytes_log_server",
                data,
                "least_operating_space_bytes_log_server",
                tags,
            )

            self.maybe_gauge("data.average_partition_size_bytes", data, "average_partition_size_bytes", tags)
            self.maybe_gauge("data.partitions_count", data, "partitions_count", tags)

            if "moving_data" in data:
                self.maybe_gauge("data.moving_data.in_flight_bytes", data["moving_data"], "in_flight_bytes", tags)
                self.maybe_gauge("data.moving_data.in_queue_bytes", data["moving_data"], "in_queue_bytes", tags)
                self.maybe_gauge(
                    "data.moving_data.total_written_bytes",
                    data["moving_data"],
                    "total_written_bytes",
                    tags,
                )

        if "datacenter_lag" in cluster:
            self.gauge("datacenter_lag.seconds", cluster["datacenter_lag"]["seconds"], tags)

        if "workload" in cluster:
            workload = cluster["workload"]
            if "transactions" in workload:
                transactions = workload["transactions"]
                for k in transactions:
                    self.maybe_hz_counter("workload.transactions." + k, transactions, k, tags)

            if "operations" in workload:
                operations = workload["operations"]
                for k in operations:
                    self.maybe_hz_counter("workload.operations." + k, operations, k, tags)

            if "keys" in workload:
                keys = workload["keys"]
                for k in keys:
                    self.maybe_hz_counter("workload.keys." + k, keys, k, tags)

            if "bytes" in workload:
                bytes = workload["bytes"]
                for k in bytes:
                    self.maybe_hz_counter("workload.bytes." + k, bytes, k, tags)

        if "latency_probe" in cluster:
            for k, v in cluster["latency_probe"].items():
                self.gauge("latency_probe." + k, v, tags)

        if "clients" in cluster:
            clients = cluster["clients"]

            if "supported_versions" in clients:
                for supported_version in clients["supported_versions"]:
                    if "count" in supported_version and "client_version" in supported_version:
                        client_version_tag = ["fdb_client_version:" + supported_version["client_version"]]
                        self.gauge("clients.connected", supported_version["count"], tags + client_version_tag)
            else:
                self.maybe_gauge("clients.connected", clients, "count", tags)

        degraded_processes = 0
        if "degraded_processes" in cluster:
            self.gauge("degraded_processes", cluster["degraded_processes"], tags)
            degraded_processes = cluster["degraded_processes"]

        if degraded_processes > 0:
            self.service_check("can_connect", AgentCheck.WARNING, message="There are degraded processes", tags=tags)
        else:
            self.service_check("can_connect", AgentCheck.OK, tags)

        if "fault_tolerance" in cluster:
            fault_tolerance = cluster["fault_tolerance"]

            self.maybe_gauge(
                "fault_tolerance.max_zone_failures_without_losing_availability",
                fault_tolerance,
                "max_zone_failures_without_losing_availability",
                tags,
            )

            self.maybe_gauge(
                "fault_tolerance.max_zone_failures_without_losing_data",
                fault_tolerance,
                "max_zone_failures_without_losing_data",
                tags,
            )

        self.maybe_gauge("maintenance_seconds_remaining", cluster, "maintenance_seconds_remaining", tags)
        self.maybe_gauge("cluster_generation", cluster, "generation", tags)

        if "qos" in cluster:
            qos = cluster["qos"]

            self.maybe_gauge("qos.transactions_per_second_limit", qos, "transactions_per_second_limit", tags)
            self.maybe_gauge(
                "qos.batch_transactions_per_second_limit", qos, "batch_transactions_per_second_limit", tags
            )
            self.maybe_gauge("qos.released_transactions_per_second", qos, "released_transactions_per_second", tags)
            self.maybe_gauge("qos.worst_queue_bytes_storage_server", qos, "worst_queue_bytes_storage_server", tags)
            self.maybe_gauge("qos.worst_queue_bytes_log_server", qos, "worst_queue_bytes_log_server", tags)

    def maybe_gauge(self, metric, obj, key, tags=None):
        if key in obj:
            self.gauge(metric, obj[key], tags=tags)

    def maybe_count(self, metric, obj, key, tags=None):
        if key in obj:
            self.monotonic_count(metric, obj[key], tags=tags)

    def maybe_hz_counter(self, metric, obj, key, tags=None):
        if key in obj:
            if "hz" in obj[key]:
                self.gauge(metric + ".hz", obj[key]["hz"], tags=tags)
            if "counter" in obj[key]:
                self.monotonic_count(metric + ".counter", obj[key]["counter"], tags=tags)

    def maybe_diff_counter(self, metric, obj, a, b, tags):
        if a in obj and "counter" in obj[a] and b in obj and "counter" in obj[b]:
            self.gauge(metric, obj[a]["counter"] - obj[b]["counter"], tags=tags)
