# (C) Datadog, Inc. 2019-present
# All rights reserved
# Licensed under Simplified BSD License (see LICENSE)
import datetime as dt  # noqa: F401
import functools
from typing import Any, Callable, List, TypeVar, cast  # noqa: F401

import vsanapiutils
from pyVim import connect
from pyVmomi import SoapStubAdapter, vim, vmodl

from datadog_checks.base.log import CheckLoggingAdapter  # noqa: F401
from datadog_checks.base.utils.http import create_ssl_context
from datadog_checks.vsphere.config import VSphereConfig  # noqa: F401
from datadog_checks.vsphere.constants import (
    ALL_PROPERTIES,
    ALL_RESOURCES,
    MAX_QUERY_METRICS_OPTION,
    MOR_TYPE_AS_STRING,
    UNLIMITED_HIST_METRICS_PER_QUERY,
    VSAN_EVENT_IDS,
)
from datadog_checks.vsphere.metrics import (
    ENTITY_REMAPPER,
)
from datadog_checks.vsphere.types import InfrastructureData
from datadog_checks.vsphere.utils import properties_to_collect

CallableT = TypeVar('CallableT', bound=Callable)


def smart_retry(f):
    # type: (CallableT) -> CallableT
    """A function decorated with this `@smart_retry` will trigger a new authentication if it fails. The function
    will then be retried.
    This is useful when the integration keeps a semi-healthy connection to the vSphere API"""

    @functools.wraps(f)
    def wrapper(api_instance, *args, **kwargs):
        # type: (VSphereAPI, *Any, **Any) -> Any
        try:
            return f(api_instance, *args, **kwargs)
        except vmodl.fault.InvalidArgument:
            # This error is raised when the api call request is invalid. This error also appear when
            # requesting non existing metrics. Retrying won't help
            # https://code.vmware.com/apis/704/vsphere/vmodl.fault.InvalidArgument.html
            raise
        except vim.fault.InvalidName:
            # For the scope of this integration, this is raised when fetching a config value from vCenter
            # that doesn't exist (especially maxQueryMetrics). Retrying won't help
            # https://code.vmware.com/apis/704/vsphere/vim.fault.InvalidName.html
            raise
        except vim.fault.RestrictedByAdministrator:
            # The operation cannot complete because of some restriction set by the server administrator.
            # Retrying won't help
            # https://code.vmware.com/apis/704/vsphere/vim.fault.RestrictedByAdministrator.html
            raise
        except Exception as e:
            api_instance.log.debug(
                "An exception occurred when executing %s: %s. Refreshing the connection to vCenter and retrying",
                f.__name__,
                e,
            )
            api_instance.smart_connect()
            return f(api_instance, *args, **kwargs)

    return cast(CallableT, wrapper)


class APIConnectionError(Exception):
    pass


class APIResponseError(Exception):
    pass


class VersionInfo(object):
    def __init__(self, about_info):
        # type: (vim.AboutInfo) -> None
        # SemVer formatted version string
        self.version_str = "{}+{}".format(about_info.version, about_info.build)

        # Text based information i.e 'VMware vCenter Server 6.7.0 build-14792544'
        self.fullName = about_info.fullName

        # 'VirtualCenter' when connected to vCenter, 'HostAgent' when connected to an ESX host directly
        self.api_type = about_info.apiType

    def is_vcenter(self):
        # type: () -> bool
        """The vSphere integration only supports connecting to a vCenter instance. It can't be used to monitor Esxi
        hosts directly."""
        return self.api_type == 'VirtualCenter'


class VSphereAPI(object):
    """Abstraction class over the vSphere SOAP api using the pyvmomi library"""

    def __init__(self, config, log):
        # type: (VSphereConfig, CheckLoggingAdapter) -> None
        self.config = config
        self.log = log

        self._conn = cast(vim.ServiceInstance, None)
        self._vsan_stub = cast(SoapStubAdapter, None)
        self.smart_connect()

    def smart_connect(self):
        # type: () -> None
        """
        Creates the connection object to the vSphere API using parameters supplied from the configuration.

        Docs for vim.ServiceInstance:
            https://vdc-download.vmware.com/vmwb-repository/dcr-public/b525fb12-61bb-4ede-b9e3-c4a1f8171510/99ba073a-60e9-4933-8690-149860ce8754/doc/vim.ServiceInstance.html
        """
        context = None
        if not self.config.ssl_verify:
            # Remove type ignore when this is merged https://github.com/python/typeshed/pull/3855
            context = create_ssl_context({"tls_verify": False})  # type: ignore
        elif self.config.ssl_capath or self.config.ssl_cafile:
            # Remove type ignore when this is merged https://github.com/python/typeshed/pull/3855
            # `check_hostname` must be enabled as well to verify the authenticity of a cert.
            context = create_ssl_context({"tls_verify": True, 'tls_check_hostname': True})  # type: ignore
            if self.config.ssl_capath:
                context.load_verify_locations(cafile=None, capath=self.config.ssl_capath)
            else:
                context.load_verify_locations(cafile=self.config.ssl_cafile, capath=None)
        try:
            # Object returned by SmartConnect is a ServerInstance
            # https://www.vmware.com/support/developer/vc-sdk/visdk2xpubs/ReferenceGuide/vim.ServiceInstance.html
            conn = connect.SmartConnect(
                host=self.config.hostname, user=self.config.username, pwd=self.config.password, sslContext=context
            )
            # Next line tries a simple API call to check the health of the connection.
            version_info = VersionInfo(conn.content.about)
        except Exception as e:
            err_msg = "Connection to {} failed: {}".format(self.config.hostname, e)
            raise APIConnectionError(err_msg)

        if not version_info.is_vcenter():
            # Connection was successful but to something that is not a VirtualCenter instance. The check won't
            # run correctly.
            # TODO: Raise an exception and stop execution here
            self.log.error(
                "%s is not a valid VirtualCenter (vCenter) instance, the vSphere API reports '%s'. "
                "Do not try to connect to ESXi hosts directly.",
                self.config.hostname,
                version_info.api_type,
            )

        if self._conn:
            connect.Disconnect(self._conn)

        self._conn = conn
        if self.config.collect_vsan:
            self._vsan_stub = vsanapiutils.GetVsanVcStub(conn._stub, context=context)
        self.log.debug("Connected to %s", version_info.fullName)

    @smart_retry
    def get_current_time(self):
        # type: () -> dt.datetime
        return self._conn.CurrentTime()

    @smart_retry
    def get_version(self):
        # type: () -> VersionInfo
        return VersionInfo(self._conn.content.about)

    @smart_retry
    def get_perf_counter_by_level(self, collection_level):
        # type: (int) -> List[vim.PerformanceManager.PerfCounterInfo]
        """
        Requests and returns the list of counter available for a given collection_level.

        https://vdc-download.vmware.com/vmwb-repository/dcr-public/fe08899f-1eec-4d8d-b3bc-a6664c168c2c/7fdf97a1-4c0d-4be0-9d43-2ceebbc174d9/doc/vim.PerformanceManager.CounterInfo.html
        """
        return self._conn.content.perfManager.QueryPerfCounterByLevel(collection_level)

    @smart_retry
    def _get_raw_infrastructure(self):
        # type: () -> List[vmodl.query.PropertyCollector.ObjectContent]
        """Traverse the whole vSphere infrastructure and returns the list of raw pyvmomi MOR objects with
        the required pre-fetched attributes."""
        content = self._conn.content  # vim.ServiceInstanceContent reference from the connection

        property_specs = []
        # Specify which attributes we want to retrieve per object
        for resource in ALL_RESOURCES:
            property_spec = vmodl.query.PropertyCollector.PropertySpec()
            property_spec.type = resource
            property_spec.pathSet = ["name", "parent"]
            if self.config.should_collect_attributes:
                property_spec.pathSet.append("customValue")
            if resource == vim.VirtualMachine:
                property_spec.pathSet.append("runtime.powerState")
                property_spec.pathSet.append("runtime.host")
                property_spec.pathSet.append("guest.hostName")

            if self.config.collect_property_metrics:
                properties = properties_to_collect(MOR_TYPE_AS_STRING.get(resource), self.config.metric_filters)
                for property in properties:
                    property_spec.pathSet.append(property)

            property_specs.append(property_spec)

        # Specify the attribute of the root object to traverse to obtain all the attributes
        traversal_spec = vmodl.query.PropertyCollector.TraversalSpec()
        traversal_spec.path = "view"
        traversal_spec.skip = False
        traversal_spec.type = vim.view.ContainerView

        retr_opts = vmodl.query.PropertyCollector.RetrieveOptions()
        # To limit the number of objects retrieved per call.
        # If batch_collector_size is 0, collect maximum number of objects.
        retr_opts.maxObjects = self.config.batch_collector_size

        # Specify the root object from where we collect the rest of the objects
        obj_spec = vmodl.query.PropertyCollector.ObjectSpec()
        obj_spec.skip = True
        obj_spec.selectSet = [traversal_spec]

        # Create our filter spec from the above specs
        filter_spec = vmodl.query.PropertyCollector.FilterSpec()
        filter_spec.propSet = property_specs

        view_ref = content.viewManager.CreateContainerView(content.rootFolder, ALL_RESOURCES, True)
        try:
            obj_spec.obj = view_ref
            filter_spec.objectSet = [obj_spec]

            # Collect the objects and their properties
            res = content.propertyCollector.RetrievePropertiesEx([filter_spec], retr_opts)
            if res is None:
                self.log.warning(
                    "Did not retrieve any properties from the vCenter. Metric collection cannot continue. "
                    "Ensure your user has correct permissions."
                )
                obj_content_list = []
                return obj_content_list

            obj_content_list = res.objects
            # Results can be paginated
            while res.token is not None:
                res = content.propertyCollector.ContinueRetrievePropertiesEx(res.token)
                obj_content_list.extend(res.objects)
        finally:
            view_ref.Destroy()

        return obj_content_list

    @smart_retry
    def _fetch_all_attributes(self):
        # type: () -> List[vim.CustomFieldsManager.FieldDef]
        """Retrieves all attributes for every single resource in vSphere. It is not possible to fetch
        only the one we needs.
        Note: Code is in a separate method so that it can be 'smart_retried' if the API call fails."""
        return self._conn.content.customFieldsManager.field

    def get_infrastructure(self):
        # type: () -> InfrastructureData
        """Traverse the whole vSphere infrastructure and outputs a dict mapping the mors to their properties.

        :return: {
            'vim.VirtualMachine-VM0': {
              'name': 'VM-0',
              ...
            }
            ...
        }
        """

        obj_content_list = self._get_raw_infrastructure()
        # Build infrastructure data
        # Each `obj_content` contains the fields:
        #   - `obj`: `ManagedEntity` aka `mor`
        #   - `propSet`: properties related to the `mor`
        infrastructure_data = {
            obj_content.obj: {prop.name: prop.val for prop in obj_content.propSet}
            for obj_content in obj_content_list
            if obj_content.propSet
        }

        # Add the root folder entity as it can't be fetched from the previous api calls.
        root_folder = self._conn.content.rootFolder
        infrastructure_data[root_folder] = {"name": root_folder.name, "parent": None}

        if self.config.should_collect_attributes or self.config.collect_property_metrics:
            # Clean up attributes in infrastructure_data,
            # at this point they are custom pyvmomi objects and the attribute keys are not resolved.

            attribute_keys = {x.key: x.name for x in self._fetch_all_attributes()}
            for props in infrastructure_data.values():
                mor_attributes = []
                if self.config.collect_property_metrics:
                    all_properties = {}
                    for attribute_name in ALL_PROPERTIES:
                        attribute_val = props.pop(attribute_name, None)
                        if attribute_val is not None:
                            all_properties[attribute_name] = attribute_val
                    props['properties'] = all_properties

                if 'customValue' not in props:
                    continue
                for attribute in props.pop('customValue'):
                    # The attribute key is always unique
                    attr_key_name = attribute_keys.get(attribute.key)
                    if attr_key_name is None:
                        self.log.debug("Unable to resolve attribute key with ID: %s", attribute.key)
                        continue
                    attr_value = attribute.value
                    mor_attributes.append("{}{}:{}".format(self.config.attr_prefix, attr_key_name, attr_value))

                props['attributes'] = mor_attributes
        return cast(InfrastructureData, infrastructure_data)

    @smart_retry
    def query_metrics(self, query_specs):
        # type: (List[vim.PerformanceManager.QuerySpec]) -> List[vim.PerformanceManager.EntityMetricBase]
        perf_manager = self._conn.content.perfManager
        values = perf_manager.QueryPerf(query_specs)
        self.log.debug("Received %s values from QueryPerf", len(values))
        self.log.trace(
            "Query metrics:\n=== QUERY ===\n%s\n=== RESPONSE ===\n%s\n=== END QUERY ===",
            query_specs,
            values,
        )
        return values

    @smart_retry
    def get_new_events(self, start_time):
        # type: (dt.datetime) -> List[vim.event.Event]
        event_manager = self._conn.content.eventManager
        query_filter = vim.event.EventFilterSpec()
        time_filter = vim.event.EventFilterSpec.ByTime(beginTime=start_time)
        query_filter.time = time_filter
        query_filter.type = [getattr(vim.event, event_type) for event_type in self.config.exclude_filters.keys()]
        try:
            events = event_manager.QueryEvents(query_filter)
        except KeyError as e:
            self.log.debug("Error parsing bulk events: %s", e)

            if self.config.use_collect_events_fallback:
                self.log.debug("Start fetching events one by one...")
                events = self._get_new_events_one_by_one(query_filter)
            else:
                raise
        return events

    def _get_new_events_one_by_one(self, query_filter):
        # type: (vim.event.EventFilterSpec) -> List[vim.event.Event]
        """
        Collecting events one by one and skip those with parsing error.

        The parsing error is triggered by unknown types like `ContentLibrary`.
        More info:
            - https://github.com/vmware/pyvmomi/issues/190
            - https://github.com/vmware/pyvmomi/issues/872

        The event collection fallback is a workaround and can be removed when the upstream issues
        mentioned above are solved.
        """
        event_manager = self._conn.content.eventManager
        events = []
        event_collector = event_manager.CreateCollectorForEvents(query_filter)
        while True:
            try:
                collected_events = event_collector.ReadNextEvents(1)  # Read with page_size=1
            except KeyError as e:
                self.log.debug("Cannot parse event, skipped: %s", e)
                continue
            if len(collected_events) == 0:
                break
            event = collected_events[0]
            self.log.debug(
                "Collect event with id:%s, type:%s: msg:%s", event.key, type(event), event.fullFormattedMessage
            )
            events.extend(collected_events)
        return events

    @smart_retry
    def get_max_query_metrics(self):
        # type: () -> float
        vcenter_settings = self._conn.content.setting.QueryOptions(MAX_QUERY_METRICS_OPTION)
        max_historical_metrics = int(vcenter_settings[0].value)
        if max_historical_metrics > 0:
            return max_historical_metrics
        else:
            return UNLIMITED_HIST_METRICS_PER_QUERY

    @smart_retry
    def get_vsan_events(self, timestamp):
        event_manager = self._conn.content.eventManager
        entity_time = vim.event.EventFilterSpec.ByTime(beginTime=timestamp)
        query_filter = vim.event.EventFilterSpec(eventTypeId=VSAN_EVENT_IDS, time=entity_time)
        events = event_manager.QueryEvents(query_filter)
        self.log.debug("Received %s vSAN events", len(events))
        return events

    @smart_retry
    def get_vsan_metrics(self, cluster_nested_elts, entity_ref_ids, id_to_tags, starting_time):
        self.log.debug('Querying vSAN metrics')
        vsan_perf_manager = vim.cluster.VsanPerformanceManager('vsan-performance-manager', self._vsan_stub)
        health_metrics = []
        performance_metrics = []
        for cluster_reference, nested_ids in cluster_nested_elts.items():
            self.log.debug("Querying vSAN metrics for cluster %s", cluster_reference.name)
            unprocessed_health_metrics = vsan_perf_manager.QueryClusterHealth(cluster_reference)
            if len(unprocessed_health_metrics) <= 0:
                self.log.debug("No health metrics returned for cluster %s", cluster_reference.name)
                continue
            processed_health_metrics = {}
            group_id = unprocessed_health_metrics[0].groupId
            group_health = unprocessed_health_metrics[0].groupHealth
            processed_health_metrics.update(
                {
                    'vsphere.vsan.cluster.health.count': {
                        'group_id': group_id,
                        'status': group_health,
                        'vsphere_cluster': cluster_reference.name,
                    }
                }
            )
            for health_test in unprocessed_health_metrics[0].groupTests:
                test_name = health_test.testId.split('.')[-1]
                processed_health_metrics.update(
                    {
                        'vsphere.vsan.cluster.health.{}.count'.format(test_name): {
                            'group_id': group_id,
                            'status': group_health,
                            'test_id': health_test.testId,
                            'test_status': health_test.testHealth,
                            'vsphere_cluster': cluster_reference.name,
                        }
                    }
                )
            health_metrics.append(processed_health_metrics)

            vsan_perf_query_spec = []
            for nested_id in nested_ids:
                for entity_type in entity_ref_ids[id_to_tags[nested_id][0]]:
                    vsan_perf_query_spec.append(
                        vim.cluster.VsanPerfQuerySpec(
                            entityRefId=(entity_type + str(nested_id)),
                            labels=list(ENTITY_REMAPPER[entity_type]),
                            startTime=starting_time,
                        )
                    )
            discovered_metrics = vsan_perf_manager.QueryVsanPerf(vsan_perf_query_spec, cluster_reference)
            for entity_type in discovered_metrics:
                for metric in entity_type.value:
                    metric.metricId.dynamicProperty.append(
                        id_to_tags[entity_type.entityRefId.replace("'", "").split(':')[-1]]
                    )
            performance_metrics.append(discovered_metrics)
        return [health_metrics, performance_metrics]
