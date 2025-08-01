# (C) Datadog, Inc. 2021-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)

# This file is autogenerated.
# To change this file you should edit assets/configuration/spec.yaml and then run the following commands:
#     ddev -x validate config -s <INTEGRATION_NAME>
#     ddev -x validate models -s <INTEGRATION_NAME>

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from datadog_checks.base.utils.functions import identity
from datadog_checks.base.utils.models import validation

from . import defaults, deprecations, validators


class Aws(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    cluster_identifier: Optional[str] = None
    instance_endpoint: Optional[str] = None


class CollectSchemas(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    collect_search_indexes: Optional[bool] = None
    collection_interval: Optional[float] = None
    enabled: Optional[bool] = None
    max_collections: Optional[float] = None
    max_depth: Optional[float] = None
    sample_size: Optional[float] = None


class Field(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    field_name: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None


class CustomQuery(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    database: Optional[str] = None
    fields: Optional[tuple[Field, ...]] = None
    metric_prefix: Optional[str] = None
    query: Optional[MappingProxyType[str, Any]] = None
    tags: Optional[tuple[str, ...]] = None


class DatabaseAutodiscovery(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    enabled: Optional[bool] = None
    exclude: Optional[tuple[str, ...]] = None
    include: Optional[tuple[str, ...]] = None
    max_collections_per_database: Optional[int] = None
    max_databases: Optional[int] = None
    refresh_interval: Optional[int] = None


class MetricPatterns(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    exclude: Optional[tuple[str, ...]] = None
    include: Optional[tuple[str, ...]] = None


class MetricsCollectionInterval(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    collection: Optional[int] = None
    collections_indexes_stats: Optional[int] = None
    db_stats: Optional[int] = None
    session_stats: Optional[int] = None
    sharded_data_distribution: Optional[int] = None


class OperationSamples(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    collection_interval: Optional[float] = None
    enabled: Optional[bool] = None
    explain_verbosity: Optional[str] = None


class Schemas(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    collect_search_indexes: Optional[bool] = None
    collection_interval: Optional[float] = None
    enabled: Optional[bool] = None
    max_collections: Optional[float] = None
    max_depth: Optional[float] = None
    sample_size: Optional[float] = None


class SlowOperations(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    collection_interval: Optional[float] = None
    enabled: Optional[bool] = None
    explain_verbosity: Optional[str] = None
    max_operations: Optional[float] = None


class InstanceConfig(BaseModel):
    model_config = ConfigDict(
        validate_default=True,
        arbitrary_types_allowed=True,
        frozen=True,
    )
    add_node_tag_to_events: Optional[bool] = None
    additional_metrics: Optional[tuple[str, ...]] = None
    aws: Optional[Aws] = None
    cluster_name: Optional[str] = None
    collect_schemas: Optional[CollectSchemas] = None
    collections: Optional[tuple[str, ...]] = None
    collections_indexes_stats: Optional[bool] = None
    connection_scheme: Optional[str] = None
    custom_queries: Optional[tuple[CustomQuery, ...]] = None
    database: Optional[str] = None
    database_autodiscovery: Optional[DatabaseAutodiscovery] = None
    database_instance_collection_interval: Optional[float] = None
    dbm: Optional[bool] = None
    dbnames: Optional[tuple[str, ...]] = None
    dbstats_tag_dbname: Optional[bool] = None
    disable_generic_tags: Optional[bool] = None
    empty_default_hostname: Optional[bool] = None
    free_storage_metrics: Optional[bool] = None
    hosts: Optional[Union[str, tuple[str, ...]]] = None
    metric_patterns: Optional[MetricPatterns] = None
    metrics_collection_interval: Optional[MetricsCollectionInterval] = None
    min_collection_interval: Optional[float] = None
    operation_samples: Optional[OperationSamples] = None
    options: Optional[MappingProxyType[str, Any]] = None
    password: Optional[str] = None
    replica_check: Optional[bool] = None
    reported_database_hostname: Optional[str] = None
    schemas: Optional[Schemas] = None
    server: Optional[str] = None
    service: Optional[str] = None
    slow_operations: Optional[SlowOperations] = None
    system_database_stats: Optional[bool] = None
    tags: Optional[tuple[str, ...]] = None
    timeout: Optional[int] = None
    tls: Optional[bool] = None
    tls_allow_invalid_certificates: Optional[bool] = None
    tls_allow_invalid_hostnames: Optional[bool] = None
    tls_ca_file: Optional[str] = None
    tls_certificate_key_file: Optional[str] = None
    username: Optional[str] = None

    @model_validator(mode='before')
    def _handle_deprecations(cls, values, info):
        fields = info.context['configured_fields']
        validation.utils.handle_deprecations('instances', deprecations.instance(), fields, info.context)
        return values

    @model_validator(mode='before')
    def _initial_validation(cls, values):
        return validation.core.initialize_config(getattr(validators, 'initialize_instance', identity)(values))

    @field_validator('*', mode='before')
    def _validate(cls, value, info):
        field = cls.model_fields[info.field_name]
        field_name = field.alias or info.field_name
        if field_name in info.context['configured_fields']:
            value = getattr(validators, f'instance_{info.field_name}', identity)(value, field=field)
        else:
            value = getattr(defaults, f'instance_{info.field_name}', lambda: value)()

        return validation.utils.make_immutable(value)

    @model_validator(mode='after')
    def _final_validation(cls, model):
        return validation.core.check_model(getattr(validators, 'check_instance', identity)(model))
