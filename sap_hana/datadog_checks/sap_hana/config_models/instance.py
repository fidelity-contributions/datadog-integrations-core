# (C) Datadog, Inc. 2021-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)

# This file is autogenerated.
# To change this file you should edit assets/configuration/spec.yaml and then run the following commands:
#     ddev -x validate config -s <INTEGRATION_NAME>
#     ddev -x validate models -s <INTEGRATION_NAME>

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from datadog_checks.base.utils.functions import identity
from datadog_checks.base.utils.models import validation

from . import defaults, validators


class CustomQuery(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    collection_interval: Optional[int] = None
    columns: Optional[tuple[MappingProxyType[str, Any], ...]] = None
    metric_prefix: Optional[str] = None
    query: Optional[str] = None
    tags: Optional[tuple[str, ...]] = None


class MetricPatterns(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    exclude: Optional[tuple[str, ...]] = None
    include: Optional[tuple[str, ...]] = None


class InstanceConfig(BaseModel):
    model_config = ConfigDict(
        validate_default=True,
        arbitrary_types_allowed=True,
        frozen=True,
    )
    batch_size: Optional[int] = None
    connection_properties: Optional[MappingProxyType[str, Any]] = None
    custom_queries: Optional[tuple[CustomQuery, ...]] = None
    disable_generic_tags: Optional[bool] = None
    empty_default_hostname: Optional[bool] = None
    metric_patterns: Optional[MetricPatterns] = None
    min_collection_interval: Optional[float] = None
    only_custom_queries: Optional[bool] = None
    password: str
    persist_db_connections: Optional[bool] = None
    port: Optional[int] = None
    schema_: Optional[str] = Field(None, alias='schema')
    server: str
    service: Optional[str] = None
    tags: Optional[tuple[str, ...]] = None
    timeout: Optional[float] = None
    tls_ca_cert: Optional[str] = None
    tls_cert: Optional[str] = None
    tls_ciphers: Optional[tuple[str, ...]] = None
    tls_private_key: Optional[str] = None
    tls_private_key_password: Optional[str] = None
    tls_validate_hostname: Optional[bool] = None
    tls_verify: Optional[bool] = None
    use_global_custom_queries: Optional[str] = None
    use_hana_hostnames: Optional[bool] = None
    use_tls: Optional[bool] = None
    username: str

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
