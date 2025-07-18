# CHANGELOG - Aerospike

<!-- towncrier release notes start -->

## 5.0.0 / 2025-07-10

***Changed***:

* Bump datadog_checks_base to 37.16.0 ([#20711](https://github.com/DataDog/integrations-core/pull/20711))

## 4.1.0 / 2025-01-16 / Agent 7.63.0

***Added***:

* Add `tls_ciphers` param to integration ([#19334](https://github.com/DataDog/integrations-core/pull/19334))

## 4.0.1 / 2024-12-26 / Agent 7.62.0

***Fixed***:

* Don't skip last index in each namespace ([#18996](https://github.com/DataDog/integrations-core/pull/18996))

## 4.0.0 / 2024-10-04 / Agent 7.59.0

***Removed***:

* Remove support for Python 2. ([#18580](https://github.com/DataDog/integrations-core/pull/18580))

***Fixed***:

* Bump the version of datadog-checks-base to 37.0.0 ([#18617](https://github.com/DataDog/integrations-core/pull/18617))

## 3.0.0 / 2024-10-01 / Agent 7.58.0

***Changed***:

* Bump minimum version of base check ([#18733](https://github.com/DataDog/integrations-core/pull/18733))

***Added***:

* Bump the python version from 3.11 to 3.12 ([#18212](https://github.com/DataDog/integrations-core/pull/18212))

## 2.2.2 / 2024-07-05 / Agent 7.55.0

***Fixed***:

* Update config model names ([#17802](https://github.com/DataDog/integrations-core/pull/17802))

## 2.2.1 / 2024-05-31

***Fixed***:

* Update the description for the `tls_ca_cert` config option to use `openssl rehash` instead of `c_rehash` ([#16981](https://github.com/DataDog/integrations-core/pull/16981))

## 2.2.0 / 2024-02-16 / Agent 7.52.0

***Added***:

* Update the configuration file to include the new oauth options parameter ([#16835](https://github.com/DataDog/integrations-core/pull/16835))

## 2.1.0 / 2024-01-05 / Agent 7.51.0

***Added***:

* Bump the Python version from py3.9 to py3.11 ([#15997](https://github.com/DataDog/integrations-core/pull/15997))
* Collect more `client_read` and `client_write` metrics by default ([#16423](https://github.com/DataDog/integrations-core/pull/16423))

## 2.0.0 / 2023-08-10 / Agent 7.48.0

***Changed***:

* Bump the minimum base check version ([#15427](https://github.com/DataDog/integrations-core/pull/15427))

***Added***:

* Update generated config models ([#15212](https://github.com/DataDog/integrations-core/pull/15212))

***Fixed***:

* Fix types for generated config models ([#15334](https://github.com/DataDog/integrations-core/pull/15334))

## 1.18.1 / 2023-07-10 / Agent 7.47.0

***Fixed***:

* Bump Python version from py3.8 to py3.9 ([#14701](https://github.com/DataDog/integrations-core/pull/14701))

## 1.18.0 / 2023-05-26 / Agent 7.46.0

***Added***:

* Add an ignore_connection_errors option to the openmetrics check ([#14504](https://github.com/DataDog/integrations-core/pull/14504))

***Fixed***:

* Update minimum datadog base package version ([#14463](https://github.com/DataDog/integrations-core/pull/14463))
* Deprecate `use_latest_spec` option ([#14446](https://github.com/DataDog/integrations-core/pull/14446))

## 1.17.2 / 2022-10-28 / Agent 7.41.0

***Fixed***:

* Update dependencies ([#13205](https://github.com/DataDog/integrations-core/pull/13205))

## 1.17.1 / 2022-09-28

***Fixed***:

* Fix missing latency metrics from other namespaces ([#12944](https://github.com/DataDog/integrations-core/pull/12944))

## 1.17.0 / 2022-09-16 / Agent 7.40.0

***Added***:

* Update HTTP config spec templates ([#12890](https://github.com/DataDog/integrations-core/pull/12890))

***Fixed***:

* Bump dependencies for 7.40 ([#12896](https://github.com/DataDog/integrations-core/pull/12896))

## 1.16.2 / 2022-08-05 / Agent 7.39.0

***Fixed***:

* Dependency updates ([#12653](https://github.com/DataDog/integrations-core/pull/12653))

## 1.16.1 / 2022-05-18 / Agent 7.37.0

***Fixed***:

* Fix extra metrics description example ([#12043](https://github.com/DataDog/integrations-core/pull/12043))

## 1.16.0 / 2022-05-15

***Added***:

* Add OpenMetricsV2 Implementation ([#11845](https://github.com/DataDog/integrations-core/pull/11845))

## 1.15.0 / 2022-04-05 / Agent 7.36.0

***Added***:

* Upgrade dependencies ([#11726](https://github.com/DataDog/integrations-core/pull/11726))
* Add metric_patterns options to filter all metric submission by a list of regexes ([#11695](https://github.com/DataDog/integrations-core/pull/11695))

***Fixed***:

* Fix broken latency metric collection ([#11752](https://github.com/DataDog/integrations-core/pull/11752))
* Support newer versions of `click` ([#11746](https://github.com/DataDog/integrations-core/pull/11746))

## 1.14.0 / 2022-02-19 / Agent 7.35.0

***Added***:

* Add `pyproject.toml` file ([#11311](https://github.com/DataDog/integrations-core/pull/11311))

***Fixed***:

* Fix namespace packaging on Python 2 ([#11532](https://github.com/DataDog/integrations-core/pull/11532))

## 1.13.1 / 2022-01-08 / Agent 7.34.0

***Fixed***:

* Add comment to autogenerated model files ([#10945](https://github.com/DataDog/integrations-core/pull/10945))

## 1.13.0 / 2021-11-13 / Agent 7.33.0

***Added***:

* Add runtime configuration validation ([#8881](https://github.com/DataDog/integrations-core/pull/8881))

## 1.12.0 / 2021-06-22 / Agent 7.30.0

***Added***:

* Stop using a deprecated method to support newer versions of Aerospike ([#9566](https://github.com/DataDog/integrations-core/pull/9566))
* Upgrade `aerospike` dependency on Python 3 ([#9552](https://github.com/DataDog/integrations-core/pull/9552))

***Fixed***:

* Handle command output with erroneous trailing separators ([#9571](https://github.com/DataDog/integrations-core/pull/9571))

## 1.11.0 / 2021-04-06 / Agent 7.28.0

***Added***:

* Support XDR metrics for Aerospike Enterprise 5.0+ ([#8696](https://github.com/DataDog/integrations-core/pull/8696))

## 1.10.1 / 2021-03-07 / Agent 7.27.0

***Fixed***:

* Return empty array instead of None ([#8532](https://github.com/DataDog/integrations-core/pull/8532))
* Fix logging ([#8515](https://github.com/DataDog/integrations-core/pull/8515))

## 1.10.0 / 2021-02-01

***Added***:

* Support Aerospike 5.3 ([#8430](https://github.com/DataDog/integrations-core/pull/8430))

***Fixed***:

* Bump minimum package ([#8443](https://github.com/DataDog/integrations-core/pull/8443))

## 1.9.0 / 2020-12-11 / Agent 7.25.0

***Added***:

* Add log pipeline ([#8068](https://github.com/DataDog/integrations-core/pull/8068))
* Update aerospike dependency ([#8044](https://github.com/DataDog/integrations-core/pull/8044))

***Fixed***:

* Update check signature ([#8157](https://github.com/DataDog/integrations-core/pull/8157))

## 1.8.3 / 2020-07-23 / Agent 7.22.0

***Fixed***:

* Fix empty result case ([#7192](https://github.com/DataDog/integrations-core/pull/7192))

## 1.8.2 / 2020-07-22

***Fixed***:

* Add debug log for get info calls ([#7182](https://github.com/DataDog/integrations-core/pull/7182))

## 1.8.1 / 2020-07-10

***Fixed***:

* Parse batch-index read latency metrics ([#6991](https://github.com/DataDog/integrations-core/pull/6991))

## 1.8.0 / 2020-05-17 / Agent 7.20.0

***Added***:

* Allow optional dependency installation for all checks ([#6589](https://github.com/DataDog/integrations-core/pull/6589))

## 1.7.1 / 2020-04-15

***Fixed***:

* Fix namespace valid chars matching regex ([#6352](https://github.com/DataDog/integrations-core/pull/6352))
* Fix namespace tagging for latency metrics ([#6345](https://github.com/DataDog/integrations-core/pull/6345))

## 1.7.0 / 2020-03-13 / Agent 7.19.0

***Added***:

* Add TLS config ([#6035](https://github.com/DataDog/integrations-core/pull/6035))

## 1.6.0 / 2020-02-22 / Agent 7.18.0

***Added***:

* Upgrade `aerospike` dependency ([#5779](https://github.com/DataDog/integrations-core/pull/5779))

## 1.5.0 / 2020-01-13 / Agent 7.17.0

***Added***:

* Use lazy logging format ([#5377](https://github.com/DataDog/integrations-core/pull/5377))
* Add latency and datacenter metrics for Aerospike also collect version metadata information ([#4969](https://github.com/DataDog/integrations-core/pull/4969))

## 1.4.0 / 2019-12-02 / Agent 7.16.0

***Added***:

* Standardize logging format ([#4897](https://github.com/DataDog/integrations-core/pull/4897))

## 1.3.0 / 2019-05-14 / Agent 6.12.0

***Added***:

* Adhere to code style ([#3484](https://github.com/DataDog/integrations-core/pull/3484))

## 1.2.0 / 2019-03-29 / Agent 6.11.0

***Added***:

* Upgrade aerospike dependency ([#3235](https://github.com/DataDog/integrations-core/pull/3235))

## 1.1.0 / 2019-02-27

***Added***:

* Add authentication and timeout options ([#3214](https://github.com/DataDog/integrations-core/pull/3214))
* Refactor check to use the official library ([#3212](https://github.com/DataDog/integrations-core/pull/3212))

## 1.0.0 / 2019-02-18 / Agent 6.10.0

***Added***:

* Officially support Aerospike ([#3078](https://github.com/DataDog/integrations-core/pull/3078))
