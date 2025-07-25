# CHANGELOG - spark

<!-- towncrier release notes start -->

## 7.0.0 / 2025-07-10

***Changed***:

* Bump datadog_checks_base to 37.16.0 ([#20711](https://github.com/DataDog/integrations-core/pull/20711))

***Fixed***:

* Allow HTTPS requests to use `tls_ciphers` parameter ([#20179](https://github.com/DataDog/integrations-core/pull/20179))
* Modernize bs4 interface by replacing deprecated 'findAll' with 'find_all' ([#20415](https://github.com/DataDog/integrations-core/pull/20415))

## 6.4.0 / 2025-05-15 / Agent 7.67.0

***Added***:

* Update dependencies ([#20215](https://github.com/DataDog/integrations-core/pull/20215))

## 6.3.0 / 2025-03-19 / Agent 7.65.0

***Added***:

* Update dependencies ([#19687](https://github.com/DataDog/integrations-core/pull/19687))

***Fixed***:

* Gracefully handle unavailable apps and their aspects. Before we would throw an exception as soon as we encountered an error, which deprived us of a lot of available metrics. ([#19750](https://github.com/DataDog/integrations-core/pull/19750))

## 6.2.0 / 2025-01-16 / Agent 7.63.0

***Added***:

* Add `tls_ciphers` param to integration ([#19334](https://github.com/DataDog/integrations-core/pull/19334))

## 6.1.0 / 2024-10-31 / Agent 7.60.0

***Added***:

* Add configuration option to disable `stage_id` tag on `spark.job` metrics and disable `spark.stage` metrics ([#18791](https://github.com/DataDog/integrations-core/pull/18791))

## 6.0.0 / 2024-10-04 / Agent 7.59.0

***Removed***:

* Remove support for Python 2. ([#18580](https://github.com/DataDog/integrations-core/pull/18580))

***Fixed***:

* Bump the version of datadog-checks-base to 37.0.0 ([#18617](https://github.com/DataDog/integrations-core/pull/18617))

## 5.0.0 / 2024-10-01 / Agent 7.58.0

***Changed***:

* Bump minimum version of base check ([#18733](https://github.com/DataDog/integrations-core/pull/18733))

***Added***:

* Bump the python version from 3.11 to 3.12 ([#18212](https://github.com/DataDog/integrations-core/pull/18212))

## 4.3.1 / 2024-05-31 / Agent 7.55.0

***Fixed***:

* Update the description for the `tls_ca_cert` config option to use `openssl rehash` instead of `c_rehash` ([#16981](https://github.com/DataDog/integrations-core/pull/16981))

## 4.3.0 / 2024-03-22 / Agent 7.53.0

***Added***:

* Adds additional Executor-level memory metrics (https://spark.apache.org/docs/latest/monitoring.html#executor-metrics) ([#17168](https://github.com/DataDog/integrations-core/pull/17168))

## 4.2.0 / 2024-02-16 / Agent 7.52.0

***Added***:

* Update dependencies ([#16788](https://github.com/DataDog/integrations-core/pull/16788))

## 4.1.0 / 2024-01-05 / Agent 7.51.0

***Added***:

* Bump the Python version from py3.9 to py3.11 ([#15997](https://github.com/DataDog/integrations-core/pull/15997))

## 4.0.0 / 2023-08-10 / Agent 7.48.0

***Changed***:

* Bump the minimum base check version ([#15427](https://github.com/DataDog/integrations-core/pull/15427))

***Added***:

* Update generated config models ([#15212](https://github.com/DataDog/integrations-core/pull/15212))

***Fixed***:

* Fix types for generated config models ([#15334](https://github.com/DataDog/integrations-core/pull/15334))

## 3.2.1 / 2023-07-10 / Agent 7.47.0

***Fixed***:

* Bump Python version from py3.8 to py3.9 ([#14701](https://github.com/DataDog/integrations-core/pull/14701))

## 3.2.0 / 2023-04-14 / Agent 7.45.0

***Added***:

* Update dependencies ([#14357](https://github.com/DataDog/integrations-core/pull/14357))

## 3.1.3 / 2022-12-09 / Agent 7.42.0

***Fixed***:

* Allow punctuation marks in query name when publishing spark structured streaming metrics ([#13246](https://github.com/DataDog/integrations-core/pull/13246)) Thanks [cpolito88](https://github.com/cpolito88).

## 3.1.2 / 2022-10-28 / Agent 7.41.0

***Fixed***:

* Add a log line when no running apps are found ([#13067](https://github.com/DataDog/integrations-core/pull/13067))

## 3.1.1 / 2022-05-15 / Agent 7.37.0

***Fixed***:

* Upgrade dependencies ([#11958](https://github.com/DataDog/integrations-core/pull/11958))

## 3.1.0 / 2022-04-05 / Agent 7.36.0

***Added***:

* Add metric_patterns options to filter all metric submission by a list of regexes ([#11695](https://github.com/DataDog/integrations-core/pull/11695))

***Fixed***:

* Remove outdated warning in the description for the `tls_ignore_warning` option ([#11591](https://github.com/DataDog/integrations-core/pull/11591))

## 3.0.0 / 2022-02-19 / Agent 7.35.0

***Changed***:

* Add tls_protocols_allowed option documentation ([#11251](https://github.com/DataDog/integrations-core/pull/11251))

***Added***:

* Add `pyproject.toml` file ([#11436](https://github.com/DataDog/integrations-core/pull/11436))

***Fixed***:

* Fix namespace packaging on Python 2 ([#11532](https://github.com/DataDog/integrations-core/pull/11532))

## 2.2.0 / 2022-01-08 / Agent 7.34.0

***Added***:

* Add `query_name` as tag for Spark Structured Streaming metrics ([#10689](https://github.com/DataDog/integrations-core/pull/10689)) Thanks [otosky](https://github.com/otosky).

***Fixed***:

* Don't submit query name tag if query is a uuid ([#11015](https://github.com/DataDog/integrations-core/pull/11015))
* Add comment to autogenerated model files ([#10945](https://github.com/DataDog/integrations-core/pull/10945))

## 2.1.0 / 2021-10-04 / Agent 7.32.0

***Added***:

* Update dependencies ([#10228](https://github.com/DataDog/integrations-core/pull/10228))
* Add HTTP option to control the size of streaming responses ([#10183](https://github.com/DataDog/integrations-core/pull/10183))
* Add allow_redirect option ([#10160](https://github.com/DataDog/integrations-core/pull/10160))
* Disable generic tags ([#10027](https://github.com/DataDog/integrations-core/pull/10027))

***Fixed***:

* Fix the description of the `allow_redirects` HTTP option ([#10195](https://github.com/DataDog/integrations-core/pull/10195))

## 2.0.0 / 2021-08-22 / Agent 7.31.0

***Changed***:

* Remove messages for integrations for OK service checks ([#9888](https://github.com/DataDog/integrations-core/pull/9888))

***Added***:

* Use `display_default` as a fallback for `default` when validating config models ([#9739](https://github.com/DataDog/integrations-core/pull/9739))

## 1.21.0 / 2021-04-19 / Agent 7.28.0

***Added***:

* Add runtime configuration validation ([#8986](https://github.com/DataDog/integrations-core/pull/8986))

## 1.20.0 / 2021-03-07 / Agent 7.27.0

***Added***:

* Rename cluster_name tag to spark_cluster ([#8592](https://github.com/DataDog/integrations-core/pull/8592))

***Fixed***:

* Skip apps which have the UI disabled ([#8558](https://github.com/DataDog/integrations-core/pull/8558))
* Bump minimum base package version ([#8443](https://github.com/DataDog/integrations-core/pull/8443))

## 1.19.1 / 2021-01-25 / Agent 7.26.0

***Fixed***:

* Update check signature ([#8259](https://github.com/DataDog/integrations-core/pull/8259))

## 1.19.0 / 2020-12-29

***Added***:

* Add metrics for structured streams ([#8078](https://github.com/DataDog/integrations-core/pull/8078))

## 1.18.0 / 2020-11-23 / Agent 7.25.0

***Added***:

* Add more granular executor metrics ([#8028](https://github.com/DataDog/integrations-core/pull/8028))

## 1.17.0 / 2020-11-06 / Agent 7.24.0

***Added***:

* Update HTTP config docs to describe dcos_auth token reader ([#7953](https://github.com/DataDog/integrations-core/pull/7953))

## 1.16.0 / 2020-10-31

***Added***:

* Add ability to dynamically get authentication information ([#7660](https://github.com/DataDog/integrations-core/pull/7660))
* [doc] Add encoding in log config sample ([#7708](https://github.com/DataDog/integrations-core/pull/7708))

## 1.15.0 / 2020-09-03 / Agent 7.23.0

***Added***:

* Add Stage and Job ID tags ([#7459](https://github.com/DataDog/integrations-core/pull/7459))
* Add RequestsWrapper option to support UTF-8 for basic auth ([#7441](https://github.com/DataDog/integrations-core/pull/7441))

***Fixed***:

* Update proxy section in conf.yaml ([#7336](https://github.com/DataDog/integrations-core/pull/7336))

## 1.14.0 / 2020-08-10 / Agent 7.22.0

***Added***:

* Add documentation for spark logs ([#7109](https://github.com/DataDog/integrations-core/pull/7109))

***Fixed***:

* Update logs config service field to optional ([#7209](https://github.com/DataDog/integrations-core/pull/7209))
* DOCS-838 Template wording ([#7038](https://github.com/DataDog/integrations-core/pull/7038))
* Update ntlm_domain example ([#7118](https://github.com/DataDog/integrations-core/pull/7118))

## 1.13.0 / 2020-06-29 / Agent 7.21.0

***Added***:

* Add note about warning concurrency ([#6967](https://github.com/DataDog/integrations-core/pull/6967))
* Add config specs ([#6921](https://github.com/DataDog/integrations-core/pull/6921))

## 1.12.0 / 2020-05-17 / Agent 7.20.0

***Added***:

* Allow optional dependency installation for all checks ([#6589](https://github.com/DataDog/integrations-core/pull/6589))

## 1.11.4 / 2020-02-22 / Agent 7.18.0

***Fixed***:

* Update documentation in example config ([#5508](https://github.com/DataDog/integrations-core/pull/5508))

## 1.11.3 / 2020-01-30

***Fixed***:

* Handle warning message from proxy ([#5525](https://github.com/DataDog/integrations-core/pull/5525))

## 1.11.2 / 2020-01-29

***Fixed***:

* Prevent crash when a single app fails ([#5552](https://github.com/DataDog/integrations-core/pull/5552))

## 1.11.1 / 2020-01-15 / Agent 7.17.0

***Fixed***:

* Make sure version collection fails gracefully ([#5465](https://github.com/DataDog/integrations-core/pull/5465))

## 1.11.0 / 2020-01-13

***Added***:

* Use lazy logging format ([#5398](https://github.com/DataDog/integrations-core/pull/5398))
* Use lazy logging format ([#5377](https://github.com/DataDog/integrations-core/pull/5377))
* Collect version metadata ([#5032](https://github.com/DataDog/integrations-core/pull/5032))

## 1.10.1 / 2019-12-06 / Agent 7.16.0

***Fixed***:

* Remove reference to Kubernetes in the service check message for `spark_driver_mode` ([#5159](https://github.com/DataDog/integrations-core/pull/5159))

## 1.10.0 / 2019-12-02

***Added***:

* Add Spark driver support ([#4631](https://github.com/DataDog/integrations-core/pull/4631)) Thanks [mrmuggymuggy](https://github.com/mrmuggymuggy).

## 1.9.0 / 2019-10-11 / Agent 6.15.0

***Added***:

* Add option to override KRB5CCNAME env var ([#4578](https://github.com/DataDog/integrations-core/pull/4578))

## 1.8.1 / 2019-07-18 / Agent 6.13.0

***Fixed***:

* Remove unused configs and code for spark check ([#4133](https://github.com/DataDog/integrations-core/pull/4133))

## 1.8.0 / 2019-07-09

***Added***:

* Use the new RequestsWrapper for connecting to services ([#4058](https://github.com/DataDog/integrations-core/pull/4058))

## 1.7.0 / 2019-05-14 / Agent 6.12.0

***Added***:

* Adhere to code style ([#3566](https://github.com/DataDog/integrations-core/pull/3566))

## 1.6.0 / 2019-01-08 / Agent 6.10.0

***Added***:

* Allow disabling of streaming metrics ([#2889](https://github.com/DataDog/integrations-core/pull/2889))
* Support Kerberos auth ([#2825](https://github.com/DataDog/integrations-core/pull/2825))

## 1.5.0 / 2018-12-20 / Agent 6.9.0

***Added***:

* Add streaming statistics metrics to the spark integration ([#2437](https://github.com/DataDog/integrations-core/pull/2437))

## 1.4.1 / 2018-09-04 / Agent 6.5.0

***Fixed***:

* Add data files to the wheel package ([#1727](https://github.com/DataDog/integrations-core/pull/1727))

## 1.4.0 / 2018-06-07

***Added***:

* Add support for HTTP authentication ([#1680](https://github.com/DataDog/integrations-core/pull/1680))

## 1.3.0 / 2018-05-11

***Added***:

* adds custom tag support to service check.

## 1.2.0 / 2018-02-13

***Added***:

* Add configuration options `ssl_verify`, `ssl_cert` and `ssl_key` to allow SSL configuration ([#1064](https://github.com/DataDog/integrations-core/pull/1064))

## 1.1.0 / 2018-01-10

***Added***:

* Filter Spark frameworks by port ([#459](https://github.com/DataDog/integrations-core/pull/459))  (Thanks [@johnjeffers](https://github.com/johnjeffers))

## 1.0.1 / 2017-07-18

***Fixed***:

* Build proxy-compatible URL  ([#437](https://github)com/DataDog/integrations-core/issues/437)

## 1.0.0 / 2017-03-22

***Added***:

* adds spark integration.
