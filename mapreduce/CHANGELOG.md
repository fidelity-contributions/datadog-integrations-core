# CHANGELOG - mapreduce

<!-- towncrier release notes start -->

## 7.0.0 / 2025-07-10

***Changed***:

* Bump datadog_checks_base to 37.16.0 ([#20711](https://github.com/DataDog/integrations-core/pull/20711))

## 6.1.0 / 2025-01-16 / Agent 7.63.0

***Added***:

* Add `tls_ciphers` param to integration ([#19334](https://github.com/DataDog/integrations-core/pull/19334))

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

## 4.2.1 / 2024-05-31 / Agent 7.55.0

***Fixed***:

* Update the description for the `tls_ca_cert` config option to use `openssl rehash` instead of `c_rehash` ([#16981](https://github.com/DataDog/integrations-core/pull/16981))

## 4.2.0 / 2024-02-16 / Agent 7.52.0

***Added***:

* Update the configuration file to include the new oauth options parameter ([#16835](https://github.com/DataDog/integrations-core/pull/16835))

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

## 3.2.0 / 2022-09-16 / Agent 7.40.0

***Added***:

* Update HTTP config spec templates ([#12890](https://github.com/DataDog/integrations-core/pull/12890))

## 3.1.0 / 2022-04-05 / Agent 7.36.0

***Added***:

* Add metric_patterns options to filter all metric submission by a list of regexes ([#11695](https://github.com/DataDog/integrations-core/pull/11695))

***Fixed***:

* Remove outdated warning in the description for the `tls_ignore_warning` option ([#11591](https://github.com/DataDog/integrations-core/pull/11591))

## 3.0.0 / 2022-02-19 / Agent 7.35.0

***Changed***:

* Add tls_protocols_allowed option documentation ([#11251](https://github.com/DataDog/integrations-core/pull/11251))

***Added***:

* Add `pyproject.toml` file ([#11393](https://github.com/DataDog/integrations-core/pull/11393))

***Fixed***:

* Fix namespace packaging on Python 2 ([#11532](https://github.com/DataDog/integrations-core/pull/11532))

## 2.1.1 / 2022-01-08 / Agent 7.34.0

***Fixed***:

* Add comment to autogenerated model files ([#10945](https://github.com/DataDog/integrations-core/pull/10945))

## 2.1.0 / 2021-10-04 / Agent 7.32.0

***Added***:

* Add runtime configuration validation ([#8951](https://github.com/DataDog/integrations-core/pull/8951))
* Add HTTP option to control the size of streaming responses ([#10183](https://github.com/DataDog/integrations-core/pull/10183))
* Add allow_redirect option ([#10160](https://github.com/DataDog/integrations-core/pull/10160))

***Fixed***:

* Fix the description of the `allow_redirects` HTTP option ([#10195](https://github.com/DataDog/integrations-core/pull/10195))

## 2.0.0 / 2021-08-22 / Agent 7.31.0

***Changed***:

* Remove messages for integrations for OK service checks ([#9888](https://github.com/DataDog/integrations-core/pull/9888))

## 1.14.0 / 2021-03-07 / Agent 7.27.0

***Added***:

* Rename cluster_name tag to mapreduce_cluster ([#8588](https://github.com/DataDog/integrations-core/pull/8588))

***Fixed***:

* Bump minimum base package version ([#8443](https://github.com/DataDog/integrations-core/pull/8443))

## 1.13.0 / 2020-10-31 / Agent 7.24.0

***Added***:

* Add ability to dynamically get authentication information ([#7660](https://github.com/DataDog/integrations-core/pull/7660))
* [doc] Add encoding in log config sample ([#7708](https://github.com/DataDog/integrations-core/pull/7708))

## 1.12.0 / 2020-09-21 / Agent 7.23.0

***Added***:

* Add RequestsWrapper option to support UTF-8 for basic auth ([#7441](https://github.com/DataDog/integrations-core/pull/7441))
* Add mapreduce config specs ([#7178](https://github.com/DataDog/integrations-core/pull/7178))

***Fixed***:

* Fix style for the latest release of Black ([#7438](https://github.com/DataDog/integrations-core/pull/7438))

## 1.11.1 / 2020-08-10 / Agent 7.22.0

***Fixed***:

* Update ntlm_domain example ([#7118](https://github.com/DataDog/integrations-core/pull/7118))

## 1.11.0 / 2020-06-29 / Agent 7.21.0

***Added***:

* Add note about warning concurrency ([#6967](https://github.com/DataDog/integrations-core/pull/6967))

## 1.10.0 / 2020-05-17 / Agent 7.20.0

***Added***:

* Allow optional dependency installation for all checks ([#6589](https://github.com/DataDog/integrations-core/pull/6589))

## 1.9.0 / 2020-04-04 / Agent 7.19.0

***Added***:

* Version metadata ([#5448](https://github.com/DataDog/integrations-core/pull/5448))

## 1.8.0 / 2020-01-13 / Agent 7.17.0

***Added***:

* Use lazy logging format ([#5398](https://github.com/DataDog/integrations-core/pull/5398))
* Use lazy logging format ([#5377](https://github.com/DataDog/integrations-core/pull/5377))

## 1.7.0 / 2019-12-02 / Agent 7.16.0

***Added***:

* Add auth type to RequestsWrapper ([#4708](https://github.com/DataDog/integrations-core/pull/4708))

## 1.6.0 / 2019-10-11 / Agent 6.15.0

***Added***:

* Add option to override KRB5CCNAME env var ([#4578](https://github.com/DataDog/integrations-core/pull/4578))

## 1.5.0 / 2019-07-12 / Agent 6.13.0

***Added***:

* Use the new RequestsWrapper for connecting to services ([#4094](https://github.com/DataDog/integrations-core/pull/4094))

## 1.4.0 / 2019-05-14 / Agent 6.12.0

***Added***:

* Adhere to code style ([#3535](https://github.com/DataDog/integrations-core/pull/3535))

## 1.3.0 / 2019-02-18 / Agent 6.10.0

***Added***:

* Support Python 3 ([#2870](https://github.com/DataDog/integrations-core/pull/2870))

## 1.2.1 / 2018-09-04 / Agent 6.5.0

***Fixed***:

* Fix bug and typo in DEFAULT_CLUSTER_NAME for YARN check ([#1814][1]) Thanks [eplanet][2].
* Add data files to the wheel package ([#1727][3])

## 1.2.0 / 2018-06-06

***Added***:

* Add support for HTTP authentication ([#1678][4])
* Adds skip ssl validation ([#1470][5])

## 1.1.0 / 2018-03-23

***Added***:

* Adds custom tag support to service check.

## 1.0.0 / 2017-03-22

***Added***:

* adds mapreduce integration.

[1]: https://github.com/DataDog/integrations-core/pull/1814
[2]: https://github.com/eplanet
[3]: https://github.com/DataDog/integrations-core/pull/1727
[4]: https://github.com/DataDog/integrations-core/pull/1678
[5]: https://github.com/DataDog/integrations-core/pull/1470
