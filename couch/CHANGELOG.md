# CHANGELOG - couch

<!-- towncrier release notes start -->

## 9.0.0 / 2025-07-10

***Changed***:

* Bump datadog_checks_base to 37.16.0 ([#20711](https://github.com/DataDog/integrations-core/pull/20711))

## 8.3.0 / 2025-03-19 / Agent 7.65.0

***Added***:

* Add support for collecting different metric aggregations for message_queue metrics. ([#19521](https://github.com/DataDog/integrations-core/pull/19521))

## 8.2.0 / 2025-01-16 / Agent 7.63.0

***Added***:

* Add `tls_ciphers` param to integration ([#19334](https://github.com/DataDog/integrations-core/pull/19334))

## 8.1.0 / 2024-11-28 / Agent 7.61.0

***Added***:

* Add support for Couch version 3.4.0 ([#19052](https://github.com/DataDog/integrations-core/pull/19052))

## 8.0.0 / 2024-10-04 / Agent 7.59.0

***Removed***:

* Remove support for Python 2. ([#18580](https://github.com/DataDog/integrations-core/pull/18580))

***Fixed***:

* Bump the version of datadog-checks-base to 37.0.0 ([#18617](https://github.com/DataDog/integrations-core/pull/18617))

## 7.0.0 / 2024-10-01 / Agent 7.58.0

***Changed***:

* Bump minimum version of base check ([#18733](https://github.com/DataDog/integrations-core/pull/18733))

***Added***:

* Bump the python version from 3.11 to 3.12 ([#18212](https://github.com/DataDog/integrations-core/pull/18212))

## 6.2.1 / 2024-05-31 / Agent 7.55.0

***Fixed***:

* Update the description for the `tls_ca_cert` config option to use `openssl rehash` instead of `c_rehash` ([#16981](https://github.com/DataDog/integrations-core/pull/16981))

## 6.2.0 / 2024-02-16 / Agent 7.52.0

***Added***:

* Added the `enable_per_db_metrics` flag, to allow the gathering of these metrics to be disabled ([#16641](https://github.com/DataDog/integrations-core/pull/16641))
* Update the configuration file to include the new oauth options parameter ([#16835](https://github.com/DataDog/integrations-core/pull/16835))

## 6.1.0 / 2024-01-05 / Agent 7.51.0

***Added***:

* Bump the Python version from py3.9 to py3.11 ([#15997](https://github.com/DataDog/integrations-core/pull/15997))

## 6.0.0 / 2023-08-10 / Agent 7.48.0

***Changed***:

* Bump the minimum base check version ([#15427](https://github.com/DataDog/integrations-core/pull/15427))

***Added***:

* Update generated config models ([#15212](https://github.com/DataDog/integrations-core/pull/15212))

***Fixed***:

* Fix types for generated config models ([#15334](https://github.com/DataDog/integrations-core/pull/15334))

## 5.2.1 / 2023-07-10 / Agent 7.47.0

***Fixed***:

* Bump Python version from py3.8 to py3.9 ([#14701](https://github.com/DataDog/integrations-core/pull/14701))

## 5.2.0 / 2022-09-16 / Agent 7.40.0

***Added***:

* Update HTTP config spec templates ([#12890](https://github.com/DataDog/integrations-core/pull/12890))

## 5.1.1 / 2022-08-05 / Agent 7.39.0

***Fixed***:

* Dependency updates ([#12653](https://github.com/DataDog/integrations-core/pull/12653))

## 5.1.0 / 2022-04-05 / Agent 7.36.0

***Added***:

* Add metric_patterns options to filter all metric submission by a list of regexes ([#11695](https://github.com/DataDog/integrations-core/pull/11695))

***Fixed***:

* Remove outdated warning in the description for the `tls_ignore_warning` option ([#11591](https://github.com/DataDog/integrations-core/pull/11591))

## 5.0.0 / 2022-02-19 / Agent 7.35.0

***Changed***:

* Add tls_protocols_allowed option documentation ([#11251](https://github.com/DataDog/integrations-core/pull/11251))

***Added***:

* Add `pyproject.toml` file ([#11333](https://github.com/DataDog/integrations-core/pull/11333))

***Fixed***:

* Fix namespace packaging on Python 2 ([#11532](https://github.com/DataDog/integrations-core/pull/11532))

## 4.2.1 / 2022-01-08 / Agent 7.34.0

***Fixed***:

* Add comment to autogenerated model files ([#10945](https://github.com/DataDog/integrations-core/pull/10945))

## 4.2.0 / 2021-11-13 / Agent 7.33.0

***Added***:

* Add runtime configuration validation ([#8901](https://github.com/DataDog/integrations-core/pull/8901))

## 4.1.0 / 2021-10-04 / Agent 7.32.0

***Added***:

* Add HTTP option to control the size of streaming responses ([#10183](https://github.com/DataDog/integrations-core/pull/10183))
* Add allow_redirect option ([#10160](https://github.com/DataDog/integrations-core/pull/10160))

***Fixed***:

* Bump base package dependency ([#10218](https://github.com/DataDog/integrations-core/pull/10218))
* Fix the description of the `allow_redirects` HTTP option ([#10195](https://github.com/DataDog/integrations-core/pull/10195))

## 4.0.0 / 2021-08-22 / Agent 7.31.0

***Changed***:

* Remove messages for integrations for OK service checks ([#9888](https://github.com/DataDog/integrations-core/pull/9888))

## 3.13.3 / 2021-07-12 / Agent 7.30.0

***Fixed***:

* Use Agent 8 signature ([#9522](https://github.com/DataDog/integrations-core/pull/9522))

## 3.13.2 / 2021-03-07 / Agent 7.27.0

***Fixed***:

* Rename config spec example consumer option `default` to `display_default` ([#8593](https://github.com/DataDog/integrations-core/pull/8593))
* Bump minimum base package version ([#8443](https://github.com/DataDog/integrations-core/pull/8443))

## 3.13.1 / 2020-11-13 / Agent 7.24.0

***Fixed***:

* Fix exception message ([#7912](https://github.com/DataDog/integrations-core/pull/7912))

## 3.13.0 / 2020-10-31

***Added***:

* Support couch v3 ([#7570](https://github.com/DataDog/integrations-core/pull/7570))
* Add ability to dynamically get authentication information ([#7660](https://github.com/DataDog/integrations-core/pull/7660))
* [doc] Add encoding in log config sample ([#7708](https://github.com/DataDog/integrations-core/pull/7708))

## 3.12.0 / 2020-09-21 / Agent 7.23.0

***Added***:

* Add RequestsWrapper option to support UTF-8 for basic auth ([#7441](https://github.com/DataDog/integrations-core/pull/7441))

***Fixed***:

* Do not render null defaults for config spec example consumer ([#7503](https://github.com/DataDog/integrations-core/pull/7503))
* Update proxy section in conf.yaml ([#7336](https://github.com/DataDog/integrations-core/pull/7336))

## 3.11.0 / 2020-08-10 / Agent 7.22.0

***Added***:

* couch config specs ([#7160](https://github.com/DataDog/integrations-core/pull/7160))

***Fixed***:

* Update logs config service field to optional ([#7209](https://github.com/DataDog/integrations-core/pull/7209))
* DOCS-838 Template wording ([#7038](https://github.com/DataDog/integrations-core/pull/7038))
* Use inclusive wording ([#7159](https://github.com/DataDog/integrations-core/pull/7159))
* Update ntlm_domain example ([#7118](https://github.com/DataDog/integrations-core/pull/7118))

## 3.10.0 / 2020-06-29 / Agent 7.21.0

***Added***:

* Add note about warning concurrency ([#6967](https://github.com/DataDog/integrations-core/pull/6967))

## 3.9.0 / 2020-05-17 / Agent 7.20.0

***Added***:

* Allow optional dependency installation for all checks ([#6589](https://github.com/DataDog/integrations-core/pull/6589))

## 3.8.1 / 2020-04-04 / Agent 7.19.0

***Fixed***:

* Update deprecated imports ([#6088](https://github.com/DataDog/integrations-core/pull/6088))
* Remove logs sourcecategory ([#6121](https://github.com/DataDog/integrations-core/pull/6121))

## 3.8.0 / 2020-02-22 / Agent 7.18.0

***Added***:

* Add version metadata ([#5615](https://github.com/DataDog/integrations-core/pull/5615))

## 3.7.0 / 2020-01-13 / Agent 7.17.0

***Added***:

* Use lazy logging format ([#5398](https://github.com/DataDog/integrations-core/pull/5398))

## 3.6.0 / 2019-12-02 / Agent 7.16.0

***Added***:

* Standardize logging format ([#4904](https://github.com/DataDog/integrations-core/pull/4904))

## 3.5.0 / 2019-10-11 / Agent 6.15.0

***Added***:

* Add option to override KRB5CCNAME env var ([#4578](https://github.com/DataDog/integrations-core/pull/4578))

## 3.4.1 / 2019-08-30 / Agent 6.14.0

***Fixed***:

* Update class signature to support the RequestsWrapper ([#4469](https://github.com/DataDog/integrations-core/pull/4469))

## 3.4.0 / 2019-08-24

***Added***:

* Add RequestsWrapper to couch ([#4118](https://github.com/DataDog/integrations-core/pull/4118))

## 3.3.0 / 2019-05-14 / Agent 6.12.0

***Added***:

* Adhere to code style ([#3493](https://github.com/DataDog/integrations-core/pull/3493))

## 3.2.1 / 2019-03-29 / Agent 6.11.0

***Fixed***:

* Include exception in connection error messages ([#3262](https://github.com/DataDog/integrations-core/pull/3262))

## 3.2.0 / 2019-02-18 / Agent 6.10.0

***Added***:

* Finish Python 3 Support ([#2911](https://github.com/DataDog/integrations-core/pull/2911))

## 3.1.0 / 2019-01-04 / Agent 6.9.0

***Added***:

* Support Python 3 ([#2721](https://github.com/DataDog/integrations-core/pull/2721))

## 3.0.0 / 2018-11-30 / Agent 6.8.0

***Removed***:

* Add CouchDB 2.2.0 compatibility by dropping the `purge_seq` metric ([#2287](https://github.com/DataDog/integrations-core/pull/2287)) Thanks [janl](https://github.com/janl).

## 2.6.1 / 2018-09-04 / Agent 6.5.0

***Fixed***:

* Make sure all checks' versions are exposed ([#1945](https://github.com/DataDog/integrations-core/pull/1945))
* Add data files to the wheel package ([#1727](https://github.com/DataDog/integrations-core/pull/1727))

## 2.6.0 / 2018-06-07

***Added***:

* Package `auto_conf.yaml` for appropriate integrations ([#1664](https://github.com/DataDog/integrations-core/pull/1664))
* Raise custom exceptions for specific errors instead of a generic `Exception`.

## 2.5.0 / 2018-05-11

***Added***:

* Hardcode the 5984 port in the Autodiscovery template ([#1444](https://github.com/DataDog/integrations-core/pull/1444) for more information)

## 2.4.0 / 2018-02-13

***Added***:

* reduces by db and by dd amplification by distributing the dbs to report on the running agents

## 2.3.0 / 2018-02-13

***Added***:

* Add custom tags to metrics and service checks ([#1034](https://github)com/DataDog/integrations-core/pull/1034)

***Fixed***:

* Handle the case where there is no database ([#1029](https://github)com/DataDog/integrations-core/pull/1029)

## 2.2.0 / 2018-01-10

***Added***:

* collects and submits CouchDB design docs metrics ([#813](https://github.com/DataDog/integrations-core/pull/813) (Thanks [@calonso](https://github)com/calonso))
* collects CouchDB active tasks stats ([#812](https://github.com/DataDog/integrations-core/pull/812) (Thanks [@calonso](https://github)com/calonso))

## 2.1.0 / 2017-11-21

***Added***:

* Update auto_conf template to support agent 6 and 5.20+ ([#860](https://github)com/DataDog/integrations-core/issues/860)
* collects Erlang VM stats from the `_system` endpoint ([#793](https://github.com/DataDog/integrations-core/issues/793) (Thanks [@calonso](https://github)com/calonso))

## 2.0.0 / 2017-09-01

***Added***:

* adds CouchDB 2.x integration.

## 1.0.1 / 2017-04-24

***Fixed***:

* Escape database names ([#268](https://github.com/DataDog/integrations-core/issues/268) (Thanks [@bernharduw](https://github)com/bernharduw))

## 1.0.0 / 2017-03-22

***Added***:

* adds couch integration.
