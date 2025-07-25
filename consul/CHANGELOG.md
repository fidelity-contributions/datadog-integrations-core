# CHANGELOG - consul

<!-- towncrier release notes start -->

## 5.0.0 / 2025-07-10

***Changed***:

* Bump datadog_checks_base to 37.16.0 ([#20711](https://github.com/DataDog/integrations-core/pull/20711))

***Added***:

* Add option to emit new `consul.check.up` metric ([#20598](https://github.com/DataDog/integrations-core/pull/20598))

## 4.1.1 / 2025-05-15 / Agent 7.67.0

***Fixed***:

* Replace deprecated `cert.not_valid_after` and `datetime.utcnow()` with `cert.not_valid_after_utc` and `datetime.now(timezone.utc)` respectively. ([#20100](https://github.com/DataDog/integrations-core/pull/20100))

## 4.1.0 / 2025-01-16 / Agent 7.63.0

***Added***:

* Add `tls_ciphers` param to integration ([#19334](https://github.com/DataDog/integrations-core/pull/19334))

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

## 2.6.1 / 2024-05-31 / Agent 7.55.0

***Fixed***:

* Update the description for the `tls_ca_cert` config option to use `openssl rehash` instead of `c_rehash` ([#16981](https://github.com/DataDog/integrations-core/pull/16981))

## 2.6.0 / 2024-03-22 / Agent 7.53.0

***Added***:

* Add configuration option to disable hosts from being created by `consul.net.node.*` metrics ([#17004](https://github.com/DataDog/integrations-core/pull/17004))

## 2.5.0 / 2024-02-16 / Agent 7.52.0

***Added***:

* Update the configuration file to include the new oauth options parameter ([#16835](https://github.com/DataDog/integrations-core/pull/16835))

## 2.4.0 / 2024-01-05 / Agent 7.51.0

***Added***:

* Bump the Python version from py3.9 to py3.11 ([#15997](https://github.com/DataDog/integrations-core/pull/15997))

## 2.3.1 / 2023-08-18 / Agent 7.48.0

***Fixed***:

* Update datadog-checks-base dependency version to 32.6.0 ([#15604](https://github.com/DataDog/integrations-core/pull/15604))

## 2.3.0 / 2023-08-10

***Added***:

* Update generated config models ([#15212](https://github.com/DataDog/integrations-core/pull/15212))

***Fixed***:

* Fix types for generated config models ([#15334](https://github.com/DataDog/integrations-core/pull/15334))

## 2.2.2 / 2023-07-10 / Agent 7.47.0

***Fixed***:

* Bump Python version from py3.8 to py3.9 ([#14701](https://github.com/DataDog/integrations-core/pull/14701))

## 2.2.1 / 2023-05-26 / Agent 7.46.0

***Fixed***:

* Add DEFAULT_METRIC_LIMIT for OpenMetrics-based checks ([#14527](https://github.com/DataDog/integrations-core/pull/14527))

## 2.2.0 / 2022-09-16 / Agent 7.40.0

***Added***:

* Update HTTP config spec templates ([#12890](https://github.com/DataDog/integrations-core/pull/12890))
* Add tag consul_node to consul service check consul.check from ConsulCheck.Node ([#12675](https://github.com/DataDog/integrations-core/pull/12675)) Thanks [hjkatz](https://github.com/hjkatz).

## 2.1.0 / 2022-04-05 / Agent 7.36.0

***Added***:

* Add metric_patterns options to filter all metric submission by a list of regexes ([#11695](https://github.com/DataDog/integrations-core/pull/11695))

***Fixed***:

* Remove outdated warning in the description for the `tls_ignore_warning` option ([#11591](https://github.com/DataDog/integrations-core/pull/11591))

## 2.0.0 / 2022-02-19 / Agent 7.35.0

***Changed***:

* Add tls_protocols_allowed option documentation ([#11251](https://github.com/DataDog/integrations-core/pull/11251))

***Added***:

* Add `pyproject.toml` file ([#11331](https://github.com/DataDog/integrations-core/pull/11331))

***Fixed***:

* Fix namespace packaging on Python 2 ([#11532](https://github.com/DataDog/integrations-core/pull/11532))

## 1.22.1 / 2022-01-08 / Agent 7.34.0

***Fixed***:

* Add comment to autogenerated model files ([#10945](https://github.com/DataDog/integrations-core/pull/10945))

## 1.22.0 / 2021-10-04 / Agent 7.32.0

***Added***:

* Add HTTP option to control the size of streaming responses ([#10183](https://github.com/DataDog/integrations-core/pull/10183))
* Add allow_redirect option ([#10160](https://github.com/DataDog/integrations-core/pull/10160))
* Disable generic tags ([#10027](https://github.com/DataDog/integrations-core/pull/10027))

***Fixed***:

* Bump base package dependency ([#10218](https://github.com/DataDog/integrations-core/pull/10218))
* Fix the description of the `allow_redirects` HTTP option ([#10195](https://github.com/DataDog/integrations-core/pull/10195))

## 1.21.0 / 2021-06-16 / Agent 7.30.0

***Added***:

* Improve performance of latency metric computation ([#9530](https://github.com/DataDog/integrations-core/pull/9530))
* Improve performance of loading JSON responses on Python 3 ([#9524](https://github.com/DataDog/integrations-core/pull/9524))
* Add missing prometheus metrics ([#9389](https://github.com/DataDog/integrations-core/pull/9389))

## 1.20.0 / 2021-05-25 / Agent 7.29.0

***Added***:

* Adding support for multiprocessing consul checks ([#9402](https://github.com/DataDog/integrations-core/pull/9402)) Thanks [lchayoun](https://github.com/lchayoun).

## 1.19.0 / 2021-04-19 / Agent 7.28.0

***Added***:

* Add runtime configuration validation ([#8899](https://github.com/DataDog/integrations-core/pull/8899))

***Fixed***:

* Rename service_whitelist to services_include ([#8802](https://github.com/DataDog/integrations-core/pull/8802))

## 1.18.0 / 2021-03-07 / Agent 7.27.0

***Added***:

* Adding services_exclude config option ([#8377](https://github.com/DataDog/integrations-core/pull/8377))

***Fixed***:

* Bump minimum base package version ([#8443](https://github.com/DataDog/integrations-core/pull/8443))

## 1.17.1 / 2020-12-11 / Agent 7.25.0

***Fixed***:

* Add consul 1.9.0 metrics ([#8095](https://github.com/DataDog/integrations-core/pull/8095))

## 1.17.0 / 2020-10-31 / Agent 7.24.0

***Added***:

* Add ability to dynamically get authentication information ([#7660](https://github.com/DataDog/integrations-core/pull/7660))
* [doc] Add encoding in log config sample ([#7708](https://github.com/DataDog/integrations-core/pull/7708))

***Fixed***:

* Add missing default HTTP headers: Accept, Accept-Encoding ([#7725](https://github.com/DataDog/integrations-core/pull/7725))

## 1.16.0 / 2020-09-21 / Agent 7.23.0

***Added***:

* Add RequestsWrapper option to support UTF-8 for basic auth ([#7441](https://github.com/DataDog/integrations-core/pull/7441))
* Support prometheus endpoint ([#7098](https://github.com/DataDog/integrations-core/pull/7098))

***Fixed***:

* Fix style for the latest release of Black ([#7438](https://github.com/DataDog/integrations-core/pull/7438))
* Update proxy section in conf.yaml ([#7336](https://github.com/DataDog/integrations-core/pull/7336))
* Use consistent formatting for boolean values ([#7405](https://github.com/DataDog/integrations-core/pull/7405))

## 1.15.1 / 2020-08-10 / Agent 7.22.0

***Fixed***:

* Update logs config service field to optional ([#7209](https://github.com/DataDog/integrations-core/pull/7209))
* DOCS-838 Template wording ([#7038](https://github.com/DataDog/integrations-core/pull/7038))
* Update ntlm_domain example ([#7118](https://github.com/DataDog/integrations-core/pull/7118))

## 1.15.0 / 2020-06-29 / Agent 7.21.0

***Added***:

* Add note about warning concurrency ([#6967](https://github.com/DataDog/integrations-core/pull/6967))

***Fixed***:

* Fix template specs typos ([#6912](https://github.com/DataDog/integrations-core/pull/6912))

## 1.14.0 / 2020-05-17 / Agent 7.20.0

***Added***:

* Allow optional dependency installation for all checks ([#6589](https://github.com/DataDog/integrations-core/pull/6589))
* Add config spec ([#6317](https://github.com/DataDog/integrations-core/pull/6317))

## 1.13.0 / 2020-04-04 / Agent 7.19.0

***Added***:

* Add option to set SNI hostname via the `Host` header for RequestsWrapper ([#5833](https://github.com/DataDog/integrations-core/pull/5833))
* Add new metric to count services ([#5992](https://github.com/DataDog/integrations-core/pull/5992))

***Fixed***:

* Remove logs sourcecategory ([#6121](https://github.com/DataDog/integrations-core/pull/6121))

## 1.12.2 / 2020-02-25 / Agent 7.18.0

***Fixed***:

* Change new added tag ([#5856](https://github.com/DataDog/integrations-core/pull/5856))

## 1.12.1 / 2020-02-25

***Fixed***:

* Bump minimun agent version ([#5834](https://github.com/DataDog/integrations-core/pull/5834))

## 1.12.0 / 2020-02-22

***Deprecated***:

* Deprecate `service` tag ([#5540](https://github.com/DataDog/integrations-core/pull/5540))

***Added***:

* Create `consul_service` tag ([#5519](https://github.com/DataDog/integrations-core/pull/5519)) Thanks [nicbono](https://github.com/nicbono).

## 1.11.0 / 2019-12-02 / Agent 7.16.0

***Added***:

* Add version metadata ([#4944](https://github.com/DataDog/integrations-core/pull/4944))
* Standardize logging format ([#4903](https://github.com/DataDog/integrations-core/pull/4903))
* Add auth type to RequestsWrapper ([#4708](https://github.com/DataDog/integrations-core/pull/4708))

## 1.10.0 / 2019-10-11 / Agent 6.15.0

***Added***:

* Add option to override KRB5CCNAME env var ([#4578](https://github.com/DataDog/integrations-core/pull/4578))

## 1.9.1 / 2019-08-30 / Agent 6.14.0

***Fixed***:

* Fix RequestsWrapper options ([#4476](https://github.com/DataDog/integrations-core/pull/4476))

## 1.9.0 / 2019-08-24

***Added***:

* Add support for proxy options ([#3363](https://github.com/DataDog/integrations-core/pull/3363))

***Fixed***:

* Fix Consul event timestamp ([#4173](https://github.com/DataDog/integrations-core/pull/4173))

## 1.8.0 / 2019-05-14 / Agent 6.12.0

***Added***:

* Adhere to code style ([#3491](https://github.com/DataDog/integrations-core/pull/3491))

## 1.7.0 / 2019-02-18 / Agent 6.10.0

***Added***:

* Add `consul.can_connect` service check for every HTTP request to consul ([#3003](https://github.com/DataDog/integrations-core/pull/3003))
* Finish supporting Py3 ([#2906](https://github.com/DataDog/integrations-core/pull/2906))

## 1.6.0 / 2018-11-30 / Agent 6.8.0

***Added***:

* Add option to run the full check on any node ([#2461](https://github.com/DataDog/integrations-core/pull/2461))
* Support Python 3 ([#2446](https://github.com/DataDog/integrations-core/pull/2446))

## 1.5.2 / 2018-10-12 / Agent 6.6.0

***Fixed***:

* Update consul timestamp to use supported python functions ([#2199](https://github.com/DataDog/integrations-core/pull/2199)) Thanks [hhansell](https://github.com/hhansell).

## 1.5.1 / 2018-09-04 / Agent 6.5.0

***Fixed***:

* Accept more standard boolean values for instance config options ([#1954](https://github.com/DataDog/integrations-core/pull/1954))
* Add data files to the wheel package ([#1727](https://github.com/DataDog/integrations-core/pull/1727))

## 1.5.0 / 2018-06-07

***Added***:

* Package `auto_conf.yaml` for appropriate integrations ([#1664](https://github.com/DataDog/integrations-core/pull/1664))
* Include consul_datacenter tag in service checks ([#1526](https://github.com/DataDog/integrations-core/pull/1526)) Thanks [TylerLubeck](https://github.com/TylerLubeck).
* Add a check to count all nodes in a consul cluster ([#1479](https://github.com/DataDog/integrations-core/pull/1479)) Thanks [TylerLubeck](https://github.com/TylerLubeck).

## 1.4.0 / 2018-05-11

***Added***:

* Hardcode the 8500 port in the Autodiscovery template ([#1444](https://github.com/DataDog/integrations-core/pull/1444) for more information)
* Include consul_datacenter tag in service checks

## 1.3.0 / 2018-01-10

***Added***:

* Add support for Consul 1.0 ([#876](https://github.com/DataDog/integrations-core/pull/876), thanks [@byronwolfman](https://github)com/byronwolfman)

***Fixed***:

* Fixes TypeError if/when services are culled ([#968](https://github)com/DataDog/integrations-core/pull/968)

## 1.2.0 / 2017-11-21

***Added***:

* Add service tags to metrics
* Update auto_conf template to support agent 6 and 5.20+ ([#860](https://github)com/DataDog/integrations-core/issues/860)

## 1.1.0 / 2017-07-18

***Added***:

* Support ACL token for authentication ([#521](https://github)com/DataDog/integrations-core/issues/521)

***Fixed***:

* Fixed duplicate service check with same tags but different status being sent (one per Node) ([#460](https://github)com/DataDog/integrations-core/issues/460)

## 1.0.0 / 2017-03-22

***Added***:

* adds Consul integration.
