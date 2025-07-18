# CHANGELOG - Kube_apiserver_metrics

<!-- towncrier release notes start -->

## 7.0.0 / 2025-07-10

***Changed***:

* Bump datadog_checks_base to 37.16.0 ([#20711](https://github.com/DataDog/integrations-core/pull/20711))

## 6.2.0 / 2025-01-25 / Agent 7.63.0

***Added***:

* Add process_cpu_total to kube_apiserver_metrics check ([#19415](https://github.com/DataDog/integrations-core/pull/19415))

## 6.1.0 / 2025-01-16

***Added***:

* Add `tls_ciphers` param to integration ([#19334](https://github.com/DataDog/integrations-core/pull/19334))

## 6.0.0 / 2024-10-04 / Agent 7.59.0

***Removed***:

* Remove support for Python 2. ([#18580](https://github.com/DataDog/integrations-core/pull/18580))

***Added***:

* Add the apiserver_admission_webhook_request_total metric ([#17690](https://github.com/DataDog/integrations-core/pull/17690))
* Bump the python version from 3.11 to 3.12 ([#18207](https://github.com/DataDog/integrations-core/pull/18207))

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

## 4.3.0 / 2024-02-16 / Agent 7.52.0

***Added***:

* Add support for kube_apiserver SLI metrics ([#16657](https://github.com/DataDog/integrations-core/pull/16657))
* Update the configuration file to include the new oauth options parameter ([#16835](https://github.com/DataDog/integrations-core/pull/16835))

## 4.2.0 / 2024-01-05 / Agent 7.51.0

***Added***:

* Bump the Python version from py3.9 to py3.11 ([#15997](https://github.com/DataDog/integrations-core/pull/15997))

## 4.1.0 / 2023-10-26 / Agent 7.50.0

***Added***:

* Add etcd.db.total_size for Kubernetes 1.26, 1.27, and 1.28 ([#15940](https://github.com/DataDog/integrations-core/pull/15940))
* Add apiserver_flowcontrol_current_inqueue_requests and apiserver_flowcontrol_dispatched_requests_total for Kubernetes >= 1.23 ([#15981](https://github.com/DataDog/integrations-core/pull/15981))
* Add etcd_requests_total and etcd_request_errors_total for Kubernetes >= 1.28 ([#15986](https://github.com/DataDog/integrations-core/pull/15986))

## 4.0.0 / 2023-08-10 / Agent 7.48.0

***Changed***:

* Bump the minimum base check version ([#15427](https://github.com/DataDog/integrations-core/pull/15427))

***Added***:

* Update generated config models ([#15212](https://github.com/DataDog/integrations-core/pull/15212))

***Fixed***:

* Fix types for generated config models ([#15334](https://github.com/DataDog/integrations-core/pull/15334))

## 3.6.2 / 2023-07-10 / Agent 7.47.0

***Fixed***:

* Deprecate apiserver_longrunning_gauge metric in favor of apiserver_longrunning_requests ([#14856](https://github.com/DataDog/integrations-core/pull/14856))
* [kube_apiserver_metrics] Rename aggregator_unavailable_apiservice `name` tag to `apiservice_name` ([#14738](https://github.com/DataDog/integrations-core/pull/14738))
* Bump Python version from py3.8 to py3.9 ([#14701](https://github.com/DataDog/integrations-core/pull/14701))

## 3.6.1 / 2023-06-14 / Agent 7.46.0

***Fixed***:

* [kube_apiserver_metrics] Rename aggregator_unavailable_apiservice `name` tag to `apiservice_name` (#14738) ([#14751](https://github.com/DataDog/integrations-core/pull/14751))

## 3.6.0 / 2023-05-26

***Added***:

* Add `flowcontrol` metrics ([#14480](https://github.com/DataDog/integrations-core/pull/14480))
* [kube-apiserver] Add new `aggregator_unavailable_apiservice` metric ([#14457](https://github.com/DataDog/integrations-core/pull/14457))

## 3.5.0 / 2023-04-14 / Agent 7.45.0

***Added***:

* Adds new metric `kubernetes_feature_enabled` ([#14147](https://github.com/DataDog/integrations-core/pull/14147))

## 3.4.0 / 2023-03-03 / Agent 7.44.0

***Added***:

* Add metric `apiserver_admission_webhook_fail_open_count` ([#13750](https://github.com/DataDog/integrations-core/pull/13750))

## 3.3.0 / 2022-09-16 / Agent 7.40.0

***Added***:

* Add new kube_apiserver_metric for deprecated API usage ([#12887](https://github.com/DataDog/integrations-core/pull/12887)) Thanks [alex-berger](https://github.com/alex-berger).
* Update HTTP config spec templates ([#12890](https://github.com/DataDog/integrations-core/pull/12890))

## 3.2.0 / 2022-05-15 / Agent 7.37.0

***Added***:

* Support dynamic bearer tokens (Bound Service Account Token Volume) ([#11915](https://github.com/DataDog/integrations-core/pull/11915))

## 3.1.0 / 2022-04-05 / Agent 7.36.0

***Added***:

* Add metric_patterns options to filter all metric submission by a list of regexes ([#11695](https://github.com/DataDog/integrations-core/pull/11695))

***Fixed***:

* Remove outdated warning in the description for the `tls_ignore_warning` option ([#11591](https://github.com/DataDog/integrations-core/pull/11591))

## 3.0.0 / 2022-02-19 / Agent 7.35.0

***Changed***:

* Add tls_protocols_allowed option documentation ([#11251](https://github.com/DataDog/integrations-core/pull/11251))

***Added***:

* Add `pyproject.toml` file ([#11380](https://github.com/DataDog/integrations-core/pull/11380))

***Fixed***:

* Fix namespace packaging on Python 2 ([#11532](https://github.com/DataDog/integrations-core/pull/11532))

## 2.0.2 / 2022-01-21 / Agent 7.34.0

***Fixed***:

* Fix license header dates in autogenerated files ([#11187](https://github.com/DataDog/integrations-core/pull/11187))

## 2.0.1 / 2022-01-18

***Fixed***:

* Fix the type of `bearer_token_auth` ([#11144](https://github.com/DataDog/integrations-core/pull/11144))

## 2.0.0 / 2022-01-08

***Changed***:

* Update the default value of the `bearer_token` parameter to send the bearer token only to secure https endpoints by default ([#10772](https://github.com/DataDog/integrations-core/pull/10772))

***Added***:

* Add new storage metrics and test for Kubernetes 1.23 ([#10906](https://github.com/DataDog/integrations-core/pull/10906))

***Fixed***:

* Add comment to autogenerated model files ([#10945](https://github.com/DataDog/integrations-core/pull/10945))

## 1.12.0 / 2021-11-13 / Agent 7.33.0

***Added***:

* Document new include_labels option ([#10617](https://github.com/DataDog/integrations-core/pull/10617))
* Document new use_process_start_time option ([#10601](https://github.com/DataDog/integrations-core/pull/10601))
* Add kube_apiserver.etcd.db.total_size metric ([#10569](https://github.com/DataDog/integrations-core/pull/10569))

## 1.11.0 / 2021-10-04 / Agent 7.32.0

***Added***:

* Add runtime configuration validation ([#8944](https://github.com/DataDog/integrations-core/pull/8944))
* Add HTTP option to control the size of streaming responses ([#10183](https://github.com/DataDog/integrations-core/pull/10183))
* Add allow_redirect option ([#10160](https://github.com/DataDog/integrations-core/pull/10160))

***Fixed***:

* Fix the description of the `allow_redirects` HTTP option ([#10195](https://github.com/DataDog/integrations-core/pull/10195))

## 1.10.0 / 2021-07-12 / Agent 7.30.0

***Added***:

* Fix auto-discovery for latest versions on Kubernetes ([#9577](https://github.com/DataDog/integrations-core/pull/9577))

## 1.9.0 / 2021-05-28 / Agent 7.29.0

***Added***:

* Support "ignore_tags" configuration ([#9392](https://github.com/DataDog/integrations-core/pull/9392))

## 1.8.0 / 2021-03-07 / Agent 7.27.0

***Added***:

* Add new metrics ([#8557](https://github.com/DataDog/integrations-core/pull/8557))

## 1.7.1 / 2021-01-25 / Agent 7.26.0

***Fixed***:

* Update metrics whose name has changed in Kubernetes 1.14 ([#8337](https://github.com/DataDog/integrations-core/pull/8337))
* Update prometheus_metrics_prefix documentation ([#8236](https://github.com/DataDog/integrations-core/pull/8236))

## 1.7.0 / 2020-10-31 / Agent 7.24.0

***Added***:

* Sync openmetrics config specs with new option ignore_metrics_by_labels ([#7823](https://github.com/DataDog/integrations-core/pull/7823))
* Add ability to dynamically get authentication information ([#7660](https://github.com/DataDog/integrations-core/pull/7660))

## 1.6.0 / 2020-09-21 / Agent 7.23.0

***Added***:

* Add RequestsWrapper option to support UTF-8 for basic auth ([#7441](https://github.com/DataDog/integrations-core/pull/7441))

***Fixed***:

* Fix style for the latest release of Black ([#7438](https://github.com/DataDog/integrations-core/pull/7438))
* Update proxy section in conf.yaml ([#7336](https://github.com/DataDog/integrations-core/pull/7336))
* Use consistent formatting for boolean values ([#7405](https://github.com/DataDog/integrations-core/pull/7405))

## 1.5.0 / 2020-08-10 / Agent 7.22.0

***Added***:

* Support "*" wildcard in type_overrides configuration ([#7071](https://github.com/DataDog/integrations-core/pull/7071))

***Fixed***:

* DOCS-838 Template wording ([#7038](https://github.com/DataDog/integrations-core/pull/7038))
* Update ntlm_domain example ([#7118](https://github.com/DataDog/integrations-core/pull/7118))

## 1.4.1 / 2020-07-02 / Agent 7.21.0

***Fixed***:

* Fix default value in example configuration file ([#7034](https://github.com/DataDog/integrations-core/pull/7034))

## 1.4.0 / 2020-06-29

***Added***:

* Add note about warning concurrency ([#6967](https://github.com/DataDog/integrations-core/pull/6967))
* kube apiserver signature and specs ([#6831](https://github.com/DataDog/integrations-core/pull/6831))

***Fixed***:

* Sync example configs ([#6920](https://github.com/DataDog/integrations-core/pull/6920))

## 1.3.0 / 2020-05-17 / Agent 7.20.0

***Added***:

* Allow optional dependency installation for all checks ([#6589](https://github.com/DataDog/integrations-core/pull/6589))
* Add admission controller metrics ([#6502](https://github.com/DataDog/integrations-core/pull/6502))

## 1.2.3 / 2020-04-04 / Agent 7.19.0

***Fixed***:

* Update deprecated imports ([#6088](https://github.com/DataDog/integrations-core/pull/6088))

## 1.2.2 / 2020-01-13 / Agent 7.17.0

***Fixed***:

* Update Kube_apiserver_metrics annotations documentation ([#5199](https://github.com/DataDog/integrations-core/pull/5199))

## 1.2.1 / 2019-12-13 / Agent 7.16.0

***Fixed***:

* Fix scrapper config cache issue ([#5202](https://github.com/DataDog/integrations-core/pull/5202))

## 1.2.0 / 2019-12-02

***Added***:

* Handle scheme in `prometheus_url` instead of the separate `scheme` option, which is now deprecated ([#4913](https://github.com/DataDog/integrations-core/pull/4913))

## 1.1.1 / 2019-10-16 / Agent 6.15.0

***Fixed***:

* Use default port for kube apiserver metrics auto conf ([#4785](https://github.com/DataDog/integrations-core/pull/4785))

## 1.1.0 / 2019-10-11

***Added***:

* Scrape apiserver_request_total metric introduced in v1.15 ([#4546](https://github.com/DataDog/integrations-core/pull/4546))

## 1.0.1 / 2019-06-06 / Agent 6.12.0

***Fixed***:

* Fix default for bearer_token and ssl_verify ([#3882](https://github.com/DataDog/integrations-core/pull/3882))

## 1.0.0 / 2019-05-31

***Added***:

* Introducing the Kubernetes APIServer metrics check ([#3746](https://github.com/DataDog/integrations-core/pull/3746))
