# CHANGELOG - teamcity

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

## 4.3.1 / 2024-08-09 / Agent 7.57.0

***Fixed***:

* Fix handling of projects with no builds. We used to refresh all the projects whenever we encountered a build config that didn't have any builds associated with it. Now we refresh only the specific build config that's lacking builds. ([#18041](https://github.com/DataDog/integrations-core/pull/18041))
* Fix handling of deleted build config ([#18122](https://github.com/DataDog/integrations-core/pull/18122))

## 4.3.0 / 2024-07-05 / Agent 7.56.0

***Added***:

* Add support for TeamCity 2023.05.4 ([#17695](https://github.com/DataDog/integrations-core/pull/17695))

***Fixed***:

* Update config model names ([#17802](https://github.com/DataDog/integrations-core/pull/17802))

## 4.2.1 / 2024-05-23

***Fixed***:

* Update the description for the `tls_ca_cert` config option to use `openssl rehash` instead of `c_rehash` ([#16981](https://github.com/DataDog/integrations-core/pull/16981))
* Check for auth_token key and skip adding guestAuth or httpAuth to url if present ([#17478](https://github.com/DataDog/integrations-core/pull/17478))

## 4.2.0 / 2024-02-16 / Agent 7.52.0

***Added***:

* Update the configuration file to include the new oauth options parameter ([#16835](https://github.com/DataDog/integrations-core/pull/16835))

## 4.1.0 / 2024-01-05 / Agent 7.51.0

***Added***:

* Bump the Python version from py3.9 to py3.11 ([#15997](https://github.com/DataDog/integrations-core/pull/15997))

## 4.0.1 / 2023-09-29 / Agent 7.49.0

***Fixed***:

* Override the default test options for some integrations ([#15779](https://github.com/DataDog/integrations-core/pull/15779))

## 4.0.0 / 2023-08-10 / Agent 7.48.0

***Changed***:

* Bump the minimum base check version ([#15427](https://github.com/DataDog/integrations-core/pull/15427))

***Added***:

* Update generated config models ([#15212](https://github.com/DataDog/integrations-core/pull/15212))

***Fixed***:

* Fix types for generated config models ([#15334](https://github.com/DataDog/integrations-core/pull/15334))

## 3.1.1 / 2023-07-10 / Agent 7.47.0

***Fixed***:

* Allow all projects to be collected in REST implementation ([#14433](https://github.com/DataDog/integrations-core/pull/14433)) Thanks [njrs92](https://github.com/njrs92).
* Bump Python version from py3.8 to py3.9 ([#14701](https://github.com/DataDog/integrations-core/pull/14701))

## 3.1.0 / 2023-05-26 / Agent 7.46.0

***Added***:

* Add an ignore_connection_errors option to the openmetrics check ([#14504](https://github.com/DataDog/integrations-core/pull/14504))

***Fixed***:

* Update minimum datadog base package version ([#14463](https://github.com/DataDog/integrations-core/pull/14463))
* Deprecate `use_latest_spec` option ([#14446](https://github.com/DataDog/integrations-core/pull/14446))

## 3.0.1 / 2022-12-16 / Agent 7.42.0

***Fixed***:

* Fix event tagging ([#13537](https://github.com/DataDog/integrations-core/pull/13537))

## 3.0.0 / 2022-12-06

***Changed***:

* Support TeamCity metrics and service checks ([#12852](https://github.com/DataDog/integrations-core/pull/12852))

## 2.2.0 / 2022-09-16 / Agent 7.40.0

***Added***:

* Update HTTP config spec templates ([#12890](https://github.com/DataDog/integrations-core/pull/12890))

## 2.1.0 / 2022-04-05 / Agent 7.36.0

***Added***:

* Add metric_patterns options to filter all metric submission by a list of regexes ([#11695](https://github.com/DataDog/integrations-core/pull/11695))

***Fixed***:

* Remove outdated warning in the description for the `tls_ignore_warning` option ([#11591](https://github.com/DataDog/integrations-core/pull/11591))

## 2.0.0 / 2022-02-19 / Agent 7.35.0

***Changed***:

* Add tls_protocols_allowed option documentation ([#11251](https://github.com/DataDog/integrations-core/pull/11251))

***Added***:

* Add `pyproject.toml` file ([#11445](https://github.com/DataDog/integrations-core/pull/11445))

***Fixed***:

* Fix namespace packaging on Python 2 ([#11532](https://github.com/DataDog/integrations-core/pull/11532))

## 1.14.1 / 2022-01-08 / Agent 7.34.0

***Fixed***:

* Add comment to autogenerated model files ([#10945](https://github.com/DataDog/integrations-core/pull/10945))

## 1.14.0 / 2021-10-04 / Agent 7.32.0

***Added***:

* Add HTTP option to control the size of streaming responses ([#10183](https://github.com/DataDog/integrations-core/pull/10183))
* Add allow_redirect option ([#10160](https://github.com/DataDog/integrations-core/pull/10160))
* Disable generic tags ([#10027](https://github.com/DataDog/integrations-core/pull/10027))

***Fixed***:

* Bump base package dependency ([#10218](https://github.com/DataDog/integrations-core/pull/10218))
* Fix the description of the `allow_redirects` HTTP option ([#10195](https://github.com/DataDog/integrations-core/pull/10195))

## 1.13.0 / 2021-05-28 / Agent 7.29.0

***Added***:

* Add runtime configuration validation ([#8995](https://github.com/DataDog/integrations-core/pull/8995))

## 1.12.1 / 2021-03-07 / Agent 7.27.0

***Fixed***:

* Bump minimum base package version ([#8443](https://github.com/DataDog/integrations-core/pull/8443))

## 1.12.0 / 2021-01-25 / Agent 7.26.0

***Added***:

* Document logs support ([#7702](https://github.com/DataDog/integrations-core/pull/7702))

## 1.11.0 / 2020-10-31 / Agent 7.24.0

***Added***:

* Add ability to dynamically get authentication information ([#7660](https://github.com/DataDog/integrations-core/pull/7660))

## 1.10.0 / 2020-09-21 / Agent 7.23.0

***Added***:

* Add config spec ([#7530](https://github.com/DataDog/integrations-core/pull/7530))

## 1.9.0 / 2020-06-29 / Agent 7.21.0

***Added***:

* Add note about warning concurrency ([#6967](https://github.com/DataDog/integrations-core/pull/6967))

## 1.8.0 / 2020-05-17 / Agent 7.20.0

***Added***:

* Allow optional dependency installation for all checks ([#6589](https://github.com/DataDog/integrations-core/pull/6589))

## 1.7.1 / 2020-04-04 / Agent 7.19.0

***Fixed***:

* Update deprecated imports ([#6088](https://github.com/DataDog/integrations-core/pull/6088))

## 1.7.0 / 2020-01-13 / Agent 7.17.0

***Added***:

* Use lazy logging format ([#5398](https://github.com/DataDog/integrations-core/pull/5398))
* Use lazy logging format ([#5377](https://github.com/DataDog/integrations-core/pull/5377))

## 1.6.0 / 2019-10-11 / Agent 6.15.0

***Added***:

* Add option to override KRB5CCNAME env var ([#4578](https://github.com/DataDog/integrations-core/pull/4578))

## 1.5.1 / 2019-08-31 / Agent 6.14.0

***Fixed***:

* Fix RequestsWrapper usage ([#4486](https://github.com/DataDog/integrations-core/pull/4486))

## 1.5.0 / 2019-08-24

***Added***:

* Add requests wrapper to teamcity ([#4209](https://github.com/DataDog/integrations-core/pull/4209))

***Fixed***:

* Update __init__ method params ([#4243](https://github.com/DataDog/integrations-core/pull/4243))
* Fix wording for config option description ([#4217](https://github.com/DataDog/integrations-core/pull/4217))

## 1.4.0 / 2019-05-14 / Agent 6.12.0

***Added***:

* Adhere to code style ([#3575](https://github.com/DataDog/integrations-core/pull/3575))

## 1.3.0 / 2019-02-18 / Agent 6.10.0

***Added***:

* Support unicode for Python 3 bindings ([#2869](https://github.com/DataDog/integrations-core/pull/2869))

***Fixed***:

* Resolve flake8 issues ([#3060](https://github.com/DataDog/integrations-core/pull/3060))

## 1.2.0 / 2019-01-04 / Agent 6.9.0

***Added***:

* Support Python 3 ([#2829][1])

## 1.1.1 / 2018-09-04 / Agent 6.5.0

***Fixed***:

* Make sure all checks' versions are exposed ([#1945][2])
* Add data files to the wheel package ([#1727][3])

## 1.1.0 / 2018-06-07

***Added***:

* Allow users to specify scheme in server URLs ([#1649][4])

## 1.0.0 / 2017-03-22

***Added***:

* adds teamcity integration.

[1]: https://github.com/DataDog/integrations-core/pull/2829
[2]: https://github.com/DataDog/integrations-core/pull/1945
[3]: https://github.com/DataDog/integrations-core/pull/1727
[4]: https://github.com/DataDog/integrations-core/pull/1649
