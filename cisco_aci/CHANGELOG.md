# CHANGELOG - cisco_aci

<!-- towncrier release notes start -->

## 4.8.0 / 2025-07-10

***Added***:

* Update dependencies ([#20561](https://github.com/DataDog/integrations-core/pull/20561))

## 4.7.0 / 2025-06-12 / Agent 7.68.0

***Added***:

* Update dependencies ([#20399](https://github.com/DataDog/integrations-core/pull/20399))

***Fixed***:

* skip topology link resolution for cisco-aci when remote port index is None ([#20418](https://github.com/DataDog/integrations-core/pull/20418))

## 4.6.0 / 2025-05-15 / Agent 7.67.0

***Added***:

* Update dependencies ([#20215](https://github.com/DataDog/integrations-core/pull/20215))

## 4.5.0 / 2025-04-17 / Agent 7.66.0

***Added***:

* Update dependencies ([#19962](https://github.com/DataDog/integrations-core/pull/19962))

## 4.4.0 / 2025-03-19 / Agent 7.65.0

***Added***:

* Update dependencies ([#19687](https://github.com/DataDog/integrations-core/pull/19687))
* Add failover support with Agent High Availability feature. ([#19776](https://github.com/DataDog/integrations-core/pull/19776))
* Collect Cisco ACI faults as logs. ([#19836](https://github.com/DataDog/integrations-core/pull/19836))

## 4.3.0 / 2025-01-16 / Agent 7.63.0

***Added***:

* Add `tls_ciphers` param to integration ([#19334](https://github.com/DataDog/integrations-core/pull/19334))

## 4.2.0 / 2024-12-26 / Agent 7.62.0

***Added***:

* [NDM] [Cisco ACI] Support submitting topology metadata (utilizing LLDP neighbor information) ([#18675](https://github.com/DataDog/integrations-core/pull/18675))

***Fixed***:

* [NDM] [Cisco ACI] Fix APIC device status ([#19204](https://github.com/DataDog/integrations-core/pull/19204))

## 4.1.0 / 2024-10-31 / Agent 7.60.0

***Added***:

* [NDM] [Cisco ACI] Utilize raw ID for interface metadata ([#18842](https://github.com/DataDog/integrations-core/pull/18842))

***Fixed***:

* [NDM] [Cisco ACI] Improve integration performance ([#18747](https://github.com/DataDog/integrations-core/pull/18747))

## 4.0.0 / 2024-10-04 / Agent 7.59.0

***Removed***:

* Remove support for Python 2. ([#18580](https://github.com/DataDog/integrations-core/pull/18580))
* [NDM] [Cisco ACI] Add check metrics ([#18748](https://github.com/DataDog/integrations-core/pull/18748))

***Fixed***:

* Bump the version of datadog-checks-base to 37.0.0 ([#18617](https://github.com/DataDog/integrations-core/pull/18617))

## 3.0.0 / 2024-10-01 / Agent 7.58.0

***Changed***:

* Bump minimum version of base check ([#18733](https://github.com/DataDog/integrations-core/pull/18733))

***Security***:

* Bump version of cryptography to 43.0.1 to address vulnerability ([#18656](https://github.com/DataDog/integrations-core/pull/18656))

***Added***:

* Bump the python version from 3.11 to 3.12 ([#18212](https://github.com/DataDog/integrations-core/pull/18212))

## 2.12.0 / 2024-09-06

***Added***:

* Add the ability to tag Cisco ACI device and interface metrics with user-defined tags. ([#18496](https://github.com/DataDog/integrations-core/pull/18496))

## 2.11.0 / 2024-09-05

***Added***:

* Update dependencies ([#18478](https://github.com/DataDog/integrations-core/pull/18478))

***Fixed***:

* [NDM] [Cisco ACI] Use name instead of node ID as device hostname ([#18375](https://github.com/DataDog/integrations-core/pull/18375))

## 2.10.2 / 2024-09-02 / Agent 7.57.0

***Fixed***:

* [NDM] [Cisco ACI] Use actual int for interface index ([#18414](https://github.com/DataDog/integrations-core/pull/18414))

## 2.10.1 / 2024-08-20

***Fixed***:

* [NDM] [Cisco ACI] Refactor batched payloads to fix incorrect status + use interface ID if name not available ([#18360](https://github.com/DataDog/integrations-core/pull/18360))

## 2.10.0 / 2024-08-09

***Added***:

* [NDM] Add NDM metadata support for Cisco ACI ([#17735](https://github.com/DataDog/integrations-core/pull/17735))
* [NDM] [Cisco ACI] Add common NDM tags to metrics ([#18017](https://github.com/DataDog/integrations-core/pull/18017))
* [NDM] [Cisco ACI] Add config flag for enabling sending metadata to NDM ([#18099](https://github.com/DataDog/integrations-core/pull/18099))
* Update dependencies ([#18187](https://github.com/DataDog/integrations-core/pull/18187))

## 2.9.0 / 2024-07-05 / Agent 7.56.0

***Added***:

* Update dependencies ([#17817](https://github.com/DataDog/integrations-core/pull/17817))

## 2.8.0 / 2024-05-31 / Agent 7.55.0

***Added***:

* Update dependencies ([#17519](https://github.com/DataDog/integrations-core/pull/17519))

***Fixed***:

* Update the description for the `tls_ca_cert` config option to use `openssl rehash` instead of `c_rehash` ([#16981](https://github.com/DataDog/integrations-core/pull/16981))

## 2.7.0 / 2024-03-07 / Agent 7.52.0

***Security***:

* Bump cryptography to 42.0.5 ([#17054](https://github.com/DataDog/integrations-core/pull/17054))

## 2.6.0 / 2024-02-16

***Added***:

* Update the configuration file to include the new oauth options parameter ([#16835](https://github.com/DataDog/integrations-core/pull/16835))

## 2.5.0 / 2024-01-05 / Agent 7.51.0

***Added***:

* Bump the Python version from py3.9 to py3.11 ([#15997](https://github.com/DataDog/integrations-core/pull/15997))
* Update dependencies ([#16448](https://github.com/DataDog/integrations-core/pull/16448))

## 2.4.2 / 2023-12-04 / Agent 7.50.0

***Fixed***:

* Bump the cryptography version to 41.0.6 ([#16322](https://github.com/DataDog/integrations-core/pull/16322))

## 2.4.1 / 2023-10-26 / Agent 7.49.0

***Fixed***:

* Bump the `cryptography` version to 41.0.5 ([#16083](https://github.com/DataDog/integrations-core/pull/16083))

## 2.4.0 / 2023-09-29

***Added***:

* Update Cryptography to 41.0.4 ([#15922](https://github.com/DataDog/integrations-core/pull/15922))

## 2.3.1 / 2023-08-18 / Agent 7.48.0

***Fixed***:

* Bump cryptography to 41.0.3 ([#15517](https://github.com/DataDog/integrations-core/pull/15517))
* Update datadog-checks-base dependency version to 32.6.0 ([#15604](https://github.com/DataDog/integrations-core/pull/15604))

## 2.3.0 / 2023-08-10

***Added***:

* Update generated config models ([#15212](https://github.com/DataDog/integrations-core/pull/15212))

***Fixed***:

* Fix types for generated config models ([#15334](https://github.com/DataDog/integrations-core/pull/15334))

## 2.2.2 / 2023-07-10 / Agent 7.47.0

***Fixed***:

* Bump Python version from py3.8 to py3.9 ([#14701](https://github.com/DataDog/integrations-core/pull/14701))

## 2.2.1 / 2022-12-09 / Agent 7.42.0

***Fixed***:

* Update cryptography dependency ([#13367](https://github.com/DataDog/integrations-core/pull/13367))
* Remove `default_backend` parameter from cryptography calls ([#13333](https://github.com/DataDog/integrations-core/pull/13333))

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

* Add `pyproject.toml` file ([#11325](https://github.com/DataDog/integrations-core/pull/11325))

***Fixed***:

* Fix namespace packaging on Python 2 ([#11532](https://github.com/DataDog/integrations-core/pull/11532))

## 1.17.2 / 2022-01-08 / Agent 7.34.0

***Fixed***:

* Add comment to autogenerated model files ([#10945](https://github.com/DataDog/integrations-core/pull/10945))

## 1.17.1 / 2021-11-16 / Agent 7.33.0

***Fixed***:

* Fix required config option ([#10629](https://github.com/DataDog/integrations-core/pull/10629))

## 1.17.0 / 2021-11-13

***Added***:

* Add runtime configuration validation ([#8894](https://github.com/DataDog/integrations-core/pull/8894))

***Fixed***:

* Harmonize check and config options ([#10488](https://github.com/DataDog/integrations-core/pull/10488))

## 1.16.0 / 2021-10-04 / Agent 7.32.0

***Added***:

* Update dependencies ([#10228](https://github.com/DataDog/integrations-core/pull/10228))
* Add HTTP option to control the size of streaming responses ([#10183](https://github.com/DataDog/integrations-core/pull/10183))
* Add allow_redirect option ([#10160](https://github.com/DataDog/integrations-core/pull/10160))

***Fixed***:

* Fix the description of the `allow_redirects` HTTP option ([#10195](https://github.com/DataDog/integrations-core/pull/10195))

## 1.15.1 / 2021-08-22 / Agent 7.31.0

***Fixed***:

* Fix typos in log lines ([#9907](https://github.com/DataDog/integrations-core/pull/9907))

## 1.15.0 / 2021-04-19 / Agent 7.28.0

***Added***:

* Upgrade cryptography to 3.4.6 on Python 3 ([#8764](https://github.com/DataDog/integrations-core/pull/8764))

## 1.14.0 / 2021-03-07 / Agent 7.27.0

***Security***:

* Upgrade cryptography python package ([#8611](https://github.com/DataDog/integrations-core/pull/8611))

***Fixed***:

* Rename config spec example consumer option `default` to `display_default` ([#8593](https://github.com/DataDog/integrations-core/pull/8593))
* Bump minimum base package version ([#8443](https://github.com/DataDog/integrations-core/pull/8443))

## 1.13.0 / 2021-01-28 / Agent 7.26.0

***Security***:

* Upgrade cryptography python package ([#8476](https://github.com/DataDog/integrations-core/pull/8476))

## 1.12.0 / 2020-10-31 / Agent 7.24.0

***Security***:

* Upgrade `cryptography` dependency ([#7869](https://github.com/DataDog/integrations-core/pull/7869))

***Added***:

* Add ability to dynamically get authentication information ([#7660](https://github.com/DataDog/integrations-core/pull/7660))

## 1.11.0 / 2020-09-21 / Agent 7.23.0

***Added***:

* Add RequestsWrapper option to support UTF-8 for basic auth ([#7441](https://github.com/DataDog/integrations-core/pull/7441))

***Fixed***:

* Fix style for the latest release of Black ([#7438](https://github.com/DataDog/integrations-core/pull/7438))
* Update proxy section in conf.yaml ([#7336](https://github.com/DataDog/integrations-core/pull/7336))

## 1.10.1 / 2020-08-10 / Agent 7.22.0

***Fixed***:

* DOCS-838 Template wording ([#7038](https://github.com/DataDog/integrations-core/pull/7038))

## 1.10.0 / 2020-06-29 / Agent 7.21.0

***Added***:

* Add note about warning concurrency ([#6967](https://github.com/DataDog/integrations-core/pull/6967))

## 1.9.0 / 2020-05-17 / Agent 7.20.0

***Added***:

* Allow optional dependency installation for all checks ([#6589](https://github.com/DataDog/integrations-core/pull/6589))
* Add config spec ([#6314](https://github.com/DataDog/integrations-core/pull/6314))

## 1.8.4 / 2020-04-04 / Agent 7.19.0

***Fixed***:

* Update deprecated imports ([#6088](https://github.com/DataDog/integrations-core/pull/6088))

## 1.8.3 / 2020-02-22 / Agent 7.18.0

***Fixed***:

* Update request wrapper with password and A6 signature ([#5684](https://github.com/DataDog/integrations-core/pull/5684))

## 1.8.2 / 2019-12-27 / Agent 7.17.0

***Fixed***:

* Ensure only one session object per url ([#5334](https://github.com/DataDog/integrations-core/pull/5334))

## 1.8.1 / 2019-12-02 / Agent 7.16.0

***Fixed***:

* Use RequestsWrapper ([#5037](https://github.com/DataDog/integrations-core/pull/5037))

## 1.8.0 / 2019-11-20

***Added***:

* Upgrade cryptography to 2.8 ([#5047](https://github.com/DataDog/integrations-core/pull/5047))
* Standardize logging format ([#4902](https://github.com/DataDog/integrations-core/pull/4902))

***Fixed***:

* Refresh auth token when it expires ([#5039](https://github.com/DataDog/integrations-core/pull/5039))

## 1.7.2 / 2019-08-24 / Agent 6.14.0

***Fixed***:

* Use utcnow instead of now ([#4192](https://github.com/DataDog/integrations-core/pull/4192))

## 1.7.1 / 2019-07-08 / Agent 6.13.0

***Fixed***:

* Fix event submission call ([#4044](https://github.com/DataDog/integrations-core/pull/4044))

## 1.7.0 / 2019-07-04

***Added***:

* Update cryptography version ([#4000](https://github.com/DataDog/integrations-core/pull/4000))

## 1.6.0 / 2019-06-01 / Agent 6.12.0

***Added***:

* Improve API logs ([#3794](https://github.com/DataDog/integrations-core/pull/3794))

***Fixed***:

* Sanitize external host tags ([#3792](https://github.com/DataDog/integrations-core/pull/3792))

## 1.5.0 / 2019-05-14

***Added***:

* Adhere to code style ([#3489](https://github.com/DataDog/integrations-core/pull/3489))

## 1.4.0 / 2019-02-18 / Agent 6.10.0

***Added***:

* Support Python 3 ([#3029](https://github.com/DataDog/integrations-core/pull/3029))

***Fixed***:

* Resolve flake8 issues ([#3060](https://github.com/DataDog/integrations-core/pull/3060))

## 1.3.0 / 2018-11-30 / Agent 6.8.0

***Added***:

* Upgrade cryptography ([#2659][1])

***Fixed***:

* Use raw string literals when \ is present ([#2465][2])

## 1.2.1 / 2018-10-12 / Agent 6.6.0

***Fixed***:

* fixes cisco for username and password ([#2267][3])

## 1.2.0 / 2018-09-04 / Agent 6.5.0

***Added***:

* Use Certs in the Cisco Check as well as Passwords ([#1986][4])

***Fixed***:

* Add data files to the wheel package ([#1727][5])

## 1.1.0 / 2018-06-21 / Agent 6.4.0

***Fixed***:

* Makes the Cisco Check more resilient ([#1785][6])

## 1.0.0 / 2018-06-07

***Added***:

* adds CiscoACI integration.

[1]: https://github.com/DataDog/integrations-core/pull/2659
[2]: https://github.com/DataDog/integrations-core/pull/2465
[3]: https://github.com/DataDog/integrations-core/pull/2267
[4]: https://github.com/DataDog/integrations-core/pull/1986
[5]: https://github.com/DataDog/integrations-core/pull/1727
[6]: https://github.com/DataDog/integrations-core/pull/1785
