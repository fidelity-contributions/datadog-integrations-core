# CHANGELOG - network

<!-- towncrier release notes start -->

## 5.3.0 / 2025-07-10

***Added***:

* Add support for mlx5_core drivers in ethtool integration [CMPT-3460] ([#20481](https://github.com/DataDog/integrations-core/pull/20481))

## 5.2.0 / 2025-03-19 / Agent 7.65.0

***Added***:

* Add new system.net.iface.up metric ([#19853](https://github.com/DataDog/integrations-core/pull/19853))

## 5.1.0 / 2024-10-31 / Agent 7.60.0

***Added***:

* Add mtu and speed tags to interfaces that support it ([#18819](https://github.com/DataDog/integrations-core/pull/18819))

## 5.0.0 / 2024-10-04 / Agent 7.59.0

***Removed***:

* Remove support for Python 2. ([#18580](https://github.com/DataDog/integrations-core/pull/18580))

***Fixed***:

* Bump the version of datadog-checks-base to 37.0.0 ([#18617](https://github.com/DataDog/integrations-core/pull/18617))

## 4.1.0 / 2024-10-04 / Agent 7.58.0

***Fixed***:

* Fix metric type in new network metrics ([#18764](https://github.com/DataDog/integrations-core/pull/18764))

## 4.0.0 / 2024-10-01

***Changed***:

* Bump minimum version of base check ([#18733](https://github.com/DataDog/integrations-core/pull/18733))

***Added***:

* Bump the python version from 3.11 to 3.12 ([#18212](https://github.com/DataDog/integrations-core/pull/18212))
* Upgrade psutil to 6.0.0 to fix performance issues addressed ([#18688](https://github.com/DataDog/integrations-core/pull/18688))

## 3.4.0 / 2024-09-05

***Added***:

* Add new TCP metrics for Network integration on Windows ([#18294](https://github.com/DataDog/integrations-core/pull/18294))
* Bump `psutil`  version to 5.9.6 ([#18491](https://github.com/DataDog/integrations-core/pull/18491))

## 3.3.0 / 2024-03-08 / Agent 7.52.0

***Added***:

* Revert "Bump psutil version to 5.9.7 (#16547)" ([#17112](https://github.com/DataDog/integrations-core/pull/17112))

## 3.2.0 / 2024-02-16

***Added***:

* Bump `psutil` version to 5.9.7 ([#16547](https://github.com/DataDog/integrations-core/pull/16547))
* Update dependencies ([#16788](https://github.com/DataDog/integrations-core/pull/16788))

## 3.1.0 / 2024-01-05 / Agent 7.51.0

***Added***:

* Bump the Python version from py3.9 to py3.11 ([#15997](https://github.com/DataDog/integrations-core/pull/15997))

## 3.0.0 / 2023-08-10 / Agent 7.48.0

***Changed***:

* Bump the minimum base check version ([#15427](https://github.com/DataDog/integrations-core/pull/15427))

***Added***:

* Update generated config models ([#15212](https://github.com/DataDog/integrations-core/pull/15212))

***Fixed***:

* Improve debug logs for ena and ethtool network metrics ([#15447](https://github.com/DataDog/integrations-core/pull/15447))
* Fix types for generated config models ([#15334](https://github.com/DataDog/integrations-core/pull/15334))

## 2.10.0 / 2023-07-10 / Agent 7.47.0

***Added***:

* Add metric conntrack_allowance_available from Amazon ENA driver ([#14634](https://github.com/DataDog/integrations-core/pull/14634)) Thanks [radykal-com](https://github.com/radykal-com).

***Fixed***:

* Bump Python version from py3.8 to py3.9 ([#14701](https://github.com/DataDog/integrations-core/pull/14701))

## 2.9.4 / 2023-04-14 / Agent 7.45.0

***Fixed***:

* Remove `distutils` dependency from python 3 version of the check ([#14245](https://github.com/DataDog/integrations-core/pull/14245))

## 2.9.3 / 2022-12-09 / Agent 7.42.0

***Fixed***:

* Fix network check for solaris ([#13319](https://github.com/DataDog/integrations-core/pull/13319))

## 2.9.2 / 2022-10-28 / Agent 7.41.0

***Fixed***:

* Extract windows check to separate class ([#13143](https://github.com/DataDog/integrations-core/pull/13143))
* Test & better document collect_cx_queues ([#13117](https://github.com/DataDog/integrations-core/pull/13117))
* Exclude loopback interface from ethtool ([#13042](https://github.com/DataDog/integrations-core/pull/13042))

## 2.9.1 / 2022-10-04 / Agent 7.40.0

***Fixed***:

* Exclude loopback interface from ethtool ([#13042](https://github.com/DataDog/integrations-core/pull/13042))

## 2.9.0 / 2022-09-16

***Added***:

* Add collection of ethtool queue stats ([#12023](https://github.com/DataDog/integrations-core/pull/12023))

## 2.8.0 / 2022-08-05 / Agent 7.39.0

***Added***:

* Add collection of ethtool queue stats ([#11056](https://github.com/DataDog/integrations-core/pull/11056))

***Fixed***:

* Dependency updates ([#12653](https://github.com/DataDog/integrations-core/pull/12653))
* Revert "Add collection of ethtool queue stats (#11056)" ([#12017](https://github.com/DataDog/integrations-core/pull/12017))

## 2.7.0 / 2022-04-05 / Agent 7.36.0

***Added***:

* Add IP and TCP metric collection ([#11170](https://github.com/DataDog/integrations-core/pull/11170))
* Add metric_patterns options to filter all metric submission by a list of regexes ([#11695](https://github.com/DataDog/integrations-core/pull/11695))

***Fixed***:

* Support newer versions of `click` ([#11746](https://github.com/DataDog/integrations-core/pull/11746))

## 2.6.0 / 2022-02-19 / Agent 7.35.0

***Added***:

* Collect additional iface metrics: mtu, num tx/rx queue and tx queue length ([#11156](https://github.com/DataDog/integrations-core/pull/11156))
* Add `pyproject.toml` file ([#11402](https://github.com/DataDog/integrations-core/pull/11402))
* Upgrade psutil to 5.9.0 ([#11139](https://github.com/DataDog/integrations-core/pull/11139))

***Fixed***:

* Fix namespace packaging on Python 2 ([#11532](https://github.com/DataDog/integrations-core/pull/11532))

## 2.5.0 / 2022-01-08 / Agent 7.34.0

***Added***:

* Add saturation metrics for network ([#10551](https://github.com/DataDog/integrations-core/pull/10551)) Thanks [luhenry](https://github.com/luhenry).
* Collect additional TcpExt metrics ([#10844](https://github.com/DataDog/integrations-core/pull/10844))

***Fixed***:

* Add comment to autogenerated model files ([#10945](https://github.com/DataDog/integrations-core/pull/10945))

## 2.4.0 / 2021-10-04 / Agent 7.32.0

***Added***:

* Disable generic tags ([#10027](https://github.com/DataDog/integrations-core/pull/10027))

## 2.3.0 / 2021-08-22 / Agent 7.31.0

***Added***:

* Use `display_default` as a fallback for `default` when validating config models ([#9739](https://github.com/DataDog/integrations-core/pull/9739))

## 2.2.0 / 2021-07-12 / Agent 7.30.0

***Added***:

* Add runtime configuration validation ([#8960](https://github.com/DataDog/integrations-core/pull/8960))

## 2.1.2 / 2021-03-07 / Agent 7.27.0

***Fixed***:

* Rename config spec example consumer option `default` to `display_default` ([#8593](https://github.com/DataDog/integrations-core/pull/8593))

## 2.1.1 / 2021-01-26 / Agent 7.26.0

***Fixed***:

* Ensure network check doesn't fail on importing fcntl on Windows ([#8459](https://github.com/DataDog/integrations-core/pull/8459))

## 2.1.0 / 2021-01-25

***Added***:

* Collect AWS ENA metrics ([#8331](https://github.com/DataDog/integrations-core/pull/8331))

***Fixed***:

* Correct default template usage ([#8233](https://github.com/DataDog/integrations-core/pull/8233))

## 2.0.0 / 2020-12-11 / Agent 7.25.0

***Changed***:

* [network] Set the collect_connection_queues parameter default value to false ([#8059](https://github.com/DataDog/integrations-core/pull/8059))

## 1.19.0 / 2020-10-31 / Agent 7.24.0

***Added***:

* Collect receive and send queue metrics ([#7861](https://github.com/DataDog/integrations-core/pull/7861))
* Collect connection state metrics on BSD/OSX ([#7659](https://github.com/DataDog/integrations-core/pull/7659))

***Fixed***:

* Fix network metric collection failure on CentOS ([#7883](https://github.com/DataDog/integrations-core/pull/7883))

## 1.18.1 / 2020-09-28 / Agent 7.23.0

***Fixed***:

* Fix procfs_path retrieval in network check ([#7652](https://github.com/DataDog/integrations-core/pull/7652))

## 1.18.0 / 2020-09-21

***Added***:

* Pass `PROC_ROOT` as environment variable to `ss` ([#7095](https://github.com/DataDog/integrations-core/pull/7095))
* Upgrade psutil to 5.7.2 ([#7395](https://github.com/DataDog/integrations-core/pull/7395))

## 1.17.0 / 2020-06-29 / Agent 7.21.0

***Added***:

* Add network spec ([#6889](https://github.com/DataDog/integrations-core/pull/6889))

## 1.16.0 / 2020-05-17 / Agent 7.20.0

***Added***:

* Allow optional dependency installation for all checks ([#6589](https://github.com/DataDog/integrations-core/pull/6589))

## 1.15.1 / 2020-04-08 / Agent 7.19.0

***Fixed***:

* Fix error message ([#6285](https://github.com/DataDog/integrations-core/pull/6285))

## 1.15.0 / 2020-04-04

***Added***:

* Upgrade psutil to 5.7.0 ([#6243](https://github.com/DataDog/integrations-core/pull/6243))

***Fixed***:

* Handle invalid type for excluded_interfaces ([#5986](https://github.com/DataDog/integrations-core/pull/5986))

## 1.14.0 / 2020-01-13 / Agent 7.17.0

***Added***:

* Use lazy logging format ([#5377](https://github.com/DataDog/integrations-core/pull/5377))

## 1.13.0 / 2020-01-02

***Added***:

* Gracefully handle /proc errors in network check ([#5245](https://github.com/DataDog/integrations-core/pull/5245))

## 1.12.2 / 2019-12-13 / Agent 7.16.0

***Fixed***:

* Bump psutil to 5.6.7 ([#5210](https://github.com/DataDog/integrations-core/pull/5210))

## 1.12.1 / 2019-12-02

***Fixed***:

* Upgrade psutil dependency to 5.6.5 ([#5059](https://github.com/DataDog/integrations-core/pull/5059))

## 1.12.0 / 2019-10-30

***Added***:

* Add use_sudo option for collecting conntrack metrics with containers ([#4920](https://github.com/DataDog/integrations-core/pull/4920))

***Fixed***:

* Fix examples in conf.yaml.default ([#4887](https://github.com/DataDog/integrations-core/pull/4887)) Thanks [q42jaap](https://github.com/q42jaap).

## 1.11.5 / 2019-10-11 / Agent 6.15.0

***Fixed***:

* Upgrade psutil dependency to 5.6.3 ([#4442](https://github.com/DataDog/integrations-core/pull/4442))

## 1.11.4 / 2019-08-30 / Agent 6.14.0

***Fixed***:

* Fix metric submission for combined connection state ([#4473](https://github.com/DataDog/integrations-core/pull/4473))

## 1.11.3 / 2019-08-14

***Fixed***:

* Drastically reduce `ss` output ([#4346](https://github.com/DataDog/integrations-core/pull/4346))

## 1.11.1 / 2019-08-02

***Fixed***:

* Fix proc location for conntrack files ([#4150](https://github.com/DataDog/integrations-core/pull/4150))

## 1.11.0 / 2019-05-14 / Agent 6.12.0

***Added***:

* Upgrade psutil dependency to 5.6.2 ([#3684](https://github.com/DataDog/integrations-core/pull/3684))
* Add conntrack metrics ([#3624](https://github.com/DataDog/integrations-core/pull/3624))
* Adhere to code style ([#3543](https://github.com/DataDog/integrations-core/pull/3543))

## 1.10.0 / 2019-03-29 / Agent 6.11.0

***Added***:

* Strip white space when reading from proc_conntrack_max_path ([#3365](https://github.com/DataDog/integrations-core/pull/3365))

## 1.9.0 / 2019-02-18 / Agent 6.10.0

***Added***:

* Add conntrack basic metrics to the network integration. ([#2981](https://github.com/DataDog/integrations-core/pull/2981)) Thanks [aerostitch](https://github.com/aerostitch).
* Upgrade psutil ([#3019](https://github.com/DataDog/integrations-core/pull/3019))
* Support Python 3 ([#3005](https://github.com/DataDog/integrations-core/pull/3005))

***Fixed***:

* Resolve flake8 issues ([#3060](https://github.com/DataDog/integrations-core/pull/3060))
* Use `device` tag instead of the deprecated `device_name` parameter ([#2945](https://github.com/DataDog/integrations-core/pull/2945)) Thanks [aerostitch](https://github.com/aerostitch).

## 1.8.0 / 2018-11-30 / Agent 6.8.0

***Added***:

* Update psutil ([#2576](https://github.com/DataDog/integrations-core/pull/2576))

***Fixed***:

* Use raw string literals when \ is present ([#2465](https://github.com/DataDog/integrations-core/pull/2465))

## 1.7.0 / 2018-10-12 / Agent 6.6.0

***Added***:

* Upgrade psutil ([#2190](https://github.com/DataDog/integrations-core/pull/2190))

## 1.6.1 / 2018-09-04 / Agent 6.5.0

***Fixed***:

* Retrieve no_proxy directly from the Datadog Agent's configuration ([#2004](https://github.com/DataDog/integrations-core/pull/2004))
* Add data files to the wheel package ([#1727](https://github.com/DataDog/integrations-core/pull/1727))

## 1.6.0 / 2018-06-07

***Added***:

* Add monotonic counts for some metrics ([#1551](https://github.com/DataDog/integrations-core/pull/1551)) Thanks [jalaziz](https://github.com/jalaziz).

## 1.5.0 / 2018-03-23

***Added***:

* Add custom tag support.

## 1.4.0 / 2018-02-13

***Added***:

* Get some host network stats when the agent is running inside a container and not in the host network namespace ([#994](https://github)com/DataDog/integrations-core/pull/994)

## 1.3.0 / 2017-09-01

***Added***:

* Collects TCPRetransFail metric from /proc/net/netstat, See [#697](https://github.com/DataDog/integrations-core/pull/697)

## 1.2.2 / 2017-08-28

***Fixed***:

* Fix incorrect `log.error` call in BSD check ([#698](https://github)com/DataDog/integrations-core/issues/698)

## 1.2.1 / 2017-07-18

***Fixed***:

* Fix TCP6 metrics overriding TCP4 metrics when monitoring non combines socket states ([#501](https://github)com/DataDog/integrations-core/issues/501)

## 1.2.0 / 2017-06-05

***Added***:

* Adds metrics from `/proc/net/netstat` in addition to the existing ones from `/proc/net/snmp` ([#299](https://github.com/DataDog/integrations-core/issues/299) and [#452](https://github.com/DataDog/integrations-core/issues/452), thanks [@cory-stripe](https://github)com/cory-stripe)

## 1.1.0 / 2017-05-03

***Fixed***:

* Work around `ss -atun` bug not differentiating tcp and udp ([#296](https://github)com/DataDog/integrations-core/issues/296)

## 1.0.0 / 2017-03-22

***Added***:

* adds network integration.
