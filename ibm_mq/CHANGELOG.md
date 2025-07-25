# CHANGELOG - IBM MQ

<!-- towncrier release notes start -->

## 8.3.0 / 2025-07-10

***Added***:

* Add ibm_mq.channel.conn_status and ibm_mq.channel.connections_active metrics with channel and connection metric tests ([#20519](https://github.com/DataDog/integrations-core/pull/20519))

***Fixed***:

* Remove relative imports for non parent modules ([#20646](https://github.com/DataDog/integrations-core/pull/20646))

## 8.2.0 / 2025-04-17 / Agent 7.66.0

***Added***:

* Allow timezone config option to be used in metric collection ([#19912](https://github.com/DataDog/integrations-core/pull/19912))

## 8.1.0 / 2025-01-25 / Agent 7.63.0

***Added***:

* Update dependencies ([#19430](https://github.com/DataDog/integrations-core/pull/19430))

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
* Upgrade psutil to 6.0.0 to fix performance issues addressed ([#18688](https://github.com/DataDog/integrations-core/pull/18688))

## 6.4.0 / 2024-09-05

***Added***:

* Bump `psutil`  version to 5.9.6 ([#18491](https://github.com/DataDog/integrations-core/pull/18491))

## 6.3.0 / 2024-03-08 / Agent 7.52.0

***Added***:

* Revert "Bump psutil version to 5.9.7 (#16547)" ([#17112](https://github.com/DataDog/integrations-core/pull/17112))

## 6.2.0 / 2024-02-16

***Added***:

* Bump `psutil` version to 5.9.7 ([#16547](https://github.com/DataDog/integrations-core/pull/16547))

## 6.1.1 / 2024-01-10 / Agent 7.51.0

***Fixed***:

* Properly drop support for Python 2 ([#16589](https://github.com/DataDog/integrations-core/pull/16589))

## 6.1.0 / 2024-01-05

***Added***:

* Bump the Python version from py3.9 to py3.11 ([#15997](https://github.com/DataDog/integrations-core/pull/15997))

## 6.0.0 / 2023-09-29 / Agent 7.49.0

***Changed***:

* Drop Python 2 support ([#15786](https://github.com/DataDog/integrations-core/pull/15786))

## 5.0.0 / 2023-08-10 / Agent 7.48.0

***Changed***:

* Bump the minimum base check version ([#15427](https://github.com/DataDog/integrations-core/pull/15427))

***Added***:

* Update generated config models ([#15212](https://github.com/DataDog/integrations-core/pull/15212))

***Fixed***:

* Fix types for generated config models ([#15334](https://github.com/DataDog/integrations-core/pull/15334))

## 4.2.0 / 2023-07-10 / Agent 7.47.0

***Added***:

* Bump dependencies for Agent 7.47 ([#15145](https://github.com/DataDog/integrations-core/pull/15145))

***Fixed***:

* Bump Python version from py3.8 to py3.9 ([#14701](https://github.com/DataDog/integrations-core/pull/14701))

## 4.1.0 / 2023-05-26 / Agent 7.46.0

***Added***:

* Tag queue metrics by usage ([#14606](https://github.com/DataDog/integrations-core/pull/14606))

## 4.0.3 / 2023-01-25 / Agent 7.43.0

***Fixed***:

* Allow setting the try basic auth value from config in all cases ([#13781](https://github.com/DataDog/integrations-core/pull/13781))

## 4.0.2 / 2023-01-20

***Fixed***:

* Prevent unnecessary non-SSL connection attempts ([#13559](https://github.com/DataDog/integrations-core/pull/13559))

## 4.0.1 / 2022-12-09 / Agent 7.42.0

***Fixed***:

* Add messaging to service checks ([#13355](https://github.com/DataDog/integrations-core/pull/13355))

## 4.0.0 / 2022-10-28 / Agent 7.41.0

***Changed***:

* Don't set any default algorithm for `ssl_cipher_spec` ([#13013](https://github.com/DataDog/integrations-core/pull/13013))

***Added***:

* Add `queue_manager_process` option ([#13107](https://github.com/DataDog/integrations-core/pull/13107))

***Fixed***:

* Add back channels_to_skip for channel status metrics ([#13113](https://github.com/DataDog/integrations-core/pull/13113))
* Update SSL connection configs used for determining SSL auth ([#12974](https://github.com/DataDog/integrations-core/pull/12974))

## 3.22.1 / 2022-08-05 / Agent 7.39.0

***Fixed***:

* Dependency updates ([#12653](https://github.com/DataDog/integrations-core/pull/12653))

## 3.22.0 / 2022-05-15 / Agent 7.37.0

***Added***:

* Add `collect_reset_queue_metrics` option ([#11818](https://github.com/DataDog/integrations-core/pull/11818))

## 3.21.0 / 2022-04-05 / Agent 7.36.0

***Added***:

* Add option to not try normal connection ([#11748](https://github.com/DataDog/integrations-core/pull/11748))
* Add metric_patterns options to filter all metric submission by a list of regexes ([#11695](https://github.com/DataDog/integrations-core/pull/11695))

***Fixed***:

* Report critical queue manager service check when connection fails ([#11737](https://github.com/DataDog/integrations-core/pull/11737))

## 3.20.0 / 2022-03-16

***Added***:

* Add `auto_discover_channels` option ([#11678](https://github.com/DataDog/integrations-core/pull/11678))

***Fixed***:

* Ensure PCFExecute disconnects ([#11677](https://github.com/DataDog/integrations-core/pull/11677))
* Improve debug logs ([#11637](https://github.com/DataDog/integrations-core/pull/11637))

## 3.19.1 / 2022-03-01 / Agent 7.35.0

***Fixed***:

* Fix unique list items and min mapping properties config validations  ([#11574](https://github.com/DataDog/integrations-core/pull/11574))

## 3.19.0 / 2022-02-19

***Added***:

* Add `pyproject.toml` file ([#11370](https://github.com/DataDog/integrations-core/pull/11370))

***Fixed***:

* Fix namespace packaging on Python 2 ([#11532](https://github.com/DataDog/integrations-core/pull/11532))

## 3.18.0 / 2022-02-01

***Added***:

* Support host override ([#11223](https://github.com/DataDog/integrations-core/pull/11223))

***Fixed***:

* Do not allow for empty values in configuration ([#11138](https://github.com/DataDog/integrations-core/pull/11138))
* Improve log message ([#11125](https://github.com/DataDog/integrations-core/pull/11125))

## 3.17.0 / 2022-01-08 / Agent 7.34.0

***Added***:

* Add Windows support to IBM MQ ([#10737](https://github.com/DataDog/integrations-core/pull/10737))

***Fixed***:

* Add comment to autogenerated model files ([#10945](https://github.com/DataDog/integrations-core/pull/10945))

## 3.16.1 / 2021-10-04 / Agent 7.32.0

***Fixed***:

* ibm mq queue pattern should have precedence over autodiscover ([#10247](https://github.com/DataDog/integrations-core/pull/10247))

## 3.16.0 / 2021-09-27

***Added***:

* Add support for queue status and last put/get time metrics ([#10129](https://github.com/DataDog/integrations-core/pull/10129))

## 3.15.0 / 2021-09-20

***Added***:

* Disable generic tags ([#10027](https://github.com/DataDog/integrations-core/pull/10027))

***Fixed***:

* Fix mypy tests ([#10134](https://github.com/DataDog/integrations-core/pull/10134))
* Stop emitting incorrect queue warning ([#10017](https://github.com/DataDog/integrations-core/pull/10017))
* Add debug lines about discovered queues ([#9969](https://github.com/DataDog/integrations-core/pull/9969))
* Add try-catch on queue closure ([#9955](https://github.com/DataDog/integrations-core/pull/9955))

## 3.14.1 / 2021-08-22 / Agent 7.31.0

***Fixed***:

* Do not store previously discovered queues ([#9821](https://github.com/DataDog/integrations-core/pull/9821))
* Fix typos in log lines ([#9907](https://github.com/DataDog/integrations-core/pull/9907))

## 3.14.0 / 2021-08-13

***Added***:

* Add `timeout` option ([#9896](https://github.com/DataDog/integrations-core/pull/9896))

***Fixed***:

* Use dedicated instance logger for connection messages ([#9887](https://github.com/DataDog/integrations-core/pull/9887))
* Do not submit critical service check when there are no messages ([#9703](https://github.com/DataDog/integrations-core/pull/9703))

## 3.13.3 / 2021-07-15 / Agent 7.30.0

***Fixed***:

* Add debug line when there are no messages available ([#9702](https://github.com/DataDog/integrations-core/pull/9702))

## 3.13.2 / 2021-06-09

***Fixed***:

* Properly close internal reply queues ([#9488](https://github.com/DataDog/integrations-core/pull/9488))

## 3.13.1 / 2021-06-01 / Agent 7.29.0

***Fixed***:

* Don't emit any warnings if NO_MSG_AVAILABLE is received ([#9452](https://github.com/DataDog/integrations-core/pull/9452))

## 3.13.0 / 2021-05-25

***Added***:

* Add runtime configuration validation ([#8935](https://github.com/DataDog/integrations-core/pull/8935))

***Fixed***:

* Try SSL connection when host not found ([#9404](https://github.com/DataDog/integrations-core/pull/9404))
* Don't emit warnings if there are no messages ([#9400](https://github.com/DataDog/integrations-core/pull/9400))

## 3.12.0 / 2021-03-07 / Agent 7.27.0

***Added***:

* Add flag to convert endianness ([#8601](https://github.com/DataDog/integrations-core/pull/8601))

***Fixed***:

* Use SSL authentication if SSL params are provided ([#8531](https://github.com/DataDog/integrations-core/pull/8531))
* Bump minimum base package version ([#8443](https://github.com/DataDog/integrations-core/pull/8443))

## 3.11.1 / 2021-01-25 / Agent 7.26.0

***Fixed***:

* Better explain ssl_key_repository_location ([#8417](https://github.com/DataDog/integrations-core/pull/8417))

## 3.11.0 / 2020-11-19 / Agent 7.25.0

***Added***:

* Add new queue stats metrics for IBM MQ ([#8032](https://github.com/DataDog/integrations-core/pull/8032))

## 3.10.0 / 2020-10-31 / Agent 7.24.0

***Added***:

* [doc] Add encoding in log config sample ([#7708](https://github.com/DataDog/integrations-core/pull/7708))

## 3.9.0 / 2020-09-21 / Agent 7.23.0

***Added***:

* Support Certificate Label and login/password for SSL conf ([#7202](https://github.com/DataDog/integrations-core/pull/7202))

***Fixed***:

* Raise exception on connection error ([#7563](https://github.com/DataDog/integrations-core/pull/7563))

## 3.8.2 / 2020-09-10

***Fixed***:

* Try normal connection before SSL connection ([#7554](https://github.com/DataDog/integrations-core/pull/7554))
* Improve error reporting when pymqi is not installed ([#7048](https://github.com/DataDog/integrations-core/pull/7048))
* Fix style for the latest release of Black ([#7438](https://github.com/DataDog/integrations-core/pull/7438))

## 3.8.1 / 2020-08-10 / Agent 7.22.0

***Fixed***:

* Update logs config service field to optional ([#7209](https://github.com/DataDog/integrations-core/pull/7209))

## 3.8.0 / 2020-07-23

***Added***:

* IBM MQ metadata ([#6979](https://github.com/DataDog/integrations-core/pull/6979))
* Collect metrics from Statistics Messages ([#6945](https://github.com/DataDog/integrations-core/pull/6945))

***Fixed***:

* Avoid shadowing depth_percent function ([#7132](https://github.com/DataDog/integrations-core/pull/7132))

## 3.7.0 / 2020-06-29 / Agent 7.21.0

***Added***:

* Add MacOS Support ([#6927](https://github.com/DataDog/integrations-core/pull/6927))

***Fixed***:

* Refactor to make encoding more consistent ([#6995](https://github.com/DataDog/integrations-core/pull/6995))
* Ensure bytes for ssl connection ([#6913](https://github.com/DataDog/integrations-core/pull/6913))
* Fix template specs typos ([#6912](https://github.com/DataDog/integrations-core/pull/6912))
* Move metrics collection logic to separate files ([#6752](https://github.com/DataDog/integrations-core/pull/6752))

## 3.6.0 / 2020-05-17 / Agent 7.20.0

***Added***:

* Allow optional dependency installation for all checks ([#6589](https://github.com/DataDog/integrations-core/pull/6589))

## 3.5.1 / 2020-04-08 / Agent 7.19.0

***Fixed***:

* Don't import pymqi unconditionally ([#6286](https://github.com/DataDog/integrations-core/pull/6286))

## 3.5.0 / 2020-04-04

***Added***:

* Apply config specs to IBM MQ ([#5903](https://github.com/DataDog/integrations-core/pull/5903))

***Fixed***:

* Update deprecated imports ([#6088](https://github.com/DataDog/integrations-core/pull/6088))
* Remove logs sourcecategory ([#6121](https://github.com/DataDog/integrations-core/pull/6121))

## 3.4.0 / 2020-03-11

***Added***:

* Add `connection_name` configuration ([#6015](https://github.com/DataDog/integrations-core/pull/6015))
* Add configuration option for the Channel Definition API version ([#5905](https://github.com/DataDog/integrations-core/pull/5905))
* Upgrade pymqi to 1.10.1 ([#5955](https://github.com/DataDog/integrations-core/pull/5955))

***Fixed***:

* IBM MQ refactor ([#5902](https://github.com/DataDog/integrations-core/pull/5902))

## 3.3.1 / 2020-01-17 / Agent 7.17.0

***Fixed***:

* Fix metric type and missing metrics in metadata.csv ([#5470](https://github.com/DataDog/integrations-core/pull/5470))

## 3.3.0 / 2020-01-13

***Added***:

* Use lazy logging format ([#5398](https://github.com/DataDog/integrations-core/pull/5398))
* Use lazy logging format ([#5377](https://github.com/DataDog/integrations-core/pull/5377))
* Add channel metrics ([#5116](https://github.com/DataDog/integrations-core/pull/5116))

## 3.2.1 / 2019-09-18 / Agent 6.15.0

***Fixed***:

* Improve IBM MQ docs and logging ([#4540](https://github.com/DataDog/integrations-core/pull/4540))
* Fix duplicate service checks ([#4525](https://github.com/DataDog/integrations-core/pull/4525))

## 3.2.0 / 2019-08-21 / Agent 6.14.0

***Added***:

* Add channel_status_mapping config ([#4395](https://github.com/DataDog/integrations-core/pull/4395))

## 3.1.1 / 2019-07-29

***Fixed***:

* Fix ibm_mq e2e import issue ([#4140](https://github.com/DataDog/integrations-core/pull/4140))

## 3.1.0 / 2019-07-04 / Agent 6.13.0

***Added***:

* Add ibm_mq.channel.count metric and ibm_mq.channel.status service check ([#3958](https://github.com/DataDog/integrations-core/pull/3958))

***Fixed***:

* Use MQCMD_INQUIRE_Q instead of queue.inquire ([#3997](https://github.com/DataDog/integrations-core/pull/3997))

## 3.0.0 / 2019-06-20

***Changed***:

* [ibm_mq] fix queue auto discovery to include any type in addition to qmodel and included regex matching on queue names ([#3893](https://github.com/DataDog/integrations-core/pull/3893))

## 2.0.0 / 2019-04-16 / Agent 6.12.0

***Changed***:

* Breaking change: Change host tag for mq_host. Dashboards and monitors may be affected ([#3608](https://github.com/DataDog/integrations-core/pull/3608))

***Added***:

* Adhere to code style ([#3519](https://github.com/DataDog/integrations-core/pull/3519))

***Fixed***:

* fix queue_manager variable naming of IBM MQ ([#3592](https://github.com/DataDog/integrations-core/pull/3592))

## 1.2.0 / 2019-03-29 / Agent 6.11.0

***Added***:

* Add ability to add additional tags to queues matching a regex ([#3399](https://github.com/DataDog/integrations-core/pull/3399))
* adds channel metrics ([#3360](https://github.com/DataDog/integrations-core/pull/3360))

***Fixed***:

* fix ssl variable naming for IBM MQ ([#3312](https://github.com/DataDog/integrations-core/pull/3312))

## 1.1.0 / 2019-02-18 / Agent 6.10.0

***Added***:

* Autodiscover queues ([#3061](https://github.com/DataDog/integrations-core/pull/3061))

## 1.0.1 / 2019-01-04 / Agent 6.9.0

***Fixed***:

* Fix Oldest Message Age ([#2859][1])

## 1.0.0 / 2018-12-09 / Agent 6.8.0

***Added***:

* IBM MQ Integration ([#2154][2])

[1]: https://github.com/DataDog/integrations-core/pull/2859
[2]: https://github.com/DataDog/integrations-core/pull/2154
