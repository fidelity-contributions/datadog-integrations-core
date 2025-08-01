# CHANGELOG - redisdb

<!-- towncrier release notes start -->

## 8.0.0 / 2025-06-12 / Agent 7.68.0

***Changed***:

* Bump `redis` dependency to `6.0.0` and stop supporting `charset`, `connection_pool`, and `errors` configuration options. ([#20265](https://github.com/DataDog/integrations-core/pull/20265))

***Added***:

* Update dependencies ([#20399](https://github.com/DataDog/integrations-core/pull/20399))

## 7.3.0 / 2025-05-15 / Agent 7.67.0

***Added***:

* Adds redisdb replication byte metrics ([#20029](https://github.com/DataDog/integrations-core/pull/20029))
* Add support of Redis 8 ([#20227](https://github.com/DataDog/integrations-core/pull/20227))

***Fixed***:

* Allow slowlog collection to fail without error to support some managed Redis that do not allow SLOWLOG GET. ([#20261](https://github.com/DataDog/integrations-core/pull/20261))

## 7.2.0 / 2025-03-19 / Agent 7.65.0

***Added***:

* Add additional memory metrics. ([#19696](https://github.com/DataDog/integrations-core/pull/19696))

## 7.1.0 / 2025-01-25 / Agent 7.63.0

***Added***:

* Update dependencies ([#19430](https://github.com/DataDog/integrations-core/pull/19430))

## 7.0.0 / 2024-10-04 / Agent 7.59.0

***Removed***:

* Remove support for Python 2. ([#18580](https://github.com/DataDog/integrations-core/pull/18580))

***Fixed***:

* Bump the version of datadog-checks-base to 37.0.0 ([#18617](https://github.com/DataDog/integrations-core/pull/18617))

## 6.0.0 / 2024-10-01 / Agent 7.58.0

***Changed***:

* Bump minimum version of base check ([#18733](https://github.com/DataDog/integrations-core/pull/18733))

***Added***:

* Bump the python version from 3.11 to 3.12 ([#18212](https://github.com/DataDog/integrations-core/pull/18212))

## 5.7.0 / 2024-08-09 / Agent 7.57.0

***Added***:

* Update dependencies ([#18187](https://github.com/DataDog/integrations-core/pull/18187))

## 5.6.0 / 2024-07-05 / Agent 7.56.0

***Added***:

* Update dependencies ([#17817](https://github.com/DataDog/integrations-core/pull/17817)), ([#17953](https://github.com/DataDog/integrations-core/pull/17953))

## 5.5.0 / 2024-05-31 / Agent 7.55.0

***Added***:

* Update dependencies ([#17519](https://github.com/DataDog/integrations-core/pull/17519))

## 5.4.0 / 2024-04-26 / Agent 7.54.0

***Added***:

* Update dependencies ([#17319](https://github.com/DataDog/integrations-core/pull/17319))

## 5.3.1 / 2024-03-05 / Agent 7.53.0

***Fixed***:

* Check existence of total_commands_processed ([#16996](https://github.com/DataDog/integrations-core/pull/16996))

## 5.3.0 / 2024-01-05 / Agent 7.51.0

***Added***:

* Bump the Python version from py3.9 to py3.11 ([#15997](https://github.com/DataDog/integrations-core/pull/15997))

## 5.2.0 / 2023-09-29 / Agent 7.49.0

***Added***:

* Update Redis to 5.0.1 ([#15922](https://github.com/DataDog/integrations-core/pull/15922))

## 5.1.0 / 2023-08-18 / Agent 7.48.0

***Added***:

* Update dependencies for Agent 7.48 ([#15585](https://github.com/DataDog/integrations-core/pull/15585))

## 5.0.0 / 2023-08-10

***Changed***:

* Bump the minimum base check version ([#15427](https://github.com/DataDog/integrations-core/pull/15427))

***Added***:

* Update generated config models ([#15212](https://github.com/DataDog/integrations-core/pull/15212))

***Fixed***:

* Fix types for generated config models ([#15334](https://github.com/DataDog/integrations-core/pull/15334))

## 4.8.0 / 2023-07-10 / Agent 7.47.0

***Added***:

* Bump dependencies for Agent 7.47 ([#15145](https://github.com/DataDog/integrations-core/pull/15145))
* Added total connections received and instantaneous kbps metrics ([#14467](https://github.com/DataDog/integrations-core/pull/14467)) Thanks [bmalec](https://github.com/bmalec).

***Fixed***:

* fix the length of key of type stream ([#14722](https://github.com/DataDog/integrations-core/pull/14722)) Thanks [sileht](https://github.com/sileht).
* Bump Python version from py3.8 to py3.9 ([#14701](https://github.com/DataDog/integrations-core/pull/14701))

## 4.7.0 / 2023-05-26 / Agent 7.46.0

***Added***:

* Add ping latency metric ([#14523](https://github.com/DataDog/integrations-core/pull/14523))

## 4.6.0 / 2023-03-31 / Agent 7.44.0

***Added***:

* Update redis to 4.5.4 ([#14270](https://github.com/DataDog/integrations-core/pull/14270))

## 4.5.3 / 2022-12-09 / Agent 7.42.0

***Fixed***:

* Stop using deprecated `distutils.version` classes ([#13408](https://github.com/DataDog/integrations-core/pull/13408))

## 4.5.2 / 2022-08-05 / Agent 7.39.0

***Fixed***:

* Dependency updates ([#12653](https://github.com/DataDog/integrations-core/pull/12653))
* Fix latency measurement to exclude agent load ([#12505](https://github.com/DataDog/integrations-core/pull/12505))

## 4.5.1 / 2022-05-15 / Agent 7.37.0

***Fixed***:

* Upgrade dependencies ([#11958](https://github.com/DataDog/integrations-core/pull/11958))

## 4.5.0 / 2022-04-05 / Agent 7.36.0

***Added***:

* Upgrade dependencies ([#11726](https://github.com/DataDog/integrations-core/pull/11726))
* Add metric_patterns options to filter all metric submission by a list of regexes ([#11695](https://github.com/DataDog/integrations-core/pull/11695))

## 4.4.0 / 2022-02-19 / Agent 7.35.0

***Added***:

* Add `pyproject.toml` file ([#11424](https://github.com/DataDog/integrations-core/pull/11424))

***Fixed***:

* Fix namespace packaging on Python 2 ([#11532](https://github.com/DataDog/integrations-core/pull/11532))

## 4.3.2 / 2022-02-01 / Agent 7.34.0

***Fixed***:

* Bump redis dependency to 4.0.2 ([#11247](https://github.com/DataDog/integrations-core/pull/11247))

## 4.3.1 / 2022-01-08

***Fixed***:

* Add comment to autogenerated model files ([#10945](https://github.com/DataDog/integrations-core/pull/10945))
* Bump redis dependency ([#9383](https://github.com/DataDog/integrations-core/pull/9383))

## 4.3.0 / 2021-11-13 / Agent 7.33.0

***Added***:

* Add support for monitoring Redis stream keys ([#10549](https://github.com/DataDog/integrations-core/pull/10549)) Thanks [jd](https://github.com/jd).

## 4.2.0 / 2021-10-04 / Agent 7.32.0

***Added***:

* Disable generic tags ([#10027](https://github.com/DataDog/integrations-core/pull/10027))

## 4.1.0 / 2021-08-22 / Agent 7.31.0

***Added***:

* Add runtime configuration validation ([#8977](https://github.com/DataDog/integrations-core/pull/8977))

## 4.0.0 / 2021-07-12 / Agent 7.30.0

***Changed***:

* Remove redis_role tag from service check ([#9540](https://github.com/DataDog/integrations-core/pull/9540))

***Added***:

* Make critical service checks more verbose ([#9280](https://github.com/DataDog/integrations-core/pull/9280))

***Fixed***:

* Expand E2E tests to account for restrictions in managed Redis environments ([#9394](https://github.com/DataDog/integrations-core/pull/9394))
* Overrides 'SLOWLOG GET' response callback with fixed/upstream version of `parse_slowlog_get` ([#9459](https://github.com/DataDog/integrations-core/pull/9459))
* Fix Redis slowlog parsing when no new slowlog entries ([#9458](https://github.com/DataDog/integrations-core/pull/9458)) Thanks [ypisetsky](https://github.com/ypisetsky).

## 3.5.0 / 2021-05-28 / Agent 7.29.0

***Added***:

* added recent_max_input_buffer and recent_max_output_buffer ([#9321](https://github.com/DataDog/integrations-core/pull/9321)) Thanks [yonatan-ess](https://github.com/yonatan-ess).

## 3.4.0 / 2021-04-19 / Agent 7.28.0

***Added***:

* Report io thread metrics ([#9018](https://github.com/DataDog/integrations-core/pull/9018)) Thanks [jlisam](https://github.com/jlisam).
* Support redis 6 cpu metrics ([#9015](https://github.com/DataDog/integrations-core/pull/9015)) Thanks [jlisam](https://github.com/jlisam).

***Fixed***:

* Skip slowlogs if there is an error ([#9147](https://github.com/DataDog/integrations-core/pull/9147))
* Correct default ssl_cert_reqs ([#9048](https://github.com/DataDog/integrations-core/pull/9048)) Thanks [wbobeirne](https://github.com/wbobeirne).

## 3.3.1 / 2021-03-07 / Agent 7.27.0

***Fixed***:

* Rename config spec example consumer option `default` to `display_default` ([#8593](https://github.com/DataDog/integrations-core/pull/8593))
* Bump minimum base package version ([#8443](https://github.com/DataDog/integrations-core/pull/8443))

## 3.3.0 / 2021-01-06 / Agent 7.25.0

***Added***:

* Update redis dependency ([#8301](https://github.com/DataDog/integrations-core/pull/8301))

## 3.2.0 / 2020-10-31 / Agent 7.24.0

***Added***:

* [doc] Add encoding in log config sample ([#7708](https://github.com/DataDog/integrations-core/pull/7708))

## 3.1.1 / 2020-09-21 / Agent 7.23.0

***Fixed***:

* Fix style for the latest release of Black ([#7438](https://github.com/DataDog/integrations-core/pull/7438))
* Handle redis role missing ([#7413](https://github.com/DataDog/integrations-core/pull/7413))

## 3.1.0 / 2020-08-10 / Agent 7.22.0

***Added***:

* Add auto_conf.yaml spec for redisdb ([#7161](https://github.com/DataDog/integrations-core/pull/7161))
* Add redis config spec ([#7091](https://github.com/DataDog/integrations-core/pull/7091))

***Fixed***:

* Update logs config service field to optional ([#7209](https://github.com/DataDog/integrations-core/pull/7209))
* Refactor instance argument ([#7018](https://github.com/DataDog/integrations-core/pull/7018))

## 3.0.0 / 2020-06-29 / Agent 7.21.0

***Changed***:

* Collect port and host from same source in _generate_instance_key ([#6680](https://github.com/DataDog/integrations-core/pull/6680))

***Added***:

* Upgrade redis dependency to support `username` in connection strings ([#6708](https://github.com/DataDog/integrations-core/pull/6708))

***Fixed***:

* Add flag to enable CLIENT command metrics ([#6877](https://github.com/DataDog/integrations-core/pull/6877))

## 2.1.1 / 2020-06-11 / Agent 7.20.2

***Fixed***:

* Add flag to enable CLIENT command metrics ([#6877](https://github.com/DataDog/integrations-core/pull/6877))

## 2.1.0 / 2020-05-17 / Agent 7.20.0

***Added***:

* Allow optional dependency installation for all checks ([#6589](https://github.com/DataDog/integrations-core/pull/6589))
* Add 'redis.net.connections' metric to count connections by client ([#6495](https://github.com/DataDog/integrations-core/pull/6495)) Thanks [remicalixte](https://github.com/remicalixte).

***Fixed***:

* Reduce slow-log log message ([#6631](https://github.com/DataDog/integrations-core/pull/6631))
* Use agent 6 signature ([#6424](https://github.com/DataDog/integrations-core/pull/6424))

## 2.0.2 / 2020-04-04 / Agent 7.19.0

***Fixed***:

* Remove logs sourcecategory ([#6121](https://github.com/DataDog/integrations-core/pull/6121))

## 2.0.1 / 2020-02-10 / Agent 7.18.0

***Fixed***:

* Handle error in config_get ([#5676](https://github.com/DataDog/integrations-core/pull/5676))

## 2.0.0 / 2020-02-05

***Changed***:

* Submit `redis.key.length` metric regardless of `warn_on_missing_keys` ([#5591](https://github.com/DataDog/integrations-core/pull/5591))

***Added***:

* Add aof loading metrics ([#5431](https://github.com/DataDog/integrations-core/pull/5431)) Thanks [tanner-bruce](https://github.com/tanner-bruce).

## 1.15.0 / 2020-01-13 / Agent 7.17.0

***Added***:

* Use lazy logging format ([#5398](https://github.com/DataDog/integrations-core/pull/5398))
* Use lazy logging format ([#5377](https://github.com/DataDog/integrations-core/pull/5377))
* Upgrade `redis` to 3.3.11 ([#5150](https://github.com/DataDog/integrations-core/pull/5150))
* Report maxclients ([#5207](https://github.com/DataDog/integrations-core/pull/5207)) Thanks [jd](https://github.com/jd).

## 1.14.0 / 2019-12-02 / Agent 7.16.0

***Added***:

* Add active defragmentation gauges ([#5022](https://github.com/DataDog/integrations-core/pull/5022))
* Use a stub class for metadata testing ([#4919](https://github.com/DataDog/integrations-core/pull/4919))

***Fixed***:

* Fix example keys config ([#4939](https://github.com/DataDog/integrations-core/pull/4939)) Thanks [sileht](https://github.com/sileht).

## 1.13.0 / 2019-10-11 / Agent 6.15.0

***Added***:

* Submit version metadata ([#4705](https://github.com/DataDog/integrations-core/pull/4705))

## 1.12.2 / 2019-10-04

***Fixed***:

* Don't display warning for default keys value ([#4641](https://github.com/DataDog/integrations-core/pull/4641))

## 1.12.1 / 2019-08-24 / Agent 6.14.0

***Fixed***:

* Always publish a value for missing keys ([#4386](https://github.com/DataDog/integrations-core/pull/4386))

## 1.12.0 / 2019-06-01 / Agent 6.12.0

***Added***:

* Add redis.mem.overhead and redis.mem.startup ([#3760](https://github.com/DataDog/integrations-core/pull/3760)) Thanks [maximebedard](https://github.com/maximebedard).

## 1.11.0 / 2019-05-14

***Added***:

* Adhere to code style ([#3562](https://github.com/DataDog/integrations-core/pull/3562))

***Fixed***:

* Adjust latency tracking in redisdb integration ([#3689](https://github.com/DataDog/integrations-core/pull/3689)) Thanks [Firehed](https://github.com/Firehed).

## 1.10.0 / 2019-02-18 / Agent 6.10.0

***Added***:

* Add redis_db tag to redis.key.length ([#3008](https://github.com/DataDog/integrations-core/pull/3008))

## 1.9.0 / 2019-01-22 / Agent 6.9.0

***Added***:

* Finish Python 3 Support ([#2951](https://github.com/DataDog/integrations-core/pull/2951))

***Fixed***:

* Only try to decode slowlog command entrypoint ([#2998](https://github.com/DataDog/integrations-core/pull/2998))

## 1.8.0 / 2018-11-30 / Agent 6.8.0

***Added***:

* Support Python 3 ([#2422](https://github.com/DataDog/integrations-core/pull/2422))

## 1.7.1 / 2018-10-12 / Agent 6.6.0

***Fixed***:

* Handle `host:` command when parsing commandstats output ([#2356](https://github.com/DataDog/integrations-core/pull/2356))
* Fix multiple db key length ([#2187](https://github.com/DataDog/integrations-core/pull/2187))

## 1.7.0 / 2018-09-04 / Agent 6.5.0

***Added***:

* Support finding key lengths on any db ([#1948](https://github.com/DataDog/integrations-core/pull/1948))

***Fixed***:

* Add data files to the wheel package ([#1727](https://github.com/DataDog/integrations-core/pull/1727))

## 1.6.0 / 2018-06-06

***Added***:

* Add a config option to disable connection cache ([#1668](https://github.com/DataDog/integrations-core/pull/1668))
* Package `auto_conf.yaml` for appropriate integrations ([#1664](https://github.com/DataDog/integrations-core/pull/1664))

## 1.5.0 / 2018-05-11

***Added***:

* Hardcode the 6379 port in the Autodiscovery template ([#1444](https://github.com/DataDog/integrations-core/pull/1444) for more information)

## 1.4.0 / 2018-01-10

***Added***:

* Keys can be expressed as patterns, see [#300](https://github.com/DataDog/integrations-core/issues/300). Thanks [@aliva](https://github.com/aliva).

***Fixed***:

* Skip non-local keys ( [#798](https://github.com/DataDog/integrations-core/issues/798)) Thanks [@chadharvey](https://github.com/chadharvey)

## 1.3.0 / 2017-11-21

***Added***:

* Update auto_conf template to support agent 6 and 5.20+ ([#860](https://github)com/DataDog/integrations-core/issues/860)

## 1.2.0 / 2017-08-28

***Added***:

* Add "redis.net.commands.instantaneous" metric ([#672](https://github)com/DataDog/integrations-core/issues/672)
* Add "redis.mem.maxmemory" metric ([#673](https://github.com/DataDog/integrations-core/issues/673), thanks [@endzyme](https://github)com/endzyme)

## 1.1.0 / 2017-07-18

***Added***:

* Add "redis_role:{master,slave}" tag.

## 1.0.0 / 2017-02-22

***Added***:

* Add redisdb integration.
