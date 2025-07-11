# (C) Datadog, Inc. 2010-present
# All rights reserved
# Licensed under Simplified BSD License (see LICENSE)
import calendar
from datetime import datetime, timedelta, timezone

import psutil

from datadog_checks.base import ConfigurationError, is_affirmative
from datadog_checks.base.checks.win.wmi import WinWMICheck, from_time, to_time
from datadog_checks.base.utils.containers import hash_mutable
from datadog_checks.base.utils.timeout import TimeoutException

EVENT_TYPE = 'win32_log_event'
SOURCE_TYPE_NAME = 'event viewer'

# Integer properties to normalize.
# Source: https://docs.microsoft.com/en-us/previous-versions/windows/desktop/eventlogprov/win32-ntlogevent
INTEGER_PROPERTIES = ['EventCode', 'EventIdentifier', 'EventType', 'RecordNumber']


class Win32EventLogWMI(WinWMICheck):
    # WMI information
    EVENT_PROPERTIES = ["EventCode", "SourceName", "TimeGenerated", "Type"]
    EXTRA_EVENT_PROPERTIES = ["InsertionStrings", "Message", "Logfile"]
    NAMESPACE = "root\\CIMV2"
    EVENT_CLASS = "Win32_NTLogEvent"
    NEW_PARAMS = (
        'tag_sid',
        'interpret_messages',
        'path',
        'start',
        'query',
        'filters',
        'included_messages',
        'excluded_messages',
        'domain',
        'timeout',
        'payload_size',
        'bookmark_frequency',
        'dd_security_events',
    )

    def __init__(self, name, init_config, instances):
        super(Win32EventLogWMI, self).__init__(name, init_config, instances)
        # Settings
        self._tag_event_id = is_affirmative(self.instance.get('tag_event_id', init_config.get('tag_event_id')))
        self._verbose = init_config.get('verbose', True)
        self._default_event_priority = init_config.get('default_event_priority', 'normal')

        # State
        self.last_ts = {}

        self.check_initializations.append(
            lambda: self.warning(
                'This version of the check is deprecated and will be removed in a future release. '
                'Set `legacy_mode` to `false` and read about the latest options, such as `query`.'
            )
        )
        for new_param in self.NEW_PARAMS:
            if new_param in self.instance:
                self.log.warning("%s config option is ignored when running legacy mode. Please remove it", new_param)

    def check(self, instance):
        # Connect to the WMI provider
        host = instance.get('host', "localhost")
        username = self.instance.get('user', self.instance.get('username', ''))
        password = instance.get('password', "")
        instance_tags = instance.get('tags', [])
        notify = instance.get('notify', [])
        event_priority = instance.get('event_priority', self._default_event_priority)
        if (event_priority.lower() != 'normal') and (event_priority.lower() != 'low'):
            event_priority = 'normal'

        user = instance.get('user')
        ltypes = instance.get('type', [])
        source_names = instance.get('source_name', [])
        log_files = instance.get('log_file', [])
        event_ids = instance.get('event_id', [])
        event_format = instance.get('event_format')
        message_filters = instance.get('message_filters', [])

        if not (source_names or event_ids or message_filters or log_files or ltypes):
            raise ConfigurationError(
                'At least one of the following filters must be set: '
                'source_name, event_id, message_filters, log_file, type'
            )

        instance_hash = hash_mutable(instance)
        instance_key = self._get_instance_key(host, self.NAMESPACE, self.EVENT_CLASS, instance_hash)

        # Store the last timestamp by instance
        if instance_key not in self.last_ts:
            # If system boot was within 600s of dd agent start then use boottime as last_ts
            uptime = datetime.now(timezone.utc) - datetime.fromtimestamp(psutil.boot_time(), timezone.utc)
            if uptime.total_seconds() <= 600:
                self.last_ts[instance_key] = datetime.now(timezone.utc) - uptime
            else:
                self.last_ts[instance_key] = datetime.now(timezone.utc)
            return

        # Event properties
        event_properties = list(self.EVENT_PROPERTIES)

        if event_format is not None:
            event_properties.extend(list(set(self.EXTRA_EVENT_PROPERTIES) & set(event_format)))
        else:
            event_properties.extend(self.EXTRA_EVENT_PROPERTIES)

        # Event filters
        query = {}
        filters = []
        last_ts = self.last_ts[instance_key]
        query['TimeGenerated'] = ('>=', self._dt_to_wmi(last_ts))
        if user:
            query['User'] = ('=', user)
        if ltypes:
            query['Type'] = []
            for ltype in ltypes:
                query['Type'].append(('=', ltype))
        if source_names:
            query['SourceName'] = []
            for source_name in source_names:
                query['SourceName'].append(('=', source_name))
        if log_files:
            query['LogFile'] = []
            for log_file in log_files:
                query['LogFile'].append(('=', log_file))
        if event_ids:
            query['EventCode'] = []
            for event_id in event_ids:
                query['EventCode'].append(('=', event_id))
        if message_filters:
            query['NOT Message'] = []
            query['Message'] = []
            for filt in message_filters:
                if filt[0] == '-':
                    query['NOT Message'].append(('LIKE', filt[1:]))
                else:
                    query['Message'].append(('LIKE', filt))

        filters.append(query)

        wmi_sampler = self._get_running_wmi_sampler(
            instance_key,
            self.EVENT_CLASS,
            event_properties,
            filters=filters,
            host=host,
            namespace=self.NAMESPACE,
            username=username,
            password=password,
            and_props=['Message'],
        )

        wmi_sampler.reset_filter(new_filters=filters)
        try:
            wmi_sampler.sample()
        except TimeoutException:
            self.log.warning(
                "[Win32EventLog] WMI query timed out. class=%s - properties=%s - filters=%s - tags=%s",
                self.EVENT_CLASS,
                event_properties,
                filters,
                instance_tags,
            )
        else:
            for ev in wmi_sampler:
                # for local events we dont need to specify a hostname
                hostname = None if (host == "localhost" or host == ".") else host
                log_ev = LogEvent(
                    ev, self.log, hostname, instance_tags, notify, self._tag_event_id, event_format, event_priority
                )

                # Since WQL only compares on the date and NOT the time, we have to
                # do a secondary check to make sure events are after the last
                # timestamp
                if log_ev.is_after(last_ts):
                    self.event(log_ev.to_event_dict())
                else:
                    self.log.debug('Skipping event after %s. ts=%s', last_ts, log_ev.timestamp)

            # Update the last time checked
            self.last_ts[instance_key] = datetime.now(timezone.utc)

    def _dt_to_wmi(self, dt):
        """A wrapper around wmi.from_time to get a WMI-formatted time from a
        time struct.
        """
        return from_time(
            year=dt.year,
            month=dt.month,
            day=dt.day,
            hours=dt.hour,
            minutes=dt.minute,
            seconds=dt.second,
            microseconds=0,
            timezone=0,
        )


class LogEvent(object):
    def __init__(self, ev, log, hostname, tags, notify_list, tag_event_id, event_format, event_priority):
        self.event = self._normalize_event(ev.copy())
        self.log = log
        self.hostname = hostname
        self.tags = self._tags(tags, self.event['EventCode']) if tag_event_id else tags
        self.notify_list = notify_list
        self.timestamp = self._wmi_to_ts(self.event['TimeGenerated'])
        self._format = event_format
        self.event_priority = event_priority

    @staticmethod
    def _normalize_event(event):
        for field in INTEGER_PROPERTIES:
            if field in event:
                event[field] = int(event[field])
        return event

    @property
    def _msg_title(self):
        return '{logfile}/{source}'.format(logfile=self.event['Logfile'], source=self.event['SourceName'])

    @property
    def _msg_text(self):
        """
        Generate the event's body to send to Datadog.

        Consider `event_format` parameter:
        * Only use the specified list of event properties.
        * If unspecified, default to the EventLog's `Message` or `InsertionStrings`.
        """
        msg_text = ""

        if self._format:
            msg_text_fields = ["%%%\n```"]

            for event_property in self._format:
                property_value = self.event.get(event_property)
                if property_value is None:
                    self.log.warning("Unrecognized `%s` event property.", event_property)
                    continue
                msg_text_fields.append(
                    "{property_name}: {property_value}".format(
                        property_name=event_property, property_value=property_value
                    )
                )

            msg_text_fields.append("```\n%%%")

            msg_text = "\n".join(msg_text_fields)
        else:
            # Override when verbosity
            if self.event.get('Message'):
                msg_text = "{message}\n".format(message=self.event['Message'])
            elif self.event.get('InsertionStrings'):
                msg_text = "\n".join([i_str for i_str in self.event['InsertionStrings'] if i_str.strip()])

        if self.notify_list:
            msg_text += "\n{notify_list}".format(notify_list=' '.join([" @" + n for n in self.notify_list]))

        return msg_text

    @property
    def _alert_type(self):
        event_type = self.event['Type']
        # Convert to a Datadog alert type
        if event_type == 'Warning':
            return 'warning'
        elif event_type == 'Error':
            return 'error'
        return 'info'

    @property
    def _aggregation_key(self):
        return self.event['SourceName']

    def to_event_dict(self):
        event_dict = {
            'timestamp': self.timestamp,
            'event_type': EVENT_TYPE,
            'priority': self.event_priority,
            'msg_title': self._msg_title,
            'msg_text': self._msg_text.strip(),
            'aggregation_key': self._aggregation_key,
            'alert_type': self._alert_type,
            'source_type_name': SOURCE_TYPE_NAME,
            'tags': self.tags,
        }
        if self.hostname:
            event_dict['host'] = self.hostname

        return event_dict

    def is_after(self, ts):
        '''Compare this event's timestamp to a give timestamp.'''
        if self.timestamp >= int(calendar.timegm(ts.timetuple())):
            return True
        return False

    def _wmi_to_ts(self, wmi_ts):
        """Convert a wmi formatted timestamp into an epoch."""
        year, month, day, hour, minute, second, microsecond, tz = to_time(wmi_ts)
        tz_delta = timedelta(minutes=int(tz))
        if '+' in wmi_ts:
            tz_delta = -tz_delta

        dt = (
            datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second, microsecond=microsecond)
            + tz_delta
        )
        return int(calendar.timegm(dt.timetuple()))

    def _tags(self, tags, event_code):
        """Inject additional tags into the list already supplied to LogEvent."""
        tags_list = []
        if tags is not None:
            tags_list += list(tags)
        tags_list.append("event_id:{event_id}".format(event_id=event_code))
        return tags_list
