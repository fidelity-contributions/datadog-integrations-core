# (C) Datadog, Inc. 2024-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
from __future__ import absolute_import

import os
import subprocess
from contextlib import contextmanager

from datadog_checks.dev.conditions import WaitForPortListening
from datadog_checks.dev.env import environment_run
from datadog_checks.dev.ssh_tunnel import KillProcess
from datadog_checks.dev.structures import LazyFunction, TempDir
from datadog_checks.dev.utils import ON_WINDOWS, find_free_port, get_ip

PID_FILE = 'ssh.pid'


def run_background_command(command, pid_filename, env=None):
    """Run `command` in the background, writing its PID in `pid_filename`."""
    if ON_WINDOWS:
        process = subprocess.Popen(command, env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        process = subprocess.Popen(command, env=env, start_new_session=True)
    with open(pid_filename, 'w') as pid_file:
        pid_file.write(str(process.pid))


@contextmanager
def socks_proxy(host, user, private_key):
    """Open a SSH connection with a SOCKS proxy."""
    set_up = SocksProxyUp(host, user, private_key)
    tear_down = KillProcess('socks_proxy', PID_FILE)

    with environment_run(up=set_up, down=tear_down) as result:
        yield result


class SocksProxyUp(LazyFunction):
    """Create a SOCKS proxy using `ssh`.

    It returns the (`ip`, `port`) on which the proxy is listening.
    """

    def __init__(self, host, user, private_key):
        self.host = host
        self.user = user
        self.private_key = private_key

    def __call__(self):
        WaitForPortListening(self.host, 22, attempts=1, wait=1)()
        with TempDir('socks_proxy') as temp_dir:
            ip = get_ip()
            local_port = find_free_port(ip)
            key_file = os.path.join(temp_dir, 'ssh_key')
            with open(key_file, 'w') as f:
                f.write(self.private_key)
            os.chmod(key_file, 0o600)
            command = [
                'ssh',
                '-N',
                '-D',
                '{}:{}'.format(ip, local_port),
                '-i',
                key_file,
                '-o',
                'BatchMode=yes',
                '-o',
                'UserKnownHostsFile={}'.format(os.devnull),
                '-o',
                'StrictHostKeyChecking=no',
                '{}@{}'.format(self.user, self.host),
            ]
            run_background_command(command, os.path.join(temp_dir, PID_FILE))

            WaitForPortListening(ip, local_port, attempts=1, wait=1)()

            return ip, local_port


@contextmanager
def tcp_tunnel(host, user, private_key, remote_port):
    """Open a SSH connection with a TCP tunnel proxy."""
    set_up = TCPTunnelUp(host, user, private_key, remote_port)
    tear_down = KillProcess('tcp_tunnel', PID_FILE)

    with environment_run(up=set_up, down=tear_down) as result:
        yield result


class TCPTunnelUp(LazyFunction):
    """Create a TCP tunnel using `ssh`.

    It returns the (`ip`, `port`) on which the tunnel is listening, connecting to `remote_port`.
    """

    def __init__(self, host, user, private_key, remote_port):
        self.host = host
        self.user = user
        self.private_key = private_key
        self.remote_port = remote_port

    def __call__(self):
        with TempDir('tcp_tunnel') as temp_dir:
            ip = get_ip()
            local_port = find_free_port(ip)
            key_file = os.path.join(temp_dir, 'ssh_key')
            with open(key_file, 'w') as f:
                f.write(self.private_key)
            os.chmod(key_file, 0o600)
            command = [
                'ssh',
                '-N',
                '-L',
                '{}:{}:localhost:{}'.format(ip, local_port, self.remote_port),
                '-i',
                key_file,
                '-o',
                'BatchMode=yes',
                '-o',
                'UserKnownHostsFile={}'.format(os.devnull),
                '-o',
                'StrictHostKeyChecking=no',
                '{}@{}'.format(self.user, self.host),
            ]
            run_background_command(command, os.path.join(temp_dir, PID_FILE))

            WaitForPortListening(ip, local_port)()

            return ip, local_port
