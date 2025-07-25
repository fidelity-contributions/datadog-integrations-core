# (C) Datadog, Inc. 2018-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
import click

from datadog_checks.dev.tooling.commands.console import (
    CONTEXT_SETTINGS,
    abort,
    echo_info,
    echo_success,
    echo_waiting,
    echo_warning,
)
from datadog_checks.dev.tooling.git import git_fetch, git_tag, git_tag_list
from datadog_checks.dev.tooling.release import get_release_tag_string
from datadog_checks.dev.tooling.utils import complete_valid_checks, get_valid_checks, get_version_string

# 0.0.1 is the initial pre-release version that is generated from the integration's template.
# Releasing any version > 0.0.1 for a core integration requires a tag.*.link file to be updated in the PR
PRERELEASE = '0.0.1'


@click.command(context_settings=CONTEXT_SETTINGS, short_help='Tag the git repo with the current release of a check')
@click.argument('check', shell_complete=complete_valid_checks)
@click.argument('version', required=False)
@click.option('--push/--no-push', default=True)
@click.option('--dry-run', '-n', is_flag=True)
@click.option('--skip-prerelease', is_flag=True)
@click.option('--fetch/--no-fetch', default=True)
def tag(check, version, push, dry_run, skip_prerelease, fetch):
    """Tag the HEAD of the git repo with the current release number for a
    specific check. The tag is pushed to origin by default.

    You can tag everything at once by setting the check to `all`.

    Notice: specifying a different version than the one in `__about__.py` is
    a maintenance task that should be run under very specific circumstances
    (e.g. re-align an old release performed on the wrong commit).

    Return codes:
    0: Success
    1: Invalid command call
    2: Nothing to tag
    3: Failed to fetch tags
    Other: Git tag command returned a non-zero exit code
    """
    tagging_all = check == 'all'

    valid_checks = get_valid_checks()
    if not tagging_all and check not in valid_checks:
        abort(f'Check `{check}` is not an Agent-based Integration')

    if tagging_all:
        if version:
            abort('You cannot tag every check with the same version')
        checks = sorted(valid_checks)
    else:
        checks = [check]

    # Check for any new tags
    tagged = False
    # Fetch all tags from the remote
    if fetch:
        echo_info('Fetching all tags from remote...')

        if (result := git_fetch(tags=True)).code != 0:
            abort(f'Failed to fetch tags: {result.stderr}', 3)

    existing_tags = git_tag_list()

    for check in checks:
        echo_info(f'{check}:')

        # get the current version
        if not version:
            version = get_version_string(check)

        if skip_prerelease and version == PRERELEASE:
            echo_warning('skipping prerelease version')
            version = None
            continue

        # get the tag name
        release_tag = get_release_tag_string(check, version)
        echo_waiting(f'Tagging HEAD with {release_tag}... ', indent=True, nl=False)

        if dry_run:
            # Get latest tag for check
            if release_tag in existing_tags:
                echo_warning('already exists (dry-run)')
            else:
                tagged = True
                echo_success("success! (dry-run)")
            version = None
            continue

        result = git_tag(release_tag, push)

        if result.code == 128 or 'already exists' in result.stderr:
            echo_warning('already exists')
        elif result.code != 0:
            abort(f'\n{result.stdout}{result.stderr}', code=result.code)
        else:
            tagged = True
            echo_success('success!')

        # Reset version
        version = None

    if not tagged:
        abort(code=2)
