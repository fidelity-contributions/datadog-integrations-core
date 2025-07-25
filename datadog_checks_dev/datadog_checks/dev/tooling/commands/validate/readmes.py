# (C) Datadog, Inc. 2020-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
import os
from tempfile import TemporaryDirectory

import click
import markdown
from bs4 import BeautifulSoup

from datadog_checks.dev.fs import chdir, create_file
from datadog_checks.dev.subprocess import run_command
from datadog_checks.dev.tooling.commands.console import (
    CONTEXT_SETTINGS,
    abort,
    annotate_display_queue,
    echo_failure,
    echo_info,
    echo_success,
)
from datadog_checks.dev.tooling.constants import get_root
from datadog_checks.dev.tooling.testing import process_checks_option
from datadog_checks.dev.tooling.utils import complete_valid_checks, get_readme_file, read_readme_file
from datadog_checks.dev.utils import download_file

IMAGE_EXTENSIONS = {".png", ".jpg"}

# Get latest format_link script from Datadog/documentation repo
DOCS_LINK_FORMAT_URL = (
    "https://raw.githubusercontent.com/DataDog/documentation/master/local/bin/py/build/actions/format_link.py"
)


@click.command(context_settings=CONTEXT_SETTINGS, short_help='Validate README.md files')
@click.pass_context
@click.argument('check', shell_complete=complete_valid_checks, required=False)
@click.option('--format-links', '-fl', is_flag=True, help='Automatically format links')
def readmes(ctx, check, format_links):
    """Validates README files.

    If `check` is specified, only the check will be validated, if check value is 'changed' will only apply to changed
    checks, an 'all' or empty `check` value will validate all README files.
    """

    repo = ctx.obj['repo_name']

    files_failed = {}
    readme_counter = set()

    integrations = process_checks_option(check, source='integrations', extend_changed=True)
    format_link_script_path = None
    if format_links:
        format_link_dir = TemporaryDirectory()

        with chdir(format_link_dir.name):
            format_link_script_path = os.path.join(format_link_dir.name, "format_link.py")
            create_file("format_link.py")
            download_file(DOCS_LINK_FORMAT_URL, format_link_script_path)

    for integration in integrations:
        display_queue = []
        readme_path = get_readme_file(integration)

        if not os.path.exists(readme_path) and repo in ('extras', 'marketplace'):
            # We are in the process of migrating extras and marketplace to manage READMEs in the Publishing Platform.
            # We'll revisit this validation once we know for sure how we handle READMEs in the new world.
            echo_info(
                f"Skipping README.md validation for {integration}: "
                "'integrations-extras' and 'marketplace' packages are migrating to the Publishing Platform "
                "for README management. If you are not using the Publishing Platform, please ensure your package "
                "includes a README.md file."
            )
            continue

        # Validate the README itself
        validate_readme(integration, repo, display_queue, files_failed, readme_counter)

        if display_queue:
            annotate_display_queue(readme_path, display_queue)
            echo_info(f'{integration}:')
            for func, message in display_queue:
                func(message)

        if format_links and format_link_script_path:
            echo_info("Formatting links in {}".format(os.path.basename(readme_path)))
            try:
                run_command(["python", format_link_script_path, "-f", readme_path])
            except Exception as e:
                echo_failure("Unable to format file: {}".format(str(e)), indent=True)

    num_files = len(readme_counter)
    files_failed = len(files_failed)
    files_passed = num_files - files_failed

    if files_failed:
        click.echo()
        echo_failure(f'Files with errors: {files_failed}')

    if files_passed:
        if files_failed:
            echo_success(f'Files valid: {files_passed}')
        else:
            echo_success(f'All {len(readme_counter)} READMEs are valid!')

    if files_failed:
        abort()


def validate_readme(integration, repo, display_queue, files_failed, readme_counter):
    readme_path = get_readme_file(integration)
    readme_contents = read_readme_file(integration)

    if repo != 'marketplace' and (error_lines := get_ascii_enforcement_error_lines(readme_contents)):
        files_failed[readme_path] = True
        display_queue.extend((echo_failure, line) for line in error_lines)

    html = markdown.markdown(readme_contents)
    soup = BeautifulSoup(html, features="html.parser")
    readme_counter.add(readme_path)

    # Check all required headers are present
    h2s = [h2.text for h2 in soup.find_all("h2")]

    for header in ("Overview", "Setup"):
        validate_header(h2s, header, files_failed, readme_path, display_queue, soup)

    if repo == 'marketplace':
        for header in ("Support", "Uninstallation"):
            validate_header(h2s, header, files_failed, readme_path, display_queue, soup)
            if header == "Uninstallation":
                validate_no_images(h2s, header, files_failed, readme_path, display_queue, soup)

    # Check all referenced images are in the `images` folder and that
    # they use the `raw.githubusercontent` format or relative paths to the `images` folder
    allow_relative = False
    if repo == "marketplace":
        allow_relative = True
    github_path = f"https://raw.githubusercontent.com/DataDog/{repo}"
    img_srcs = [img.attrs.get("src") for img in soup.find_all("img")]
    for img_src in img_srcs:
        image_name = os.path.split(img_src)[-1]
        file_path = os.path.join(get_root(), integration, "images", image_name)
        if img_src.startswith(github_path) or (img_src.startswith("images/") and allow_relative):
            if not os.path.exists(file_path):
                files_failed[readme_path] = True
                display_queue.append(
                    (echo_failure, f"     image: {img_src} is linked in its readme but does not exist")
                )
        else:
            error_msg = (
                f"     All images must be checked into the repo under the `{integration}/images` folder. "
                f"This image path must be in the form: "
                f"https://raw.githubusercontent.com/DataDog/{repo}/master/{integration}/images/<IMAGE_NAME>"
            )
            if allow_relative:
                error_msg += "or be a relative path to the `images/` folder (without a `/` prefix)."
            error_msg += f" Image currently is: {img_src}"

            display_queue.append((echo_failure, error_msg))


# Method to validate header exists and there is text under it
def validate_header(h2s, header, files_failed, readme_path, display_queue, soup):
    if header not in h2s:
        files_failed[readme_path] = True
        display_queue.append((echo_failure, f"     readme is missing a {header} H2 (##) section"))
    else:
        curr_header = soup.find('h2', text=header)
        curr_instructions = curr_header.find_next()
        if curr_instructions.name == "h2":
            files_failed[readme_path] = True
            display_queue.append((echo_failure, f"     readme has an empty {header} H2 (##) section"))


# Method to validate no images are present within the header
def validate_no_images(h2s, header, files_failed, readme_path, display_queue, soup):
    if header not in h2s:
        return
    curr_header = soup.find('h2', text=header)
    curr_text = curr_header.find_next()
    while curr_text is not None and curr_text.name != "h2":
        if curr_text.name == "img":
            files_failed[readme_path] = True
            display_queue.append((echo_failure, f"     readme has an image in {header} H2 (##) section"))
        curr_text = curr_text.find_next()


def get_ascii_enforcement_error_lines(contents):
    errors_lines = []
    for i, line in enumerate(contents.splitlines()):
        # Don't print newlines
        line = line.rstrip('\n')
        invalid_code_unit_indices = []
        indicator_code_units = []
        for code_unit in line:
            if ord(code_unit) > 256:
                invalid_code_unit_indices.append(i)
                indicator_code_units.append('^')
            else:
                indicator_code_units.append(' ')

        if invalid_code_unit_indices:
            errors_lines.append(f'    | {line}')
            errors_lines.append(f'    | {"".join(indicator_code_units)}')

    if errors_lines:
        errors_lines.insert(0, '    readme contains non-ASCII character(s)')

    return errors_lines
