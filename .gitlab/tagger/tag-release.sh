#!/bin/bash
# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euxo pipefail
IFS=$'\n\t'

echo "Grabbing GitHub deploy key and starting ssh-agent"
set +x
eval "$(ssh-agent -s)"
aws ssm get-parameter --region us-east-1 --name ci.integrations-core.github_deploy_key --with-decryption --query "Parameter.Value" --out text | ssh-add -
set -x

git remote set-url origin git@github.com:DataDog/integrations-core.git
git config --global user.email "$TAGGER_EMAIL"
git config --global user.name "$TAGGER_NAME"

set +e
ddev release tag all --skip-prerelease
status=$?
# If we fail to fetch tags, fallback to the old behavior of getting list of tags even if they're not fetched
if [[ $status -eq 3 ]]; then
    echo "Failed to fetch tags, falling back to list tags without fetching"
    ddev release tag all --skip-prerelease --no-fetch
    status=$?
fi
set -e

# Only build packages if there were new releases
if [[ $status -eq 0 ]]; then
    ./.gitlab/tagger/build-packages.sh
elif [[ $status -eq 2 ]]; then
    echo "No new releases, skipping the build pipeline trigger"
else
    exit $status
fi
