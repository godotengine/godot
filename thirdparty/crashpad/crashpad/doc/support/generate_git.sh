#!/bin/bash

# Copyright 2015 The Crashpad Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# Run from the Crashpad project root directory.
cd "$(dirname "${0}")/../.."

source doc/support/compat.sh

basename="$(basename "${0}")"

status="$(git status --porcelain)"
if [[ -n "${status}" ]]; then
  echo "${basename}: the working directory must be clean" >& 2
  git status
  exit 1
fi

# git symbolic-ref gives the current branch name, but fails for detached HEAD.
# In that case, git rev-parse will give the current hash.
original_branch="$(git symbolic-ref --short --quiet HEAD || git rev-parse HEAD)"

local_branch="\
$(${sed_ext} -e 's/(.*)\..*/\1/' <<< "${basename}").${$}.${RANDOM}"

remote_name=origin
remote_master_branch_name=master
remote_master_branch="${remote_name}/${remote_master_branch_name}"
remote_doc_branch_name=doc
remote_doc_branch="${remote_name}/${remote_doc_branch_name}"

git fetch

git checkout -b "${local_branch}" "${remote_doc_branch}"

dirty=

function cleanup() {
  if [[ "${dirty}" ]]; then
    git reset --hard HEAD
    git clean --force
  fi

  git checkout "${original_branch}"
  git branch -D "${local_branch}"
}

trap cleanup EXIT

master_hash=$(git rev-parse --short=12 "${remote_master_branch}")
git merge "${remote_master_branch}" \
    -m "Merge ${remote_master_branch_name} ${master_hash} into doc"

dirty=y

doc/support/generate.sh

git add -A doc/generated

count="$(git diff --staged --numstat | wc -l)"
if [[ $count -gt 0 ]]; then
  git commit \
      -m "Update documentation to ${remote_master_branch_name} ${master_hash}"
  dirty=

  git push "${remote_name}" "HEAD:${remote_doc_branch_name}"
else
  dirty=
fi

# cleanup will run on exit
