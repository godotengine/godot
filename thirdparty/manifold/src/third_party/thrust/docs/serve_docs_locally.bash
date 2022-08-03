#! /usr/bin/env bash

###############################################################################
# Copyright (c) 2018-2021 NVIDIA Corporation
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
###############################################################################

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

REPO_PATH=${SCRIPT_PATH}/..

BUILD_DOCS_PATH=build_docs
BUILD_GITHUB_PAGES_PATH=${BUILD_DOCS_PATH}/github_pages

cd ${REPO_PATH}/${BUILD_GITHUB_PAGES_PATH}

bundle install
bundle exec jekyll serve \
  --verbose              \
  --incremental          \
  --profile              \
  --baseurl "/thrust"    \
  ${@}

