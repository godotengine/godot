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

set -e

function usage {
  echo "Usage: ${0} [flags...]"
  echo
  echo "Generate Thrust documentation markdown with Doxygen and Doxybook that "
  echo "can be served with Jekyll."
  echo
  echo "-h, -help, --help"
  echo "  Print this message."
  echo
  echo "-c, --clean"
  echo "  Delete the all existing build artifacts before generating the "
  echo "  markdown."

  exit -3
}

LOCAL=0
CLEAN=0

while test ${#} != 0
do
  case "${1}" in
  -h) ;&
  -help) ;&
  --help) usage ;;
  -c) ;&
  --clean) CLEAN=1 ;;
  esac
  shift
done

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

REPO_PATH=${SCRIPT_PATH}/..

BUILD_DOCS_PATH=build_docs
BUILD_DOXYGEN_PATH=${BUILD_DOCS_PATH}/doxygen
BUILD_GITHUB_PAGES_PATH=${BUILD_DOCS_PATH}/github_pages

cd ${REPO_PATH}

if [[ "${CLEAN}" == 1 ]]; then
  rm -rf ${BUILD_DOXYGEN_PATH}
  rm -rf ${BUILD_GITHUB_PAGES_PATH}
fi

mkdir -p ${BUILD_DOXYGEN_PATH}/xml
mkdir -p ${BUILD_GITHUB_PAGES_PATH}
mkdir -p ${BUILD_GITHUB_PAGES_PATH}/api
mkdir -p ${BUILD_GITHUB_PAGES_PATH}/contributing
mkdir -p ${BUILD_GITHUB_PAGES_PATH}/releases

# Copy all the documentation sources and Jekyll configuration into
# `{BUILD_GITHUB_PAGES_PATH}`.
cp -ur docs/github_pages/* ${BUILD_GITHUB_PAGES_PATH}/
cp README.md               ${BUILD_GITHUB_PAGES_PATH}/overview.md
cp CODE_OF_CONDUCT.md      ${BUILD_GITHUB_PAGES_PATH}/contributing/code_of_conduct.md
cp CHANGELOG.md            ${BUILD_GITHUB_PAGES_PATH}/releases/changelog.md

doxygen docs/doxygen/config.dox

# `--debug-templates` will cause JSON output to be generated, which is useful
# for debugging.
doxybook2 --config docs/doxybook/config.json  \
          --templates docs/doxybook/templates \
          --debug-templates                   \
          --input ${BUILD_DOXYGEN_PATH}/xml   \
          --output ${BUILD_GITHUB_PAGES_PATH}/api

# Doxygen and Doxybook don't give us a way to disable all the things we'd like,
# so it's important to purge Doxybook Markdown output that we don't need:
# 0) We want our Jekyll build to be as fast as possible and avoid wasting time
#    on stuff we don't need.
# 1) We don't want content that we don't plan to use to either show up on the
#    site index or appear in search results.
rm -rf ${BUILD_GITHUB_PAGES_PATH}/api/files
rm -rf ${BUILD_GITHUB_PAGES_PATH}/api/index_files.md
rm -rf ${BUILD_GITHUB_PAGES_PATH}/api/pages
rm -rf ${BUILD_GITHUB_PAGES_PATH}/api/index_pages.md
rm -rf ${BUILD_GITHUB_PAGES_PATH}/api/examples
rm -rf ${BUILD_GITHUB_PAGES_PATH}/api/index_examples.md
rm -rf ${BUILD_GITHUB_PAGES_PATH}/api/images
rm -rf ${BUILD_GITHUB_PAGES_PATH}/api/index_namespaces.md
rm -rf ${BUILD_GITHUB_PAGES_PATH}/api/index_groups.md
rm -rf ${BUILD_GITHUB_PAGES_PATH}/api/index_classes.md

