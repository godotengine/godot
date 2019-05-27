#!/bin/bash
# Copyright (c) 2017 Google Inc.

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
#
# Script to determine if source code in Pull Request is properly formatted.
# Exits with non 0 exit code if formatting is needed.
#
# This script assumes to be invoked at the project root directory.

FILES_TO_CHECK=$(git diff --name-only master | grep -E ".*\.(cpp|cc|c\+\+|cxx|c|h|hpp)$")

if [ -z "${FILES_TO_CHECK}" ]; then
  echo "No source code to check for formatting."
  exit 0
fi

FORMAT_DIFF=$(git diff -U0 master -- ${FILES_TO_CHECK} | python ./utils/clang-format-diff.py -p1 -style=file)

if [ -z "${FORMAT_DIFF}" ]; then
  echo "All source code in PR properly formatted."
  exit 0
else
  echo "Found formatting errors!"
  echo "${FORMAT_DIFF}"
  exit 1
fi
