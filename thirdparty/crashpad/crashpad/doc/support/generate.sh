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

function maybe_mkdir() {
  local dir="${1}"
  if [[ ! -d "${dir}" ]]; then
    mkdir "${dir}"
  fi
}

# Run from the Crashpad project root directory.
cd "$(dirname "${0}")/../.."

source doc/support/compat.sh

python doc/support/generate_doxygen.py

output_dir=doc/generated
maybe_mkdir "${output_dir}"

maybe_mkdir "${output_dir}/doxygen"
rsync -Ilr --delete --exclude .git "out/doc/doxygen/html/" \
    "${output_dir}/doxygen"

# Ensure a favicon exists at the root since the browser will always request it.
cp doc/favicon.ico "${output_dir}/"
