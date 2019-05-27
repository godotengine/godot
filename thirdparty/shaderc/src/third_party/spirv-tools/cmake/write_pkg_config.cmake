# Copyright (c) 2017 Pierre Moreau
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

# First, retrieve the current version from CHANGES
file(STRINGS ${CHANGES_FILE} CHANGES_CONTENT)
string(
REGEX
  MATCH "v[0-9]+(.[0-9]+)?(-dev)? [0-9]+-[0-9]+-[0-9]+"
  FIRST_VERSION_LINE
  ${CHANGES_CONTENT})
string(
REGEX
  REPLACE "^v([^ ]+) .+$" "\\1"
  CURRENT_VERSION
  "${FIRST_VERSION_LINE}")
# If this is a development version, replace "-dev" by ".0" as pkg-config nor
# CMake support "-dev" in the version.
# If it's not a "-dev" version then ensure it ends with ".1"
string(REGEX REPLACE "-dev.1" ".0" CURRENT_VERSION "${CURRENT_VERSION}.1")
configure_file(${TEMPLATE_FILE} ${OUT_FILE} @ONLY)
