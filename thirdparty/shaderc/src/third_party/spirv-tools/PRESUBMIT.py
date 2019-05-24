# Copyright (c) 2018 The Khronos Group Inc.
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

"""Presubmit script for SPIRV-Tools.

See http://dev.chromium.org/developers/how-tos/depottools/presubmit-scripts
for more details about the presubmit API built into depot_tools.
"""

LINT_FILTERS = [
  "-build/storage_class",
  "-readability/casting",
  "-readability/fn_size",
  "-readability/todo",
  "-runtime/explicit",
  "-runtime/int",
  "-runtime/printf",
  "-runtime/references",
  "-runtime/string",
]


def CheckChangeOnUpload(input_api, output_api):
  results = []
  results += input_api.canned_checks.CheckPatchFormatted(input_api, output_api)
  results += input_api.canned_checks.CheckChangeLintsClean(
      input_api, output_api, None, LINT_FILTERS)

  return results
