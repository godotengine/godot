# Copyright (c) 2018 Google LLC.
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

if(NOT ${SPIRV_SKIP_TESTS})
  if(${PYTHONINTERP_FOUND})
    add_test(NAME spirv_opt_cli_tools_tests
      COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/../spirv_test_framework.py
      $<TARGET_FILE:spirv-opt> $<TARGET_FILE:spirv-as> $<TARGET_FILE:spirv-dis>
      --test-dir ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    message("Skipping CLI tools tests - Python executable not found")
  endif()
endif()
