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


add_spvtools_unittest(TARGET link
  SRCS
       binary_version_test.cpp
       entry_points_test.cpp
       global_values_amount_test.cpp
       ids_limit_test.cpp
       matching_imports_to_exports_test.cpp
       memory_model_test.cpp
       partial_linkage_test.cpp
       unique_ids_test.cpp
  LIBS SPIRV-Tools-opt SPIRV-Tools-link
)
