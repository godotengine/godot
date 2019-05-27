# Copyright (c) 2017 Google Inc.
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


add_spvtools_unittest(TARGET dominator_analysis
  SRCS ../function_utils.h
       common_dominators.cpp
       generated.cpp
       nested_ifs.cpp
       nested_ifs_post.cpp
       nested_loops.cpp
       nested_loops_with_unreachables.cpp
       post.cpp
       simple.cpp
       switch_case_fallthrough.cpp
       unreachable_for.cpp
       unreachable_for_post.cpp
  LIBS SPIRV-Tools-opt
  PCH_FILE pch_test_opt_dom
)
