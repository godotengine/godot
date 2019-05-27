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


add_spvtools_unittest(TARGET opt_loops
  SRCS ../function_utils.h
       dependence_analysis.cpp
       dependence_analysis_helpers.cpp
       fusion_compatibility.cpp
       fusion_illegal.cpp
       fusion_legal.cpp
       fusion_pass.cpp
       hoist_all_loop_types.cpp
       hoist_double_nested_loops.cpp
       hoist_from_independent_loops.cpp
       hoist_simple_case.cpp
       hoist_single_nested_loops.cpp
       hoist_without_preheader.cpp
       lcssa.cpp
       loop_descriptions.cpp
       loop_fission.cpp
       nested_loops.cpp
       peeling.cpp
       peeling_pass.cpp
       unroll_assumptions.cpp
       unroll_simple.cpp
       unswitch.cpp
  LIBS SPIRV-Tools-opt
  PCH_FILE pch_test_opt_loop
)
