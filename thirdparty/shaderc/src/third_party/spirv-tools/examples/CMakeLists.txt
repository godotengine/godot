# Copyright (c) 2016 Google Inc.
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

# Add a SPIR-V Tools example. Signature:
#   add_spvtools_example(
#     TARGET target_name
#     SRCS   src_file1.cpp src_file2.cpp
#     LIBS   lib_target1 lib_target2
#   )
function(add_spvtools_example)
  if (NOT ${SPIRV_SKIP_EXECUTABLES})
    set(one_value_args TARGET)
    set(multi_value_args SRCS LIBS)
    cmake_parse_arguments(
        ARG "" "${one_value_args}" "${multi_value_args}" ${ARGN})

    add_executable(${ARG_TARGET} ${ARG_SRCS})
    spvtools_default_compile_options(${ARG_TARGET})
    target_link_libraries(${ARG_TARGET} PRIVATE ${ARG_LIBS})
    set_property(TARGET ${ARG_TARGET} PROPERTY FOLDER "SPIRV-Tools examples")
  endif()
endfunction()

add_subdirectory(cpp-interface)
