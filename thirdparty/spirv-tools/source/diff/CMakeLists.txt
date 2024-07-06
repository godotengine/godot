# Copyright (c) 2022 Google LLC.

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
set(SPIRV_TOOLS_DIFF_SOURCES
  diff.h
  lcs.h

  diff.cpp
)

add_library(SPIRV-Tools-diff ${SPIRV_TOOLS_LIBRARY_TYPE} ${SPIRV_TOOLS_DIFF_SOURCES})

spvtools_default_compile_options(SPIRV-Tools-diff)
target_include_directories(SPIRV-Tools-diff
  PUBLIC
	$<BUILD_INTERFACE:${spirv-tools_SOURCE_DIR}/include>
	$<BUILD_INTERFACE:${SPIRV_HEADER_INCLUDE_DIR}>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${spirv-tools_BINARY_DIR}
)
# We need the assembling and disassembling functionalities in the main library.
target_link_libraries(SPIRV-Tools-diff
  PUBLIC ${SPIRV_TOOLS_FULL_VISIBILITY})
# We need the internals of spirv-opt.
target_link_libraries(SPIRV-Tools-diff
  PUBLIC SPIRV-Tools-opt)

set_property(TARGET SPIRV-Tools-diff PROPERTY FOLDER "SPIRV-Tools libraries")
spvtools_check_symbol_exports(SPIRV-Tools-diff)

if(ENABLE_SPIRV_TOOLS_INSTALL)
  install(TARGETS SPIRV-Tools-diff EXPORT SPIRV-Tools-diffTargets)
  export(EXPORT SPIRV-Tools-diffTargets FILE SPIRV-Tools-diffTargets.cmake)

  spvtools_config_package_dir(SPIRV-Tools-diff PACKAGE_DIR)
  install(EXPORT SPIRV-Tools-diffTargets FILE SPIRV-Tools-diffTargets.cmake
  	DESTINATION ${PACKAGE_DIR})

  spvtools_generate_config_file(SPIRV-Tools-diff)
  install(FILES ${CMAKE_BINARY_DIR}/SPIRV-Tools-diffConfig.cmake DESTINATION ${PACKAGE_DIR})
endif(ENABLE_SPIRV_TOOLS_INSTALL)
