# Copyright (c) 2021 Google LLC.

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
set(SPIRV_TOOLS_LINT_SOURCES
  divergence_analysis.h
  lints.h

  linter.cpp
  divergence_analysis.cpp
  lint_divergent_derivatives.cpp
)

if(MSVC AND (NOT ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")))
  # Enable parallel builds across four cores for this lib.
  add_definitions(/MP4)
endif()

add_library(SPIRV-Tools-lint ${SPIRV_TOOLS_LIBRARY_TYPE} ${SPIRV_TOOLS_LINT_SOURCES})

spvtools_default_compile_options(SPIRV-Tools-lint)
target_include_directories(SPIRV-Tools-lint
  PUBLIC
	$<BUILD_INTERFACE:${spirv-tools_SOURCE_DIR}/include>
	$<BUILD_INTERFACE:${SPIRV_HEADER_INCLUDE_DIR}>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${spirv-tools_BINARY_DIR}
)
# We need the assembling and disassembling functionalities in the main library.
target_link_libraries(SPIRV-Tools-lint
  PUBLIC ${SPIRV_TOOLS_FULL_VISIBILITY})
# We need the internals of spirv-opt.
target_link_libraries(SPIRV-Tools-lint
  PUBLIC SPIRV-Tools-opt)

set_property(TARGET SPIRV-Tools-lint PROPERTY FOLDER "SPIRV-Tools libraries")
spvtools_check_symbol_exports(SPIRV-Tools-lint)

if(ENABLE_SPIRV_TOOLS_INSTALL)
  install(TARGETS SPIRV-Tools-lint EXPORT SPIRV-Tools-lintTargets)
  export(EXPORT SPIRV-Tools-lintTargets FILE SPIRV-Tools-lintTargets.cmake)

  spvtools_config_package_dir(SPIRV-Tools-lint PACKAGE_DIR)
  install(EXPORT SPIRV-Tools-lintTargets FILE SPIRV-Tools-lintTargets.cmake
  	DESTINATION ${PACKAGE_DIR})

  spvtools_generate_config_file(SPIRV-Tools-lint)
  install(FILES ${CMAKE_BINARY_DIR}/SPIRV-Tools-lintConfig.cmake DESTINATION ${PACKAGE_DIR})
endif(ENABLE_SPIRV_TOOLS_INSTALL)
