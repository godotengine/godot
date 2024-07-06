# Copyright (c) 2017 Pierre Moreau

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
add_library(SPIRV-Tools-link ${SPIRV_TOOLS_LIBRARY_TYPE}
  linker.cpp
)

spvtools_default_compile_options(SPIRV-Tools-link)
target_include_directories(SPIRV-Tools-link
  PUBLIC
    $<BUILD_INTERFACE:${spirv-tools_SOURCE_DIR}/include>
	$<BUILD_INTERFACE:${SPIRV_HEADER_INCLUDE_DIR}>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${spirv-tools_BINARY_DIR}
)
# We need the IR functionalities from the optimizer
target_link_libraries(SPIRV-Tools-link
  PUBLIC SPIRV-Tools-opt)

set_property(TARGET SPIRV-Tools-link PROPERTY FOLDER "SPIRV-Tools libraries")
spvtools_check_symbol_exports(SPIRV-Tools-link)

if(ENABLE_SPIRV_TOOLS_INSTALL)
  install(TARGETS SPIRV-Tools-link EXPORT SPIRV-Tools-linkTargets)
  export(EXPORT SPIRV-Tools-linkTargets FILE SPIRV-Tools-linkTargets.cmake)

  spvtools_config_package_dir(SPIRV-Tools-link PACKAGE_DIR)
  install(EXPORT SPIRV-Tools-linkTargets FILE SPIRV-Tools-linkTargets.cmake
  	DESTINATION ${PACKAGE_DIR})

  spvtools_generate_config_file(SPIRV-Tools-link)
  install(FILES ${CMAKE_BINARY_DIR}/SPIRV-Tools-linkConfig.cmake DESTINATION ${PACKAGE_DIR})
endif(ENABLE_SPIRV_TOOLS_INSTALL)
