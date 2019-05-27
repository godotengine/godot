# Copyright (c) 2015-2016 The Khronos Group Inc.
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

add_subdirectory(lesspipe)
add_subdirectory(emacs)

# Add a SPIR-V Tools command line tool. Signature:
#   add_spvtools_tool(
#     TARGET target_name
#     SRCS   src_file1.cpp src_file2.cpp
#     LIBS   lib_target1 lib_target2
#   )
function(add_spvtools_tool)
  set(one_value_args TARGET)
  set(multi_value_args SRCS LIBS)
  cmake_parse_arguments(
    ARG "" "${one_value_args}" "${multi_value_args}" ${ARGN})

  add_executable(${ARG_TARGET} ${ARG_SRCS})
  spvtools_default_compile_options(${ARG_TARGET})
  target_link_libraries(${ARG_TARGET} PRIVATE ${ARG_LIBS})
  target_include_directories(${ARG_TARGET} PRIVATE
    ${spirv-tools_SOURCE_DIR}
    ${spirv-tools_BINARY_DIR}
  )
  set_property(TARGET ${ARG_TARGET} PROPERTY FOLDER "SPIRV-Tools executables")
endfunction()

if (NOT ${SPIRV_SKIP_EXECUTABLES})
  add_spvtools_tool(TARGET spirv-as SRCS as/as.cpp LIBS ${SPIRV_TOOLS})
  add_spvtools_tool(TARGET spirv-dis SRCS dis/dis.cpp LIBS ${SPIRV_TOOLS})
  add_spvtools_tool(TARGET spirv-val SRCS val/val.cpp util/cli_consumer.cpp LIBS ${SPIRV_TOOLS})
  add_spvtools_tool(TARGET spirv-opt SRCS opt/opt.cpp util/cli_consumer.cpp LIBS SPIRV-Tools-opt ${SPIRV_TOOLS})
  if (NOT DEFINED IOS_PLATFORM) # iOS does not allow std::system calls which spirv-reduce requires
    add_spvtools_tool(TARGET spirv-reduce SRCS reduce/reduce.cpp util/cli_consumer.cpp LIBS SPIRV-Tools-reduce ${SPIRV_TOOLS})
  endif()
  add_spvtools_tool(TARGET spirv-link SRCS link/linker.cpp LIBS SPIRV-Tools-link ${SPIRV_TOOLS})
  add_spvtools_tool(TARGET spirv-stats
	            SRCS stats/stats.cpp
		               stats/stats_analyzer.cpp
                   stats/stats_analyzer.h
                   stats/spirv_stats.cpp
                   stats/spirv_stats.h
		    LIBS ${SPIRV_TOOLS})
  add_spvtools_tool(TARGET spirv-cfg
                    SRCS cfg/cfg.cpp
                         cfg/bin_to_dot.h
                         cfg/bin_to_dot.cpp
                    LIBS ${SPIRV_TOOLS})
  target_include_directories(spirv-cfg PRIVATE ${spirv-tools_SOURCE_DIR}
                                               ${SPIRV_HEADER_INCLUDE_DIR})
  target_include_directories(spirv-stats PRIVATE ${spirv-tools_SOURCE_DIR}
                                                 ${SPIRV_HEADER_INCLUDE_DIR})

  set(SPIRV_INSTALL_TARGETS spirv-as spirv-dis spirv-val spirv-opt spirv-stats
                            spirv-cfg spirv-link)
  if(NOT DEFINED IOS_PLATFORM)
    set(SPIRV_INSTALL_TARGETS ${SPIRV_INSTALL_TARGETS} spirv-reduce)
  endif()

  if(SPIRV_BUILD_COMPRESSION)
    add_spvtools_tool(TARGET spirv-markv
                      SRCS comp/markv.cpp
                           comp/markv_model_factory.cpp
                           comp/markv_model_shader.cpp
	              LIBS SPIRV-Tools-comp SPIRV-Tools-opt ${SPIRV_TOOLS})
    target_include_directories(spirv-markv PRIVATE ${spirv-tools_SOURCE_DIR}
                                                   ${SPIRV_HEADER_INCLUDE_DIR})
    set(SPIRV_INSTALL_TARGETS ${SPIRV_INSTALL_TARGETS} spirv-markv)
  endif(SPIRV_BUILD_COMPRESSION)

  if(ENABLE_SPIRV_TOOLS_INSTALL)
    install(TARGETS ${SPIRV_INSTALL_TARGETS}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif(ENABLE_SPIRV_TOOLS_INSTALL)
endif()
