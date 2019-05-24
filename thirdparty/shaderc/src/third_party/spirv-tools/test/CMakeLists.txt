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

# Add a SPIR-V Tools unit test. Signature:
#   add_spvtools_unittest(
#     TARGET target_name
#     SRCS   src_file.h src_file.cpp
#     LIBS   lib1 lib2
#   )

if (NOT "${SPIRV_SKIP_TESTS}")
  if (TARGET gmock_main)
    message(STATUS "Found Google Mock, building tests.")
  else()
    message(STATUS "Did not find googletest, tests will not be built. "
      "To enable tests place googletest in '<spirv-dir>/external/googletest'.")
  endif()
endif()

function(add_spvtools_unittest)
  if (NOT "${SPIRV_SKIP_TESTS}" AND TARGET gmock_main)
    set(one_value_args TARGET PCH_FILE)
    set(multi_value_args SRCS LIBS ENVIRONMENT)
    cmake_parse_arguments(
      ARG "" "${one_value_args}" "${multi_value_args}" ${ARGN})
    set(target test_${ARG_TARGET})
    set(SRC_COPY ${ARG_SRCS})
    if (DEFINED ARG_PCH_FILE)
      spvtools_pch(SRC_COPY ${ARG_PCH_FILE})
    endif()
    add_executable(${target} ${SRC_COPY})
    spvtools_default_compile_options(${target})
    if(${COMPILER_IS_LIKE_GNU})
      target_compile_options(${target} PRIVATE -Wno-undef)
      # Effcee and RE2 headers exhibit shadowing.
      target_compile_options(${target} PRIVATE -Wno-shadow)
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
      # Disable C4503 "decorated name length exceeded" warning,
      # triggered by some heavily templated types.
      # We don't care much about that in test code.
      # Important to do since we have warnings-as-errors.
      target_compile_options(${target} PRIVATE /wd4503)
      # Googletest accidentally turns off support for ::testing::Combine
      # in VS 2017.  See https://github.com/google/googletest/issues/1352
      # Forcibly turn it on again.
      target_compile_options(${target} PRIVATE /DGTEST_HAS_COMBINE=1)
    endif()
    target_include_directories(${target} PRIVATE
      ${SPIRV_HEADER_INCLUDE_DIR}
      ${spirv-tools_SOURCE_DIR}
      ${spirv-tools_SOURCE_DIR}/include
      ${spirv-tools_SOURCE_DIR}/test
      ${spirv-tools_BINARY_DIR}
      ${gtest_SOURCE_DIR}/include
      ${gmock_SOURCE_DIR}/include
    )
    if (TARGET effcee)
      # If using Effcee for testing, then add its include directory.
      target_include_directories(${target} PRIVATE ${effcee_SOURCE_DIR})
    endif()
    target_link_libraries(${target} PRIVATE ${ARG_LIBS})
    if (TARGET effcee)
      target_link_libraries(${target} PRIVATE effcee)
    endif()
    target_link_libraries(${target} PRIVATE gmock_main)
    add_test(NAME spirv-tools-${target} COMMAND ${target})
    if (DEFINED ARG_ENVIRONMENT)
      set_tests_properties(spirv-tools-${target} PROPERTIES ENVIRONMENT ${ARG_ENVIRONMENT})
    endif()
    set_property(TARGET ${target} PROPERTY FOLDER "SPIRV-Tools tests")
  endif()
endfunction()

set(TEST_SOURCES
  test_fixture.h
  unit_spirv.h

  assembly_context_test.cpp
  assembly_format_test.cpp
  binary_destroy_test.cpp
  binary_endianness_test.cpp
  binary_header_get_test.cpp
  binary_parse_test.cpp
  binary_strnlen_s_test.cpp
  binary_to_text_test.cpp
  binary_to_text.literal_test.cpp
  comment_test.cpp
  diagnostic_test.cpp
  enum_string_mapping_test.cpp
  enum_set_test.cpp
  ext_inst.debuginfo_test.cpp
  ext_inst.glsl_test.cpp
  ext_inst.opencl_test.cpp
  fix_word_test.cpp
  generator_magic_number_test.cpp
  hex_float_test.cpp
  immediate_int_test.cpp
  libspirv_macros_test.cpp
  log_test.cpp
  named_id_test.cpp
  name_mapper_test.cpp
  opcode_make_test.cpp
  opcode_require_capabilities_test.cpp
  opcode_split_test.cpp
  opcode_table_get_test.cpp
  operand_capabilities_test.cpp
  operand_test.cpp
  operand_pattern_test.cpp
  parse_number_test.cpp
  preserve_numeric_ids_test.cpp
  software_version_test.cpp
  string_utils_test.cpp
  target_env_test.cpp
  text_advance_test.cpp
  text_destroy_test.cpp
  text_literal_test.cpp
  text_start_new_inst_test.cpp
  text_to_binary.annotation_test.cpp
  text_to_binary.barrier_test.cpp
  text_to_binary.constant_test.cpp
  text_to_binary.control_flow_test.cpp
  text_to_binary_test.cpp
  text_to_binary.debug_test.cpp
  text_to_binary.device_side_enqueue_test.cpp
  text_to_binary.extension_test.cpp
  text_to_binary.function_test.cpp
  text_to_binary.group_test.cpp
  text_to_binary.image_test.cpp
  text_to_binary.literal_test.cpp
  text_to_binary.memory_test.cpp
  text_to_binary.misc_test.cpp
  text_to_binary.mode_setting_test.cpp
  text_to_binary.pipe_storage_test.cpp
  text_to_binary.type_declaration_test.cpp
  text_to_binary.subgroup_dispatch_test.cpp
  text_to_binary.reserved_sampling_test.cpp
  text_word_get_test.cpp

  unit_spirv.cpp
)

spvtools_pch(TEST_SOURCES pch_test)

add_spvtools_unittest(
  TARGET spirv_unit_tests
  SRCS ${TEST_SOURCES}
  LIBS ${SPIRV_TOOLS})

add_spvtools_unittest(
  TARGET c_interface
  SRCS c_interface_test.cpp
  LIBS ${SPIRV_TOOLS})

add_spvtools_unittest(
  TARGET c_interface_shared
  SRCS c_interface_test.cpp
  LIBS ${SPIRV_TOOLS}-shared
  ENVIRONMENT PATH=$<TARGET_FILE_DIR:${SPIRV_TOOLS}-shared>)

add_spvtools_unittest(
  TARGET cpp_interface
  SRCS cpp_interface_test.cpp
  LIBS SPIRV-Tools-opt)

if (${SPIRV_TIMER_ENABLED})
add_spvtools_unittest(
  TARGET timer
  SRCS timer_test.cpp
  LIBS ${SPIRV_TOOLS})
endif()


add_spvtools_unittest(
  TARGET bit_stream
  SRCS bit_stream.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../source/comp/bit_stream.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../source/comp/bit_stream.h
  LIBS ${SPIRV_TOOLS})

add_spvtools_unittest(
  TARGET huffman_codec
  SRCS huffman_codec.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../source/comp/bit_stream.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../source/comp/bit_stream.h
      ${CMAKE_CURRENT_SOURCE_DIR}/../source/comp/huffman_codec.h
  LIBS ${SPIRV_TOOLS})

add_spvtools_unittest(
  TARGET move_to_front
  SRCS move_to_front_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../source/comp/move_to_front.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../source/comp/move_to_front.cpp
  LIBS ${SPIRV_TOOLS})

add_subdirectory(comp)
add_subdirectory(link)
add_subdirectory(opt)
add_subdirectory(reduce)
add_subdirectory(stats)
add_subdirectory(tools)
add_subdirectory(util)
add_subdirectory(val)
