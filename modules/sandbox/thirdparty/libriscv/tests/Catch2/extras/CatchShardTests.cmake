
#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)

# SPDX-License-Identifier: BSL-1.0

# Supported optional args:
#  * SHARD_COUNT - number of shards to split target's tests into
#  * REPORTER    - reporter spec to use for tests
#  * TEST_SPEC   - test spec used for filtering tests
function(catch_add_sharded_tests TARGET)
  if(${CMAKE_VERSION} VERSION_LESS "3.10.0")
    message(FATAL_ERROR "add_sharded_catch_tests only supports CMake versions 3.10.0 and up")
  endif()

  cmake_parse_arguments(
    ""
    ""
    "SHARD_COUNT;REPORTER;TEST_SPEC"
    ""
    ${ARGN}
  )
  
  if(NOT DEFINED _SHARD_COUNT)
    set(_SHARD_COUNT 2)
  endif()

  # Generate a unique name based on the extra arguments
  string(SHA1 args_hash "${_TEST_SPEC} ${_EXTRA_ARGS} ${_REPORTER} ${_OUTPUT_DIR} ${_OUTPUT_PREFIX} ${_OUTPUT_SUFFIX} ${_SHARD_COUNT}")
  string(SUBSTRING ${args_hash} 0 7 args_hash)

  set(ctest_include_file "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}-sharded-tests-include-${args_hash}.cmake")
  set(ctest_tests_file "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}-sharded-tests-impl-${args_hash}.cmake")

  file(WRITE "${ctest_include_file}"
    "if(EXISTS \"${ctest_tests_file}\")\n"
    "  include(\"${ctest_tests_file}\")\n"
    "else()\n"
    "  add_test(${TARGET}_NOT_BUILT-${args_hash} ${TARGET}_NOT_BUILT-${args_hash})\n"
    "endif()\n"
  )

  set_property(DIRECTORY
    APPEND PROPERTY TEST_INCLUDE_FILES "${ctest_include_file}"
  )

  set(shard_impl_script_file "${_CATCH_DISCOVER_SHARD_TESTS_IMPL_SCRIPT}")

  add_custom_command(
    TARGET ${TARGET} POST_BUILD
    BYPRODUCTS "${ctest_tests_file}"
    COMMAND "${CMAKE_COMMAND}"
            -D "TARGET_NAME=${TARGET}"
            -D "TEST_BINARY=$<TARGET_FILE:${TARGET}>"
            -D "CTEST_FILE=${ctest_tests_file}"
            -D "SHARD_COUNT=${_SHARD_COUNT}"
            -D "REPORTER_SPEC=${_REPORTER}"
            -D "TEST_SPEC=${_TEST_SPEC}"
            -P "${shard_impl_script_file}"
    VERBATIM
  )
endfunction()


###############################################################################

set(_CATCH_DISCOVER_SHARD_TESTS_IMPL_SCRIPT
    ${CMAKE_CURRENT_LIST_DIR}/CatchShardTestsImpl.cmake
  CACHE INTERNAL "Catch2 full path to CatchShardTestsImpl.cmake helper file"
)
