thrust_update_system_found_flags()

set(extra_cmake_flags)

# Need to pass these when testing NVC++.
if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(extra_cmake_flags
    -D "CMAKE_CUDA_COMPILER_ID=${CMAKE_CUDA_COMPILER_ID}"
    -D "CMAKE_CUDA_COMPILER_FORCED=${CMAKE_CUDA_COMPILER_FORCED}"
  )
endif()

if (THRUST_CPP_FOUND AND THRUST_CUDA_FOUND)
  # Test that we can use `find_package` on an installed Thrust:
  add_test(
    NAME thrust.test.cmake.test_install
    COMMAND "${CMAKE_COMMAND}"
      --log-level=VERBOSE
      -G "${CMAKE_GENERATOR}"
      -S "${CMAKE_CURRENT_SOURCE_DIR}/test_install"
      -B "${CMAKE_CURRENT_BINARY_DIR}/test_install"
      -D "THRUST_BINARY_DIR=${Thrust_BINARY_DIR}"
      -D "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      -D "CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
      -D "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      ${extra_cmake_flags}
  )
endif()

# Check source code for issues that can be found by pattern matching:
add_test(
  NAME thrust.test.cmake.check_source_files
  COMMAND
    "${CMAKE_COMMAND}"
      -D "Thrust_SOURCE_DIR=${Thrust_SOURCE_DIR}"
      -P "${CMAKE_CURRENT_LIST_DIR}/check_source_files.cmake"
)
