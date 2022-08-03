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
  # Do a basic check of the cmake/ThrustAddSubdir.cmake mechanism:
  add_test(
    NAME thrust.example.cmake.add_subdir
    COMMAND "${CMAKE_COMMAND}"
      --log-level=VERBOSE
      -G "${CMAKE_GENERATOR}"
      -S "${CMAKE_CURRENT_SOURCE_DIR}/add_subdir"
      -B "${CMAKE_CURRENT_BINARY_DIR}/add_subdir"
      -D "THRUST_ROOT=${Thrust_SOURCE_DIR}"
      -D "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      -D "CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
      -D "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      ${extra_cmake_flags}
  )
endif()
