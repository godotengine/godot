# 3.15 is the minimum for including the project with add_subdirectory.
# 3.17 for building the project's standalone tests/examples/etc.
# 3.18.3 for C++17 + CUDA
cmake_minimum_required(VERSION 3.15)

# Remove this when we use the new CUDA_ARCHITECTURES properties with both
# nvcc and nvc++.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

project(Thrust NONE)

# Determine whether Thrust is the top-level project or included into
# another project via add_subdirectory()
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(THRUST_TOPLEVEL_PROJECT ON)
else()
  set(THRUST_TOPLEVEL_PROJECT OFF)
endif()

# This must be done before any languages are enabled:
if (THRUST_TOPLEVEL_PROJECT)
  include(cmake/ThrustCompilerHacks.cmake)
endif()

# This must appear after our Compiler Hacks or else CMake will delete the cache
# and reconfigure from scratch.
# This must also appear before the installation rules, as it is required by the
# GNUInstallDirs CMake module.
enable_language(CXX)

# Optionally include installation rules for non-top-level builds:
option(THRUST_ENABLE_INSTALL_RULES "Enable installation of Thrust" ${THRUST_TOPLEVEL_PROJECT})
if (THRUST_ENABLE_INSTALL_RULES)
  include(cmake/ThrustInstallRules.cmake)
endif()

# Support adding Thrust to a parent project via add_subdirectory.
# See examples/cmake/add_subdir/CMakeLists.txt for details.
if (NOT THRUST_TOPLEVEL_PROJECT)
  include(cmake/ThrustAddSubdir.cmake)
  return()
endif()

# We use 3.17 features when building our tests, etc.
cmake_minimum_required(VERSION 3.17)

option(THRUST_ENABLE_HEADER_TESTING "Test that all public headers compile." "ON")
option(THRUST_ENABLE_TESTING "Build Thrust testing suite." "ON")
option(THRUST_ENABLE_EXAMPLES "Build Thrust examples." "ON")
option(THRUST_ENABLE_BENCHMARKS "Build Thrust runtime benchmarks." "OFF")
option(THRUST_INCLUDE_CUB_CMAKE "Build CUB tests and examples. (Requires CUDA)." "OFF")

# Mark this option as advanced for now. We'll revisit this later once the new
# benchmarks are ready. For now, we just need to expose a way to compile
# bench.cu from CMake for NVIDIA's internal builds.
mark_as_advanced(THRUST_ENABLE_BENCHMARKS)

# Check if we're actually building anything before continuing. If not, no need
# to search for deps, etc. This is a common approach for packagers that just
# need the install rules. See GH issue NVIDIA/thrust#1211.
if (NOT (THRUST_ENABLE_HEADER_TESTING OR
         THRUST_ENABLE_TESTING OR
         THRUST_ENABLE_EXAMPLES OR
         THRUST_ENABLE_BENCHMARKS OR
         THRUST_INCLUDE_CUB_CMAKE))
  return()
endif()

include(cmake/AppendOptionIfAvailable.cmake)
include(cmake/ThrustBuildCompilerTargets.cmake)
include(cmake/ThrustBuildTargetList.cmake)
include(cmake/ThrustFindThrust.cmake)
include(cmake/ThrustMultiConfig.cmake)
include(cmake/ThrustUtilities.cmake)

# Add cache string options for CMAKE_BUILD_TYPE and default to RelWithDebInfo.
if ("" STREQUAL "${CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)

  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel
  )
endif ()

# Disable compiler extensions:
set(CMAKE_CXX_EXTENSIONS OFF)

# Where to put build outputs. Use CMAKE_BINARY_DIR so they'll show up in the
# top-level project's dir when building Thrust via add_subdirectory.
set(THRUST_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
set(THRUST_EXECUTABLE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

thrust_configure_multiconfig()
thrust_find_thrust()
thrust_build_compiler_targets()
thrust_update_system_found_flags()
if (THRUST_CUDA_FOUND)
  include(cmake/ThrustCudaConfig.cmake)
endif()
thrust_build_target_list()

message(STATUS "CPP system found?  ${THRUST_CPP_FOUND}")
message(STATUS "CUDA system found? ${THRUST_CUDA_FOUND}")
message(STATUS "TBB system found?  ${THRUST_TBB_FOUND}")
message(STATUS "OMP system found?  ${THRUST_OMP_FOUND}")

if (THRUST_ENABLE_HEADER_TESTING)
  include(cmake/ThrustHeaderTesting.cmake)
endif()

# Both testing and examples use ctest
if (THRUST_ENABLE_TESTING OR THRUST_ENABLE_EXAMPLES)
  include(CTest)
  enable_testing()
endif()

if (THRUST_ENABLE_TESTING)
  add_subdirectory(testing)
endif()

if (THRUST_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (THRUST_ENABLE_BENCHMARKS)
  add_subdirectory(internal/benchmark)
endif()

if (THRUST_INCLUDE_CUB_CMAKE AND THRUST_CUDA_FOUND)
  set(CUB_IN_THRUST ON)
  # CUB's path is specified generically to support both GitHub and Perforce
  # source tree layouts. The include directory used by cub-config.cmake
  # for source layouts is the same as the project root.
  add_subdirectory("${_CUB_INCLUDE_DIR}" dependencies/cub)
endif()
