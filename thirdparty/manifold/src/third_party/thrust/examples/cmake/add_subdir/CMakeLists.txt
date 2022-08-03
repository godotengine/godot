# This example demonstrates / tests adding thrust via a CMake add_subdirectory
# call from a parent project.
#
# The variables THRUST_REQUIRED_SYSTEMS and THRUST_OPTIONAL_SYSTEMS must be
# set prior to add_subdirectory(thrust), and afterwards the thrust_create_target
# function may be used to create targets with the desired systems. See
# NVIDIA/thrust/cmake/README.md for more details on thrust_create_target.

cmake_minimum_required(VERSION 3.15)

# Silence warnings about empty CUDA_ARCHITECTURES properties on example targets:
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

project(ThrustAddSubDirExample CXX)

# Add required Thrust systems to THRUST_REQUIRED_SYSTEMS.
# Options are: CPP, CUDA, TBB or OMP.
# Multiple systems may be specified.
# An error is emitted if the system is not found.
set(THRUST_REQUIRED_SYSTEMS CPP)

# Add optional Thrust systems to THRUST_OPTIONAL_SYSTEMS.
# Options are: CPP, CUDA, TBB or OMP.
# Multiple systems may be specified.
# No error is emitted if not found.
set(THRUST_OPTIONAL_SYSTEMS CUDA)

# Use your project's checkout of Thrust here, for most cases
# `add_subdirectory(thrust)` will be sufficient.
add_subdirectory("${THRUST_ROOT}" thrust)

# Create a thrust target that only uses the serial CPP backend.
# See thrust/thrust/cmake/README.md for details and additional options:
thrust_create_target(ThrustCPP HOST CPP DEVICE CPP)

# Create an executable that uses the CPP-only thrust target:
add_executable(ExecWithCPP dummy.cpp)
target_link_libraries(ExecWithCPP ThrustCPP)

# To test for optional systems, first call thrust_update_system_found_flags to
# set the THRUST_${system}_FOUND flags in current scope.
# Required due to CMake scoping rules.
thrust_update_system_found_flags()

# Create and use a Thrust target configured to use CUDA acceleration if CUDA
# is available:
if (THRUST_CUDA_FOUND)
  enable_language(CUDA)
  thrust_create_target(ThrustCUDA HOST CPP DEVICE CUDA)
  add_executable(ExecWithCUDA dummy.cu)
  target_link_libraries(ExecWithCUDA ThrustCUDA)
endif()

#
# Validation
#

function(assert_boolean var_name expect)
  if (expect)
    if (NOT ${var_name})
      message(FATAL_ERROR "'${var_name}' is false, expected true.")
    endif()
  else()
    if (${var_name})
      message(FATAL_ERROR "'${var_name}' is true, expected false.")
    endif()
  endif()
endfunction()

function(assert_target target_name)
  if (NOT TARGET "${target_name}")
    message(FATAL_ERROR "Target '${target_name}' not defined.")
  endif()
endfunction()

assert_boolean(THRUST_CPP_FOUND TRUE)
assert_boolean(THRUST_CUDA_FOUND TRUE)
assert_boolean(THRUST_OMP_FOUND FALSE)
assert_boolean(THRUST_TBB_FOUND FALSE)

assert_target(ThrustCPP)
assert_target(ThrustCUDA)
assert_target(ExecWithCPP)
assert_target(ExecWithCUDA)

thrust_debug_target(ThrustCPP "")
thrust_debug_target(ThrustCUDA "")
thrust_debug_target(ExecWithCPP "")
thrust_debug_target(ExecWithCUDA "")
