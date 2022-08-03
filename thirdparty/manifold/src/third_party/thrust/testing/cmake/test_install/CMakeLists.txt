# Test that an installation of the project can be located by find_package() call
# with appropriate prefix settings.
#
# Expects THRUST_BINARY_DIR to be set to an existing thrust build directory.

cmake_minimum_required(VERSION 3.15)

project(ThrustTestInstall CXX CUDA)

# This will eventually get deleted recursively -- keep that in mind if modifying
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install_prefix/")

function(do_manual_install)
  # Inspired by the VTK-m install tests, we can just glob up all of the
  # cmake_install.cmake, include (ie. run) them, and they'll effectively
  # install the project into the current value of CMAKE_INSTALL_PREFIX.

  # Gather all of the install files from Thrust's root:
  file(GLOB_RECURSE install_files
    LIST_DIRECTORIES False
    "${THRUST_BINARY_DIR}/cmake_install.cmake"
  )

  message(STATUS "Locating install files...")
  foreach (install_file IN LISTS install_files)
    message(STATUS "  * ${install_file}")
  endforeach()

  message(STATUS "Building install tree...")
  foreach(install_file IN LISTS install_files)
    include("${install_file}")
  endforeach()
endfunction()

function(do_cleanup)
  message(STATUS "Removing ${CMAKE_INSTALL_PREFIX}")
  file(REMOVE_RECURSE "${CMAKE_INSTALL_PREFIX}")
endfunction()

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

function(find_installed_project)
  set(CMAKE_PREFIX_PATH "${CMAKE_INSTALL_PREFIX}")
  find_package(Thrust CONFIG COMPONENTS CPP CUDA)

  if (NOT Thrust_FOUND)
    message(FATAL_ERROR
      "find_package(Thrust) failed. "
      "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
    )
  endif()

  # Test some internal config vars to check that this is the expected install:
  # TODO The cmake_path (3.19) command will provide more robust ways to do this

  # Escape regex special characters in the install prefix, see
  # https://gitlab.kitware.com/cmake/cmake/-/issues/18580
  string(REGEX REPLACE "([][+.*()^])" "\\\\\\1"
    prefix_regex
    "${CMAKE_INSTALL_PREFIX}"
  )
  if (NOT _THRUST_INCLUDE_DIR MATCHES "^${prefix_regex}")
    message(FATAL_ERROR
      "Found Thrust in unexpected location: "
      " * _THRUST_INCLUDE_DIR=${_THRUST_INCLUDE_DIR} "
      " * ExpectedPrefix=${CMAKE_INSTALL_DIR}"
    )
  endif()
  if (NOT _CUB_INCLUDE_DIR MATCHES "^${prefix_regex}")
    message(FATAL_ERROR
      "Found CUB in unexpected location: "
      " * _CUB_INCLUDE_DIR=${_CUB_INCLUDE_DIR} "
      " * ExpectedPrefix=${CMAKE_INSTALL_DIR}"
    )
  endif()

  thrust_create_target(Thrust)
  assert_target(Thrust)
  assert_target(CUB::CUB)
  assert_target(Thrust::CPP::Host)
  assert_target(Thrust::CUDA::Device)

  thrust_update_system_found_flags()
  assert_boolean(THRUST_CPP_FOUND TRUE)
  assert_boolean(THRUST_CUDA_FOUND TRUE)
  assert_boolean(THRUST_OMP_FOUND FALSE)
  assert_boolean(THRUST_TBB_FOUND FALSE)

endfunction()

do_cleanup() # Prepare for new installation
do_manual_install()
find_installed_project()
do_cleanup() # Clean up if successful
