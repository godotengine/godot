# CMakeLists.txt -- Build system for the pybind11 modules
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 3.4)

# The `cmake_minimum_required(VERSION 3.4...3.22)` syntax does not work with
# some versions of VS that have a patched CMake 3.11. This forces us to emulate
# the behavior using the following workaround:
if(${CMAKE_VERSION} VERSION_LESS 3.22)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.22)
endif()

# Avoid infinite recursion if tests include this as a subdirectory
if(DEFINED PYBIND11_MASTER_PROJECT)
  return()
endif()

# Extract project version from source
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/pybind11/detail/common.h"
     pybind11_version_defines REGEX "#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) ")

foreach(ver ${pybind11_version_defines})
  if(ver MATCHES [[#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$]])
    set(PYBIND11_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
  endif()
endforeach()

if(PYBIND11_VERSION_PATCH MATCHES [[\.([a-zA-Z0-9]+)$]])
  set(pybind11_VERSION_TYPE "${CMAKE_MATCH_1}")
endif()
string(REGEX MATCH "^[0-9]+" PYBIND11_VERSION_PATCH "${PYBIND11_VERSION_PATCH}")

project(
  pybind11
  LANGUAGES CXX
  VERSION "${PYBIND11_VERSION_MAJOR}.${PYBIND11_VERSION_MINOR}.${PYBIND11_VERSION_PATCH}")

# Standard includes
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)

if(NOT pybind11_FIND_QUIETLY)
  message(STATUS "pybind11 v${pybind11_VERSION} ${pybind11_VERSION_TYPE}")
endif()

# Check if pybind11 is being used directly or via add_subdirectory
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  ### Warn if not an out-of-source builds
  if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
    set(lines
        "You are building in-place. If that is not what you intended to "
        "do, you can clean the source directory with:\n"
        "rm -r CMakeCache.txt CMakeFiles/ cmake_uninstall.cmake pybind11Config.cmake "
        "pybind11ConfigVersion.cmake tests/CMakeFiles/\n")
    message(AUTHOR_WARNING ${lines})
  endif()

  set(PYBIND11_MASTER_PROJECT ON)

  if(OSX AND CMAKE_VERSION VERSION_LESS 3.7)
    # Bug in macOS CMake < 3.7 is unable to download catch
    message(WARNING "CMAKE 3.7+ needed on macOS to download catch, and newer HIGHLY recommended")
  elseif(WINDOWS AND CMAKE_VERSION VERSION_LESS 3.8)
    # Only tested with 3.8+ in CI.
    message(WARNING "CMAKE 3.8+ tested on Windows, previous versions untested")
  endif()

  message(STATUS "CMake ${CMAKE_VERSION}")

  if(CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif()

  set(pybind11_system "")

  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
else()
  set(PYBIND11_MASTER_PROJECT OFF)
  set(pybind11_system SYSTEM)
endif()

# Options
option(PYBIND11_INSTALL "Install pybind11 header files?" ${PYBIND11_MASTER_PROJECT})
option(PYBIND11_TEST "Build pybind11 test suite?" ${PYBIND11_MASTER_PROJECT})
option(PYBIND11_NOPYTHON "Disable search for Python" OFF)
option(PYBIND11_SIMPLE_GIL_MANAGEMENT
       "Use simpler GIL management logic that does not support disassociation" OFF)
set(PYBIND11_INTERNALS_VERSION
    ""
    CACHE STRING "Override the ABI version, may be used to enable the unstable ABI.")

if(PYBIND11_SIMPLE_GIL_MANAGEMENT)
  add_compile_definitions(PYBIND11_SIMPLE_GIL_MANAGEMENT)
endif()

cmake_dependent_option(
  USE_PYTHON_INCLUDE_DIR
  "Install pybind11 headers in Python include directory instead of default installation prefix"
  OFF "PYBIND11_INSTALL" OFF)

cmake_dependent_option(PYBIND11_FINDPYTHON "Force new FindPython" OFF
                       "NOT CMAKE_VERSION VERSION_LESS 3.12" OFF)

# NB: when adding a header don't forget to also add it to setup.py
set(PYBIND11_HEADERS
    include/pybind11/detail/class.h
    include/pybind11/detail/common.h
    include/pybind11/detail/descr.h
    include/pybind11/detail/init.h
    include/pybind11/detail/internals.h
    include/pybind11/detail/type_caster_base.h
    include/pybind11/detail/typeid.h
    include/pybind11/attr.h
    include/pybind11/buffer_info.h
    include/pybind11/cast.h
    include/pybind11/chrono.h
    include/pybind11/common.h
    include/pybind11/complex.h
    include/pybind11/options.h
    include/pybind11/eigen.h
    include/pybind11/eigen/matrix.h
    include/pybind11/eigen/tensor.h
    include/pybind11/embed.h
    include/pybind11/eval.h
    include/pybind11/gil.h
    include/pybind11/iostream.h
    include/pybind11/functional.h
    include/pybind11/numpy.h
    include/pybind11/operators.h
    include/pybind11/pybind11.h
    include/pybind11/pytypes.h
    include/pybind11/stl.h
    include/pybind11/stl_bind.h
    include/pybind11/stl/filesystem.h)

# Compare with grep and warn if mismatched
if(PYBIND11_MASTER_PROJECT AND NOT CMAKE_VERSION VERSION_LESS 3.12)
  file(
    GLOB_RECURSE _pybind11_header_check
    LIST_DIRECTORIES false
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    CONFIGURE_DEPENDS "include/pybind11/*.h")
  set(_pybind11_here_only ${PYBIND11_HEADERS})
  set(_pybind11_disk_only ${_pybind11_header_check})
  list(REMOVE_ITEM _pybind11_here_only ${_pybind11_header_check})
  list(REMOVE_ITEM _pybind11_disk_only ${PYBIND11_HEADERS})
  if(_pybind11_here_only)
    message(AUTHOR_WARNING "PYBIND11_HEADERS has extra files:" ${_pybind11_here_only})
  endif()
  if(_pybind11_disk_only)
    message(AUTHOR_WARNING "PYBIND11_HEADERS is missing files:" ${_pybind11_disk_only})
  endif()
endif()

# CMake 3.12 added list(TRANSFORM <list> PREPEND
# But we can't use it yet
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/" PYBIND11_HEADERS
               "${PYBIND11_HEADERS}")

# Cache variable so this can be used in parent projects
set(pybind11_INCLUDE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/include"
    CACHE INTERNAL "Directory where pybind11 headers are located")

# Backward compatible variable for add_subdirectory mode
if(NOT PYBIND11_MASTER_PROJECT)
  set(PYBIND11_INCLUDE_DIR
      "${pybind11_INCLUDE_DIR}"
      CACHE INTERNAL "")
endif()

# Note: when creating targets, you cannot use if statements at configure time -
# you need generator expressions, because those will be placed in the target file.
# You can also place ifs *in* the Config.in, but not here.

# This section builds targets, but does *not* touch Python
# Non-IMPORT targets cannot be defined twice
if(NOT TARGET pybind11_headers)
  # Build the headers-only target (no Python included):
  # (long name used here to keep this from clashing in subdirectory mode)
  add_library(pybind11_headers INTERFACE)
  add_library(pybind11::pybind11_headers ALIAS pybind11_headers) # to match exported target
  add_library(pybind11::headers ALIAS pybind11_headers) # easier to use/remember

  target_include_directories(
    pybind11_headers ${pybind11_system} INTERFACE $<BUILD_INTERFACE:${pybind11_INCLUDE_DIR}>
                                                  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  target_compile_features(pybind11_headers INTERFACE cxx_inheriting_constructors cxx_user_literals
                                                     cxx_right_angle_brackets)
  if(NOT "${PYBIND11_INTERNALS_VERSION}" STREQUAL "")
    target_compile_definitions(
      pybind11_headers INTERFACE "PYBIND11_INTERNALS_VERSION=${PYBIND11_INTERNALS_VERSION}")
  endif()
else()
  # It is invalid to install a target twice, too.
  set(PYBIND11_INSTALL OFF)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/tools/pybind11Common.cmake")
# https://github.com/jtojnar/cmake-snips/#concatenating-paths-when-building-pkg-config-files
# TODO: cmake 3.20 adds the cmake_path() function, which obsoletes this snippet
include("${CMAKE_CURRENT_SOURCE_DIR}/tools/JoinPaths.cmake")

# Relative directory setting
if(USE_PYTHON_INCLUDE_DIR AND DEFINED Python_INCLUDE_DIRS)
  file(RELATIVE_PATH CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX} ${Python_INCLUDE_DIRS})
elseif(USE_PYTHON_INCLUDE_DIR AND DEFINED PYTHON_INCLUDE_DIR)
  file(RELATIVE_PATH CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX} ${PYTHON_INCLUDE_DIRS})
endif()

if(PYBIND11_INSTALL)
  install(DIRECTORY ${pybind11_INCLUDE_DIR}/pybind11 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  set(PYBIND11_CMAKECONFIG_INSTALL_DIR
      "${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}"
      CACHE STRING "install path for pybind11Config.cmake")

  if(IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
    set(pybind11_INCLUDEDIR "${CMAKE_INSTALL_FULL_INCLUDEDIR}")
  else()
    set(pybind11_INCLUDEDIR "\$\{PACKAGE_PREFIX_DIR\}/${CMAKE_INSTALL_INCLUDEDIR}")
  endif()

  configure_package_config_file(
    tools/${PROJECT_NAME}Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})

  if(CMAKE_VERSION VERSION_LESS 3.14)
    # Remove CMAKE_SIZEOF_VOID_P from ConfigVersion.cmake since the library does
    # not depend on architecture specific settings or libraries.
    set(_PYBIND11_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
    unset(CMAKE_SIZEOF_VOID_P)

    write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY AnyNewerVersion)

    set(CMAKE_SIZEOF_VOID_P ${_PYBIND11_CMAKE_SIZEOF_VOID_P})
  else()
    # CMake 3.14+ natively supports header-only libraries
    write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY AnyNewerVersion ARCH_INDEPENDENT)
  endif()

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
          tools/FindPythonLibsNew.cmake
          tools/pybind11Common.cmake
          tools/pybind11Tools.cmake
          tools/pybind11NewTools.cmake
    DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})

  if(NOT PYBIND11_EXPORT_NAME)
    set(PYBIND11_EXPORT_NAME "${PROJECT_NAME}Targets")
  endif()

  install(TARGETS pybind11_headers EXPORT "${PYBIND11_EXPORT_NAME}")

  install(
    EXPORT "${PYBIND11_EXPORT_NAME}"
    NAMESPACE "pybind11::"
    DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})

  # pkg-config support
  if(NOT prefix_for_pc_file)
    set(prefix_for_pc_file "${CMAKE_INSTALL_PREFIX}")
  endif()
  join_paths(includedir_for_pc_file "\${prefix}" "${CMAKE_INSTALL_INCLUDEDIR}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tools/pybind11.pc.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/pybind11.pc" @ONLY)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pybind11.pc"
          DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig/")

  # Uninstall target
  if(PYBIND11_MASTER_PROJECT)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake_uninstall.cmake.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)

    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P
                                        ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
  endif()
endif()

# BUILD_TESTING takes priority, but only if this is the master project
if(PYBIND11_MASTER_PROJECT AND DEFINED BUILD_TESTING)
  if(BUILD_TESTING)
    if(_pybind11_nopython)
      message(FATAL_ERROR "Cannot activate tests in NOPYTHON mode")
    else()
      add_subdirectory(tests)
    endif()
  endif()
else()
  if(PYBIND11_TEST)
    if(_pybind11_nopython)
      message(FATAL_ERROR "Cannot activate tests in NOPYTHON mode")
    else()
      add_subdirectory(tests)
    endif()
  endif()
endif()

# Better symmetry with find_package(pybind11 CONFIG) mode.
if(NOT PYBIND11_MASTER_PROJECT)
  set(pybind11_FOUND
      TRUE
      CACHE INTERNAL "True if pybind11 and all required components found on the system")
endif()
