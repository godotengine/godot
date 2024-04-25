# tools/pybind11Tools.cmake -- Build system for the pybind11 modules
#
# Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

# include_guard(global) (pre-CMake 3.10)
if(TARGET pybind11::python_headers)
  return()
endif()

# Built-in in CMake 3.5+
include(CMakeParseArguments)

if(pybind11_FIND_QUIETLY)
  set(_pybind11_quiet QUIET)
else()
  set(_pybind11_quiet "")
endif()

# If this is the first run, PYTHON_VERSION can stand in for PYBIND11_PYTHON_VERSION
if(NOT DEFINED PYBIND11_PYTHON_VERSION AND DEFINED PYTHON_VERSION)
  message(WARNING "Set PYBIND11_PYTHON_VERSION to search for a specific version, not "
                  "PYTHON_VERSION (which is an output). Assuming that is what you "
                  "meant to do and continuing anyway.")
  set(PYBIND11_PYTHON_VERSION
      "${PYTHON_VERSION}"
      CACHE STRING "Python version to use for compiling modules")
  unset(PYTHON_VERSION)
  unset(PYTHON_VERSION CACHE)
elseif(DEFINED PYBIND11_PYTHON_VERSION)
  # If this is set as a normal variable, promote it
  set(PYBIND11_PYTHON_VERSION
      "${PYBIND11_PYTHON_VERSION}"
      CACHE STRING "Python version to use for compiling modules")
else()
  # Make an empty cache variable.
  set(PYBIND11_PYTHON_VERSION
      ""
      CACHE STRING "Python version to use for compiling modules")
endif()

# A user can set versions manually too
set(Python_ADDITIONAL_VERSIONS
    "3.11;3.10;3.9;3.8;3.7;3.6"
    CACHE INTERNAL "")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_package(PythonLibsNew ${PYBIND11_PYTHON_VERSION} MODULE REQUIRED ${_pybind11_quiet})
list(REMOVE_AT CMAKE_MODULE_PATH -1)

# Makes a normal variable a cached variable
macro(_PYBIND11_PROMOTE_TO_CACHE NAME)
  set(_tmp_ptc "${${NAME}}")
  # CMake 3.21 complains if a cached variable is shadowed by a normal one
  unset(${NAME})
  set(${NAME}
      "${_tmp_ptc}"
      CACHE INTERNAL "")
endmacro()

# Cache variables so pybind11_add_module can be used in parent projects
_pybind11_promote_to_cache(PYTHON_INCLUDE_DIRS)
_pybind11_promote_to_cache(PYTHON_LIBRARIES)
_pybind11_promote_to_cache(PYTHON_MODULE_PREFIX)
_pybind11_promote_to_cache(PYTHON_MODULE_EXTENSION)
_pybind11_promote_to_cache(PYTHON_VERSION_MAJOR)
_pybind11_promote_to_cache(PYTHON_VERSION_MINOR)
_pybind11_promote_to_cache(PYTHON_VERSION)
_pybind11_promote_to_cache(PYTHON_IS_DEBUG)

if(PYBIND11_MASTER_PROJECT)
  if(PYTHON_MODULE_EXTENSION MATCHES "pypy")
    if(NOT DEFINED PYPY_VERSION)
      execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c
                [=[import sys; sys.stdout.write(".".join(map(str, sys.pypy_version_info[:3])))]=]
        OUTPUT_VARIABLE pypy_version)
      set(PYPY_VERSION
          ${pypy_version}
          CACHE INTERNAL "")
    endif()
    message(STATUS "PYPY ${PYPY_VERSION} (Py ${PYTHON_VERSION})")
  else()
    message(STATUS "PYTHON ${PYTHON_VERSION}")
  endif()
endif()

# Only add Python for build - must be added during the import for config since
# it has to be re-discovered.
#
# This needs to be an target to it is included after the local pybind11
# directory, just in case there are multiple versions of pybind11, we want the
# one we expect.
add_library(pybind11::python_headers INTERFACE IMPORTED)
set_property(TARGET pybind11::python_headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                      "$<BUILD_INTERFACE:${PYTHON_INCLUDE_DIRS}>")
set_property(
  TARGET pybind11::pybind11
  APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES pybind11::python_headers)

set(pybind11_INCLUDE_DIRS
    "${pybind11_INCLUDE_DIR}" "${PYTHON_INCLUDE_DIRS}"
    CACHE INTERNAL "Directories where pybind11 and possibly Python headers are located")

# Python debug libraries expose slightly different objects before 3.8
# https://docs.python.org/3.6/c-api/intro.html#debugging-builds
# https://stackoverflow.com/questions/39161202/how-to-work-around-missing-pymodule-create2-in-amd64-win-python35-d-lib
if(PYTHON_IS_DEBUG)
  set_property(
    TARGET pybind11::pybind11
    APPEND
    PROPERTY INTERFACE_COMPILE_DEFINITIONS Py_DEBUG)
endif()

# The <3.11 code here does not support release/debug builds at the same time, like on vcpkg
if(CMAKE_VERSION VERSION_LESS 3.11)
  set_property(
    TARGET pybind11::module
    APPEND
    PROPERTY
      INTERFACE_LINK_LIBRARIES
      pybind11::python_link_helper
      "$<$<OR:$<PLATFORM_ID:Windows>,$<PLATFORM_ID:Cygwin>>:$<BUILD_INTERFACE:${PYTHON_LIBRARIES}>>"
  )

  set_property(
    TARGET pybind11::embed
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES pybind11::pybind11 $<BUILD_INTERFACE:${PYTHON_LIBRARIES}>)
else()
  # The IMPORTED INTERFACE library here is to ensure that "debug" and "release" get processed outside
  # of a generator expression - https://gitlab.kitware.com/cmake/cmake/-/issues/18424, as they are
  # target_link_library keywords rather than real libraries.
  add_library(pybind11::_ClassicPythonLibraries IMPORTED INTERFACE)
  target_link_libraries(pybind11::_ClassicPythonLibraries INTERFACE ${PYTHON_LIBRARIES})
  target_link_libraries(
    pybind11::module
    INTERFACE
      pybind11::python_link_helper
      "$<$<OR:$<PLATFORM_ID:Windows>,$<PLATFORM_ID:Cygwin>>:pybind11::_ClassicPythonLibraries>")

  target_link_libraries(pybind11::embed INTERFACE pybind11::pybind11
                                                  pybind11::_ClassicPythonLibraries)
endif()

function(pybind11_extension name)
  # The prefix and extension are provided by FindPythonLibsNew.cmake
  set_target_properties(${name} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                           SUFFIX "${PYTHON_MODULE_EXTENSION}")
endfunction()

# Build a Python extension module:
# pybind11_add_module(<name> [MODULE | SHARED] [EXCLUDE_FROM_ALL]
#                     [NO_EXTRAS] [THIN_LTO] [OPT_SIZE] source1 [source2 ...])
#
function(pybind11_add_module target_name)
  set(options "MODULE;SHARED;EXCLUDE_FROM_ALL;NO_EXTRAS;SYSTEM;THIN_LTO;OPT_SIZE")
  cmake_parse_arguments(ARG "${options}" "" "" ${ARGN})

  if(ARG_MODULE AND ARG_SHARED)
    message(FATAL_ERROR "Can't be both MODULE and SHARED")
  elseif(ARG_SHARED)
    set(lib_type SHARED)
  else()
    set(lib_type MODULE)
  endif()

  if(ARG_EXCLUDE_FROM_ALL)
    set(exclude_from_all EXCLUDE_FROM_ALL)
  else()
    set(exclude_from_all "")
  endif()

  add_library(${target_name} ${lib_type} ${exclude_from_all} ${ARG_UNPARSED_ARGUMENTS})

  target_link_libraries(${target_name} PRIVATE pybind11::module)

  if(ARG_SYSTEM)
    message(
      STATUS
        "Warning: this does not have an effect - use NO_SYSTEM_FROM_IMPORTED if using imported targets"
    )
  endif()

  pybind11_extension(${target_name})

  # -fvisibility=hidden is required to allow multiple modules compiled against
  # different pybind versions to work properly, and for some features (e.g.
  # py::module_local).  We force it on everything inside the `pybind11`
  # namespace; also turning it on for a pybind module compilation here avoids
  # potential warnings or issues from having mixed hidden/non-hidden types.
  if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)
    set_target_properties(${target_name} PROPERTIES CXX_VISIBILITY_PRESET "hidden")
  endif()

  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
    set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
  endif()

  if(ARG_NO_EXTRAS)
    return()
  endif()

  if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    if(ARG_THIN_LTO)
      target_link_libraries(${target_name} PRIVATE pybind11::thin_lto)
    else()
      target_link_libraries(${target_name} PRIVATE pybind11::lto)
    endif()
  endif()

  # Use case-insensitive comparison to match the result of $<CONFIG:cfgs>
  string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
  if(NOT MSVC AND NOT "${uppercase_CMAKE_BUILD_TYPE}" MATCHES DEBUG|RELWITHDEBINFO)
    pybind11_strip(${target_name})
  endif()

  if(MSVC)
    target_link_libraries(${target_name} PRIVATE pybind11::windows_extras)
  endif()

  if(ARG_OPT_SIZE)
    target_link_libraries(${target_name} PRIVATE pybind11::opt_size)
  endif()
endfunction()

# Provide general way to call common Python commands in "common" file.
set(_Python
    PYTHON
    CACHE INTERNAL "" FORCE)
