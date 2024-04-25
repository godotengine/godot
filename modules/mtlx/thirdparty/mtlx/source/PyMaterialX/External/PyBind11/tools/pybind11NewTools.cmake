# tools/pybind11NewTools.cmake -- Build system for the pybind11 modules
#
# Copyright (c) 2020 Wenzel Jakob <wenzel@inf.ethz.ch> and Henry Schreiner
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

if(CMAKE_VERSION VERSION_LESS 3.12)
  message(FATAL_ERROR "You cannot use the new FindPython module with CMake < 3.12")
endif()

include_guard(DIRECTORY)

get_property(
  is_config
  TARGET pybind11::headers
  PROPERTY IMPORTED)

if(pybind11_FIND_QUIETLY)
  set(_pybind11_quiet QUIET)
else()
  set(_pybind11_quiet "")
endif()

if(NOT Python_FOUND AND NOT Python3_FOUND)
  if(NOT DEFINED Python_FIND_IMPLEMENTATIONS)
    set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
  endif()

  # GitHub Actions like activation
  if(NOT DEFINED Python_ROOT_DIR AND DEFINED ENV{pythonLocation})
    set(Python_ROOT_DIR "$ENV{pythonLocation}")
  endif()

  find_package(Python 3.6 REQUIRED COMPONENTS Interpreter Development ${_pybind11_quiet})

  # If we are in submodule mode, export the Python targets to global targets.
  # If this behavior is not desired, FindPython _before_ pybind11.
  if(NOT is_config)
    set_property(TARGET Python::Python PROPERTY IMPORTED_GLOBAL TRUE)
    set_property(TARGET Python::Interpreter PROPERTY IMPORTED_GLOBAL TRUE)
    if(TARGET Python::Module)
      set_property(TARGET Python::Module PROPERTY IMPORTED_GLOBAL TRUE)
    endif()
  endif()
endif()

if(Python_FOUND)
  set(_Python
      Python
      CACHE INTERNAL "" FORCE)
elseif(Python3_FOUND)
  set(_Python
      Python3
      CACHE INTERNAL "" FORCE)
endif()

if(PYBIND11_MASTER_PROJECT)
  if(${_Python}_INTERPRETER_ID MATCHES "PyPy")
    message(STATUS "PyPy ${${_Python}_PyPy_VERSION} (Py ${${_Python}_VERSION})")
  else()
    message(STATUS "${_Python} ${${_Python}_VERSION}")
  endif()
endif()

# If a user finds Python, they may forget to include the Interpreter component
# and the following two steps require it. It is highly recommended by CMake
# when finding development libraries anyway, so we will require it.
if(NOT DEFINED ${_Python}_EXECUTABLE)
  message(
    FATAL_ERROR
      "${_Python} was found without the Interpreter component. Pybind11 requires this component.")

endif()

if(NOT ${_Python}_EXECUTABLE STREQUAL PYBIND11_PYTHON_EXECUTABLE_LAST)
  # Detect changes to the Python version/binary in subsequent CMake runs, and refresh config if needed
  unset(PYTHON_IS_DEBUG CACHE)
  unset(PYTHON_MODULE_EXTENSION CACHE)
  set(PYBIND11_PYTHON_EXECUTABLE_LAST
      "${${_Python}_EXECUTABLE}"
      CACHE INTERNAL "Python executable during the last CMake run")
endif()

if(NOT DEFINED PYTHON_IS_DEBUG)
  # Debug check - see https://stackoverflow.com/questions/646518/python-how-to-detect-debug-Interpreter
  execute_process(
    COMMAND "${${_Python}_EXECUTABLE}" "-c"
            "import sys; sys.exit(hasattr(sys, 'gettotalrefcount'))"
    RESULT_VARIABLE _PYTHON_IS_DEBUG)
  set(PYTHON_IS_DEBUG
      "${_PYTHON_IS_DEBUG}"
      CACHE INTERNAL "Python debug status")
endif()

# Get the suffix - SO is deprecated, should use EXT_SUFFIX, but this is
# required for PyPy3 (as of 7.3.1)
if(NOT DEFINED PYTHON_MODULE_EXTENSION)
  execute_process(
    COMMAND
      "${${_Python}_EXECUTABLE}" "-c"
      "import sys, importlib; s = importlib.import_module('distutils.sysconfig' if sys.version_info < (3, 10) else 'sysconfig'); print(s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO'))"
    OUTPUT_VARIABLE _PYTHON_MODULE_EXTENSION
    ERROR_VARIABLE _PYTHON_MODULE_EXTENSION_ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(_PYTHON_MODULE_EXTENSION STREQUAL "")
    message(
      FATAL_ERROR "pybind11 could not query the module file extension, likely the 'distutils'"
                  "package is not installed. Full error message:\n${_PYTHON_MODULE_EXTENSION_ERR}")
  endif()

  # This needs to be available for the pybind11_extension function
  set(PYTHON_MODULE_EXTENSION
      "${_PYTHON_MODULE_EXTENSION}"
      CACHE INTERNAL "")
endif()

# Python debug libraries expose slightly different objects before 3.8
# https://docs.python.org/3.6/c-api/intro.html#debugging-builds
# https://stackoverflow.com/questions/39161202/how-to-work-around-missing-pymodule-create2-in-amd64-win-python35-d-lib
if(PYTHON_IS_DEBUG)
  set_property(
    TARGET pybind11::pybind11
    APPEND
    PROPERTY INTERFACE_COMPILE_DEFINITIONS Py_DEBUG)
endif()

# Check on every access - since Python can change - do nothing in that case.

if(DEFINED ${_Python}_INCLUDE_DIRS)
  # Only add Python for build - must be added during the import for config
  # since it has to be re-discovered.
  #
  # This needs to be a target to be included after the local pybind11
  # directory, just in case there there is an installed pybind11 sitting
  # next to Python's includes. It also ensures Python is a SYSTEM library.
  add_library(pybind11::python_headers INTERFACE IMPORTED)
  set_property(
    TARGET pybind11::python_headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                             "$<BUILD_INTERFACE:${${_Python}_INCLUDE_DIRS}>")
  set_property(
    TARGET pybind11::pybind11
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES pybind11::python_headers)
  set(pybind11_INCLUDE_DIRS
      "${pybind11_INCLUDE_DIR}" "${${_Python}_INCLUDE_DIRS}"
      CACHE INTERNAL "Directories where pybind11 and possibly Python headers are located")
endif()

# In CMake 3.18+, you can find these separately, so include an if
if(TARGET ${_Python}::Python)
  set_property(
    TARGET pybind11::embed
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES ${_Python}::Python)
endif()

# CMake 3.15+ has this
if(TARGET ${_Python}::Module)
  set_property(
    TARGET pybind11::module
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES ${_Python}::Module)
else()
  set_property(
    TARGET pybind11::module
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES pybind11::python_link_helper)
endif()

# WITHOUT_SOABI and WITH_SOABI will disable the custom extension handling used by pybind11.
# WITH_SOABI is passed on to python_add_library.
function(pybind11_add_module target_name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG
                        "STATIC;SHARED;MODULE;THIN_LTO;OPT_SIZE;NO_EXTRAS;WITHOUT_SOABI" "" "")

  if(ARG_STATIC)
    set(lib_type STATIC)
  elseif(ARG_SHARED)
    set(lib_type SHARED)
  else()
    set(lib_type MODULE)
  endif()

  if("${_Python}" STREQUAL "Python")
    python_add_library(${target_name} ${lib_type} ${ARG_UNPARSED_ARGUMENTS})
  elseif("${_Python}" STREQUAL "Python3")
    python3_add_library(${target_name} ${lib_type} ${ARG_UNPARSED_ARGUMENTS})
  else()
    message(FATAL_ERROR "Cannot detect FindPython version: ${_Python}")
  endif()

  target_link_libraries(${target_name} PRIVATE pybind11::headers)

  if(lib_type STREQUAL "MODULE")
    target_link_libraries(${target_name} PRIVATE pybind11::module)
  else()
    target_link_libraries(${target_name} PRIVATE pybind11::embed)
  endif()

  if(MSVC)
    target_link_libraries(${target_name} PRIVATE pybind11::windows_extras)
  endif()

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

  # If we don't pass a WITH_SOABI or WITHOUT_SOABI, use our own default handling of extensions
  if(NOT ARG_WITHOUT_SOABI AND NOT "WITH_SOABI" IN_LIST ARG_UNPARSED_ARGUMENTS)
    pybind11_extension(${target_name})
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
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(${target_name})
  endif()

  if(MSVC)
    target_link_libraries(${target_name} PRIVATE pybind11::windows_extras)
  endif()

  if(ARG_OPT_SIZE)
    target_link_libraries(${target_name} PRIVATE pybind11::opt_size)
  endif()
endfunction()

function(pybind11_extension name)
  # The extension is precomputed
  set_target_properties(${name} PROPERTIES PREFIX "" SUFFIX "${PYTHON_MODULE_EXTENSION}")

endfunction()
