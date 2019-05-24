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

cmake_minimum_required(VERSION 2.8.12)
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
if (POLICY CMP0054)
  # Avoid dereferencing variables or interpret keywords that have been
  # quoted or bracketed.
  # https://cmake.org/cmake/help/v3.1/policy/CMP0054.html
  cmake_policy(SET CMP0054 NEW)
endif()
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(spirv-tools)
enable_testing()
set(SPIRV_TOOLS "SPIRV-Tools")

include(GNUInstallDirs)
include(cmake/setup_build.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 11)

option(SPIRV_ALLOW_TIMERS "Allow timers via clock_gettime on supported platforms" ON)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  add_definitions(-DSPIRV_LINUX)
  set(SPIRV_TIMER_ENABLED ${SPIRV_ALLOW_TIMERS})
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
  add_definitions(-DSPIRV_WINDOWS)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "CYGWIN")
  add_definitions(-DSPIRV_WINDOWS)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  add_definitions(-DSPIRV_MAC)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
  add_definitions(-DSPIRV_ANDROID)
  set(SPIRV_TIMER_ENABLED ${SPIRV_ALLOW_TIMERS})
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "FreeBSD")
  add_definitions(-DSPIRV_FREEBSD)
else()
  message(FATAL_ERROR "Your platform '${CMAKE_SYSTEM_NAME}' is not supported!")
endif()

if (${SPIRV_TIMER_ENABLED})
  add_definitions(-DSPIRV_TIMER_ENABLED)
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

option(SKIP_SPIRV_TOOLS_INSTALL "Skip installation" ${SKIP_SPIRV_TOOLS_INSTALL})
if(NOT ${SKIP_SPIRV_TOOLS_INSTALL})
  set(ENABLE_SPIRV_TOOLS_INSTALL ON)
endif()

option(SPIRV_BUILD_COMPRESSION "Build SPIR-V compressing codec" OFF)

option(SPIRV_WERROR "Enable error on warning" ON)
if(("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU") OR (("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") AND (NOT CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")))
  set(COMPILER_IS_LIKE_GNU TRUE)
endif()
if(${COMPILER_IS_LIKE_GNU})
  set(SPIRV_WARNINGS -Wall -Wextra -Wnon-virtual-dtor -Wno-missing-field-initializers)

  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(SPIRV_WARNINGS ${SPIRV_WARNINGS} -Wno-self-assign)
  endif()

  option(SPIRV_WARN_EVERYTHING "Enable -Weverything" ${SPIRV_WARN_EVERYTHING})
  if(${SPIRV_WARN_EVERYTHING})
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(SPIRV_WARNINGS ${SPIRV_WARNINGS}
        -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(SPIRV_WARNINGS ${SPIRV_WARNINGS} -Wpedantic -pedantic-errors)
    else()
      message(STATUS "Unknown compiler ${CMAKE_CXX_COMPILER_ID}, "
                     "so SPIRV_WARN_EVERYTHING has no effect")
    endif()
  endif()

  if(${SPIRV_WERROR})
    set(SPIRV_WARNINGS ${SPIRV_WARNINGS} -Werror)
  endif()
elseif(MSVC)
  set(SPIRV_WARNINGS -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS /wd4800)

  if(${SPIRV_WERROR})
    set(SPIRV_WARNINGS ${SPIRV_WARNINGS} /WX)
  endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

option(SPIRV_COLOR_TERMINAL "Enable color terminal output" ON)
if(${SPIRV_COLOR_TERMINAL})
  add_definitions(-DSPIRV_COLOR_TERMINAL)
endif()

option(SPIRV_LOG_DEBUG "Enable excessive debug output" OFF)
if(${SPIRV_LOG_DEBUG})
  add_definitions(-DSPIRV_LOG_DEBUG)
endif()

if (DEFINED SPIRV_TOOLS_EXTRA_DEFINITIONS)
  add_definitions(${SPIRV_TOOLS_EXTRA_DEFINITIONS})
endif()

function(spvtools_default_compile_options TARGET)
  target_compile_options(${TARGET} PRIVATE ${SPIRV_WARNINGS})

  if (${COMPILER_IS_LIKE_GNU})
    target_compile_options(${TARGET} PRIVATE
      -std=c++11 -fno-exceptions -fno-rtti)
    target_compile_options(${TARGET} PRIVATE
      -Wall -Wextra -Wno-long-long -Wshadow -Wundef -Wconversion
      -Wno-sign-conversion)
    # For good call stacks in profiles, keep the frame pointers.
    if(NOT "${SPIRV_PERF}" STREQUAL "")
      target_compile_options(${TARGET} PRIVATE -fno-omit-frame-pointer)
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(SPIRV_USE_SANITIZER "" CACHE STRING
        "Use the clang sanitizer [address|memory|thread|...]")
      if(NOT "${SPIRV_USE_SANITIZER}" STREQUAL "")
        target_compile_options(${TARGET} PRIVATE
          -fsanitize=${SPIRV_USE_SANITIZER})
      endif()
      target_compile_options(${TARGET} PRIVATE
         -ftemplate-depth=1024)
    else()
      target_compile_options(${TARGET} PRIVATE
         -Wno-missing-field-initializers)
    endif()
  endif()

  if (MSVC)
    # Specify /EHs for exception handling. This makes using SPIRV-Tools as
    # dependencies in other projects easier.
    target_compile_options(${TARGET} PRIVATE /EHs)
  endif()

  # For MinGW cross compile, statically link to the C++ runtime.
  # But it still depends on MSVCRT.dll.
  if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
      set_target_properties(${TARGET} PROPERTIES
        LINK_FLAGS -static -static-libgcc -static-libstdc++)
    endif()
  endif()
endfunction()

if(NOT COMMAND find_host_package)
  macro(find_host_package)
    find_package(${ARGN})
  endmacro()
endif()
if(NOT COMMAND find_host_program)
  macro(find_host_program)
    find_program(${ARGN})
  endmacro()
endif()

find_host_package(PythonInterp)

# Check for symbol exports on Linux.
# At the moment, this check will fail on the OSX build machines for the Android NDK.
# It appears they don't have objdump.
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  macro(spvtools_check_symbol_exports TARGET)
    if (NOT "${SPIRV_SKIP_TESTS}")
      add_test(NAME spirv-tools-symbol-exports-${TARGET}
               COMMAND ${PYTHON_EXECUTABLE}
               ${spirv-tools_SOURCE_DIR}/utils/check_symbol_exports.py "$<TARGET_FILE:${TARGET}>")
    endif()
  endmacro()
else()
  macro(spvtools_check_symbol_exports TARGET)
    if (NOT "${SPIRV_SKIP_TESTS}")
      message("Skipping symbol exports test for ${TARGET}")
    endif()
  endmacro()
endif()

# Defaults to OFF if the user didn't set it.
option(SPIRV_SKIP_EXECUTABLES
  "Skip building the executable and tests along with the library"
  ${SPIRV_SKIP_EXECUTABLES})
option(SPIRV_SKIP_TESTS
  "Skip building tests along with the library" ${SPIRV_SKIP_TESTS})
if ("${SPIRV_SKIP_EXECUTABLES}")
  set(SPIRV_SKIP_TESTS ON)
endif()

# Defaults to ON.  The checks can be time consuming.
# Turn off if they take too long.
option(SPIRV_CHECK_CONTEXT "In a debug build, check if the IR context is in a valid state." ON)
if (${SPIRV_CHECK_CONTEXT})
  add_definitions(-DSPIRV_CHECK_CONTEXT)
endif()

# Precompiled header macro. Parameters are source file list and filename for pch cpp file.
macro(spvtools_pch SRCS PCHPREFIX)
  if(MSVC AND CMAKE_GENERATOR MATCHES "^Visual Studio")
    set(PCH_NAME "$(IntDir)\\${PCHPREFIX}.pch")
    # make source files use/depend on PCH_NAME
    set_source_files_properties(${${SRCS}} PROPERTIES COMPILE_FLAGS "/Yu${PCHPREFIX}.h /FI${PCHPREFIX}.h /Fp${PCH_NAME} /Zm300" OBJECT_DEPENDS "${PCH_NAME}")
    # make PCHPREFIX.cpp file compile and generate PCH_NAME
    set_source_files_properties("${PCHPREFIX}.cpp" PROPERTIES COMPILE_FLAGS "/Yc${PCHPREFIX}.h /Fp${PCH_NAME} /Zm300" OBJECT_OUTPUTS "${PCH_NAME}")
    list(APPEND ${SRCS} "${PCHPREFIX}.cpp")
  endif()
endmacro(spvtools_pch)

add_subdirectory(external)

add_subdirectory(source)
add_subdirectory(tools)

add_subdirectory(test)
add_subdirectory(examples)

if(ENABLE_SPIRV_TOOLS_INSTALL)
  install(
    FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/include/spirv-tools/libspirv.h
      ${CMAKE_CURRENT_SOURCE_DIR}/include/spirv-tools/libspirv.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/include/spirv-tools/optimizer.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/include/spirv-tools/linker.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/include/spirv-tools/instrument.hpp
    DESTINATION
      ${CMAKE_INSTALL_INCLUDEDIR}/spirv-tools/)
endif(ENABLE_SPIRV_TOOLS_INSTALL)

if (NOT "${SPIRV_SKIP_TESTS}")
  add_test(NAME spirv-tools-copyrights
           COMMAND ${PYTHON_EXECUTABLE} utils/check_copyright.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(SPIRV_LIBRARIES "-lSPIRV-Tools -lSPIRV-Tools-link -lSPIRV-Tools-opt")
set(SPIRV_SHARED_LIBRARIES "-lSPIRV-Tools-shared")
if(SPIRV_BUILD_COMPRESSION)
  set(SPIRV_LIBRARIES "${SPIRV_LIBRARIES} -lSPIRV-Tools-comp")
endif(SPIRV_BUILD_COMPRESSION)

# Build pkg-config file
# Use a first-class target so it's regenerated when relevant files are updated.
add_custom_target(spirv-tools-pkg-config ALL
        COMMAND ${CMAKE_COMMAND}
                      -DCHANGES_FILE=${CMAKE_CURRENT_SOURCE_DIR}/CHANGES
                      -DTEMPLATE_FILE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/SPIRV-Tools.pc.in
                      -DOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Tools.pc
                      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                      -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
                      -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
                      -DSPIRV_LIBRARIES=${SPIRV_LIBRARIES}
                      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_pkg_config.cmake
        DEPENDS "CHANGES" "cmake/SPIRV-Tools.pc.in" "cmake/write_pkg_config.cmake")
add_custom_target(spirv-tools-shared-pkg-config ALL
        COMMAND ${CMAKE_COMMAND}
                      -DCHANGES_FILE=${CMAKE_CURRENT_SOURCE_DIR}/CHANGES
                      -DTEMPLATE_FILE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/SPIRV-Tools-shared.pc.in
                      -DOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Tools-shared.pc
                      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                      -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
                      -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
                      -DSPIRV_SHARED_LIBRARIES=${SPIRV_SHARED_LIBRARIES}
                      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_pkg_config.cmake
        DEPENDS "CHANGES" "cmake/SPIRV-Tools-shared.pc.in" "cmake/write_pkg_config.cmake")

# Install pkg-config file
if (ENABLE_SPIRV_TOOLS_INSTALL)
  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Tools.pc
      ${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Tools-shared.pc
    DESTINATION
      ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()
