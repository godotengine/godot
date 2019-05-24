# Copyright 2017 The Effcee Authors.
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

# For cross-compilation, we need to use find_host_package
# in the remaining setup. But if not cross-compiling, then we
# need to alias find_host_package to find_package.
# Similar for find_host_program.
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

if (ANDROID)
  # For android let's preemptively find the correct packages so that
  # child projects (e.g. googletest) do not fail to find them.
  find_host_package(PythonInterp)
endif()

foreach(PROGRAM echo python)
  string(TOUPPER ${PROGRAM} PROG_UC)
  if (ANDROID)
    find_host_program(${PROG_UC}_EXE ${PROGRAM} REQUIRED)
  else()
    find_program(${PROG_UC}_EXE ${PROGRAM} REQUIRED)
  endif()
endforeach(PROGRAM)

option(DISABLE_RTTI "Disable RTTI in builds")
if(DISABLE_RTTI)
  if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fno-rtti")
  endif(UNIX)
endif(DISABLE_RTTI)

option(DISABLE_EXCEPTIONS "Disables exceptions in builds")
if(DISABLE_EXCEPTIONS)
  if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
  endif(UNIX)
endif(DISABLE_EXCEPTIONS)

if(WIN32)
  # Ensure that gmock compiles the same as the rest of the code, otherwise
  # failures will occur.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif(WIN32)

if(WIN32)
# On Windows, CMake by default compiles with the shared CRT.
# Default it to the static CRT.
  option(EFFCEE_ENABLE_SHARED_CRT
	 "Use the shared CRT with MSVC instead of the static CRT"
	 ${EFFCEE_ENABLE_SHARED_CRT})
  if (NOT EFFCEE_ENABLE_SHARED_CRT)
    if(MSVC)
      # Link executables statically by replacing /MD with /MT everywhere.
      foreach(flag_var
	  CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
	  CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
	if(${flag_var} MATCHES "/MD")
	  string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
	endif(${flag_var} MATCHES "/MD")
      endforeach(flag_var)
    endif(MSVC)
  endif(NOT EFFCEE_ENABLE_SHARED_CRT)
endif(WIN32)
