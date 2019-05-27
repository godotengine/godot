# Copyright (c) 2015-2016 The Khronos Group Inc.
#
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

if (DEFINED SPIRV-Headers_SOURCE_DIR)
  # This allows flexible position of the SPIRV-Headers repo.
  set(SPIRV_HEADER_DIR ${SPIRV-Headers_SOURCE_DIR})
else()
  if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Headers)
    set(SPIRV_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Headers)
  else()
    set(SPIRV_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spirv-headers)
  endif()
endif()

if (IS_DIRECTORY ${SPIRV_HEADER_DIR})
  set(SPIRV_HEADER_INCLUDE_DIR ${SPIRV_HEADER_DIR}/include PARENT_SCOPE)
else()
  message(FATAL_ERROR
    "SPIRV-Headers was not found - please checkout a copy under external/.")
endif()

if (NOT ${SPIRV_SKIP_TESTS})
  # Find gmock if we can. If it's not already configured, then try finding
  # it in external/googletest.
  if (TARGET gmock)
    message(STATUS "Google Mock already configured")
  else()
    set(GMOCK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googlemock)
    if(EXISTS ${GMOCK_DIR})
      if(MSVC)
        # Our tests use ::testing::Combine.  Work around a compiler
        # detection problem in googletest, where that template is
        # accidentally disabled for VS 2017.
        # See https://github.com/google/googletest/issues/1352
        add_definitions(-DGTEST_HAS_COMBINE=1)
      endif()
      if(WIN32)
        option(gtest_force_shared_crt
          "Use shared (DLL) run-time lib even when Google Test is built as static lib."
          ON)
      endif()
      add_subdirectory(${GMOCK_DIR} EXCLUDE_FROM_ALL)
    endif()
  endif()
  if (TARGET gmock)
    set(GTEST_TARGETS
      gtest
      gtest_main
      gmock
      gmock_main
    )
    foreach(target ${GTEST_TARGETS})
      set_property(TARGET ${target} PROPERTY FOLDER GoogleTest)
    endforeach()
  endif()

  # Find Effcee and RE2, for testing.

  # First find RE2, since Effcee depends on it.
  # If already configured, then use that.  Otherwise, prefer to find it under 're2'
  # in this directory.
  if (NOT TARGET re2)
    # If we are configuring RE2, then turn off its testing.  It takes a long time and
    # does not add much value for us.  If an enclosing project configured RE2, then it
    # has already chosen whether to enable RE2 testing.
    set(RE2_BUILD_TESTING OFF CACHE STRING "Run RE2 Tests")
    if (NOT RE2_SOURCE_DIR)
      if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/re2)
	set(RE2_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/re2" CACHE STRING "RE2 source dir" )
      endif()
    endif()
  endif()

  if (NOT TARGET effcee)
    # Expect to find effcee in this directory.
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/effcee)
      # If we're configuring RE2 (via Effcee), then turn off RE2 testing.
      if (NOT TARGET re2)
	set(RE2_BUILD_TESTING OFF)
      endif()
      if (MSVC)
	# SPIRV-Tools uses the shared CRT with MSVC.  Tell Effcee to do the same.
	set(EFFCEE_ENABLE_SHARED_CRT ON)
      endif()
      add_subdirectory(effcee)
      set_property(TARGET effcee PROPERTY FOLDER Effcee)
      # Turn off warnings for effcee and re2
      set_property(TARGET effcee APPEND PROPERTY COMPILE_OPTIONS -w)
      set_property(TARGET re2 APPEND PROPERTY COMPILE_OPTIONS -w)
    endif()
  endif()
endif()
