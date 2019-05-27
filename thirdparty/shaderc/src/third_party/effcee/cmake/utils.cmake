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

# Utility functions

function(effcee_default_c_compile_options TARGET)
  if (NOT "${MSVC}")
    target_compile_options(${TARGET} PRIVATE -Wall -Werror)
    if (ENABLE_CODE_COVERAGE)
      # The --coverage option is a synonym for -fprofile-arcs -ftest-coverage
      # when compiling.
      target_compile_options(${TARGET} PRIVATE -g -O0 --coverage)
      # The --coverage option is a synonym for -lgcov when linking for gcc.
      # For clang, it links in a different library, libclang_rt.profile, which
      # requires clang to be built with compiler-rt.
      target_link_libraries(${TARGET} PRIVATE --coverage)
    endif()
    if (NOT EFFCEE_ENABLE_SHARED_CRT)
      if (WIN32)
        # For MinGW cross compile, statically link to the libgcc runtime.
        # But it still depends on MSVCRT.dll.
        set_target_properties(${TARGET} PROPERTIES
          LINK_FLAGS "-static -static-libgcc")
      endif(WIN32)
    endif(NOT EFFCEE_ENABLE_SHARED_CRT)
    if (UNIX AND NOT MINGW)
      target_link_libraries(${TARGET} PUBLIC -pthread)
    endif()
  else()
    # disable warning C4800: 'int' : forcing value to bool 'true' or 'false'
    # (performance warning)
    target_compile_options(${TARGET} PRIVATE /wd4800)
  endif()
endfunction(effcee_default_c_compile_options)

function(effcee_default_compile_options TARGET)
  effcee_default_c_compile_options(${TARGET})
  if (NOT "${MSVC}")
    # RE2's public header requires C++11.  So publicly required C++11
    target_compile_options(${TARGET} PUBLIC -std=c++11)
    if (NOT EFFCEE_ENABLE_SHARED_CRT)
      if (WIN32)
        # For MinGW cross compile, statically link to the C++ runtime.
        # But it still depends on MSVCRT.dll.
        set_target_properties(${TARGET} PROPERTIES
          LINK_FLAGS "-static -static-libgcc -static-libstdc++")
      endif(WIN32)
    endif(NOT EFFCEE_ENABLE_SHARED_CRT)
  endif()
endfunction(effcee_default_compile_options)
