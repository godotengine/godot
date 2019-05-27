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

SET(CMAKE_SYSTEM_NAME Windows)

set(MINGW_COMPILER_PREFIX "i686-w64-mingw32" CACHE STRING
    "What compiler prefix to use for mingw")

set(MINGW_SYSROOT "/usr/${MINGW_COMPILER_PREFIX}" CACHE STRING
    "What sysroot to use for mingw")

# Which compilers to use for C and C++
find_program(CMAKE_RC_COMPILER NAMES ${MINGW_COMPILER_PREFIX}-windres)
find_program(CMAKE_C_COMPILER NAMES ${MINGW_COMPILER_PREFIX}-gcc)
find_program(CMAKE_CXX_COMPILER NAMES ${MINGW_COMPILER_PREFIX}-g++)

SET(CMAKE_FIND_ROOT_PATH ${MINGW_SYSROOT})

# Adjust the default behaviour of the FIND_XXX() commands:
# Search headers and libraries in the target environment; search
# programs in the host environment.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
