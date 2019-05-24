#!/usr/bin/env sh
# Copyright (c) 2016 The Khronos Group Inc.

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

# A script for automatically disassembling a .spv file
# for less(1).  This assumes spirv-dis is on our PATH.
#
# See https://github.com/KhronosGroup/SPIRV-Tools/issues/359

case "$1" in
    *.spv) spirv-dis "$@" 2>/dev/null;;
    *) exit 1;;
esac

exit $?

