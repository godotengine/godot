# Copyright (c) 2016 Google Inc.
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

# Install a script for use with the LESSOPEN of less(1).
# For example, after installation into /usr/local do:
#    export LESSOPEN='|/usr/local/bin "%s"'
#    less -R foo.spv
#
# See https://github.com/KhronosGroup/SPIRV-Tools/issues/359

# The script will be installed with everyone having read and execute
# permissions.
# We have a .sh extension because Windows users often configure
# executable settings via filename extension.
if(ENABLE_SPIRV_TOOLS_INSTALL)
  install(PROGRAMS spirv-lesspipe.sh DESTINATION ${CMAKE_INSTALL_BINDIR})
endif(ENABLE_SPIRV_TOOLS_INSTALL)
