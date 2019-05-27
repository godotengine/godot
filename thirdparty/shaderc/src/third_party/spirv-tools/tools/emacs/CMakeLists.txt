# Copyright (c) 2016 LunarG Inc.
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

# Install a script for use with the auto-compression feature of emacs(1).
# Upon loading a file with the .spv extension, the file will be disassembled
# using spirv-dis, and the result colorized with asm-mode in emacs.  The file
# may be edited within the constraints of validity, and when re-saved will be
# re-assembled using spirv-as.

# It is required that those tools be in your PATH.  If that is not the case
# when starting emacs, the path can be modified as in this example:
# (setenv "PATH" (concat (getenv "PATH") ":/path/to/spirv/tools"))
#
# See https://github.com/KhronosGroup/SPIRV-Tools/issues/359

# This is an absolute directory, and ignores CMAKE_INSTALL_PREFIX, or
# it will not be found by emacs upon startup.  It is only installed if
# both of the following are true:
#   1. SPIRV_TOOLS_INSTALL_EMACS_HELPERS is defined
#   2. The directory /etc/emacs/site-start.d already exists at the time of
#      cmake invocation (not at the time of make install).  This is
#      typically true if emacs is installed on the system.

# Note that symbol IDs are not preserved through a load/edit/save operation.
# This may change if the ability is added to spirv-as.

option(SPIRV_TOOLS_INSTALL_EMACS_HELPERS
  "Install Emacs helper to disassemble/assemble SPIR-V binaries on file load/save."
  ${SPIRV_TOOLS_INSTALL_EMACS_HELPERS})
if (${SPIRV_TOOLS_INSTALL_EMACS_HELPERS})
  if(EXISTS /etc/emacs/site-start.d)
    if(ENABLE_SPIRV_TOOLS_INSTALL)
      install(FILES 50spirv-tools.el DESTINATION /etc/emacs/site-start.d)
    endif(ENABLE_SPIRV_TOOLS_INSTALL)
  endif()
endif()

