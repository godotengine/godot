// Copyright (c) 2019, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef COMMON_WINDOWS_MODULE_INFO_H_
#define COMMON_WINDOWS_MODULE_INFO_H_

#include <string>

namespace google_breakpad {

using std::wstring;
// A structure that carries information that identifies a module.
struct PDBModuleInfo {
public:
  // The basename of the pe/pdb file from which information was loaded.
  wstring debug_file;

  // The module's identifier.  For recent pe/pdb files, the identifier consists
  // of the pe/pdb's guid, in uppercase hexadecimal form without any dashes
  // or separators, followed immediately by the pe/pdb's age, also in
  // uppercase hexadecimal form.  For older pe/pdb files which have no guid,
  // the identifier is the pe/pdb's 32-bit signature value, in zero-padded
  // hexadecimal form, followed immediately by the pe/pdb's age, in lowercase
  // hexadecimal form.
  wstring debug_identifier;

  // A string identifying the cpu that the pe/pdb is associated with.
  // Currently, this may be "x86" or "unknown".
  wstring cpu;
};

// A structure that carries information that identifies a PE file,
// either an EXE or a DLL.
struct PEModuleInfo {
  // The basename of the PE file.
  wstring code_file;

  // The PE file's code identifier, which consists of its timestamp
  // and file size concatenated together into a single hex string.
  // (The fields IMAGE_OPTIONAL_HEADER::SizeOfImage and
  // IMAGE_FILE_HEADER::TimeDateStamp, as defined in the ImageHlp
  // documentation.) This is not well documented, if it's documented
  // at all, but it's what symstore does and what DbgHelp supports.
  wstring code_identifier;
};

}  // namespace google_breakpad

#endif  // COMMON_WINDOWS_MODULE_INFO_H_
