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

#ifndef COMMON_WINDOWS_PE_SOURCE_LINE_WRITER_H_
#define COMMON_WINDOWS_PE_SOURCE_LINE_WRITER_H_

#include <string>

#include "common/basictypes.h"
#include "common/windows/module_info.h"

namespace google_breakpad {

using std::wstring;

// PESourceLineWriter uses a pe file produced by Visual C++ to output
// a line/address map for use with BasicSourceLineResolver.
// NOTE: Only supports PE32+ format, ie. a 64bit PE file.
class PESourceLineWriter {
public:
  explicit PESourceLineWriter(const wstring& pe_file);
  ~PESourceLineWriter();

  // Writes Breakpad symbols from the pe file to |symbol_file|.
  // Returns true on success.
  bool WriteSymbols(FILE* symbol_file);

  // Retrieves information about the module. Returns true on success.
  bool GetModuleInfo(PDBModuleInfo* info);

  // Retrieves information about the module's PE file.  Returns
  // true on success.
  bool GetPEInfo(PEModuleInfo* info);

private:
  const wstring pe_file_;

  DISALLOW_COPY_AND_ASSIGN(PESourceLineWriter);
};

}  // namespace google_breakpad

#endif  // COMMON_WINDOWS_PE_SOURCE_LINE_WRITER_H_
