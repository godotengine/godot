// Copyright 2019 Google LLC
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
//     * Neither the name of Google LLC nor the names of its
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

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/windows/pe_source_line_writer.h"

#include "common/windows/pe_util.h"

namespace google_breakpad {
PESourceLineWriter::PESourceLineWriter(const wstring& pe_file) :
  pe_file_(pe_file) {
}

PESourceLineWriter::~PESourceLineWriter() {
}

bool PESourceLineWriter::WriteSymbols(FILE* symbol_file) {
  PDBModuleInfo module_info;
  if (!GetModuleInfo(&module_info)) {
    return false;
  }
  // Hard-code "windows" for the OS because that's the only thing that makes
  // sense for PDB files.  (This might not be strictly correct for Windows CE
  // support, but we don't care about that at the moment.)
  fprintf(symbol_file, "MODULE windows %ws %ws %ws\n",
    module_info.cpu.c_str(), module_info.debug_identifier.c_str(),
    module_info.debug_file.c_str());

  PEModuleInfo pe_info;
  if (!GetPEInfo(&pe_info)) {
    return false;
  }
  fprintf(symbol_file, "INFO CODE_ID %ws %ws\n",
    pe_info.code_identifier.c_str(),
    pe_info.code_file.c_str());

  if (!PrintPEFrameData(pe_file_, symbol_file)) {
    return false;
  }

  return true;
}

bool PESourceLineWriter::GetModuleInfo(PDBModuleInfo* info) {
  return ReadModuleInfo(pe_file_, info);
}

bool PESourceLineWriter::GetPEInfo(PEModuleInfo* info) {
  return ReadPEInfo(pe_file_, info);
}

}  // namespace google_breakpad
