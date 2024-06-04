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

#ifndef COMMON_WINDOWS_PE_UTIL_H_
#define COMMON_WINDOWS_PE_UTIL_H_

#include <windows.h>

#include "common/windows/module_info.h"

namespace google_breakpad {

using std::wstring;

// Reads |pe_file| and populates |info|. Returns true on success.
// Only supports PE32+ format, ie. a 64bit PE file.
// Will fail if |pe_file| does not contain a valid CodeView record.
bool ReadModuleInfo(const wstring& pe_file, PDBModuleInfo* info);

// Reads |pe_file| and populates |info|. Returns true on success.
bool ReadPEInfo(const wstring& pe_file, PEModuleInfo* info);

// Reads |pe_file| and prints frame data (aka. unwind info) to |out_file|.
// Only supports PE32+ format, ie. a 64bit PE file.
bool PrintPEFrameData(const wstring& pe_file, FILE* out_file);

// Combines a GUID |signature| and DWORD |age| to create a Breakpad debug
// identifier.
wstring GenerateDebugIdentifier(DWORD age, GUID signature);

// Combines a DWORD |signature| and DWORD |age| to create a Breakpad debug
// identifier.
wstring GenerateDebugIdentifier(DWORD age, DWORD signature);

// Converts |machine| enum value to the corresponding string used by Breakpad.
// The enum is IMAGE_FILE_MACHINE_*, contained in winnt.h.
constexpr const wchar_t* FileHeaderMachineToCpuString(WORD machine) {
  switch (machine) {
    case IMAGE_FILE_MACHINE_I386: {
      return L"x86";
    }
    case IMAGE_FILE_MACHINE_IA64:
    case IMAGE_FILE_MACHINE_AMD64: {
      return L"x86_64";
    }
    default: { return L"unknown"; }
  }
}

}  // namespace google_breakpad

#endif  // COMMON_WINDOWS_PE_UTIL_H_
