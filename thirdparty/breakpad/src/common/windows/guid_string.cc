// Copyright (c) 2006, Google Inc.
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

// guid_string.cc: Convert GUIDs to strings.
//
// See guid_string.h for documentation.

#include <wchar.h>

#include "common/windows/string_utils-inl.h"

#include "common/windows/guid_string.h"

namespace google_breakpad {

// static
wstring GUIDString::GUIDToWString(GUID *guid) {
  wchar_t guid_string[37];
  swprintf(
      guid_string, sizeof(guid_string) / sizeof(guid_string[0]),
      L"%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      guid->Data1, guid->Data2, guid->Data3,
      guid->Data4[0], guid->Data4[1], guid->Data4[2],
      guid->Data4[3], guid->Data4[4], guid->Data4[5],
      guid->Data4[6], guid->Data4[7]);

  // remove when VC++7.1 is no longer supported
  guid_string[sizeof(guid_string) / sizeof(guid_string[0]) - 1] = L'\0';

  return wstring(guid_string);
}

// static
wstring GUIDString::GUIDToSymbolServerWString(GUID *guid) {
  wchar_t guid_string[33];
  swprintf(
      guid_string, sizeof(guid_string) / sizeof(guid_string[0]),
      L"%08X%04X%04X%02X%02X%02X%02X%02X%02X%02X%02X",
      guid->Data1, guid->Data2, guid->Data3,
      guid->Data4[0], guid->Data4[1], guid->Data4[2],
      guid->Data4[3], guid->Data4[4], guid->Data4[5],
      guid->Data4[6], guid->Data4[7]);

  // remove when VC++7.1 is no longer supported
  guid_string[sizeof(guid_string) / sizeof(guid_string[0]) - 1] = L'\0';

  return wstring(guid_string);
}

}  // namespace google_breakpad
