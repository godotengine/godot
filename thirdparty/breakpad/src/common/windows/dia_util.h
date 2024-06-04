// Copyright 2013 Google Inc. All rights reserved.
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

// Utilities for loading debug streams and tables from a PDB file.

#ifndef COMMON_WINDOWS_DIA_UTIL_H_
#define COMMON_WINDOWS_DIA_UTIL_H_

#include <Windows.h>
#include <dia2.h>

namespace google_breakpad {

// Find the debug stream of the given |name| in the given |session|. Returns
// true on success, false on error of if the stream does not exist. On success
// the stream will be returned via |debug_stream|.
bool FindDebugStream(const wchar_t* name,
                     IDiaSession* session,
                     IDiaEnumDebugStreamData** debug_stream);

// Finds the first table implementing the COM interface with ID |iid| in the
// given |session|. Returns true on success, false on error or if no such
// table is found. On success the table will be returned via |table|.
bool FindTable(REFIID iid, IDiaSession* session, void** table);

// A templated version of FindTable. Finds the first table implementing type
// |InterfaceType| in the given |session|. Returns true on success, false on
// error or if no such table is found. On success the table will be returned via
// |table|.
template<typename InterfaceType>
bool FindTable(IDiaSession* session, InterfaceType** table) {
  return FindTable(__uuidof(InterfaceType),
                   session,
                   reinterpret_cast<void**>(table));
}

}  // namespace google_breakpad

#endif  // COMMON_WINDOWS_DIA_UTIL_H_
