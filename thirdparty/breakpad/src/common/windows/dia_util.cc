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

#include "common/windows/dia_util.h"

#include <atlbase.h>

namespace google_breakpad {

bool FindDebugStream(const wchar_t* name,
                     IDiaSession* session,
                     IDiaEnumDebugStreamData** debug_stream) {
  CComPtr<IDiaEnumDebugStreams> enum_debug_streams;
  if (FAILED(session->getEnumDebugStreams(&enum_debug_streams))) {
    fprintf(stderr, "IDiaSession::getEnumDebugStreams failed\n");
    return false;
  }

  CComPtr<IDiaEnumDebugStreamData> temp_debug_stream;
  ULONG fetched = 0;
  while (SUCCEEDED(enum_debug_streams->Next(1, &temp_debug_stream, &fetched)) &&
         fetched == 1) {
    CComBSTR stream_name;
    if (FAILED(temp_debug_stream->get_name(&stream_name))) {
      fprintf(stderr, "IDiaEnumDebugStreamData::get_name failed\n");
      return false;
    }

    // Found the stream?
    if (wcsncmp((LPWSTR)stream_name, name, stream_name.Length()) == 0) {
      *debug_stream = temp_debug_stream.Detach();
      return true;
    }

    temp_debug_stream.Release();
  }

  // No table was found.
  return false;
}

bool FindTable(REFIID iid, IDiaSession* session, void** table) {
  // Get the table enumerator.
  CComPtr<IDiaEnumTables> enum_tables;
  if (FAILED(session->getEnumTables(&enum_tables))) {
    fprintf(stderr, "IDiaSession::getEnumTables failed\n");
    return false;
  }

  // Iterate through the tables.
  CComPtr<IDiaTable> temp_table;
  ULONG fetched = 0;
  while (SUCCEEDED(enum_tables->Next(1, &temp_table, &fetched)) &&
         fetched == 1) {
    void* temp = NULL;
    if (SUCCEEDED(temp_table->QueryInterface(iid, &temp))) {
      *table = temp;
      return true;
    }
    temp_table.Release();
  }

  // The table was not found.
  return false;
}

}  // namespace google_breakpad