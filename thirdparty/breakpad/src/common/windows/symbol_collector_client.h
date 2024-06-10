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

#ifndef COMMON_WINDOWS_SYMBOL_COLLECTOR_CLIENT_H_
#define COMMON_WINDOWS_SYMBOL_COLLECTOR_CLIENT_H_

#include <string>

namespace google_breakpad {

  using std::wstring;

  struct UploadUrlResponse {
    // URL at which to HTTP PUT symbol file.
    wstring upload_url;
    // Unique key used to complete upload of symbol file.
    wstring upload_key;
  };

  enum SymbolStatus {
    Found,
    Missing,
    Unknown
  };

  enum CompleteUploadResult {
    Ok,
    DuplicateData,
    Error
  };

  // Client to interact with sym-upload-v2 API server via HTTP/REST.
  class SymbolCollectorClient {
  public:
    // Returns a URL at which a symbol file can be HTTP PUT without
    // authentication, along with an upload key that can be used to
    // complete the upload process with CompleteUpload.
    static bool CreateUploadUrl(
        wstring& api_url,
        wstring& api_key,
        int* timeout_ms,
        UploadUrlResponse *uploadUrlResponse);

    // Notify the API that symbol file upload is finished and its contents
    // are ready to be read and/or used for further processing.
    static CompleteUploadResult CompleteUpload(wstring& api_url,
                                               wstring& api_key,
                                               int* timeout_ms,
                                               const wstring& upload_key,
                                               const wstring& debug_file,
                                               const wstring& debug_id,
                                               const wstring& type,
                                               const wstring& product_name);

    // Returns whether or not a symbol file corresponding to the debug_file/
    // debug_id pair is already present in symbol storage.
    static SymbolStatus CheckSymbolStatus(
        wstring& api_url,
        wstring& api_key,
        int* timeout_ms,
        const wstring& debug_file,
        const wstring& debug_id);
  };

}  // namespace google_breakpad

#endif  // COMMON_WINDOWS_SYMBOL_COLLECTOR_CLIENT_H_
