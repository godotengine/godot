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

#ifndef COMMON_LINUX_SYMBOL_COLLECTOR_CLIENT_H_
#define COMMON_LINUX_SYMBOL_COLLECTOR_CLIENT_H_

#include <string>

#include "common/linux/libcurl_wrapper.h"
#include "common/using_std_string.h"

namespace google_breakpad {
namespace sym_upload {

struct UploadUrlResponse {
  string upload_url;
  string upload_key;
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

// Helper class to communicate with a sym-upload-v2 service over HTTP/REST,
// via libcurl.
class SymbolCollectorClient {
 public:
  static bool CreateUploadUrl(
      LibcurlWrapper* libcurl_wrapper,
      const string& api_url,
      const string& api_key,
      UploadUrlResponse* uploadUrlResponse);

  static CompleteUploadResult CompleteUpload(
      LibcurlWrapper* libcurl_wrapper,
      const string& api_url,
      const string& api_key,
      const string& upload_key,
      const string& debug_file,
      const string& debug_id,
      const string& type);

  static SymbolStatus CheckSymbolStatus(
      LibcurlWrapper* libcurl_wrapper,
      const string& api_url,
      const string& api_key,
      const string& debug_file,
      const string& debug_id);
};

}  // namespace sym_upload
}  // namespace google_breakpad

#endif  // COMMON_LINUX_SYMBOL_COLLECTOR_CLIENT_H_
