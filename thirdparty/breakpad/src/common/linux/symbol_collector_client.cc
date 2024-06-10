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

#include "common/linux/symbol_collector_client.h"

#include <stdio.h>

#include <iostream>
#include <regex>

#include "common/linux/libcurl_wrapper.h"

namespace google_breakpad {
namespace sym_upload {

// static
bool SymbolCollectorClient::CreateUploadUrl(
    LibcurlWrapper* libcurl_wrapper,
    const string& api_url,
    const string& api_key,
    UploadUrlResponse* uploadUrlResponse) {
  string header, response;
  long response_code;

  string url = api_url + "/v1/uploads:create";
  if (!api_key.empty()) {
    url += "?key=" + api_key;
  }

  if (!libcurl_wrapper->SendSimplePostRequest(url,
                                              /*body=*/"",
                                              /*content_type=*/"",
                                              &response_code,
                                              &header,
                                              &response)) {
    printf("Failed to create upload url.\n");
    printf("Response code: %ld\n", response_code);
    printf("Response:\n");
    printf("%s\n", response.c_str());
    return false;
  }

  // Note camel-case rather than underscores.
  std::regex upload_url_regex("\"uploadUrl\": \"([^\"]+)\"");
  std::regex upload_key_regex("\"uploadKey\": \"([^\"]+)\"");

  std::smatch upload_url_match;
  if (!std::regex_search(response, upload_url_match, upload_url_regex) ||
      upload_url_match.size() != 2) {
    printf("Failed to parse create url response.");
    printf("Response:\n");
    printf("%s\n", response.c_str());
    return false;
  }
  string upload_url = upload_url_match[1].str();

  std::smatch upload_key_match;
  if (!std::regex_search(response, upload_key_match, upload_key_regex) ||
      upload_key_match.size() != 2) {
    printf("Failed to parse create url response.");
    printf("Response:\n");
    printf("%s\n", response.c_str());
    return false;
  }
  string upload_key = upload_key_match[1].str();

  uploadUrlResponse->upload_url = upload_url;
  uploadUrlResponse->upload_key = upload_key;
  return true;
}

// static
CompleteUploadResult SymbolCollectorClient::CompleteUpload(
    LibcurlWrapper* libcurl_wrapper,
    const string& api_url,
    const string& api_key,
    const string& upload_key,
    const string& debug_file,
    const string& debug_id,
    const string& type) {
  string header, response;
  long response_code;

  string url = api_url + "/v1/uploads/" + upload_key + ":complete";
  if (!api_key.empty()) {
    url += "?key=" + api_key;
  }
  string body =
      "{ symbol_id: {"
      "debug_file: \"" + debug_file + "\", "
      "debug_id: \"" + debug_id + "\" }, "
      "symbol_upload_type: \"" + type + "\" }";

  if (!libcurl_wrapper->SendSimplePostRequest(url,
                                              body,
                                              "application/son",
                                              &response_code,
                                              &header,
                                              &response)) {
    printf("Failed to complete upload.\n");
    printf("Response code: %ld\n", response_code);
    printf("Response:\n");
    printf("%s\n", response.c_str());
    return CompleteUploadResult::Error;
  }

  std::regex result_regex("\"result\": \"([^\"]+)\"");
  std::smatch result_match;
  if (!std::regex_search(response, result_match, result_regex) ||
      result_match.size() != 2) {
    printf("Failed to parse complete upload response.");
    printf("Response:\n");
    printf("%s\n", response.c_str());
    return CompleteUploadResult::Error;
  }
  string result = result_match[1].str();

  if (result.compare("DUPLICATE_DATA") == 0) {
    return CompleteUploadResult::DuplicateData;
  }

  return CompleteUploadResult::Ok;
}

// static
SymbolStatus SymbolCollectorClient::CheckSymbolStatus(
    LibcurlWrapper* libcurl_wrapper,
    const string& api_url,
    const string& api_key,
    const string& debug_file,
    const string& debug_id) {
  string header, response;
  long response_code;
  string url = api_url +
               "/v1/symbols/" + debug_file + "/" + debug_id + ":checkStatus";
  if (!api_key.empty()) {
    url += "?key=" + api_key;
  }

  if (!libcurl_wrapper->SendGetRequest(
      url,
      &response_code,
      &header,
      &response)) {
    printf("Failed to check symbol status, error message.\n");
    printf("Response code: %ld\n", response_code);
    printf("Response:\n");
    printf("%s\n", response.c_str());
    return SymbolStatus::Unknown;
  }

  std::regex status_regex("\"status\": \"([^\"]+)\"");
  std::smatch status_match;
  if (!std::regex_search(response, status_match, status_regex) ||
      status_match.size() != 2) {
    printf("Failed to parse check symbol status response.");
    printf("Response:\n");
    printf("%s\n", response.c_str());
    return SymbolStatus::Unknown;
  }
  string status = status_match[1].str();

  return (status.compare("FOUND") == 0) ?
      SymbolStatus::Found :
      SymbolStatus::Missing;
}

}  // namespace sym_upload
}  // namespace google_breakpad
