// Copyright 2014 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "util/net/http_multipart_builder.h"

#include <sys/types.h>

#include <utility>
#include <vector>

#include "base/logging.h"
#include "base/rand_util.h"
#include "base/strings/stringprintf.h"
#include "util/net/http_body.h"
#include "util/net/http_body_gzip.h"

namespace crashpad {

namespace {

constexpr char kCRLF[] = "\r\n";

constexpr char kBoundaryCRLF[] = "\r\n\r\n";

// Generates a random string suitable for use as a multipart boundary.
std::string GenerateBoundaryString() {
  // RFC 2046 §5.1.1 says that the boundary string may be 1 to 70 characters
  // long, choosing from the set of alphanumeric characters along with
  // characters from the set “'()+_,-./:=? ”, and not ending in a space.
  // However, some servers have been observed as dealing poorly with certain
  // nonalphanumeric characters. See
  // blink/Source/platform/network/FormDataEncoder.cpp
  // blink::FormDataEncoder::GenerateUniqueBoundaryString().
  //
  // This implementation produces a 56-character string with over 190 bits of
  // randomness (62^32 > 2^190).
  std::string boundary_string = "---MultipartBoundary-";
  for (int index = 0; index < 32; ++index) {
    static constexpr char kCharacters[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    int random_value =
        base::RandInt(0, static_cast<int>(strlen(kCharacters)) - 1);
    boundary_string += kCharacters[random_value];
  }
  boundary_string += "---";
  return boundary_string;
}

// Escapes the specified name to be suitable for the name field of a
// form-data part.
std::string EncodeMIMEField(const std::string& name) {
  // This URL-escapes the quote character and newline characters, per Blink. See
  // blink/Source/platform/network/FormDataEncoder.cpp
  // blink::AppendQuotedString(). %-encoding is endorsed by RFC 7578 §2, with
  // approval for otherwise unencoded UTF-8 given by RFC 7578 §5.1. Blink does
  // not escape the '%' character, but it seems appropriate to do so in order to
  // be able to decode the string properly.
  std::string encoded;
  for (char character : name) {
    switch (character) {
      case '\r':
      case '\n':
      case '"':
      case '%':
        encoded += base::StringPrintf("%%%02x", character);
        break;
      default:
        encoded += character;
        break;
    }
  }

  return encoded;
}

// Returns a string, formatted with a multipart boundary and a field name,
// after which the contents of the part at |name| can be appended.
std::string GetFormDataBoundary(const std::string& boundary,
                                const std::string& name) {
  return base::StringPrintf(
      "--%s%sContent-Disposition: form-data; name=\"%s\"",
      boundary.c_str(),
      kCRLF,
      EncodeMIMEField(name).c_str());
}

void AssertSafeMIMEType(const std::string& string) {
  for (size_t i = 0; i < string.length(); ++i) {
    char c = string[i];
    CHECK((c >= 'a' && c <= 'z') ||
          (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') ||
          c == '/' ||
          c == '.' ||
          c == '_' ||
          c == '+' ||
          c == '-');
  }
}

}  // namespace

HTTPMultipartBuilder::HTTPMultipartBuilder()
    : boundary_(GenerateBoundaryString()),
      form_data_(),
      file_attachments_(),
      gzip_enabled_(false) {}

HTTPMultipartBuilder::~HTTPMultipartBuilder() {
}

void HTTPMultipartBuilder::SetGzipEnabled(bool gzip_enabled) {
  gzip_enabled_ = gzip_enabled;
}

void HTTPMultipartBuilder::SetFormData(const std::string& key,
                                       const std::string& value) {
  EraseKey(key);
  form_data_[key] = value;
}

void HTTPMultipartBuilder::SetFileAttachment(
    const std::string& key,
    const std::string& upload_file_name,
    FileReaderInterface* reader,
    const std::string& content_type) {
  EraseKey(upload_file_name);

  FileAttachment attachment;
  attachment.filename = EncodeMIMEField(upload_file_name);
  attachment.reader = reader;

  if (content_type.empty()) {
    attachment.content_type = "application/octet-stream";
  } else {
    AssertSafeMIMEType(content_type);
    attachment.content_type = content_type;
  }

  file_attachments_[key] = attachment;
}

std::unique_ptr<HTTPBodyStream> HTTPMultipartBuilder::GetBodyStream() {
  // The objects inserted into this vector will be owned by the returned
  // CompositeHTTPBodyStream. Take care to not early-return without deleting
  // this memory.
  std::vector<HTTPBodyStream*> streams;

  for (const auto& pair : form_data_) {
    std::string field = GetFormDataBoundary(boundary_, pair.first);
    field += kBoundaryCRLF;
    field += pair.second;
    field += kCRLF;
    streams.push_back(new StringHTTPBodyStream(field));
  }

  for (const auto& pair : file_attachments_) {
    const FileAttachment& attachment = pair.second;
    std::string header = GetFormDataBoundary(boundary_, pair.first);
    header += base::StringPrintf("; filename=\"%s\"%s",
        attachment.filename.c_str(), kCRLF);
    header += base::StringPrintf("Content-Type: %s%s",
        attachment.content_type.c_str(), kBoundaryCRLF);

    streams.push_back(new StringHTTPBodyStream(header));
    streams.push_back(new FileReaderHTTPBodyStream(attachment.reader));
    streams.push_back(new StringHTTPBodyStream(kCRLF));
  }

  streams.push_back(
      new StringHTTPBodyStream("--"  + boundary_ + "--" + kCRLF));

  auto composite =
      std::unique_ptr<HTTPBodyStream>(new CompositeHTTPBodyStream(streams));
  if (gzip_enabled_) {
    return std::unique_ptr<HTTPBodyStream>(
        new GzipHTTPBodyStream(std::move(composite)));
  }
  return composite;
}

void HTTPMultipartBuilder::PopulateContentHeaders(
    HTTPHeaders* http_headers) const {
  std::string content_type =
      base::StringPrintf("multipart/form-data; boundary=%s", boundary_.c_str());
  (*http_headers)[kContentType] = content_type;

  if (gzip_enabled_) {
    (*http_headers)[kContentEncoding] = "gzip";
  }
}

void HTTPMultipartBuilder::EraseKey(const std::string& key) {
  auto data_it = form_data_.find(key);
  if (data_it != form_data_.end())
    form_data_.erase(data_it);

  auto file_it = file_attachments_.find(key);
  if (file_it != file_attachments_.end())
    file_attachments_.erase(file_it);
}

}  // namespace crashpad
