// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "util/net/url.h"

#include <string.h>

#include "base/logging.h"
#include "base/strings/stringprintf.h"

namespace crashpad {

std::string URLEncode(const std::string& url) {
  const char kSafeCharacters[] = "-_.~";
  std::string encoded;
  encoded.reserve(url.length());

  for (unsigned char character : url) {
    if (((character >= 'A') && (character <= 'Z')) ||
        ((character >= 'a') && (character <= 'z')) ||
        ((character >= '0') && (character <= '9')) ||
        (strchr(kSafeCharacters, character) != nullptr)) {
      // Copy unreserved character.
      encoded += character;
    } else {
      // Escape character.
      encoded += base::StringPrintf("%%%02X", character);
    }
  }

  return encoded;
}

bool CrackURL(const std::string& url,
              std::string* scheme,
              std::string* host,
              std::string* port,
              std::string* rest) {
  std::string result_scheme;
  std::string result_port;

  size_t host_start;
  static constexpr const char kHttp[] = "http://";
  static constexpr const char kHttps[] = "https://";
  if (url.compare(0, strlen(kHttp), kHttp) == 0) {
    result_scheme = "http";
    result_port = "80";
    host_start = strlen(kHttp);
  } else if (url.compare(0, strlen(kHttps), kHttps) == 0) {
    result_scheme = "https";
    result_port = "443";
    host_start = strlen(kHttps);
  } else {
    LOG(ERROR) << "expecting http or https";
    return false;
  }

  // Find the start of the resource.
  size_t resource_start = url.find('/', host_start);
  if (resource_start == std::string::npos) {
    LOG(ERROR) << "no resource component";
    return false;
  }

  scheme->swap(result_scheme);
  port->swap(result_port);
  std::string host_and_possible_port =
      url.substr(host_start, resource_start - host_start);
  size_t colon = host_and_possible_port.find(':');
  if (colon == std::string::npos) {
    *host = host_and_possible_port;
  } else {
    *host = host_and_possible_port.substr(0, colon);
    *port = host_and_possible_port.substr(colon + 1);
  }

  *rest = url.substr(resource_start);
  return true;
}

}  // namespace crashpad
