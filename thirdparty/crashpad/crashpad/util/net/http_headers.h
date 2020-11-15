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

#ifndef CRASHPAD_UTIL_NET_HTTP_HEADERS_H_
#define CRASHPAD_UTIL_NET_HTTP_HEADERS_H_

#include <map>
#include <string>

namespace crashpad {

//! \brief A map of HTTP header fields to their values.
using HTTPHeaders = std::map<std::string, std::string>;

//! \brief The header name `"Content-Type"`.
constexpr char kContentType[] = "Content-Type";

//! \brief The header name `"Content-Length"`.
constexpr char kContentLength[] = "Content-Length";

//! \brief The header name `"Content-Encoding"`.
constexpr char kContentEncoding[] = "Content-Encoding";

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_NET_HTTP_HEADERS_H_
