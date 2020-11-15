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

#ifndef CRASHPAD_UTIL_NET_URL_H_
#define CRASHPAD_UTIL_NET_URL_H_

#include <string>

namespace crashpad {

//! \brief Performs percent-encoding (URL encoding) on the input string,
//!     following RFC 3986 paragraph 2.
//!
//! \param[in] url The string to be encoded.
//! \return The encoded string.
std::string URLEncode(const std::string& url);

//! \brief Crack a URL into component parts.
//!
//! This is not a general function, and works only on the limited style of URLs
//! that are expected to be used by HTTPTransport::SetURL().
//!
//! \param[in] url The URL to crack.
//! \param[out] scheme The request scheme, either http or https.
//! \param[out] host The hostname.
//! \param[out] port The port.
//! \param[out] rest The remainder of the URL (both resource and URL params).
//! \return `true` on success in which case all output parameters will be filled
//!     out, or `false` on failure, in which case the output parameters will be
//!     unmodified and an error will be logged.
bool CrackURL(const std::string& url,
              std::string* scheme,
              std::string* host,
              std::string* port,
              std::string* rest);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_NET_URL_H_
