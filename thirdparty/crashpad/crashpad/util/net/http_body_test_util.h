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

#ifndef CRASHPAD_UTIL_NET_HTTP_BODY_TEST_UTIL_H_
#define CRASHPAD_UTIL_NET_HTTP_BODY_TEST_UTIL_H_

#include <sys/types.h>

#include <string>

namespace crashpad {

class HTTPBodyStream;

namespace test {

//! \brief Reads a HTTPBodyStream to a string. If an error occurs, adds a
//!     test failure and returns an empty string.
//!
//! \param[in] stream The stream from which to read.
//!
//! \return The contents of the stream, or an empty string on failure.
std::string ReadStreamToString(HTTPBodyStream* stream);

//! \brief Reads a HTTPBodyStream to a string. If an error occurs, adds a
//!     test failure and returns an empty string.
//!
//! \param[in] stream The stream from which to read.
//! \param[in] buffer_size The size of the buffer to use when reading from the
//!     stream.
//!
//! \return The contents of the stream, or an empty string on failure.
std::string ReadStreamToString(HTTPBodyStream* stream, size_t buffer_size);

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_UTIL_NET_HTTP_BODY_TEST_UTIL_H_
