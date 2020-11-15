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

#ifndef CRASHPAD_UTIL_MISC_ZLIB_H_
#define CRASHPAD_UTIL_MISC_ZLIB_H_

#include <string>

namespace crashpad {

//! \brief Obtain a \a window_bits parameter to pass to `deflateInit2()` or
//!     `inflateInit2()` that specifies a `gzip` wrapper instead of the default
//!     zlib wrapper.
//!
//! \param[in] window_bits A \a window_bits value that only specifies the base-2
//!     logarithm of the deflate sliding window size.
//!
//! \return \a window_bits adjusted to specify a `gzip` wrapper, to be passed to
//!     `deflateInit2()` or `inflateInit2()`.
int ZlibWindowBitsWithGzipWrapper(int window_bits);

//! \brief Formats a string for an error received from the zlib library.
//!
//! \param[in] zr A zlib result code, such as `Z_STREAM_ERROR`.
//!
//! \return A formatted string.
std::string ZlibErrorString(int zr);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_ZLIB_H_
