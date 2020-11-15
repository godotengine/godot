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

#ifndef CRASHPAD_TEST_ERRORS_H_
#define CRASHPAD_TEST_ERRORS_H_

#include <string>

#include "build/build_config.h"

namespace crashpad {
namespace test {

// These functions format messages in a similar way to the PLOG and PCHECK
// family of logging macros in base/logging.h. They exist to interoperate with
// gtest assertions, which don’t interoperate with logging but can be streamed
// to.
//
// Where non-test code could do:
//   PCHECK(rv == 0) << "close";
// gtest-based test code can do:
//   EXPECT_EQ(rv, 0) << ErrnoMessage("close");

//! \brief Formats an error message using an `errno` value.
//!
//! The returned string will combine the \a base string, if supplied, with a
//! textual and numeric description of the error.
//!
//! The message is formatted using `strerror()`. \a err may be `0` or outside of
//! the range of known error codes, and the message returned will contain the
//! string that `strerror()` uses in these cases.
//!
//! \param[in] err The error code, usable as an `errno` value.
//! \param[in] base A string to prepend to the error description.
//!
//! \return A string of the format `"Operation not permitted (1)"` if \a err has
//!     the value `EPERM` on a system where this is defined to be `1`. If \a
//!     base is not empty, it will be prepended to this string, separated by a
//!     colon.
std::string ErrnoMessage(int err, const std::string& base = std::string());

//! \brief Formats an error message using `errno`.
//!
//! The returned string will combine the \a base string, if supplied, with a
//! textual and numeric description of the error.
//!
//! The message is formatted using `strerror()`. `errno` may be `0` or outside
//! of the range of known error codes, and the message returned will contain the
//! string that `strerror()` uses in these cases.
//!
//! \param[in] base A string to prepend to the error description.
//!
//! \return A string of the format `"Operation not permitted (1)"` if `errno`
//!     has the value `EPERM` on a system where this is defined to be `1`. If
//!     \a base is not empty, it will be prepended to this string, separated by
//!     a colon.
std::string ErrnoMessage(const std::string& base = std::string());

#if defined(OS_WIN) || DOXYGEN
//! \brief Formats an error message using `GetLastError()`.
//!
//! The returned string will combine the \a base string, if supplied, with a
//! textual and numeric description of the error. The format is the same as the
//! `PLOG()` formatting in base.
std::string ErrorMessage(const std::string& base = std::string());
#endif

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_ERRORS_H_
