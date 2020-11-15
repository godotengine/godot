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

#ifndef CRASHPAD_MINIDUMP_TEST_MINIDUMP_STRING_WRITER_TEST_UTIL_H_
#define CRASHPAD_MINIDUMP_TEST_MINIDUMP_STRING_WRITER_TEST_UTIL_H_

#include <windows.h>
#include <dbghelp.h>

#include <string>

#include "base/strings/string16.h"

namespace crashpad {

struct MinidumpUTF8String;

namespace test {

//! \brief Returns a MINIDUMP_STRING located within a minidump file’s contents.
//!
//! If \a rva points outside of the range of \a file_contents, if the string has
//! an incorrect length or is not `NUL`-terminated, or if any of the string data
//! would lie outside of the range of \a file_contents, this function will fail.
//!
//! \param[in] file_contents The contents of the minidump file.
//! \param[in] rva The offset within the minidump file of the desired
//!     MINIDUMP_STRING.
//!
//! \return On success, a pointer to the MINIDUMP_STRING in \a file_contents. On
//!     failure, raises a gtest assertion and returns `nullptr`.
//!
//! \sa MinidumpStringAtRVAAsString()
//! \sa MinidumpUTF8StringAtRVA()
const MINIDUMP_STRING* MinidumpStringAtRVA(const std::string& file_contents,
                                           RVA rva);

//! \brief Returns a MinidumpUTF8String located within a minidump file’s
//!     contents.
//!
//! If \a rva points outside of the range of \a file_contents, if the string has
//! an incorrect length or is not `NUL`-terminated, or if any of the string data
//! would lie outside of the range of \a file_contents, this function will fail.
//!
//! \param[in] file_contents The contents of the minidump file.
//! \param[in] rva The offset within the minidump file of the desired
//!     MinidumpUTF8String.
//!
//! \return On success, a pointer to the MinidumpUTF8String in \a file_contents.
//!     On failure, raises a gtest assertion and returns `nullptr`.
//!
//! \sa MinidumpUTF8StringAtRVAAsString()
//! \sa MinidumpStringAtRVA()
const MinidumpUTF8String* MinidumpUTF8StringAtRVA(
    const std::string& file_contents,
    RVA rva);

//! \brief Returns the contents of a MINIDUMP_STRING as a `string16`.
//!
//! This function uses MinidumpStringAtRVA() to obtain a MINIDUMP_STRING, and
//! returns the string data as a `string16`.
//!
//! \param[in] file_contents The contents of the minidump file.
//! \param[in] rva The offset within the minidump file of the desired
//!     MINIDUMP_STRING.
//!
//! \return On success, the string read from \a file_writer at offset \a rva. On
//!     failure, raises a gtest assertion and returns an empty string.
//!
//! \sa MinidumpUTF8StringAtRVAAsString()
base::string16 MinidumpStringAtRVAAsString(const std::string& file_contents,
                                           RVA rva);

//! \brief Returns the contents of a MinidumpUTF8String as a `std::string`.
//!
//! This function uses MinidumpUTF8StringAtRVA() to obtain a MinidumpUTF8String,
//! and returns the string data as a `std::string`.
//!
//! \param[in] file_contents The contents of the minidump file.
//! \param[in] rva The offset within the minidump file of the desired
//!     MinidumpUTF8String.
//!
//! \return On success, the string read from \a file_writer at offset \a rva. On
//!     failure, raises a gtest assertion and returns an empty string.
//!
//! \sa MinidumpStringAtRVAAsString()
std::string MinidumpUTF8StringAtRVAAsString(const std::string& file_contents,
                                            RVA rva);

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_TEST_MINIDUMP_STRING_WRITER_TEST_UTIL_H_
