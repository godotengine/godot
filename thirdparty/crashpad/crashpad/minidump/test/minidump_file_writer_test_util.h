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

#ifndef CRASHPAD_MINIDUMP_TEST_MINIDUMP_FILE_WRITER_TEST_UTIL_H_
#define CRASHPAD_MINIDUMP_TEST_MINIDUMP_FILE_WRITER_TEST_UTIL_H_

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>

#include <string>

namespace crashpad {
namespace test {

//! \brief Returns the MINIDUMP_HEADER at the start of a minidump file, along
//!     with the MINIDUMP_DIRECTORY it references.
//!
//! This function validates the MINIDUMP_HEADER::Signature and
//! MINIDUMP_HEADER::Version fields.
//!
//! \param[in] file_contents The contents of the minidump file.
//! \param[out] directory The MINIDUMP_DIRECTORY referenced by the
//!     MINIDUMP_HEADER. If the MINIDUMP_HEADER does not reference a
//!     MINIDUMP_DIRECTORY, `nullptr` without raising a gtest assertion. If the
//!     referenced MINIDUMP_DIRECTORY is not valid, `nullptr` with a gtest
//!     assertion raised. On failure, `nullptr`.
//!
//! \return On success, the MINIDUMP_HEADER at the beginning of the minidump
//!     file. On failure, raises a gtest assertion and returns `nullptr`.
const MINIDUMP_HEADER* MinidumpHeaderAtStart(
    const std::string& file_contents,
    const MINIDUMP_DIRECTORY** directory);

//! \brief Verifies, via gtest assertions, that a MINIDUMP_HEADER contains
//!     expected values.
//!
//! All fields in the MINIDUMP_HEADER will be evaluated except for the Signature
//! and Version fields, because those are checked by MinidumpHeaderAtStart().
//! Most other fields are are compared to their correct default values.
//! MINIDUMP_HEADER::NumberOfStreams is compared to \a streams, and
//! MINIDUMP_HEADER::TimeDateStamp is compared to \a timestamp. Most fields are
//! checked with nonfatal EXPECT-style assertions, but
//! MINIDUMP_HEADER::NumberOfStreams and MINIDUMP_HEADER::StreamDirectoryRva are
//! checked with fatal ASSERT-style assertions, because they must be correct in
//! order for processing of the minidump to continue.
void VerifyMinidumpHeader(const MINIDUMP_HEADER* header,
                          uint32_t streams,
                          uint32_t timestamp);

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_TEST_MINIDUMP_FILE_WRITER_TEST_UTIL_H_
