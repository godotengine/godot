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

#ifndef MINIDUMP_TEST_MINIDUMP_BYTE_ARRAY_WRITER_TEST_UTIL_H_
#define MINIDUMP_TEST_MINIDUMP_BYTE_ARRAY_WRITER_TEST_UTIL_H_

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>

#include <string>
#include <vector>

namespace crashpad {
namespace test {

//! \brief Returns the bytes referenced by a MinidumpByteArray object located
//!     in a minidump file at the specified RVA.
//!
//! \param[in] file_contents The contents of the minidump file.
//! \param[in] rva The offset in the minidump file of the MinidumpByteArray.
//!
//! \return The MinidumpByteArray::data referenced by the \a rva. Note that
//!       this function does not check that the data are within the bounds of
//!       the \a file_contents.
std::vector<uint8_t> MinidumpByteArrayAtRVA(const std::string& file_contents,
                                            RVA rva);

}  // namespace test
}  // namespace crashpad

#endif  // MINIDUMP_TEST_MINIDUMP_BYTE_ARRAY_WRITER_TEST_UTIL_H_
