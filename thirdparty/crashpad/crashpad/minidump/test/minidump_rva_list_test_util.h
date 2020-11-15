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

#ifndef CRASHPAD_MINIDUMP_TEST_MINIDUMP_RVA_LIST_TEST_UTIL_H_
#define CRASHPAD_MINIDUMP_TEST_MINIDUMP_RVA_LIST_TEST_UTIL_H_

#include <sys/types.h>

#include <string>

namespace crashpad {

struct MinidumpRVAList;

namespace test {

//! \brief Returns the MinidumpRVAList at the start of a minidump file.
//!
//! \param[in] file_contents The contents of the minidump file.
//! \param[in] count The number of ::RVA objects expected in the
//!     MinidumpRVAList. This function will only be successful if exactly this
//!     many objects are present, and if space for them exists in \a
//!     file_contents.
//!
//! \return On success, the MinidumpRVAList at the beginning of the file. On
//!     failure, raises a gtest assertion and returns `nullptr`.
const MinidumpRVAList* MinidumpRVAListAtStart(const std::string& file_contents,
                                              size_t count);

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_TEST_MINIDUMP_RVA_LIST_TEST_UTIL_H_
