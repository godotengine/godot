// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_TEST_FILE_H_
#define CRASHPAD_TEST_FILE_H_

#include "base/files/file_path.h"
#include "util/file/file_io.h"

namespace crashpad {
namespace test {

//! \brief Determines whether a file exists.
//!
//! \param[in] path The path to check for existence.
//!
//! \return `true` if \a path exists. `false` if it does not exist. If an error
//!     other than “file not found” occurs when searching for \a path, returns
//!     `false` with a gtest failure added.
bool FileExists(const base::FilePath& path);

//! \brief Determines the size of a file.
//!
//! \param[in] path The path of the file to check. The file must exist.
//!
//! \return The size of the file at \a path. If the file does not exist, or an
//!     error occurs when attempting to determine its size, returns `-1` with a
//!     gtest failure added.
FileOffset FileSize(const base::FilePath& path);

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_FILE_H_
