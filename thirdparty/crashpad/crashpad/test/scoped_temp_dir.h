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

#ifndef CRASHPAD_TEST_SCOPED_TEMP_DIR_
#define CRASHPAD_TEST_SCOPED_TEMP_DIR_

#include "base/files/file_path.h"
#include "base/macros.h"

namespace crashpad {
namespace test {

//! \brief A RAII object that creates a temporary directory for testing.
//!
//! Upon construction, a temporary directory will be created. Failure to create
//! the directory is fatal. On destruction, the directory and all its contents
//! will be removed.
class ScopedTempDir {
 public:
  ScopedTempDir();
  ~ScopedTempDir();

  //! \brief Returns the path of the temporary directory.
  //!
  //! \return The temporary directory path.
  const base::FilePath& path() const { return path_; }

  //! \brief Moves the temporary directory to a new temporary location.
  void Rename();

 private:
  //! \brief Creates the temporary directory and asserts success of the
  //!     operation.
  static base::FilePath CreateTemporaryDirectory();

  //! \brief Removes all files and subdirectories at the given \a path,
  //!     including the \a path itself.
  //!
  //! Failures are recorded by gtest expectations.
  //!
  //! \param[in] path The path to delete, along with its contents. This must
  //!     reference a directory.
  static void RecursivelyDeleteTemporaryDirectory(const base::FilePath& path);

  base::FilePath path_;

  DISALLOW_COPY_AND_ASSIGN(ScopedTempDir);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_SCOPED_TEMP_DIR_
