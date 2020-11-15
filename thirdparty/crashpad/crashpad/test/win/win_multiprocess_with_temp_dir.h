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

#ifndef CRASHPAD_TEST_WIN_WIN_MULTIPROCESS_WITH_TEMPDIR_H_
#define CRASHPAD_TEST_WIN_WIN_MULTIPROCESS_WITH_TEMPDIR_H_

#include <wchar.h>
#include <windows.h>

#include <memory>
#include <string>

#include "base/files/file_path.h"
#include "base/macros.h"
#include "test/scoped_temp_dir.h"
#include "test/win/win_multiprocess.h"

namespace crashpad {
namespace test {

//! \brief Manages a multiprocess test on Windows with a parent-created
//!     temporary directory.
//!
//! This class creates a temp directory in the parent process for the use of
//! the subprocess and its children. To ensure a raceless rundown, it waits on
//! the child process and any processes directly created by the child before
//! deleting the temporary directory.
class WinMultiprocessWithTempDir : public WinMultiprocess {
 public:
  WinMultiprocessWithTempDir();

 protected:
  void WinMultiprocessParentBeforeChild() override;
  void WinMultiprocessParentAfterChild(HANDLE child) override;

  //! \brief Returns the path of the temp directory.
  base::FilePath GetTempDirPath() const;

 private:
  class ScopedEnvironmentVariable {
   public:
    explicit ScopedEnvironmentVariable(const wchar_t* name);
    ~ScopedEnvironmentVariable();

    std::wstring GetValue() const;

    // Sets this environment variable to |new_value|. If |new_value| is nullptr
    // this environment variable will be undefined.
    void SetValue(const wchar_t* new_value) const;

   private:
    std::wstring GetValueImpl(bool* is_defined) const;

    std::wstring original_value_;
    const wchar_t* name_;
    bool was_defined_;

    DISALLOW_COPY_AND_ASSIGN(ScopedEnvironmentVariable);
  };

  std::unique_ptr<ScopedTempDir> temp_dir_;
  ScopedEnvironmentVariable temp_dir_env_;

  DISALLOW_COPY_AND_ASSIGN(WinMultiprocessWithTempDir);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_WIN_WIN_MULTIPROCESS_WITH_TEMPDIR_H_
