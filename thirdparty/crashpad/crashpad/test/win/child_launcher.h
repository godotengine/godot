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

#ifndef CRASHPAD_TEST_WIN_CHILD_LAUNCHER_H_
#define CRASHPAD_TEST_WIN_CHILD_LAUNCHER_H_

#include <windows.h>

#include <string>

#include "base/files/file_path.h"
#include "util/win/scoped_handle.h"

namespace crashpad {
namespace test {

//! \brief Creates a child process for testing. Uses gtest `ASSERT_*` to
//!     indicate failure. The child's output is passed through a pipe and is
//!     available via stdout_read_handle(), and the child's input is attached to
//!     a second pipe available via stdin_write_handle().
class ChildLauncher {
 public:
  //! \brief Creates the object. \a executable will be escaped and prepended to
  //!     \a command_line to build the command line of the child.
  ChildLauncher(const base::FilePath& executable,
                const std::wstring& command_line);

  ~ChildLauncher();

  //! \brief Starts the child process, after which the handle functions below
  //!     will be valid.
  //!
  //! Errors are signaled via gtest assertions. This method may be invoked via
  //! `ASSERT_NO_FATAL_FAILURE()` to assert that it succeeds.
  void Start();

  //! \brief Waits for the child process to exit.
  //!
  //! \return The process exit code.
  DWORD WaitForExit();

  //! \brief The child process's `HANDLE`.
  HANDLE process_handle() const { return process_handle_.get(); }

  //! \brief The child process's main thread's `HANDLE`.
  HANDLE main_thread_handle() const { return main_thread_handle_.get(); }

  //! \brief The read end of a pipe attached to the child's stdout.
  HANDLE stdout_read_handle() const { return stdout_read_handle_.get(); }

  //! \brief The write end of a pipe attached to the child's stdin.
  HANDLE stdin_write_handle() const { return stdin_write_handle_.get(); }

 private:
  base::FilePath executable_;
  std::wstring command_line_;
  ScopedKernelHANDLE process_handle_;
  ScopedKernelHANDLE main_thread_handle_;
  ScopedFileHANDLE stdout_read_handle_;
  ScopedFileHANDLE stdin_write_handle_;
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_WIN_CHILD_LAUNCHER_H_
