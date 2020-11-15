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

#ifndef CRASHPAD_TEST_WIN_WIN_CHILD_PROCESS_H_
#define CRASHPAD_TEST_WIN_WIN_CHILD_PROCESS_H_

#include <memory>

#include "base/macros.h"
#include "util/file/file_io.h"
#include "util/win/scoped_handle.h"

namespace crashpad {
namespace test {

//! \brief Facilitates the launching of child processes from unit tests.
class WinChildProcess {
 public:
  //! \brief Groups handles used to communicate with, observe, and manage a
  //!     child process.
  struct Handles {
    //! \brief A handle to read from an anonymous pipe shared with the child
    //!     process.
    ScopedFileHANDLE read;
    //! \brief A handle to write to an anonymous pipe shared with the child
    //!     process.
    ScopedFileHANDLE write;
    //! \brief A handle to the child process.
    ScopedKernelHANDLE process;
  };

  WinChildProcess();
  virtual ~WinChildProcess() {}

  //! \brief Returns true if the current process is a child process.
  static bool IsChildProcess();

  //! \brief Runs the child process defined by T if the current process is a
  //!     child process; does not return in that case. Otherwise, returns.
  template <class T>
  static void EntryPoint() {
    if (IsChildProcess()) {
      // The static_cast here will cause a compiler failure if T is not a
      // subclass of WinChildProcess. It's the constructor of WinChildProcess
      // that performs the pipe handshake with the parent process (without which
      // we would have a hang).
      T child_process;
      int result = static_cast<WinChildProcess*>(&child_process)->Run();
      exit(result);
    }
  }

  //! \brief Launches a child process and returns the Handles for that process.
  //!     The process is guaranteed to be executing by the time this method
  //!     returns. Returns null and logs a GTest failure in case of failure.
  static std::unique_ptr<Handles> Launch();

 protected:
  //! \brief Returns a handle to read from an anonymous pipe shared with the
  //!     parent process.
  //!
  //! It is an error to call this after CloseReadPipe() has been called.
  //!
  //! \return The read pipe's file handle.
  FileHandle ReadPipeHandle() const;

  //! \brief Returns a handle to write to an anonymous pipe shared with the
  //!     parent process.
  //!
  //! It is an error to call this after CloseWritePipe() has been called.
  //!
  //! \return The write pipe's file handle.
  FileHandle WritePipeHandle() const;

  //! \brief Closes the read pipe.
  //!
  //! ReadPipeHandle() must not be called after this.
  void CloseReadPipe();

  //! \brief Closes the write pipe.
  //!
  //! An attempt to read from the read pipe in the parent process will indicate
  //! end-of-file. WritePipeHandle() must not be called after this.
  void CloseWritePipe();

 private:
  //! \brief The subclass-provided child routine.
  //!
  //! Subclasses must implement this method to define how the child operates.
  //! Subclasses may exit with a failure status by using `LOG(FATAL)`,
  //! `abort()`, or similar. They may also exit by returning their exit code
  //! from this method. It is up to the client to observe and interpret the
  //! child's exit code.
  //!
  //! \return The child process exit code.
  virtual int Run() = 0;

  ScopedFileHANDLE pipe_read_;
  ScopedFileHANDLE pipe_write_;

  DISALLOW_COPY_AND_ASSIGN(WinChildProcess);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_WIN_WIN_CHILD_PROCESS_H_
