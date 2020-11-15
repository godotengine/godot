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

#ifndef CRASHPAD_TEST_WIN_WIN_MULTIPROCESS_H_
#define CRASHPAD_TEST_WIN_WIN_MULTIPROCESS_H_

#include <windows.h>

#include "base/macros.h"
#include "gtest/gtest.h"
#include "test/win/win_child_process.h"
#include "util/file/file_io.h"
#include "util/win/scoped_handle.h"

namespace crashpad {
namespace test {

//! \brief Manages a multiprocess test on Windows.
class WinMultiprocess {
 public:
  WinMultiprocess();

  //! \brief Runs the test.
  //!
  //! This method establishes the testing environment by respawning the process
  //! as a child with additional flags.
  //!
  //! In the parent process, WinMultiprocessParent() is run, and in the child
  //! WinMultiprocessChild().
  template <class T>
  static void Run() {
    ASSERT_NO_FATAL_FAILURE(
        WinChildProcess::EntryPoint<ChildProcessHelper<T>>());
    // If WinChildProcess::EntryPoint returns, we are in the parent process.
    T parent_process;
    WinMultiprocess* parent_multiprocess = &parent_process;

    parent_multiprocess->WinMultiprocessParentBeforeChild();

    std::unique_ptr<WinChildProcess::Handles> child_handles =
        WinChildProcess::Launch();
    ASSERT_TRUE(child_handles.get());
    parent_process.child_handles_ = child_handles.get();
    parent_multiprocess->WinMultiprocessParent();

    // Close our side of the handles now that we're done. The child can
    // use this to know when it's safe to complete.
    child_handles->read.reset();
    child_handles->write.reset();

    // Wait for the child to complete.
    ASSERT_EQ(WaitForSingleObject(child_handles->process.get(), INFINITE),
              WAIT_OBJECT_0);

    DWORD exit_code;
    ASSERT_TRUE(GetExitCodeProcess(child_handles->process.get(), &exit_code));
    ASSERT_EQ(exit_code, parent_process.exit_code_);

    parent_multiprocess->WinMultiprocessParentAfterChild(
        child_handles->process.get());
  }

 protected:
  virtual ~WinMultiprocess();

  //! \brief Sets the expected exit code of the child process.
  //!
  //! The default expected termination code is `EXIT_SUCCESS` (`0`).
  //!
  //! \param[in] exit_code The expected exit status of the child.
  void SetExpectedChildExitCode(unsigned int exit_code);

  //! \brief Returns the read pipe's file handle.
  //!
  //! This method may be called by either the parent or the child process.
  //! Anything written to the write pipe in the partner process will appear
  //! on this file handle in this process.
  //!
  //! It is an error to call this after CloseReadPipe() has been called.
  //!
  //! \return The read pipe's file handle.
  FileHandle ReadPipeHandle() const;

  //! \brief Returns the write pipe's file handle.
  //!
  //! This method may be called by either the parent or the child process.
  //! Anything written to this file handle in this process will appear on
  //! the read pipe in the partner process.
  //!
  //! It is an error to call this after CloseWritePipe() has been called.
  //!
  //! \return The write pipe's file handle.
  FileHandle WritePipeHandle() const;

  //! \brief Closes the read pipe.
  //!
  //! This method may be called by either the parent or the child process.
  //! ReadPipeHandle() must not be called after this.
  void CloseReadPipe();

  //! \brief Closes the write pipe.
  //!
  //! This method may be called by either the parent or the child process. An
  //! attempt to read from the read pipe in the partner process will indicate
  //! end-of-file. WritePipeHandle() must not be called after this.
  void CloseWritePipe();

  //! \brief Returns a handle to the child process.
  //!
  //! This method may only be called by the parent process.
  HANDLE ChildProcess() const;

 private:
  // Implements an adapter to provide WinMultiprocess with access to the
  // anonymous pipe handles from WinChildProcess.
  class ChildProcessHelperBase : public WinChildProcess {
   public:
    ChildProcessHelperBase() {}
    ~ChildProcessHelperBase() override {}

    void CloseWritePipeForwarder() { CloseWritePipe(); }
    void CloseReadPipeForwarder() { CloseReadPipe(); }
    FileHandle ReadPipeHandleForwarder() const { return ReadPipeHandle(); }
    FileHandle WritePipeHandleForwarder() const { return WritePipeHandle(); }

   private:
    DISALLOW_COPY_AND_ASSIGN(ChildProcessHelperBase);
  };

  // Forwards WinChildProcess::Run to T::WinMultiprocessChild.
  template <class T>
  class ChildProcessHelper : public ChildProcessHelperBase {
   public:
    ChildProcessHelper() {}
    ~ChildProcessHelper() override {}

   private:
    int Run() override {
      T child_process;
      child_process.child_process_helper_ = this;
      static_cast<WinMultiprocess*>(&child_process)->WinMultiprocessChild();
      if (testing::Test::HasFailure())
        return 255;
      return EXIT_SUCCESS;
    }

    DISALLOW_COPY_AND_ASSIGN(ChildProcessHelper);
  };

  //! \brief The subclass-provided parent routine.
  //!
  //! Test failures should be reported via gtest: `EXPECT_*()`, `ASSERT_*()`,
  //! `FAIL()`, etc.
  //!
  //! This method need not use `WaitForSingleObject()`-family call to wait for
  //! the child process to exit, as this is handled by this class.
  //!
  //! Subclasses must implement this method to define how the parent operates.
  virtual void WinMultiprocessParent() = 0;

  //! \brief The optional routine run in parent before the child is spawned.
  //!
  //! Subclasses may implement this method to prepare the environment for
  //! the child process.
  virtual void WinMultiprocessParentBeforeChild() {}

  //! \brief The optional routine run in parent after the child exits.
  //!
  //! Subclasses may implement this method to clean up the environment after
  //! the child process has exited.
  //!
  //! \param[in] child A handle to the exited child process.
  virtual void WinMultiprocessParentAfterChild(HANDLE child) {}

  //! \brief The subclass-provided child routine.
  //!
  //! Test failures should be reported via gtest: `EXPECT_*()`, `ASSERT_*()`,
  //! `FAIL()`, etc.
  //!
  //! Subclasses must implement this method to define how the child operates.
  //! Subclasses may exit with a failure status by using `LOG(FATAL)`,
  //! `abort()`, or similar. They may exit cleanly by returning from this
  //! method.
  virtual void WinMultiprocessChild() = 0;

  unsigned int exit_code_;
  WinChildProcess::Handles* child_handles_;
  ChildProcessHelperBase* child_process_helper_;

  DISALLOW_COPY_AND_ASSIGN(WinMultiprocess);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_WIN_WIN_MULTIPROCESS_H_
