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

#ifndef CRASHPAD_TEST_MULTIPROCESS_H_
#define CRASHPAD_TEST_MULTIPROCESS_H_

#include <sys/types.h>

#include "base/macros.h"
#include "build/build_config.h"
#include "util/file/file_io.h"

namespace crashpad {
namespace test {

namespace internal {
struct MultiprocessInfo;
};

//! \brief Manages a multiprocess test.
//!
//! These tests are `fork()`-based. The parent and child processes are able to
//! communicate via a pair of POSIX pipes.
//!
//! Subclasses are expected to implement the parent and child by overriding the
//! appropriate methods.
//!
//! On Windows and Fuchsia, this class is only an internal implementation
//! detail of MultiprocessExec and all tests must use that class.
class Multiprocess {
 public:
  //! \brief The termination type for a child process.
  enum TerminationReason : bool {
    //! \brief The child terminated normally.
    //!
    //! A normal return happens when a test returns from RunChild(), or for
    //! tests that `exec()`, returns from `main()`. This also happens for tests
    //! that call `exit()` or `_exit()`.
    kTerminationNormal = false,

#if !defined(OS_FUCHSIA)  // There are no signals on Fuchsia.
    //! \brief The child terminated by signal.
    //!
    //! Signal termination happens as a result of a crash, a call to `abort()`,
    //! assertion failure (including gtest assertions), etc.
    kTerminationSignal,
#endif  // !defined(OS_FUCHSIA)
  };

  Multiprocess();

  //! \brief Runs the test.
  //!
  //! This method establishes the proper testing environment by calling
  //! PreFork(), then calls `fork()`. In the parent process, it calls
  //! RunParent(), and in the child process, it calls RunChild().
  //!
  //! This method uses gtest assertions to validate the testing environment. If
  //! the testing environment cannot be set up properly, it is possible that
  //! MultiprocessParent() or MultiprocessChild() will not be called. In the
  //! parent process, this method also waits for the child process to exit after
  //! MultiprocessParent() returns, and verifies that it exited in accordance
  //! with the expectations set by SetExpectedChildTermination().
  void Run();

  //! \brief Sets the expected termination reason and code.
  //!
  //! The default expected termination reasaon is
  //! TerminationReason::kTerminationNormal, and the default expected
  //! termination code is `EXIT_SUCCESS` (`0`).
  //!
  //! This method does not need to be called if the default termination
  //! expectation is appropriate, but if this method is called, it must be
  //! called before Run().
  //!
  //! \param[in] reason Whether to expect the child to terminate normally or
  //!     as a result of a signal.
  //! \param[in] code If \a reason is TerminationReason::kTerminationNormal,
  //!     this is the expected exit status of the child. If \a reason is
  //!     TerminationReason::kTerminationSignal, this is the signal that is
  //!     expected to kill the child. On Linux platforms, SIG_DFL will be
  //!     installed for \a code in the child process.
  void SetExpectedChildTermination(TerminationReason reason, int code);

#if !defined(OS_WIN)
  //! \brief Sets termination reason and code appropriately for a child that
  //!     terminates via `__builtin_trap()`.
  void SetExpectedChildTerminationBuiltinTrap();
#endif  // !OS_WIN

 protected:
  ~Multiprocess();

  //! \brief Establishes the proper testing environment prior to forking.
  //!
  //! Subclasses that solely implement a test should not need to override this
  //! method. Subclasses that do not implement tests but instead implement
  //! additional testing features on top of this class may override this method
  //! provided that they call the superclass’ implementation first as follows:
  //!
  //! \code
  //!   void PreFork() override {
  //!     ASSERT_NO_FATAL_FAILURE(Multiprocess::PreFork());
  //!
  //!     // Place subclass-specific pre-fork code here.
  //!   }
  //! \endcode
  //!
  //! Subclass implementations may signal failure by raising their own fatal
  //! gtest assertions.
  virtual void PreFork()
#if defined(OS_WIN) || defined(OS_FUCHSIA)
      = 0
#endif  // OS_WIN || OS_FUCHSIA
      ;

#if !defined(OS_WIN) && !defined(OS_FUCHSIA)
  //! \brief Returns the child process’ process ID.
  //!
  //! This method may only be called by the parent process.
  pid_t ChildPID() const;
#endif  // !OS_WIN && !OS_FUCHSIA

  //! \brief Returns the read pipe’s file handle.
  //!
  //! This method may be called by either the parent or the child process.
  //! Anything written to the write pipe in the partner process will appear
  //! on this file handle in this process.
  //!
  //! It is an error to call this after CloseReadPipe() has been called.
  //!
  //! \return The read pipe’s file handle.
  FileHandle ReadPipeHandle() const;

  //! \brief Returns the write pipe’s file handle.
  //!
  //! This method may be called by either the parent or the child process.
  //! Anything written to this file handle in this process will appear on
  //! the read pipe in the partner process.
  //!
  //! It is an error to call this after CloseWritePipe() has been called.
  //!
  //! \return The write pipe’s file handle.
  FileHandle WritePipeHandle() const;

  //! \brief Closes the read pipe.
  //!
  //! This method may be called by either the parent or the child process. An
  //! attempt to write to the write pipe in the partner process will fail with
  //! `EPIPE` or `SIGPIPE`. ReadPipeHandle() must not be called after this.
  void CloseReadPipe();

  //! \brief Closes the write pipe.
  //!
  //! This method may be called by either the parent or the child process. An
  //! attempt to read from the read pipe in the partner process will indicate
  //! end-of-file. WritePipeHandle() must not be called after this.
  void CloseWritePipe();

  void set_info(internal::MultiprocessInfo* info) { info_ = info; }
  internal::MultiprocessInfo* info() { return info_; }

 private:
  //! \brief Runs the parent side of the test.
  //!
  //! This method establishes the parent’s environment and calls
  //! MultiprocessParent().
  void RunParent();

  //! \brief Runs the child side of the test.
  //!
  //! This method establishes the child’s environment, calls
  //! MultiprocessChild(), and exits cleanly by calling `_exit(0)`. However, if
  //! any failure (via fatal or nonfatal gtest assertion) is detected, the child
  //! will exit with a failure status.
  void RunChild();

  //! \brief The subclass-provided parent routine.
  //!
  //! Test failures should be reported via gtest: `EXPECT_*()`, `ASSERT_*()`,
  //! `FAIL()`, etc.
  //!
  //! This method must not use a `wait()`-family system call to wait for the
  //! child process to exit, as this is handled by this class.
  //!
  //! Subclasses must implement this method to define how the parent operates.
  virtual void MultiprocessParent() = 0;

  //! \brief The subclass-provided child routine.
  //!
  //! Test failures should be reported via gtest: `EXPECT_*()`, `ASSERT_*()`,
  //! `FAIL()`, etc.
  //!
  //! Subclasses must implement this method to define how the child operates.
  //! Subclasses may exit with a failure status by using `LOG(FATAL)`,
  //! `abort()`, or similar. They may exit cleanly by returning from this method
  //! or by calling `_exit(0)`. Under no circumstances may `exit()` be called
  //! by the child without having the child process `exec()`. Use
  //! MultiprocessExec if the child should call `exec()`.
  virtual void MultiprocessChild() = 0;

  internal::MultiprocessInfo* info_;
  int code_;
  TerminationReason reason_;

  DISALLOW_COPY_AND_ASSIGN(Multiprocess);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_MULTIPROCESS_H_
