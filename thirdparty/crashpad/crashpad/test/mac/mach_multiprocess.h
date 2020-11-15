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

#ifndef CRASHPAD_TEST_MAC_MACH_MULTIPROCESS_H_
#define CRASHPAD_TEST_MAC_MACH_MULTIPROCESS_H_

#include <mach/mach.h>
#include <unistd.h>

#include "base/macros.h"
#include "test/multiprocess.h"

namespace crashpad {
namespace test {

namespace internal {
struct MachMultiprocessInfo;
}  // namespace internal

//! \brief Manages a Mach-aware multiprocess test.
//!
//! This is similar to the base Multiprocess test, but adds Mach features. The
//! parent process has access to the child process’ task port. The parent and
//! child processes are able to communicate via Mach IPC: each process has a
//! receive right to its “local port” and a send right to a “remote port”, and
//! messages sent to the remote port in one process can be received on the local
//! port in the partner process.
//!
//! Subclasses are expected to implement the parent and child by overriding the
//! appropriate methods.
class MachMultiprocess : public Multiprocess {
 public:
  MachMultiprocess();

  void Run();

 protected:
  ~MachMultiprocess();

  // Multiprocess:
  void PreFork() override;

  //! \brief Returns a receive right for the local port.
  //!
  //! This method may be called by either the parent or the child process. It
  //! returns a receive right, with a corresponding send right held in the
  //! opposing process.
  mach_port_t LocalPort() const;

  //! \brief Returns a send right for the remote port.
  //!
  //! This method may be called by either the parent or the child process. It
  //! returns a send right, with the corresponding receive right held in the
  //! opposing process.
  mach_port_t RemotePort() const;

  //! \brief Returns a send right for the child’s task port.
  //!
  //! This method may only be called by the parent process.
  task_t ChildTask() const;

 private:
  // Multiprocess:

  //! \brief Runs the parent side of the test.
  //!
  //! This method establishes the parent’s environment and calls
  //! MachMultiprocessParent().
  //!
  //! Subclasses must override MachMultiprocessParent() instead of this method.
  void MultiprocessParent() final;

  //! \brief Runs the child side of the test.
  //!
  //! This method establishes the child’s environment and calls
  //! MachMultiprocessChild(). If any failure (via fatal or nonfatal gtest
  //! assertion) is detected, the child will exit with a failure status.
  //!
  //! Subclasses must override MachMultiprocessChild() instead of this method.
  void MultiprocessChild() final;

  //! \brief The subclass-provided parent routine.
  //!
  //! Test failures should be reported via gtest: `EXPECT_*()`, `ASSERT_*()`,
  //! `FAIL()`, etc.
  //!
  //! This method must not use a `wait()`-family system call to wait for the
  //! child process to exit, as this is handled by the superclass.
  //!
  //! Subclasses must implement this method to define how the parent operates.
  virtual void MachMultiprocessParent() = 0;

  //! \brief The subclass-provided child routine.
  //!
  //! Test failures should be reported via gtest: `EXPECT_*()`, `ASSERT_*()`,
  //! `FAIL()`, etc.
  //!
  //! Subclasses must implement this method to define how the child operates.
  virtual void MachMultiprocessChild() = 0;

  internal::MachMultiprocessInfo* info_;

  DISALLOW_COPY_AND_ASSIGN(MachMultiprocess);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_MAC_MACH_MULTIPROCESS_H_
