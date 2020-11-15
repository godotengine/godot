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

#ifndef CRASHPAD_TEST_MAC_EXCEPTION_SWALLOWER_H_
#define CRASHPAD_TEST_MAC_EXCEPTION_SWALLOWER_H_

#include <memory>

#include "base/macros.h"

namespace crashpad {
namespace test {

//! \brief Swallows `EXC_CRASH` and `EXC_CORPSE_NOTIFY` exceptions in test child
//!     processes.
//!
//! This class is intended to be used by test code that crashes intentionally.
//!
//! On macOS, the system’s crash reporter normally saves crash reports for all
//! crashes in test code, by virtue of being set as the `EXC_CRASH` or
//! `EXC_CORPSE_NOTIFY` handler. This litters the user’s
//! `~/Library/Logs/DiagnosticReports` directory and can be time-consuming.
//! Reports generated for code that crashes intentionally have no value, and
//! many Crashpad tests do crash intentionally.
//!
//! Instantiate an ExceptionSwallower object in a parent test process (a process
//! where `TEST()`, `TEST_F()`, and `TEST_P()` execute) to create an exception
//! swallower server running on a dedicated thread. A service mapping for this
//! server will be published with the bootstrap server and made available in the
//! `CRASHPAD_EXCEPTION_SWALLOWER_SERVICE` environment variable. In a child
//! process, call SwallowExceptions() to look up this service and set it as the
//! `EXC_CRASH` and `EXC_CORPSE_NOTIFY` handler. When these exceptions are
//! raised in the child process, they’ll be handled by the exception swallower
//! server, which performs no action but reports that exceptions were
//! successfully handled so that the system’s crash reporter, ReportCrash, will
//! not be invoked.
//!
//! At most one ExceptionSwallower may be instantiated in a process at a time.
//! If `CRASHPAD_EXCEPTION_SWALLOWER_SERVICE` is already set, ExceptionSwallower
//! leaves it in place and takes no additional action.
//!
//! Crashpad’s ASSERT_DEATH_CRASH(), EXPECT_DEATH_CRASH(), ASSERT_DEATH_CHECK(),
//! and EXPECT_DEATH_CHECK() macros make use of this class on macOS, as does the
//! Multiprocess test interface.
class ExceptionSwallower {
 public:
  ExceptionSwallower();
  ~ExceptionSwallower();

  //! \brief In a test child process, arranges to swallow `EXC_CRASH` and
  //!     `EXC_CORPSE_NOTIFY` exceptions.
  //!
  //! This must be called in a test child process. It must not be called from a
  //! parent test process directly. Parent test processes are those that execute
  //! `TEST()`, `TEST_F()`, and `TEST_P()`. Test child processes execute
  //! ASSERT_DEATH_CRASH(), EXPECT_DEATH_CRASH(), ASSERT_DEATH_CHECK(),
  //! EXPECT_DEATH_CHECK(), and Multiprocess::RunChild().
  //!
  //! It is an error to call this in a test child process without having first
  //! instantiated an ExceptionSwallower object in a parent test project. It is
  //! also an error to call this in a parent test process.
  static void SwallowExceptions();

 private:
  class ExceptionSwallowerThread;

  std::unique_ptr<ExceptionSwallowerThread> exception_swallower_thread_;

  DISALLOW_COPY_AND_ASSIGN(ExceptionSwallower);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_MAC_EXCEPTION_SWALLOWER_H_
