// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_WIN_TERMINATION_CODES_H_
#define CRASHPAD_UTIL_WIN_TERMINATION_CODES_H_

namespace crashpad {

//! \brief Crashpad-specific codes that are used as arguments to
//!     SafeTerminateProcess() or `TerminateProcess()` in unusual circumstances.
enum TerminationCodes : unsigned int {
  //! \brief The crash handler did not respond, and the client self-terminated.
  kTerminationCodeCrashNoDump = 0xffff7001,

  //! \brief The initial process snapshot failed, so the correct client
  //!     termination code could not be retrieved.
  kTerminationCodeSnapshotFailed = 0xffff7002,

  //! \brief A dump was requested for a client that was never registered with
  //!     the crash handler.
  kTerminationCodeNotConnectedToHandler = 0xffff7003,
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_TERMINATION_CODES_H_
