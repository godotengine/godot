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

#ifndef CRASHPAD_UTIL_WIN_SCOPED_SET_EVENT_H_
#define CRASHPAD_UTIL_WIN_SCOPED_SET_EVENT_H_

#include <windows.h>

#include "base/macros.h"

namespace crashpad {

//! \brief Calls `SetEvent()` on destruction at latest.
//!
//! Does not assume ownership of the event handle. Use ScopedKernelHANDLE for
//! ownership.
class ScopedSetEvent {
 public:
  explicit ScopedSetEvent(HANDLE event);
  ~ScopedSetEvent();

  //! \brief Calls `SetEvent()` immediately.
  //!
  //! `SetEvent()` will not be called on destruction.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  bool Set();

 private:
  HANDLE event_;  // weak

  DISALLOW_COPY_AND_ASSIGN(ScopedSetEvent);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_SCOPED_SET_EVENT_H_
