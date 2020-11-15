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

#ifndef CRASHPAD_CLIENT_SIMULATE_CRASH_MAC_H_
#define CRASHPAD_CLIENT_SIMULATE_CRASH_MAC_H_

#include <mach/mach.h>

#include "util/misc/capture_context.h"

//! \file

namespace crashpad {

//! \brief Simulates a exception without crashing.
//!
//! This function searches for an `EXC_CRASH` handler in the same manner that
//! the kernel does, and sends it an exception message to that handler in the
//! format that the handler expects, considering the behavior and thread state
//! flavor that are registered for it. The exception sent to the handler will be
//! ::kMachExceptionSimulated, not `EXC_CRASH`.
//!
//! Typically, the CRASHPAD_SIMULATE_CRASH() macro will be used in preference to
//! this function, because it combines the context-capture operation with the
//! raising of a simulated exception.
//!
//! This function returns normally after the exception message is processed. If
//! no valid handler was found, or no handler processed the exception
//! successfully, a warning will be logged, but these conditions are not
//! considered fatal.
//!
//! \param[in] cpu_context The thread state to pass to the exception handler as
//!     the exception context, provided that it is compatible with the thread
//!     state flavor that the exception handler accepts. If it is not
//!     compatible, the correct thread state for the handler will be obtained by
//!     calling `thread_get_state()`.
void SimulateCrash(const NativeCPUContext& cpu_context);

}  // namespace crashpad

//! \brief Captures the CPU context and simulates an exception without crashing.
#define CRASHPAD_SIMULATE_CRASH()           \
  do {                                      \
    crashpad::NativeCPUContext cpu_context; \
    crashpad::CaptureContext(&cpu_context); \
    crashpad::SimulateCrash(cpu_context);   \
  } while (false)

#endif  // CRASHPAD_CLIENT_SIMULATE_CRASH_MAC_H_
