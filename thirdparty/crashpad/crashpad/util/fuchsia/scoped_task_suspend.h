// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_FUCHSIA_SCOPED_TASK_SUSPEND_H_
#define CRASHPAD_UTIL_FUCHSIA_SCOPED_TASK_SUSPEND_H_

#include <lib/zx/process.h>
#include <lib/zx/suspend_token.h>
#include <lib/zx/thread.h>

#include <vector>

#include "base/macros.h"

namespace crashpad {

//! \brief Manages the suspension of another task.
//!
//! The underlying API only supports suspending threads (despite its name) not
//! entire tasks. As a result, it's possible some threads may not be correctly
//! suspended/resumed as their creation might race enumeration.
//!
//! Additionally, suspending a thread is asynchronous and may take an
//! arbitrary amount of time.
//!
//! Because of these limitations, this class is limited to being a best-effort,
//! and correct suspension/resumption cannot be relied upon.
//!
//! Callers should not attempt to suspend the current task as obtained via
//! `zx_process_self()`.
class ScopedTaskSuspend {
 public:
  explicit ScopedTaskSuspend(const zx::process& process);
  explicit ScopedTaskSuspend(const zx::thread& thread);
  ~ScopedTaskSuspend() = default;

 private:
  // Could be one (for a thread) or many (for every thread in a process).
  std::vector<zx::suspend_token> suspend_tokens_;

  DISALLOW_COPY_AND_ASSIGN(ScopedTaskSuspend);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_FUCHSIA_SCOPED_TASK_SUSPEND_H_
