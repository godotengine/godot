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

#ifndef CRASHPAD_UTIL_MACH_SCOPED_TASK_SUSPEND_H_
#define CRASHPAD_UTIL_MACH_SCOPED_TASK_SUSPEND_H_

#include <mach/mach.h>

#include "base/macros.h"

namespace crashpad {

//! \brief Manages the suspension of another task.
//!
//! While an object of this class exists, the other task will be suspended. Once
//! the object is destroyed, the other task will become eligible for resumption.
//! Note that suspensions are counted, and the task will not actually resume
//! unless its suspend count drops to 0.
//!
//! Callers should not attempt to suspend the current task (`mach_task_self()`).
class ScopedTaskSuspend {
 public:
  explicit ScopedTaskSuspend(task_t task);
  ~ScopedTaskSuspend();

 private:
  task_t task_;

  DISALLOW_COPY_AND_ASSIGN(ScopedTaskSuspend);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_SCOPED_TASK_SUSPEND_H_
