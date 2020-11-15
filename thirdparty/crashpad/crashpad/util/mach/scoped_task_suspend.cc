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

#include "util/mach/scoped_task_suspend.h"

#include "base/logging.h"
#include "base/mac/mach_logging.h"

namespace crashpad {

ScopedTaskSuspend::ScopedTaskSuspend(task_t task) : task_(task) {
  DCHECK_NE(task_, mach_task_self());

  kern_return_t kr = task_suspend(task_);
  if (kr != KERN_SUCCESS) {
    task_ = TASK_NULL;
    MACH_LOG(ERROR, kr) << "task_suspend";
  }
}

ScopedTaskSuspend::~ScopedTaskSuspend() {
  if (task_ != TASK_NULL) {
    kern_return_t kr = task_resume(task_);
    MACH_LOG_IF(ERROR, kr != KERN_SUCCESS, kr) << "task_resume";
  }
}

}  // namespace crashpad
