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

#include "util/fuchsia/scoped_task_suspend.h"

#include <lib/zx/time.h>

#include <vector>

#include "base/fuchsia/fuchsia_logging.h"
#include "base/logging.h"
#include "util/fuchsia/koid_utilities.h"

namespace crashpad {

namespace {

// Returns the suspend token of the suspended thread. This function attempts
// to wait a short time for the thread to actually suspend before returning
// but this is not guaranteed.
zx::suspend_token SuspendThread(const zx::thread& thread) {
  zx::suspend_token token;
  zx_status_t status = thread.suspend(&token);
  if (status != ZX_OK) {
    ZX_LOG(ERROR, status) << "zx_task_suspend";
    return zx::suspend_token();
  }

  zx_signals_t observed = 0u;
  if (thread.wait_one(ZX_THREAD_SUSPENDED,
                      zx::deadline_after(zx::msec(50)),
                      &observed) != ZX_OK) {
    LOG(ERROR) << "thread failed to suspend";
  }
  return token;
}

}  // namespace

ScopedTaskSuspend::ScopedTaskSuspend(const zx::process& process) {
  DCHECK_NE(process.get(), zx::process::self()->get());
  for (const auto& thread : GetThreadHandles(process))
    suspend_tokens_.push_back(SuspendThread(thread));
}

ScopedTaskSuspend::ScopedTaskSuspend(const zx::thread& thread) {
  DCHECK_NE(thread.get(), zx::thread::self()->get());
  suspend_tokens_.push_back(SuspendThread(thread));
}

}  // namespace crashpad
