// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#include "util/thread/thread.h"

#include "base/logging.h"

namespace crashpad {

void Thread::Start() {
  DCHECK(!platform_thread_);
  platform_thread_ =
      CreateThread(nullptr, 0, ThreadEntryThunk, this, 0, nullptr);
  PCHECK(platform_thread_) << "CreateThread";
}

void Thread::Join() {
  DCHECK(platform_thread_);
  DWORD result = WaitForSingleObject(platform_thread_, INFINITE);
  PCHECK(result == WAIT_OBJECT_0) << "WaitForSingleObject";
  platform_thread_ = 0;
}

// static
DWORD WINAPI Thread::ThreadEntryThunk(void* argument) {
  Thread* self = reinterpret_cast<Thread*>(argument);
  self->ThreadMain();
  return 0;
}

}  // namespace crashpad
