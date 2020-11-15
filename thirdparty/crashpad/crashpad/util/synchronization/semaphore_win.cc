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

#include "util/synchronization/semaphore.h"

#include <cmath>
#include <limits>

#include "base/logging.h"

namespace crashpad {

Semaphore::Semaphore(int value)
    : semaphore_(CreateSemaphore(nullptr,
                                 value,
                                 std::numeric_limits<LONG>::max(),
                                 nullptr)) {
  PCHECK(semaphore_) << "CreateSemaphore";
}

Semaphore::~Semaphore() {
  PCHECK(CloseHandle(semaphore_));
}

void Semaphore::Wait() {
  PCHECK(WaitForSingleObject(semaphore_, INFINITE) == WAIT_OBJECT_0);
}

bool Semaphore::TimedWait(double seconds) {
  DCHECK_GE(seconds, 0.0);

  if (std::isinf(seconds)) {
    Wait();
    return true;
  }

  DWORD rv = WaitForSingleObject(semaphore_, static_cast<DWORD>(seconds * 1E3));
  PCHECK(rv == WAIT_OBJECT_0 || rv == WAIT_TIMEOUT) << "WaitForSingleObject";
  return rv == WAIT_OBJECT_0;
}

void Semaphore::Signal() {
  PCHECK(ReleaseSemaphore(semaphore_, 1, nullptr));
}

}  // namespace crashpad
