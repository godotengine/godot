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

#include "util/process/process_memory_fuchsia.h"

#include <limits>

#include "base/logging.h"
#include "base/fuchsia/fuchsia_logging.h"

namespace crashpad {

ProcessMemoryFuchsia::ProcessMemoryFuchsia()
    : ProcessMemory(), process_(), initialized_() {}

ProcessMemoryFuchsia::~ProcessMemoryFuchsia() {}

bool ProcessMemoryFuchsia::Initialize(const zx::unowned_process& process) {
  return Initialize(*process);
}

bool ProcessMemoryFuchsia::Initialize(const zx::process& process) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  process_ = zx::unowned_process(process);
  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

ssize_t ProcessMemoryFuchsia::ReadUpTo(VMAddress address,
                                       size_t size,
                                       void* buffer) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  DCHECK_LE(size, size_t{std::numeric_limits<ssize_t>::max()});

  size_t actual;
  zx_status_t status = process_->read_memory(address, buffer, size, &actual);

  if (status != ZX_OK) {
    ZX_LOG(ERROR, status) << "zx_process_read_memory";
    return -1;
  }

  return actual;
}

}  // namespace crashpad
