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

#ifndef CRASHPAD_UTIL_PROCESS_PROCESS_MEMORY_FUCHSIA_H_
#define CRASHPAD_UTIL_PROCESS_PROCESS_MEMORY_FUCHSIA_H_

#include <lib/zx/process.h>

#include <string>

#include "base/macros.h"
#include "util/misc/address_types.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/process/process_memory.h"

namespace crashpad {

//! \brief Accesses the memory of another Fuchsia process.
class ProcessMemoryFuchsia final : public ProcessMemory {
 public:
  ProcessMemoryFuchsia();
  ~ProcessMemoryFuchsia();

  //! \brief Initializes this object to read the memory of a process by handle.
  //!
  //! This method must be called successfully prior to calling any other method
  //! in this class.
  //!
  //! \param[in] process The handle to the target process.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  bool Initialize(const zx::process& process);
  // TODO(wez): Remove this overload when zx::unowned_process allows implicit
  // copy.
  bool Initialize(const zx::unowned_process& process);

 private:
  ssize_t ReadUpTo(VMAddress address, size_t size, void* buffer) const override;

  zx::unowned_process process_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcessMemoryFuchsia);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_PROCESS_PROCESS_MEMORY_FUCHSIA_H_
