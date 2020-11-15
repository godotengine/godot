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

#ifndef CRASHPAD_UTIL_PROCESS_PROCESS_MEMORY_LINUX_H_
#define CRASHPAD_UTIL_PROCESS_PROCESS_MEMORY_LINUX_H_

#include <sys/types.h>

#include <string>

#include "base/files/scoped_file.h"
#include "base/macros.h"
#include "util/misc/address_types.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/process/process_memory.h"

namespace crashpad {

//! \brief Accesses the memory of another Linux process.
class ProcessMemoryLinux final : public ProcessMemory {
 public:
  ProcessMemoryLinux();
  ~ProcessMemoryLinux();

  //! \brief Initializes this object to read the memory of a process whose ID
  //!     is \a pid.
  //!
  //! This method must be called successfully prior to calling any other method
  //! in this class.
  //!
  //! \param[in] pid The process ID of a target process.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  bool Initialize(pid_t pid);

 private:
  ssize_t ReadUpTo(VMAddress address, size_t size, void* buffer) const override;

  base::ScopedFD mem_fd_;
  pid_t pid_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcessMemoryLinux);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_PROCESS_PROCESS_MEMORY_LINUX_H_
