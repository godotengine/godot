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

#ifndef CRASHPAD_SNAPSHOT_LINUX_DEBUG_RENDEZVOUS_H_
#define CRASHPAD_SNAPSHOT_LINUX_DEBUG_RENDEZVOUS_H_

#include <string>
#include <vector>

#include "base/macros.h"
#include "util/linux/address_types.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/process/process_memory_range.h"

namespace crashpad {

//! \brief Reads an `r_debug` struct defined in `<link.h>` via
//!     ProcessMemoryRange.
class DebugRendezvous {
 public:
  //! \brief An entry in the dynamic linker's list of loaded objects.
  //!
  //! All of these values should be checked before use. Whether and how they are
  //! populated may vary by dynamic linker.
  struct LinkEntry {
    LinkEntry();

    //! \brief A filename identifying the object.
    std::string name;

    //! \brief The difference between the preferred load address in the ELF file
    //!     and the actual loaded address in memory.
    LinuxVMOffset load_bias;

    //! \brief The address of the dynamic array for this object.
    LinuxVMAddress dynamic_array;
  };

  DebugRendezvous();
  ~DebugRendezvous();

  //! \brief Initializes this object by reading an `r_debug` struct from a
  //!     target process.
  //!
  //! This method must be called successfully prior to calling any other method
  //! in this class.
  //!
  //! \param[in] memory A memory reader for the remote process.
  //! \param[in] address The address of an `r_debug` struct in the remote
  //!     process.
  //! \return `true` on success. `false` on failure with a message logged.
  bool Initialize(const ProcessMemoryRange& memory, LinuxVMAddress address);

  //! \brief Returns the LinkEntry for the main executable.
  const LinkEntry* Executable() const;

  //! \brief Returns a vector of modules found in the link map.
  //!
  //! This list excludes the entry for the executable and may include entries
  //! for the VDSO and loader.
  const std::vector<LinkEntry>& Modules() const;

 private:
  template <typename Traits>
  bool InitializeSpecific(const ProcessMemoryRange& memory,
                          LinuxVMAddress address);

  std::vector<LinkEntry> modules_;
  LinkEntry executable_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(DebugRendezvous);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_LINUX_DEBUG_RENDEZVOUS_H_
