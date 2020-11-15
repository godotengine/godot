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

#ifndef CRASHPAD_SNAPSHOT_FUCHSIA_MEMORY_MAP_FUCHSIA_H_
#define CRASHPAD_SNAPSHOT_FUCHSIA_MEMORY_MAP_FUCHSIA_H_

#include <lib/zx/process.h>
#include <zircon/syscalls/object.h>

#include <vector>

#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

//! \brief A list of mappings in the address space of a Fuchsia process.
class MemoryMapFuchsia {
 public:
  MemoryMapFuchsia();
  ~MemoryMapFuchsia();

  //! \brief Initializes this object with information about the mapped memory
  //!     regions in the given process.
  //!
  //! \return `true` on success, or `false`, with an error logged.
  bool Initialize(const zx::process& process);

  //! \brief Searches through the previously retrieved memory map for the given
  //!     address. If found, returns the deepest `zx_info_maps_t` mapping that
  //!     contains \a address.
  //!
  //! \param[in] address The address to locate.
  //! \param[out] map The `zx_info_maps_t` data corresponding to the address.
  //! \return `true` if a mapping for \a address was found, in which case \a map
  //!     will be filled out, otherwise `false` and \a map will be unchanged.
  bool FindMappingForAddress(zx_vaddr_t address, zx_info_maps_t* map) const;

 private:
  std::vector<zx_info_maps_t> map_entries_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(MemoryMapFuchsia);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_FUCHSIA_MEMORY_MAP_FUCHSIA_H_
