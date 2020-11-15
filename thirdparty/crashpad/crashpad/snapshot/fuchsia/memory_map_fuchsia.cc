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

#include "snapshot/fuchsia/memory_map_fuchsia.h"

#include "base/fuchsia/fuchsia_logging.h"
#include "util/numeric/checked_range.h"

namespace crashpad {

MemoryMapFuchsia::MemoryMapFuchsia() = default;

MemoryMapFuchsia::~MemoryMapFuchsia() = default;

bool MemoryMapFuchsia::Initialize(const zx::process& process) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  // There's no way to know what an appropriate buffer size is before starting.
  // Start at a size that should be more than enough for any reasonable process.
  map_entries_.resize(4096);

  // Retrieving the maps is racy with new mappings being created, so retry this
  // loop up to |tries| times until the number of actual mappings retrieved
  // matches those available.
  int tries = 5;
  for (;;) {
    size_t actual;
    size_t available;
    zx_status_t status =
        process.get_info(ZX_INFO_PROCESS_MAPS,
                         &map_entries_[0],
                         map_entries_.size() * sizeof(map_entries_[0]),
                         &actual,
                         &available);
    if (status != ZX_OK) {
      ZX_LOG(ERROR, status) << "zx_object_get_info ZX_INFO_PROCESS_MAPS";
      map_entries_.clear();
      return false;
    }
    if (actual < available && tries-- > 0) {
      // Make the buffer slightly larger than |available| to attempt to account
      // for the race between here and the next retrieval.
      map_entries_.resize(available + 20);
      continue;
    }

    map_entries_.resize(actual);

    INITIALIZATION_STATE_SET_VALID(initialized_);
    return true;
  }
}

bool MemoryMapFuchsia::FindMappingForAddress(zx_vaddr_t address,
                                             zx_info_maps_t* map) const {
  bool found = false;
  zx_info_maps_t result = {};
  for (const auto& m : map_entries_) {
    CheckedRange<zx_vaddr_t, size_t> range(m.base, m.size);
    if (range.ContainsValue(address)) {
      if (!found || m.depth > result.depth) {
        result = m;
        found = true;
      }
    }
  }

  if (found) {
    *map = result;
  }
  return found;
}

}  // namespace crashpad
