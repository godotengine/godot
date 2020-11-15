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

#include "minidump/minidump_thread_id_map.h"

#include <limits>
#include <set>
#include <utility>

#include "base/logging.h"
#include "base/numerics/safe_conversions.h"
#include "snapshot/thread_snapshot.h"

namespace crashpad {

void BuildMinidumpThreadIDMap(
    const std::vector<const ThreadSnapshot*>& thread_snapshots,
    MinidumpThreadIDMap* thread_id_map) {
  DCHECK(thread_id_map->empty());

  // First, try truncating each 64-bit thread ID to 32 bits. If that’s possible
  // for each unique 64-bit thread ID, then this will be used as the mapping.
  // This preserves as much of the original thread ID as possible when feasible.
  bool collision = false;
  std::set<uint32_t> thread_ids_32;
  for (const ThreadSnapshot* thread_snapshot : thread_snapshots) {
    uint64_t thread_id_64 = thread_snapshot->ThreadID();
    if (thread_id_map->find(thread_id_64) == thread_id_map->end()) {
      uint32_t thread_id_32 = static_cast<uint32_t>(thread_id_64);
      if (!thread_ids_32.insert(thread_id_32).second) {
        collision = true;
        break;
      }
      thread_id_map->insert(std::make_pair(thread_id_64, thread_id_32));
    }
  }

  if (collision) {
    // Since there was a collision, go back and assign each unique 64-bit thread
    // ID its own sequential 32-bit equivalent. The 32-bit thread IDs will not
    // bear any resemblance to the original 64-bit thread IDs.
    thread_id_map->clear();
    for (const ThreadSnapshot* thread_snapshot : thread_snapshots) {
      uint64_t thread_id_64 = thread_snapshot->ThreadID();
      if (thread_id_map->find(thread_id_64) == thread_id_map->end()) {
        uint32_t thread_id_32 =
            base::checked_cast<uint32_t>(thread_id_map->size());
        thread_id_map->insert(std::make_pair(thread_id_64, thread_id_32));
      }
    }

    DCHECK_LE(thread_id_map->size(), std::numeric_limits<uint32_t>::max());
  }
}

}  // namespace crashpad
