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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_THREAD_ID_MAP_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_THREAD_ID_MAP_H_

#include <stdint.h>

#include <map>
#include <vector>

namespace crashpad {

class ThreadSnapshot;

//! \brief A map that connects 64-bit snapshot thread IDs to 32-bit minidump
//!     thread IDs.
//!
//! 64-bit snapshot thread IDs are obtained from ThreadSnapshot::ThreadID().
//! 32-bit minidump thread IDs are stored in MINIDUMP_THREAD::ThreadId.
//!
//! A ThreadIDMap ensures that there are no collisions among the set of 32-bit
//! minidump thread IDs.
using MinidumpThreadIDMap = std::map<uint64_t, uint32_t>;

//! \brief Builds a MinidumpThreadIDMap for a group of ThreadSnapshot objects.
//!
//! \param[in] thread_snapshots The thread snapshots to use as source data.
//! \param[out] thread_id_map A MinidumpThreadIDMap to be built by this method.
//!     This map must be empty when this function is called.
//!
//! The map ensures that for any unique 64-bit thread ID found in a
//! ThreadSnapshot, the 32-bit thread ID used in a minidump file will also be
//! unique.
void BuildMinidumpThreadIDMap(
    const std::vector<const ThreadSnapshot*>& thread_snapshots,
    MinidumpThreadIDMap* thread_id_map);

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_THREAD_ID_MAP_H_
