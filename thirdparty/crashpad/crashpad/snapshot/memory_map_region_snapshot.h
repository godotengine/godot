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

#ifndef CRASHPAD_SNAPSHOT_MEMORY_MAP_REGION_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_MEMORY_MAP_REGION_SNAPSHOT_H_

#include <windows.h>
#include <dbghelp.h>

namespace crashpad {

//! \brief An abstract interface to a snapshot representing a region of the
//!     memory map present in the snapshot process.
class MemoryMapRegionSnapshot {
 public:
  virtual ~MemoryMapRegionSnapshot() {}

  //! \brief Gets a MINIDUMP_MEMORY_INFO representing the region.
  virtual const MINIDUMP_MEMORY_INFO& AsMinidumpMemoryInfo() const = 0;
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MEMORY_MAP_REGION_SNAPSHOT_H_
