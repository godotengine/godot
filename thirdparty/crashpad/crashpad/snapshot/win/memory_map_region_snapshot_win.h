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

#ifndef CRASHPAD_SNAPSHOT_WIN_MEMORY_MAP_REGION_SNAPSHOT_WIN_H_
#define CRASHPAD_SNAPSHOT_WIN_MEMORY_MAP_REGION_SNAPSHOT_WIN_H_

#include "snapshot/memory_map_region_snapshot.h"

namespace crashpad {
namespace internal {

class MemoryMapRegionSnapshotWin : public MemoryMapRegionSnapshot {
 public:
  explicit MemoryMapRegionSnapshotWin(const MEMORY_BASIC_INFORMATION64& mbi);
  ~MemoryMapRegionSnapshotWin() override;

  virtual const MINIDUMP_MEMORY_INFO& AsMinidumpMemoryInfo() const override;

 private:
  MINIDUMP_MEMORY_INFO memory_info_;
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_WIN_MEMORY_MAP_REGION_SNAPSHOT_WIN_H_
