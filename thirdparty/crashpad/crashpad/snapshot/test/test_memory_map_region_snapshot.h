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

#ifndef CRASHPAD_SNAPSHOT_TEST_TEST_MEMORY_MAP_REGION_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_TEST_TEST_MEMORY_MAP_REGION_SNAPSHOT_H_

#include <vector>

#include "base/macros.h"
#include "snapshot/memory_map_region_snapshot.h"

namespace crashpad {
namespace test {

//! \brief A test MemoryMapRegionSnapshot that can carry arbitrary data for
//!     testing purposes.
class TestMemoryMapRegionSnapshot final : public MemoryMapRegionSnapshot {
 public:
  TestMemoryMapRegionSnapshot();
  ~TestMemoryMapRegionSnapshot() override;

  void SetMindumpMemoryInfo(const MINIDUMP_MEMORY_INFO& mmi);

  // MemoryMapRegionSnapshot:
  const MINIDUMP_MEMORY_INFO& AsMinidumpMemoryInfo() const override;

 private:
  MINIDUMP_MEMORY_INFO memory_info_;

  DISALLOW_COPY_AND_ASSIGN(TestMemoryMapRegionSnapshot);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_TEST_TEST_MEMORY_MAP_REGION_SNAPSHOT_H_
