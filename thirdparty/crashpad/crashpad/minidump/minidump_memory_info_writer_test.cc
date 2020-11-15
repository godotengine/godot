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

#include "minidump/minidump_memory_info_writer.h"

#include <memory>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "minidump/minidump_file_writer.h"
#include "minidump/test/minidump_file_writer_test_util.h"
#include "minidump/test/minidump_writable_test_util.h"
#include "snapshot/test/test_memory_map_region_snapshot.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

// The memory info list is expected to be the only stream.
void GetMemoryInfoListStream(
    const std::string& file_contents,
    const MINIDUMP_MEMORY_INFO_LIST** memory_info_list) {
  constexpr size_t kDirectoryOffset = sizeof(MINIDUMP_HEADER);
  constexpr size_t kMemoryInfoListStreamOffset =
      kDirectoryOffset + sizeof(MINIDUMP_DIRECTORY);

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(file_contents, &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 1, 0));
  ASSERT_TRUE(directory);

  constexpr size_t kDirectoryIndex = 0;

  ASSERT_EQ(directory[kDirectoryIndex].StreamType,
            kMinidumpStreamTypeMemoryInfoList);
  EXPECT_EQ(directory[kDirectoryIndex].Location.Rva,
            kMemoryInfoListStreamOffset);

  *memory_info_list =
      MinidumpWritableAtLocationDescriptor<MINIDUMP_MEMORY_INFO_LIST>(
          file_contents, directory[kDirectoryIndex].Location);
  ASSERT_TRUE(*memory_info_list);
}

TEST(MinidumpMemoryInfoWriter, Empty) {
  MinidumpFileWriter minidump_file_writer;
  auto memory_info_list_writer =
      std::make_unique<MinidumpMemoryInfoListWriter>();
  ASSERT_TRUE(
      minidump_file_writer.AddStream(std::move(memory_info_list_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_EQ(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY) +
                sizeof(MINIDUMP_MEMORY_INFO_LIST));

  const MINIDUMP_MEMORY_INFO_LIST* memory_info_list = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      GetMemoryInfoListStream(string_file.string(), &memory_info_list));

  EXPECT_EQ(memory_info_list->NumberOfEntries, 0u);
}

TEST(MinidumpMemoryInfoWriter, OneRegion) {
  MinidumpFileWriter minidump_file_writer;
  auto memory_info_list_writer =
      std::make_unique<MinidumpMemoryInfoListWriter>();

  auto memory_map_region = std::make_unique<TestMemoryMapRegionSnapshot>();

  MINIDUMP_MEMORY_INFO mmi;
  mmi.BaseAddress = 0x12340000;
  mmi.AllocationBase = 0x12000000;
  mmi.AllocationProtect = PAGE_READWRITE;
  mmi.__alignment1 = 0;
  mmi.RegionSize = 0x6000;
  mmi.State = MEM_COMMIT;
  mmi.Protect = PAGE_NOACCESS;
  mmi.Type = MEM_PRIVATE;
  mmi.__alignment2 = 0;
  memory_map_region->SetMindumpMemoryInfo(mmi);

  std::vector<const MemoryMapRegionSnapshot*> memory_map;
  memory_map.push_back(memory_map_region.get());
  memory_info_list_writer->InitializeFromSnapshot(memory_map);

  ASSERT_TRUE(
      minidump_file_writer.AddStream(std::move(memory_info_list_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_EQ(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY) +
                sizeof(MINIDUMP_MEMORY_INFO_LIST) +
                sizeof(MINIDUMP_MEMORY_INFO));

  const MINIDUMP_MEMORY_INFO_LIST* memory_info_list = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      GetMemoryInfoListStream(string_file.string(), &memory_info_list));

  EXPECT_EQ(memory_info_list->NumberOfEntries, 1u);
  const MINIDUMP_MEMORY_INFO* memory_info =
      reinterpret_cast<const MINIDUMP_MEMORY_INFO*>(&memory_info_list[1]);
  EXPECT_EQ(memory_info->BaseAddress, mmi.BaseAddress);
  EXPECT_EQ(memory_info->AllocationBase, mmi.AllocationBase);
  EXPECT_EQ(memory_info->AllocationProtect, mmi.AllocationProtect);
  EXPECT_EQ(memory_info->RegionSize, mmi.RegionSize);
  EXPECT_EQ(memory_info->State, mmi.State);
  EXPECT_EQ(memory_info->Protect, mmi.Protect);
  EXPECT_EQ(memory_info->Type, mmi.Type);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
