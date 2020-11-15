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

#include "snapshot/crashpad_types/crashpad_info_reader.h"

#include <sys/types.h>
#include <unistd.h>

#include <memory>

#include "build/build_config.h"
#include "client/annotation_list.h"
#include "client/crashpad_info.h"
#include "client/simple_address_range_bag.h"
#include "client/simple_string_dictionary.h"
#include "gtest/gtest.h"
#include "test/multiprocess_exec.h"
#include "test/process_type.h"
#include "util/file/file_io.h"
#include "util/misc/from_pointer_cast.h"
#include "util/process/process_memory_native.h"

#if defined(OS_FUCHSIA)
#include <zircon/process.h>
#endif

namespace crashpad {
namespace test {
namespace {

constexpr TriState kCrashpadHandlerBehavior = TriState::kEnabled;
constexpr TriState kSystemCrashReporterForwarding = TriState::kDisabled;
constexpr TriState kGatherIndirectlyReferencedMemory = TriState::kUnset;

constexpr uint32_t kIndirectlyReferencedMemoryCap = 42;

class ScopedUnsetCrashpadInfo {
 public:
  explicit ScopedUnsetCrashpadInfo(CrashpadInfo* crashpad_info)
      : crashpad_info_(crashpad_info) {}

  ~ScopedUnsetCrashpadInfo() {
    crashpad_info_->set_crashpad_handler_behavior(TriState::kUnset);
    crashpad_info_->set_system_crash_reporter_forwarding(TriState::kUnset);
    crashpad_info_->set_gather_indirectly_referenced_memory(TriState::kUnset,
                                                            0);
    crashpad_info_->set_extra_memory_ranges(nullptr);
    crashpad_info_->set_simple_annotations(nullptr);
    crashpad_info_->set_annotations_list(nullptr);
  }

 private:
  CrashpadInfo* crashpad_info_;

  DISALLOW_COPY_AND_ASSIGN(ScopedUnsetCrashpadInfo);
};

class CrashpadInfoTestDataSetup {
 public:
  CrashpadInfoTestDataSetup() {
    CrashpadInfo* info = CrashpadInfo::GetCrashpadInfo();
    unset_.reset(new ScopedUnsetCrashpadInfo(info));

    info->set_extra_memory_ranges(&extra_memory_);
    info->set_simple_annotations(&simple_annotations_);
    info->set_annotations_list(&annotation_list_);
    info->set_crashpad_handler_behavior(kCrashpadHandlerBehavior);
    info->set_system_crash_reporter_forwarding(kSystemCrashReporterForwarding);
    info->set_gather_indirectly_referenced_memory(
        kGatherIndirectlyReferencedMemory, kIndirectlyReferencedMemoryCap);
  }

  void GetAddresses(VMAddress* info_address,
                    VMAddress* extra_memory_address,
                    VMAddress* simple_annotations_address,
                    VMAddress* annotations_list_address) {
    *info_address = FromPointerCast<VMAddress>(CrashpadInfo::GetCrashpadInfo());
    *extra_memory_address = FromPointerCast<VMAddress>(&extra_memory_);
    *simple_annotations_address =
        FromPointerCast<VMAddress>(&simple_annotations_);
    *annotations_list_address = FromPointerCast<VMAddress>(&annotation_list_);
  }

 private:
  std::unique_ptr<ScopedUnsetCrashpadInfo> unset_;
  SimpleAddressRangeBag extra_memory_;
  SimpleStringDictionary simple_annotations_;
  AnnotationList annotation_list_;

  DISALLOW_COPY_AND_ASSIGN(CrashpadInfoTestDataSetup);
};

void ExpectCrashpadInfo(ProcessType process,
                        bool is_64_bit,
                        VMAddress info_address,
                        VMAddress extra_memory_address,
                        VMAddress simple_annotations_address,
                        VMAddress annotations_list_address) {
  ProcessMemoryNative memory;
  ASSERT_TRUE(memory.Initialize(process));

  ProcessMemoryRange range;
  ASSERT_TRUE(range.Initialize(&memory, is_64_bit));

  CrashpadInfoReader reader;
  ASSERT_TRUE(reader.Initialize(&range, info_address));
  EXPECT_EQ(reader.CrashpadHandlerBehavior(), kCrashpadHandlerBehavior);
  EXPECT_EQ(reader.SystemCrashReporterForwarding(),
            kSystemCrashReporterForwarding);
  EXPECT_EQ(reader.GatherIndirectlyReferencedMemory(),
            kGatherIndirectlyReferencedMemory);
  EXPECT_EQ(reader.IndirectlyReferencedMemoryCap(),
            kIndirectlyReferencedMemoryCap);
  EXPECT_EQ(reader.ExtraMemoryRanges(), extra_memory_address);
  EXPECT_EQ(reader.SimpleAnnotations(), simple_annotations_address);
  EXPECT_EQ(reader.AnnotationsList(), annotations_list_address);
}

TEST(CrashpadInfoReader, ReadFromSelf) {
#if defined(ARCH_CPU_64_BITS)
  constexpr bool am_64_bit = true;
#else
  constexpr bool am_64_bit = false;
#endif

  CrashpadInfoTestDataSetup test_data_setup;
  VMAddress info_address;
  VMAddress extra_memory_address;
  VMAddress simple_annotations_address;
  VMAddress annotations_list_address;
  test_data_setup.GetAddresses(&info_address,
                               &extra_memory_address,
                               &simple_annotations_address,
                               &annotations_list_address);
  ExpectCrashpadInfo(GetSelfProcess(),
                     am_64_bit,
                     info_address,
                     extra_memory_address,
                     simple_annotations_address,
                     annotations_list_address);
}

CRASHPAD_CHILD_TEST_MAIN(ReadFromChildTestMain) {
  CrashpadInfoTestDataSetup test_data_setup;
  VMAddress info_address;
  VMAddress extra_memory_address;
  VMAddress simple_annotations_address;
  VMAddress annotations_list_address;
  test_data_setup.GetAddresses(&info_address,
                               &extra_memory_address,
                               &simple_annotations_address,
                               &annotations_list_address);

  FileHandle out = StdioFileHandle(StdioStream::kStandardOutput);
  CheckedWriteFile(out, &info_address, sizeof(info_address));
  CheckedWriteFile(out, &extra_memory_address, sizeof(extra_memory_address));
  CheckedWriteFile(
      out, &simple_annotations_address, sizeof(simple_annotations_address));
  CheckedWriteFile(
      out, &annotations_list_address, sizeof(annotations_list_address));
  CheckedReadFileAtEOF(StdioFileHandle(StdioStream::kStandardInput));
  return 0;
}

class ReadFromChildTest : public MultiprocessExec {
 public:
  ReadFromChildTest() : MultiprocessExec() {
    SetChildTestMainFunction("ReadFromChildTestMain");
  }

  ~ReadFromChildTest() = default;

 private:
  void MultiprocessParent() {
#if defined(ARCH_CPU_64_BITS)
    constexpr bool am_64_bit = true;
#else
    constexpr bool am_64_bit = false;
#endif

    VMAddress info_address;
    VMAddress extra_memory_address;
    VMAddress simple_annotations_address;
    VMAddress annotations_list_address;
    ASSERT_TRUE(
        ReadFileExactly(ReadPipeHandle(), &info_address, sizeof(info_address)));
    ASSERT_TRUE(ReadFileExactly(
        ReadPipeHandle(), &extra_memory_address, sizeof(extra_memory_address)));
    ASSERT_TRUE(ReadFileExactly(ReadPipeHandle(),
                                &simple_annotations_address,
                                sizeof(simple_annotations_address)));
    ASSERT_TRUE(ReadFileExactly(ReadPipeHandle(),
                                &annotations_list_address,
                                sizeof(annotations_list_address)));
    ExpectCrashpadInfo(ChildProcess(),
                       am_64_bit,
                       info_address,
                       extra_memory_address,
                       simple_annotations_address,
                       annotations_list_address);
  }

  DISALLOW_COPY_AND_ASSIGN(ReadFromChildTest);
};

TEST(CrashpadInfoReader, ReadFromChild) {
  ReadFromChildTest test;
  test.Run();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
