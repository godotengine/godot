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

#include "snapshot/win/process_snapshot_win.h"

#include "base/files/file_path.h"
#include "build/build_config.h"
#include "gtest/gtest.h"
#include "snapshot/win/pe_image_reader.h"
#include "snapshot/win/process_reader_win.h"
#include "test/errors.h"
#include "test/gtest_disabled.h"
#include "test/test_paths.h"
#include "test/win/child_launcher.h"
#include "util/file/file_io.h"
#include "util/win/scoped_handle.h"
#include "util/win/scoped_process_suspend.h"
#include "util/win/scoped_set_event.h"

namespace crashpad {
namespace test {
namespace {

void TestImageReaderChild(const TestPaths::Architecture architecture) {
  UUID done_uuid;
  done_uuid.InitializeWithNew();
  ScopedKernelHANDLE done(
      CreateEvent(nullptr, true, false, done_uuid.ToString16().c_str()));
  ASSERT_TRUE(done.is_valid()) << ErrorMessage("CreateEvent");

  base::FilePath child_test_executable =
      TestPaths::BuildArtifact(L"snapshot",
                               L"image_reader",
                               TestPaths::FileType::kExecutable,
                               architecture);
  ChildLauncher child(child_test_executable, done_uuid.ToString16());
  ASSERT_NO_FATAL_FAILURE(child.Start());

  ScopedSetEvent set_done(done.get());

  char c;
  ASSERT_TRUE(
      LoggingReadFileExactly(child.stdout_read_handle(), &c, sizeof(c)));
  ASSERT_EQ(c, ' ');

  {
    ScopedProcessSuspend suspend(child.process_handle());

    ProcessSnapshotWin process_snapshot;
    ASSERT_TRUE(process_snapshot.Initialize(
        child.process_handle(), ProcessSuspensionState::kSuspended, 0, 0));

    ASSERT_GE(process_snapshot.Modules().size(), 2u);

    UUID uuid;
    DWORD age;
    std::string pdbname;
    const std::string suffix(".pdb");

    // Check the main .exe to see that we can retrieve its sections.
    auto module = reinterpret_cast<const internal::ModuleSnapshotWin*>(
        process_snapshot.Modules()[0]);
    ASSERT_TRUE(module->pe_image_reader().DebugDirectoryInformation(
        &uuid, &age, &pdbname));
    EXPECT_NE(pdbname.find("crashpad_snapshot_test_image_reader"),
              std::string::npos);
    EXPECT_EQ(
        pdbname.compare(pdbname.size() - suffix.size(), suffix.size(), suffix),
        0);

    // Check the dll it loads too.
    module = reinterpret_cast<const internal::ModuleSnapshotWin*>(
        process_snapshot.Modules().back());
    ASSERT_TRUE(module->pe_image_reader().DebugDirectoryInformation(
        &uuid, &age, &pdbname));
    EXPECT_NE(pdbname.find("crashpad_snapshot_test_image_reader_module"),
              std::string::npos);
    EXPECT_EQ(
        pdbname.compare(pdbname.size() - suffix.size(), suffix.size(), suffix),
        0);

    // Sum the size of the extra memory in all the threads and confirm it's near
    // the limit that the child process set in its CrashpadInfo.
    EXPECT_GE(process_snapshot.Threads().size(), 100u);

    size_t extra_memory_total = 0;
    for (const auto* thread : process_snapshot.Threads()) {
      for (const auto* extra_memory : thread->ExtraMemory()) {
        extra_memory_total += extra_memory->Size();
      }
    }

    // Confirm that less than 1M of extra data was gathered. The cap is set to
    // only 100K, but there are other "extra memory" regions that aren't
    // included in the cap. (Completely uncapped it would be > 10M.)
    EXPECT_LT(extra_memory_total, 1000000u);
  }

  // Tell the child it can terminate.
  EXPECT_TRUE(set_done.Set());

  EXPECT_EQ(child.WaitForExit(), 0u);
}

TEST(ProcessSnapshotTest, CrashpadInfoChild) {
  TestImageReaderChild(TestPaths::Architecture::kDefault);
}

#if defined(ARCH_CPU_64_BITS)
TEST(ProcessSnapshotTest, CrashpadInfoChildWOW64) {
  if (!TestPaths::Has32BitBuildArtifacts()) {
    DISABLED_TEST();
  }

  TestImageReaderChild(TestPaths::Architecture::k32Bit);
}
#endif

}  // namespace
}  // namespace test
}  // namespace crashpad
