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

#include "snapshot/win/system_snapshot_win.h"

#include <sys/time.h>
#include <time.h>

#include <string>

#include "build/build_config.h"
#include "gtest/gtest.h"
#include "snapshot/win/process_reader_win.h"

namespace crashpad {
namespace test {
namespace {

class SystemSnapshotWinTest : public testing::Test {
 public:
  SystemSnapshotWinTest()
      : Test(),
        process_reader_(),
        system_snapshot_() {
  }

  const internal::SystemSnapshotWin& system_snapshot() const {
    return system_snapshot_;
  }

  // testing::Test:
  void SetUp() override {
    ASSERT_TRUE(process_reader_.Initialize(GetCurrentProcess(),
                                           ProcessSuspensionState::kRunning));
    system_snapshot_.Initialize(&process_reader_);
  }

 private:
  ProcessReaderWin process_reader_;
  internal::SystemSnapshotWin system_snapshot_;

  DISALLOW_COPY_AND_ASSIGN(SystemSnapshotWinTest);
};

TEST_F(SystemSnapshotWinTest, GetCPUArchitecture) {
  CPUArchitecture cpu_architecture = system_snapshot().GetCPUArchitecture();

#if defined(ARCH_CPU_X86)
  EXPECT_EQ(cpu_architecture, kCPUArchitectureX86);
#elif defined(ARCH_CPU_X86_64)
  EXPECT_EQ(cpu_architecture, kCPUArchitectureX86_64);
#endif
}

TEST_F(SystemSnapshotWinTest, CPUCount) {
  EXPECT_GE(system_snapshot().CPUCount(), 1);
}

TEST_F(SystemSnapshotWinTest, CPUVendor) {
  std::string cpu_vendor = system_snapshot().CPUVendor();

  // There are a variety of other values, but we don't expect to run our tests
  // on them.
  EXPECT_TRUE(cpu_vendor == "GenuineIntel" || cpu_vendor == "AuthenticAMD");
}

TEST_F(SystemSnapshotWinTest, CPUX86SupportsDAZ) {
  // Most SSE2+ machines support Denormals-Are-Zero. This may fail if run on
  // older machines.
  EXPECT_TRUE(system_snapshot().CPUX86SupportsDAZ());
}

TEST_F(SystemSnapshotWinTest, GetOperatingSystem) {
  EXPECT_EQ(system_snapshot().GetOperatingSystem(),
            SystemSnapshot::kOperatingSystemWindows);
}

TEST_F(SystemSnapshotWinTest, OSVersion) {
  int major;
  int minor;
  int bugfix;
  std::string build;
  system_snapshot().OSVersion(&major, &minor, &bugfix, &build);

  EXPECT_GE(major, 5);
  if (major == 5)
    EXPECT_GE(minor, 1);
  if (major == 6)
    EXPECT_TRUE(minor >= 0 && minor <= 3);
}

TEST_F(SystemSnapshotWinTest, OSVersionFull) {
  EXPECT_FALSE(system_snapshot().OSVersionFull().empty());
}

TEST_F(SystemSnapshotWinTest, MachineDescription) {
  EXPECT_TRUE(system_snapshot().MachineDescription().empty());
}

TEST_F(SystemSnapshotWinTest, TimeZone) {
  SystemSnapshot::DaylightSavingTimeStatus dst_status;
  int standard_offset_seconds;
  int daylight_offset_seconds;
  std::string standard_name;
  std::string daylight_name;

  system_snapshot().TimeZone(&dst_status,
                             &standard_offset_seconds,
                             &daylight_offset_seconds,
                             &standard_name,
                             &daylight_name);

  // |standard_offset_seconds| gives seconds east of UTC, and |timezone| gives
  // seconds west of UTC.
  long timezone = 0;
  _get_timezone(&timezone);
  EXPECT_EQ(standard_offset_seconds, -timezone);

  // In contemporary usage, most time zones have an integer hour offset from
  // UTC, although several are at a half-hour offset, and two are at 15-minute
  // offsets. Throughout history, other variations existed. See
  // https://www.timeanddate.com/time/time-zones-interesting.html.
  EXPECT_EQ(standard_offset_seconds % (15 * 60), 0)
      << "standard_offset_seconds " << standard_offset_seconds;

  // dst_status of kDoesNotObserveDaylightSavingTime can mean only that the
  // adjustment is not automatic, as opposed to daylight/standard differences
  // not existing at all. So it cannot be asserted that the two offsets are the
  // same in that case.

  EXPECT_EQ(daylight_offset_seconds % (15 * 60), 0)
      << "daylight_offset_seconds " << daylight_offset_seconds;

  // In contemporary usage, dst_delta_seconds will almost always be one hour,
  // except for Lord Howe Island, Australia, which uses a 30-minute delta.
  // Throughout history, other variations existed. See
  // https://www.timeanddate.com/time/dst/.
  int dst_delta_seconds = daylight_offset_seconds - standard_offset_seconds;
  if (dst_delta_seconds != 60 * 60 && dst_delta_seconds != 30 * 60) {
    FAIL() << "dst_delta_seconds " << dst_delta_seconds;
  }

  if (dst_status != SystemSnapshot::kDoesNotObserveDaylightSavingTime) {
    EXPECT_NE(standard_name, daylight_name);
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
