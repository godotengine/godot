// Copyright (c) 2018 Google LLC.
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

#include <unistd.h>
#include <sstream>

#include "gtest/gtest.h"
#include "source/util/timer.h"

namespace spvtools {
namespace utils {
namespace {

// A mock class to mimic Timer class for a testing purpose. It has fixed
// CPU/WALL/USR/SYS time, RSS delta, and the delta of the number of page faults.
class MockTimer : public Timer {
 public:
  MockTimer(std::ostream* out, bool measure_mem_usage = false)
      : Timer(out, measure_mem_usage) {}
  double CPUTime() override { return 0.019123; }
  double WallTime() override { return 0.019723; }
  double UserTime() override { return 0.012723; }
  double SystemTime() override { return 0.002723; }
  long RSS() const override { return 360L; }
  long PageFault() const override { return 3600L; }
};

// This unit test checks whether the actual output of MockTimer::Report() is the
// same as fixed CPU/WALL/USR/SYS time, RSS delta, and the delta of the number
// of page faults that are returned by MockTimer.
TEST(MockTimer, DoNothing) {
  std::ostringstream buf;

  PrintTimerDescription(&buf);
  MockTimer timer(&buf);
  timer.Start();

  // Do nothing.

  timer.Stop();
  timer.Report("TimerTest");

  EXPECT_EQ(0.019123, timer.CPUTime());
  EXPECT_EQ(0.019723, timer.WallTime());
  EXPECT_EQ(0.012723, timer.UserTime());
  EXPECT_EQ(0.002723, timer.SystemTime());
  EXPECT_EQ(
      "                     PASS name    CPU time   WALL time    USR time"
      "    SYS time\n                     TimerTest        0.02        0.02"
      "        0.01        0.00\n",
      buf.str());
}

// This unit test checks whether the ScopedTimer<MockTimer> correctly reports
// the fixed CPU/WALL/USR/SYS time, RSS delta, and the delta of the number of
// page faults that are returned by MockTimer.
TEST(MockTimer, TestScopedTimer) {
  std::ostringstream buf;

  {
    ScopedTimer<MockTimer> scopedtimer(&buf, "ScopedTimerTest");
    // Do nothing.
  }

  EXPECT_EQ(
      "               ScopedTimerTest        0.02        0.02        0.01"
      "        0.00\n",
      buf.str());
}

// A mock class to mimic CumulativeTimer class for a testing purpose. It has
// fixed CPU/WALL/USR/SYS time, RSS delta, and the delta of the number of page
// faults for each measurement (i.e., a pair of Start() and Stop()). If the
// number of measurements increases, it increases |count_stop_| by the number of
// calling Stop() and the amount of each resource usage is proportional to
// |count_stop_|.
class MockCumulativeTimer : public CumulativeTimer {
 public:
  MockCumulativeTimer(std::ostream* out, bool measure_mem_usage = false)
      : CumulativeTimer(out, measure_mem_usage), count_stop_(0) {}
  double CPUTime() override { return count_stop_ * 0.019123; }
  double WallTime() override { return count_stop_ * 0.019723; }
  double UserTime() override { return count_stop_ * 0.012723; }
  double SystemTime() override { return count_stop_ * 0.002723; }
  long RSS() const override { return count_stop_ * 360L; }
  long PageFault() const override { return count_stop_ * 3600L; }

  // Calling Stop() does nothing but just increases |count_stop_| by 1.
  void Stop() override { ++count_stop_; };

 private:
  unsigned int count_stop_;
};

// This unit test checks whether the MockCumulativeTimer correctly reports the
// cumulative CPU/WALL/USR/SYS time, RSS delta, and the delta of the number of
// page faults whose values are fixed for each measurement (i.e., a pair of
// Start() and Stop()).
TEST(MockCumulativeTimer, DoNothing) {
  CumulativeTimer* ctimer;
  std::ostringstream buf;

  {
    ctimer = new MockCumulativeTimer(&buf);
    ctimer->Start();

    // Do nothing.

    ctimer->Stop();
  }

  {
    ctimer->Start();

    // Do nothing.

    ctimer->Stop();
    ctimer->Report("CumulativeTimerTest");
  }

  EXPECT_EQ(
      "           CumulativeTimerTest        0.04        0.04        0.03"
      "        0.01\n",
      buf.str());

  if (ctimer) delete ctimer;
}

}  // namespace
}  // namespace utils
}  // namespace spvtools
