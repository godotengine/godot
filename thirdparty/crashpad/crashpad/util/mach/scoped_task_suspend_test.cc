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

#include "util/mach/scoped_task_suspend.h"

#include <mach/mach.h>

#include "gtest/gtest.h"
#include "test/mac/mach_errors.h"
#include "test/mac/mach_multiprocess.h"

namespace crashpad {
namespace test {
namespace {

int SuspendCount(task_t task) {
  // As of the 10.8 SDK, the preferred routine is MACH_TASK_BASIC_INFO.
  // TASK_BASIC_INFO_64 is equivalent and works on earlier systems.
  task_basic_info_64 task_basic_info;
  mach_msg_type_number_t task_basic_info_count = TASK_BASIC_INFO_64_COUNT;
  kern_return_t kr = task_info(task,
                               TASK_BASIC_INFO_64,
                               reinterpret_cast<task_info_t>(&task_basic_info),
                               &task_basic_info_count);
  if (kr != KERN_SUCCESS) {
    ADD_FAILURE() << MachErrorMessage(kr, "task_info");
    return -1;
  }

  return task_basic_info.suspend_count;
}

class ScopedTaskSuspendTest final : public MachMultiprocess {
 public:
  ScopedTaskSuspendTest() : MachMultiprocess() {}
  ~ScopedTaskSuspendTest() {}

 private:
  // MachMultiprocess:

  void MachMultiprocessParent() override {
    task_t child_task = ChildTask();

    EXPECT_EQ(SuspendCount(child_task), 0);

    {
      ScopedTaskSuspend suspend(child_task);
      EXPECT_EQ(SuspendCount(child_task), 1);

      {
        ScopedTaskSuspend suspend_again(child_task);
        EXPECT_EQ(SuspendCount(child_task), 2);
      }

      EXPECT_EQ(SuspendCount(child_task), 1);
    }

    EXPECT_EQ(SuspendCount(child_task), 0);
  }

  void MachMultiprocessChild() override {
  }

  DISALLOW_COPY_AND_ASSIGN(ScopedTaskSuspendTest);
};

TEST(ScopedTaskSuspend, ScopedTaskSuspend) {
  ScopedTaskSuspendTest scoped_task_suspend_test;
  scoped_task_suspend_test.Run();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
