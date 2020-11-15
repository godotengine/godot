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

#include "util/mach/exception_behaviors.h"

#include <sys/types.h>

#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "gtest/gtest.h"
#include "util/mach/mach_extensions.h"

namespace crashpad {
namespace test {
namespace {

TEST(ExceptionBehaviors, ExceptionBehaviors) {
  static constexpr struct {
    exception_behavior_t behavior;
    bool state;
    bool identity;
    bool mach_exception_codes;
    exception_behavior_t basic_behavior;
  } kTestData[] = {
      {EXCEPTION_DEFAULT, false, true, false, EXCEPTION_DEFAULT},
      {EXCEPTION_STATE, true, false, false, EXCEPTION_STATE},
      {EXCEPTION_STATE_IDENTITY, true, true, false, EXCEPTION_STATE_IDENTITY},
      {kMachExceptionCodes | EXCEPTION_DEFAULT,
       false,
       true,
       true,
       EXCEPTION_DEFAULT},
      {kMachExceptionCodes | EXCEPTION_STATE,
       true,
       false,
       true,
       EXCEPTION_STATE},
      {kMachExceptionCodes | EXCEPTION_STATE_IDENTITY,
       true,
       true,
       true,
       EXCEPTION_STATE_IDENTITY},
  };

  for (size_t index = 0; index < arraysize(kTestData); ++index) {
    const auto& test_data = kTestData[index];
    SCOPED_TRACE(base::StringPrintf(
        "index %zu, behavior %d", index, test_data.behavior));

    EXPECT_EQ(ExceptionBehaviorHasState(test_data.behavior), test_data.state);
    EXPECT_EQ(ExceptionBehaviorHasIdentity(test_data.behavior),
              test_data.identity);
    EXPECT_EQ(ExceptionBehaviorHasMachExceptionCodes(test_data.behavior),
              test_data.mach_exception_codes);
    EXPECT_EQ(ExceptionBehaviorBasic(test_data.behavior),
              test_data.basic_behavior);
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
