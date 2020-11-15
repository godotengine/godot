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

#include "util/misc/scoped_forbid_return.h"

#include "base/compiler_specific.h"
#include "gtest/gtest.h"
#include "test/gtest_death.h"

namespace crashpad {
namespace test {
namespace {

enum ForbidReturnType {
  kForbidReturnDefault = 0,
  kForbidReturnArmed,
  kForbidReturnDisarmed,
};

void ScopedForbidReturnHelper(ForbidReturnType type) {
  ScopedForbidReturn forbid_return;

  switch (type) {
    case kForbidReturnDefault:
      break;
    case kForbidReturnArmed:
      forbid_return.Arm();
      break;
    case kForbidReturnDisarmed:
      forbid_return.Disarm();
      break;
  }
}

constexpr char kForbiddenMessage[] = "attempt to exit scope forbidden";

TEST(ScopedForbidReturnDeathTest, Default) {
  // kForbiddenMessage may appear to be unused if ASSERT_DEATH_CHECK() throws it
  // away.
  ALLOW_UNUSED_LOCAL(kForbiddenMessage);

  ASSERT_DEATH_CHECK(ScopedForbidReturnHelper(kForbidReturnDefault),
                     kForbiddenMessage);
}

TEST(ScopedForbidReturnDeathTest, Armed) {
  ASSERT_DEATH_CHECK(ScopedForbidReturnHelper(kForbidReturnArmed),
                     kForbiddenMessage);
}

TEST(ScopedForbidReturn, Disarmed) {
  ScopedForbidReturnHelper(kForbidReturnDisarmed);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
