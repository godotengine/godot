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

#include "util/win/critical_section_with_debug_info.h"

#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(CriticalSectionWithDebugInfo, CriticalSectionWithDebugInfo) {
  CRITICAL_SECTION critical_section;
  ASSERT_TRUE(
      InitializeCriticalSectionWithDebugInfoIfPossible(&critical_section));
  EnterCriticalSection(&critical_section);
  LeaveCriticalSection(&critical_section);
  DeleteCriticalSection(&critical_section);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
