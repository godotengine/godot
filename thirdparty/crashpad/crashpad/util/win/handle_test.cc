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

#include "util/win/handle.h"

#include <stdint.h>

#include <limits>

#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(Handle, HandleToInt) {
  EXPECT_EQ(HandleToInt(nullptr), 0);
  EXPECT_EQ(HandleToInt(INVALID_HANDLE_VALUE), -1);
  EXPECT_EQ(HandleToInt(reinterpret_cast<HANDLE>(1)), 1);
  EXPECT_EQ(HandleToInt(reinterpret_cast<HANDLE>(
                static_cast<intptr_t>(std::numeric_limits<int>::max()))),
            std::numeric_limits<int>::max());
  EXPECT_EQ(HandleToInt(reinterpret_cast<HANDLE>(
                static_cast<intptr_t>(std::numeric_limits<int>::min()))),
            std::numeric_limits<int>::min());
}

TEST(Handle, IntToHandle) {
  EXPECT_EQ(IntToHandle(0), nullptr);
  EXPECT_EQ(IntToHandle(-1), INVALID_HANDLE_VALUE);
  EXPECT_EQ(IntToHandle(1), reinterpret_cast<HANDLE>(1));
  EXPECT_EQ(IntToHandle(std::numeric_limits<int>::max()),
            reinterpret_cast<HANDLE>(
                static_cast<intptr_t>(std::numeric_limits<int>::max())));
  EXPECT_EQ(IntToHandle(std::numeric_limits<int>::min()),
            reinterpret_cast<HANDLE>(
                static_cast<intptr_t>(std::numeric_limits<int>::min())));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
