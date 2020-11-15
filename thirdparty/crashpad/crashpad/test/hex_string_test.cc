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

#include "test/hex_string.h"

#include "base/macros.h"
#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(HexString, HexString) {
  EXPECT_EQ(BytesToHexString(nullptr, 0), "");

  static constexpr char kBytes[] = "Abc123xyz \x0a\x7f\xf0\x9f\x92\xa9_";
  EXPECT_EQ(BytesToHexString(kBytes, arraysize(kBytes)),
            "41626331323378797a200a7ff09f92a95f00");
}

}  // namespace
}  // namespace test
}  // namespace crashpad
