// Copyright (c) 2016 Google Inc.
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

#include "test/unit_spirv.h"

namespace spvtools {
namespace {

TEST(Strnlen, Samples) {
  EXPECT_EQ(0u, spv_strnlen_s(nullptr, 0));
  EXPECT_EQ(0u, spv_strnlen_s(nullptr, 5));
  EXPECT_EQ(0u, spv_strnlen_s("abc", 0));
  EXPECT_EQ(1u, spv_strnlen_s("abc", 1));
  EXPECT_EQ(3u, spv_strnlen_s("abc", 3));
  EXPECT_EQ(3u, spv_strnlen_s("abc\0", 5));
  EXPECT_EQ(0u, spv_strnlen_s("\0", 5));
  EXPECT_EQ(1u, spv_strnlen_s("a\0c", 5));
}

}  // namespace
}  // namespace spvtools
