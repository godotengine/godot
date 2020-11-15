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

#include "util/numeric/in_range_cast.h"

#include <stdint.h>

#include <limits>

#include "gtest/gtest.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {
namespace test {
namespace {

constexpr int32_t kInt32Min = std::numeric_limits<int32_t>::min();
constexpr int64_t kInt64Min = std::numeric_limits<int64_t>::min();

TEST(InRangeCast, Uint32) {
  EXPECT_EQ(InRangeCast<uint32_t>(0, 1), 0u);
  EXPECT_EQ(InRangeCast<uint32_t>(1, 1), 1u);
  EXPECT_EQ(InRangeCast<uint32_t>(2, 1), 2u);
  EXPECT_EQ(InRangeCast<uint32_t>(-1, 0), 0u);
  EXPECT_EQ(InRangeCast<uint32_t>(-1, 1), 1u);
  EXPECT_EQ(InRangeCast<uint32_t>(-1, 2), 2u);
  EXPECT_EQ(InRangeCast<uint32_t>(0xffffffffu, 1), 0xffffffffu);
  EXPECT_EQ(InRangeCast<uint32_t>(UINT64_C(0xffffffff), 1), 0xffffffffu);
  EXPECT_EQ(InRangeCast<uint32_t>(UINT64_C(0x100000000), 1), 1u);
  EXPECT_EQ(InRangeCast<uint32_t>(UINT64_C(0x100000001), 1), 1u);
  EXPECT_EQ(InRangeCast<uint32_t>(kInt32Min, 1), 1u);
  EXPECT_EQ(InRangeCast<uint32_t>(kInt64Min, 1), 1u);
  EXPECT_EQ(InRangeCast<uint32_t>(-1, 0xffffffffu), 0xffffffffu);
}

TEST(InRangeCast, Int32) {
  EXPECT_EQ(InRangeCast<int32_t>(0, 1), 0);
  EXPECT_EQ(InRangeCast<int32_t>(1, 1), 1);
  EXPECT_EQ(InRangeCast<int32_t>(2, 1), 2);
  EXPECT_EQ(InRangeCast<int32_t>(-1, 1), -1);
  EXPECT_EQ(InRangeCast<int32_t>(0x7fffffff, 1), 0x7fffffff);
  EXPECT_EQ(InRangeCast<int32_t>(0x7fffffffu, 1), 0x7fffffff);
  EXPECT_EQ(InRangeCast<int32_t>(0x80000000u, 1), 1);
  EXPECT_EQ(InRangeCast<int32_t>(0xffffffffu, 1), 1);
  EXPECT_EQ(InRangeCast<int32_t>(INT64_C(0x80000000), 1), 1);
  EXPECT_EQ(InRangeCast<int32_t>(INT64_C(0xffffffff), 1), 1);
  EXPECT_EQ(InRangeCast<int32_t>(INT64_C(0x100000000), 1), 1);
  EXPECT_EQ(InRangeCast<int32_t>(kInt32Min, 1), kInt32Min);
  EXPECT_EQ(InRangeCast<int32_t>(implicit_cast<int64_t>(kInt32Min), 1),
            kInt32Min);
  EXPECT_EQ(InRangeCast<int32_t>(implicit_cast<int64_t>(kInt32Min) - 1, 1), 1);
  EXPECT_EQ(InRangeCast<int32_t>(kInt64Min, 1), 1);
  EXPECT_EQ(InRangeCast<int32_t>(0xffffffffu, 0), 0);
  EXPECT_EQ(InRangeCast<int32_t>(0xffffffffu, -1), -1);
  EXPECT_EQ(InRangeCast<int32_t>(0xffffffffu, kInt32Min), kInt32Min);
  EXPECT_EQ(InRangeCast<int32_t>(0xffffffffu, 0x7fffffff), 0x7fffffff);
}

TEST(InRangeCast, Uint64) {
  EXPECT_EQ(InRangeCast<uint64_t>(0, 1), 0u);
  EXPECT_EQ(InRangeCast<uint64_t>(1, 1), 1u);
  EXPECT_EQ(InRangeCast<uint64_t>(2, 1), 2u);
  EXPECT_EQ(InRangeCast<uint64_t>(-1, 0), 0u);
  EXPECT_EQ(InRangeCast<uint64_t>(-1, 1), 1u);
  EXPECT_EQ(InRangeCast<uint64_t>(-1, 2), 2u);
  EXPECT_EQ(InRangeCast<uint64_t>(0xffffffffu, 1), 0xffffffffu);
  EXPECT_EQ(InRangeCast<uint64_t>(UINT64_C(0xffffffff), 1), 0xffffffffu);
  EXPECT_EQ(InRangeCast<uint64_t>(UINT64_C(0x100000000), 1),
            UINT64_C(0x100000000));
  EXPECT_EQ(InRangeCast<uint64_t>(UINT64_C(0x100000001), 1),
            UINT64_C(0x100000001));
  EXPECT_EQ(InRangeCast<uint64_t>(kInt32Min, 1), 1u);
  EXPECT_EQ(InRangeCast<uint64_t>(INT64_C(-1), 1), 1u);
  EXPECT_EQ(InRangeCast<uint64_t>(kInt64Min, 1), 1u);
  EXPECT_EQ(InRangeCast<uint64_t>(-1, UINT64_C(0xffffffffffffffff)),
            UINT64_C(0xffffffffffffffff));
}

TEST(InRangeCast, Int64) {
  EXPECT_EQ(InRangeCast<int64_t>(0, 1), 0);
  EXPECT_EQ(InRangeCast<int64_t>(1, 1), 1);
  EXPECT_EQ(InRangeCast<int64_t>(2, 1), 2);
  EXPECT_EQ(InRangeCast<int64_t>(-1, 1), -1);
  EXPECT_EQ(InRangeCast<int64_t>(0x7fffffff, 1), 0x7fffffff);
  EXPECT_EQ(InRangeCast<int64_t>(0x7fffffffu, 1), 0x7fffffff);
  EXPECT_EQ(InRangeCast<int64_t>(0x80000000u, 1), INT64_C(0x80000000));
  EXPECT_EQ(InRangeCast<int64_t>(0xffffffffu, 1), INT64_C(0xffffffff));
  EXPECT_EQ(InRangeCast<int64_t>(INT64_C(0x80000000), 1), INT64_C(0x80000000));
  EXPECT_EQ(InRangeCast<int64_t>(INT64_C(0xffffffff), 1), INT64_C(0xffffffff));
  EXPECT_EQ(InRangeCast<int64_t>(INT64_C(0x100000000), 1),
            INT64_C(0x100000000));
  EXPECT_EQ(InRangeCast<int64_t>(INT64_C(0x7fffffffffffffff), 1),
            INT64_C(0x7fffffffffffffff));
  EXPECT_EQ(InRangeCast<int64_t>(UINT64_C(0x7fffffffffffffff), 1),
            INT64_C(0x7fffffffffffffff));
  EXPECT_EQ(InRangeCast<int64_t>(UINT64_C(0x8000000000000000), 1), 1);
  EXPECT_EQ(InRangeCast<int64_t>(UINT64_C(0xffffffffffffffff), 1), 1);
  EXPECT_EQ(InRangeCast<int64_t>(kInt32Min, 1), kInt32Min);
  EXPECT_EQ(InRangeCast<int64_t>(implicit_cast<int64_t>(kInt32Min), 1),
            kInt32Min);
  EXPECT_EQ(InRangeCast<int64_t>(kInt64Min, 1), kInt64Min);
  EXPECT_EQ(InRangeCast<int64_t>(UINT64_C(0xffffffffffffffff), 0), 0);
  EXPECT_EQ(InRangeCast<int64_t>(UINT64_C(0xffffffffffffffff), -1), -1);
  EXPECT_EQ(InRangeCast<int64_t>(UINT64_C(0xffffffffffffffff), kInt64Min),
            kInt64Min);
  EXPECT_EQ(InRangeCast<int64_t>(UINT64_C(0xffffffffffffffff),
                                 INT64_C(0x7fffffffffffffff)),
            INT64_C(0x7fffffffffffffff));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
