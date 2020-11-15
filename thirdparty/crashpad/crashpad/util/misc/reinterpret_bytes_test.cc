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

#include "util/misc/reinterpret_bytes.h"

#include <stdint.h>

#include <limits>

#include "base/bit_cast.h"
#include "gtest/gtest.h"
#include "util/numeric/int128.h"

namespace crashpad {
namespace test {
namespace {

template <typename From, typename To>
void ExpectReinterpret(From from, To* to, To expected) {
  ASSERT_TRUE(ReinterpretBytes(from, to));
  EXPECT_EQ(*to, expected);
}

template <typename From, typename To>
void ExpectUnsignedEqual(From from, To* to) {
  To expected = static_cast<To>(from);
  ExpectReinterpret(from, to, expected);
}

TEST(ReinterpretBytes, ToUnsigned) {
  uint64_t from64, to64;
  uint32_t from32, to32;

  from32 = 0;
  ExpectUnsignedEqual(from32, &to32);
  ExpectUnsignedEqual(from32, &to64);

  from32 = std::numeric_limits<uint32_t>::max();
  ExpectUnsignedEqual(from32, &to32);
  ExpectUnsignedEqual(from32, &to64);

  from64 = 0;
  ExpectUnsignedEqual(from64, &to32);
  ExpectUnsignedEqual(from64, &to64);

  from64 = std::numeric_limits<uint64_t>::max();
  ExpectUnsignedEqual(from64, &to64);
  EXPECT_FALSE(ReinterpretBytes(from64, &to32));

  uint8_t to8 = std::numeric_limits<uint8_t>::max();
  uint128_struct from128;
  from128.lo = to8;
  from128.hi = 0;
  ExpectReinterpret(from128, &to8, to8);
}

TEST(ReinterpretBytes, ToSigned) {
  uint64_t from64;
  int64_t to64;
  int32_t to32;

  from64 = 0;
  ExpectReinterpret(from64, &to32, static_cast<int32_t>(0));
  ExpectReinterpret(from64, &to64, static_cast<int64_t>(0));

  to32 = -1;
  from64 = bit_cast<uint32_t>(to32);
  ExpectReinterpret(from64, &to32, to32);

  to32 = std::numeric_limits<int32_t>::max();
  from64 = bit_cast<uint32_t>(to32);
  ExpectReinterpret(from64, &to32, to32);

  to32 = std::numeric_limits<int32_t>::min();
  from64 = bit_cast<uint32_t>(to32);
  ExpectReinterpret(from64, &to32, to32);

  to64 = -1;
  from64 = bit_cast<uint64_t>(to64);
  ExpectReinterpret(from64, &to64, to64);

  to64 = std::numeric_limits<int64_t>::max();
  from64 = bit_cast<uint64_t>(to64);
  ExpectReinterpret(from64, &to64, to64);

  to64 = std::numeric_limits<int64_t>::min();
  from64 = bit_cast<uint64_t>(to64);
  ExpectReinterpret(from64, &to64, to64);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
