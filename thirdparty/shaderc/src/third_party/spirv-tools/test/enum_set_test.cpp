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

#include <algorithm>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "source/enum_set.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::ElementsIn;
using ::testing::Eq;
using ::testing::ValuesIn;

TEST(EnumSet, IsEmpty1) {
  EnumSet<uint32_t> set;
  EXPECT_TRUE(set.IsEmpty());
  set.Add(0);
  EXPECT_FALSE(set.IsEmpty());
}

TEST(EnumSet, IsEmpty2) {
  EnumSet<uint32_t> set;
  EXPECT_TRUE(set.IsEmpty());
  set.Add(150);
  EXPECT_FALSE(set.IsEmpty());
}

TEST(EnumSet, IsEmpty3) {
  EnumSet<uint32_t> set(4);
  EXPECT_FALSE(set.IsEmpty());
}

TEST(EnumSet, IsEmpty4) {
  EnumSet<uint32_t> set(300);
  EXPECT_FALSE(set.IsEmpty());
}

TEST(EnumSetHasAnyOf, EmptySetEmptyQuery) {
  const EnumSet<uint32_t> set;
  const EnumSet<uint32_t> empty;
  EXPECT_TRUE(set.HasAnyOf(empty));
  EXPECT_TRUE(EnumSet<uint32_t>().HasAnyOf(EnumSet<uint32_t>()));
}

TEST(EnumSetHasAnyOf, MaskSetEmptyQuery) {
  EnumSet<uint32_t> set;
  const EnumSet<uint32_t> empty;
  set.Add(5);
  set.Add(8);
  EXPECT_TRUE(set.HasAnyOf(empty));
}

TEST(EnumSetHasAnyOf, OverflowSetEmptyQuery) {
  EnumSet<uint32_t> set;
  const EnumSet<uint32_t> empty;
  set.Add(200);
  set.Add(300);
  EXPECT_TRUE(set.HasAnyOf(empty));
}

TEST(EnumSetHasAnyOf, EmptyQuery) {
  EnumSet<uint32_t> set;
  const EnumSet<uint32_t> empty;
  set.Add(5);
  set.Add(8);
  set.Add(200);
  set.Add(300);
  EXPECT_TRUE(set.HasAnyOf(empty));
}

TEST(EnumSetHasAnyOf, EmptyQueryAlwaysTrue) {
  EnumSet<uint32_t> set;
  const EnumSet<uint32_t> empty;
  EXPECT_TRUE(set.HasAnyOf(empty));
  set.Add(5);
  EXPECT_TRUE(set.HasAnyOf(empty));

  EXPECT_TRUE(EnumSet<uint32_t>(100).HasAnyOf(EnumSet<uint32_t>()));
}

TEST(EnumSetHasAnyOf, ReflexiveMask) {
  EnumSet<uint32_t> set(3);
  set.Add(24);
  set.Add(30);
  EXPECT_TRUE(set.HasAnyOf(set));
}

TEST(EnumSetHasAnyOf, ReflexiveOverflow) {
  EnumSet<uint32_t> set(200);
  set.Add(300);
  set.Add(400);
  EXPECT_TRUE(set.HasAnyOf(set));
}

TEST(EnumSetHasAnyOf, Reflexive) {
  EnumSet<uint32_t> set(3);
  set.Add(24);
  set.Add(300);
  set.Add(400);
  EXPECT_TRUE(set.HasAnyOf(set));
}

TEST(EnumSetHasAnyOf, EmptySetHasNone) {
  EnumSet<uint32_t> set;
  EnumSet<uint32_t> items;
  for (uint32_t i = 0; i < 200; ++i) {
    items.Add(i);
    EXPECT_FALSE(set.HasAnyOf(items));
    EXPECT_FALSE(set.HasAnyOf(EnumSet<uint32_t>(i)));
  }
}

TEST(EnumSetHasAnyOf, MaskSetMaskQuery) {
  EnumSet<uint32_t> set(0);
  EnumSet<uint32_t> items(1);
  EXPECT_FALSE(set.HasAnyOf(items));
  set.Add(2);
  items.Add(3);
  EXPECT_FALSE(set.HasAnyOf(items));
  set.Add(3);
  EXPECT_TRUE(set.HasAnyOf(items));
  set.Add(4);
  EXPECT_TRUE(set.HasAnyOf(items));
}

TEST(EnumSetHasAnyOf, OverflowSetOverflowQuery) {
  EnumSet<uint32_t> set(100);
  EnumSet<uint32_t> items(200);
  EXPECT_FALSE(set.HasAnyOf(items));
  set.Add(300);
  items.Add(400);
  EXPECT_FALSE(set.HasAnyOf(items));
  set.Add(200);
  EXPECT_TRUE(set.HasAnyOf(items));
  set.Add(500);
  EXPECT_TRUE(set.HasAnyOf(items));
}

TEST(EnumSetHasAnyOf, GeneralCase) {
  EnumSet<uint32_t> set(0);
  EnumSet<uint32_t> items(100);
  EXPECT_FALSE(set.HasAnyOf(items));
  set.Add(300);
  items.Add(4);
  EXPECT_FALSE(set.HasAnyOf(items));
  set.Add(5);
  items.Add(500);
  EXPECT_FALSE(set.HasAnyOf(items));
  set.Add(500);
  EXPECT_TRUE(set.HasAnyOf(items));
  EXPECT_FALSE(set.HasAnyOf(EnumSet<uint32_t>(20)));
  EXPECT_FALSE(set.HasAnyOf(EnumSet<uint32_t>(600)));
  EXPECT_TRUE(set.HasAnyOf(EnumSet<uint32_t>(5)));
  EXPECT_TRUE(set.HasAnyOf(EnumSet<uint32_t>(300)));
  EXPECT_TRUE(set.HasAnyOf(EnumSet<uint32_t>(0)));
}

TEST(EnumSet, DefaultIsEmpty) {
  EnumSet<uint32_t> set;
  for (uint32_t i = 0; i < 1000; ++i) {
    EXPECT_FALSE(set.Contains(i));
  }
}

TEST(CapabilitySet, ConstructSingleMemberMatrix) {
  CapabilitySet s(SpvCapabilityMatrix);
  EXPECT_TRUE(s.Contains(SpvCapabilityMatrix));
  EXPECT_FALSE(s.Contains(SpvCapabilityShader));
  EXPECT_FALSE(s.Contains(static_cast<SpvCapability>(1000)));
}

TEST(CapabilitySet, ConstructSingleMemberMaxInMask) {
  CapabilitySet s(static_cast<SpvCapability>(63));
  EXPECT_FALSE(s.Contains(SpvCapabilityMatrix));
  EXPECT_FALSE(s.Contains(SpvCapabilityShader));
  EXPECT_TRUE(s.Contains(static_cast<SpvCapability>(63)));
  EXPECT_FALSE(s.Contains(static_cast<SpvCapability>(64)));
  EXPECT_FALSE(s.Contains(static_cast<SpvCapability>(1000)));
}

TEST(CapabilitySet, ConstructSingleMemberMinOverflow) {
  // Check the first one that forces overflow beyond the mask.
  CapabilitySet s(static_cast<SpvCapability>(64));
  EXPECT_FALSE(s.Contains(SpvCapabilityMatrix));
  EXPECT_FALSE(s.Contains(SpvCapabilityShader));
  EXPECT_FALSE(s.Contains(static_cast<SpvCapability>(63)));
  EXPECT_TRUE(s.Contains(static_cast<SpvCapability>(64)));
  EXPECT_FALSE(s.Contains(static_cast<SpvCapability>(1000)));
}

TEST(CapabilitySet, ConstructSingleMemberMaxOverflow) {
  // Check the max 32-bit signed int.
  CapabilitySet s(static_cast<SpvCapability>(0x7fffffffu));
  EXPECT_FALSE(s.Contains(SpvCapabilityMatrix));
  EXPECT_FALSE(s.Contains(SpvCapabilityShader));
  EXPECT_FALSE(s.Contains(static_cast<SpvCapability>(1000)));
  EXPECT_TRUE(s.Contains(static_cast<SpvCapability>(0x7fffffffu)));
}

TEST(CapabilitySet, AddEnum) {
  CapabilitySet s(SpvCapabilityShader);
  s.Add(SpvCapabilityKernel);
  s.Add(static_cast<SpvCapability>(42));
  EXPECT_FALSE(s.Contains(SpvCapabilityMatrix));
  EXPECT_TRUE(s.Contains(SpvCapabilityShader));
  EXPECT_TRUE(s.Contains(SpvCapabilityKernel));
  EXPECT_TRUE(s.Contains(static_cast<SpvCapability>(42)));
}

TEST(CapabilitySet, InitializerListEmpty) {
  CapabilitySet s{};
  for (uint32_t i = 0; i < 1000; i++) {
    EXPECT_FALSE(s.Contains(static_cast<SpvCapability>(i)));
  }
}

struct ForEachCase {
  CapabilitySet capabilities;
  std::vector<SpvCapability> expected;
};

using CapabilitySetForEachTest = ::testing::TestWithParam<ForEachCase>;

TEST_P(CapabilitySetForEachTest, CallsAsExpected) {
  EXPECT_THAT(ElementsIn(GetParam().capabilities), Eq(GetParam().expected));
}

TEST_P(CapabilitySetForEachTest, CopyConstructor) {
  CapabilitySet copy(GetParam().capabilities);
  EXPECT_THAT(ElementsIn(copy), Eq(GetParam().expected));
}

TEST_P(CapabilitySetForEachTest, MoveConstructor) {
  // We need a writable copy to move from.
  CapabilitySet copy(GetParam().capabilities);
  CapabilitySet moved(std::move(copy));
  EXPECT_THAT(ElementsIn(moved), Eq(GetParam().expected));

  // The moved-from set is empty.
  EXPECT_THAT(ElementsIn(copy), Eq(std::vector<SpvCapability>{}));
}

TEST_P(CapabilitySetForEachTest, OperatorEquals) {
  CapabilitySet assigned = GetParam().capabilities;
  EXPECT_THAT(ElementsIn(assigned), Eq(GetParam().expected));
}

TEST_P(CapabilitySetForEachTest, OperatorEqualsSelfAssign) {
  CapabilitySet assigned{GetParam().capabilities};
  assigned = assigned;
  EXPECT_THAT(ElementsIn(assigned), Eq(GetParam().expected));
}

INSTANTIATE_TEST_SUITE_P(Samples, CapabilitySetForEachTest,
                         ValuesIn(std::vector<ForEachCase>{
                             {{}, {}},
                             {{SpvCapabilityMatrix}, {SpvCapabilityMatrix}},
                             {{SpvCapabilityKernel, SpvCapabilityShader},
                              {SpvCapabilityShader, SpvCapabilityKernel}},
                             {{static_cast<SpvCapability>(999)},
                              {static_cast<SpvCapability>(999)}},
                             {{static_cast<SpvCapability>(0x7fffffff)},
                              {static_cast<SpvCapability>(0x7fffffff)}},
                             // Mixture and out of order
                             {{static_cast<SpvCapability>(0x7fffffff),
                               static_cast<SpvCapability>(100),
                               SpvCapabilityShader, SpvCapabilityMatrix},
                              {SpvCapabilityMatrix, SpvCapabilityShader,
                               static_cast<SpvCapability>(100),
                               static_cast<SpvCapability>(0x7fffffff)}},
                         }));

}  // namespace
}  // namespace spvtools
