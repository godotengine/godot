// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include <string>

#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::AutoText;

TEST(TextStartsWithOp, YesAtStart) {
  EXPECT_TRUE(AssemblyContext(AutoText("OpFoo"), nullptr).isStartOfNewInst());
  EXPECT_TRUE(AssemblyContext(AutoText("OpFoo"), nullptr).isStartOfNewInst());
  EXPECT_TRUE(AssemblyContext(AutoText("OpEnCL"), nullptr).isStartOfNewInst());
}

TEST(TextStartsWithOp, YesAtMiddle) {
  {
    AutoText text("  OpFoo");
    AssemblyContext dat(text, nullptr);
    dat.seekForward(2);
    EXPECT_TRUE(dat.isStartOfNewInst());
  }
  {
    AutoText text("xx OpFoo");
    AssemblyContext dat(text, nullptr);
    dat.seekForward(2);
    EXPECT_TRUE(dat.isStartOfNewInst());
  }
}

TEST(TextStartsWithOp, NoIfTooFar) {
  AutoText text("  OpFoo");
  AssemblyContext dat(text, nullptr);
  dat.seekForward(3);
  EXPECT_FALSE(dat.isStartOfNewInst());
}

TEST(TextStartsWithOp, NoRegular) {
  EXPECT_FALSE(
      AssemblyContext(AutoText("Fee Fi Fo Fum"), nullptr).isStartOfNewInst());
  EXPECT_FALSE(AssemblyContext(AutoText("123456"), nullptr).isStartOfNewInst());
  EXPECT_FALSE(AssemblyContext(AutoText("123456"), nullptr).isStartOfNewInst());
  EXPECT_FALSE(AssemblyContext(AutoText("OpenCL"), nullptr).isStartOfNewInst());
}

TEST(TextStartsWithOp, YesForValueGenerationForm) {
  EXPECT_TRUE(
      AssemblyContext(AutoText("%foo = OpAdd"), nullptr).isStartOfNewInst());
  EXPECT_TRUE(
      AssemblyContext(AutoText("%foo  =  OpAdd"), nullptr).isStartOfNewInst());
}

TEST(TextStartsWithOp, NoForNearlyValueGeneration) {
  EXPECT_FALSE(
      AssemblyContext(AutoText("%foo = "), nullptr).isStartOfNewInst());
  EXPECT_FALSE(AssemblyContext(AutoText("%foo "), nullptr).isStartOfNewInst());
  EXPECT_FALSE(AssemblyContext(AutoText("%foo"), nullptr).isStartOfNewInst());
}

}  // namespace
}  // namespace spvtools
