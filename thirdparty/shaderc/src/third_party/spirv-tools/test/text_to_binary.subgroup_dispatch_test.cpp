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

// Assembler tests for instructions in the "Barrier Instructions" section
// of the SPIR-V spec.

#include "test/unit_spirv.h"

#include "gmock/gmock.h"
#include "test/test_fixture.h"

namespace spvtools {
namespace {

using ::spvtest::MakeInstruction;
using ::testing::Eq;

using OpGetKernelLocalSizeForSubgroupCountTest = spvtest::TextToBinaryTest;

// We should be able to assemble it.  Validation checks are in another test
// file.
TEST_F(OpGetKernelLocalSizeForSubgroupCountTest, OpcodeAssemblesInV10) {
  EXPECT_THAT(
      CompiledInstructions("%res = OpGetKernelLocalSizeForSubgroupCount %type "
                           "%sgcount %invoke %param %param_size %param_align",
                           SPV_ENV_UNIVERSAL_1_0),
      Eq(MakeInstruction(SpvOpGetKernelLocalSizeForSubgroupCount,
                         {1, 2, 3, 4, 5, 6, 7})));
}

TEST_F(OpGetKernelLocalSizeForSubgroupCountTest, ArgumentCount) {
  EXPECT_THAT(CompileFailure("OpGetKernelLocalSizeForSubgroupCount",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected <result-id> at the beginning of an instruction, "
                 "found 'OpGetKernelLocalSizeForSubgroupCount'."));
  EXPECT_THAT(CompileFailure("%res = OpGetKernelLocalSizeForSubgroupCount",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected operand, found end of stream."));
  EXPECT_THAT(
      CompileFailure("%1 = OpGetKernelLocalSizeForSubgroupCount %2 %3 %4 %5 %6",
                     SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected operand, found end of stream."));
  EXPECT_THAT(
      CompiledInstructions("%res = OpGetKernelLocalSizeForSubgroupCount %type "
                           "%sgcount %invoke %param %param_size %param_align",
                           SPV_ENV_UNIVERSAL_1_1),
      Eq(MakeInstruction(SpvOpGetKernelLocalSizeForSubgroupCount,
                         {1, 2, 3, 4, 5, 6, 7})));
  EXPECT_THAT(
      CompileFailure("%res = OpGetKernelLocalSizeForSubgroupCount %type "
                     "%sgcount %invoke %param %param_size %param_align %extra",
                     SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected '=', found end of stream."));
}

TEST_F(OpGetKernelLocalSizeForSubgroupCountTest, ArgumentTypes) {
  EXPECT_THAT(CompileFailure(
                  "%1 = OpGetKernelLocalSizeForSubgroupCount 2 %3 %4 %5 %6 %7",
                  SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected id to start with %."));
  EXPECT_THAT(
      CompileFailure(
          "%1 = OpGetKernelLocalSizeForSubgroupCount %2 %3 %4 %5 %6 \"abc\"",
          SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected id to start with %."));
}

using OpGetKernelMaxNumSubgroupsTest = spvtest::TextToBinaryTest;

TEST_F(OpGetKernelMaxNumSubgroupsTest, OpcodeAssemblesInV10) {
  EXPECT_THAT(
      CompiledInstructions("%res = OpGetKernelMaxNumSubgroups %type "
                           "%invoke %param %param_size %param_align",
                           SPV_ENV_UNIVERSAL_1_0),
      Eq(MakeInstruction(SpvOpGetKernelMaxNumSubgroups, {1, 2, 3, 4, 5, 6})));
}

TEST_F(OpGetKernelMaxNumSubgroupsTest, ArgumentCount) {
  EXPECT_THAT(
      CompileFailure("OpGetKernelMaxNumSubgroups", SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected <result-id> at the beginning of an instruction, found "
         "'OpGetKernelMaxNumSubgroups'."));
  EXPECT_THAT(CompileFailure("%res = OpGetKernelMaxNumSubgroups",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected operand, found end of stream."));
  EXPECT_THAT(CompileFailure("%1 = OpGetKernelMaxNumSubgroups %2 %3 %4 %5",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected operand, found end of stream."));
  EXPECT_THAT(
      CompiledInstructions("%res = OpGetKernelMaxNumSubgroups %type "
                           "%invoke %param %param_size %param_align",
                           SPV_ENV_UNIVERSAL_1_1),
      Eq(MakeInstruction(SpvOpGetKernelMaxNumSubgroups, {1, 2, 3, 4, 5, 6})));
  EXPECT_THAT(CompileFailure("%res = OpGetKernelMaxNumSubgroups %type %invoke "
                             "%param %param_size %param_align %extra",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected '=', found end of stream."));
}

TEST_F(OpGetKernelMaxNumSubgroupsTest, ArgumentTypes) {
  EXPECT_THAT(CompileFailure("%1 = OpGetKernelMaxNumSubgroups 2 %3 %4 %5 %6",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected id to start with %."));
  EXPECT_THAT(
      CompileFailure("%1 = OpGetKernelMaxNumSubgroups %2 %3 %4 %5 \"abc\"",
                     SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected id to start with %."));
}

}  // namespace
}  // namespace spvtools
