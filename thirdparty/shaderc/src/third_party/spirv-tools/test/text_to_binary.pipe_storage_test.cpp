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

#include "gmock/gmock.h"
#include "test/test_fixture.h"

namespace spvtools {
namespace {

using ::spvtest::MakeInstruction;
using ::testing::Eq;

using OpTypePipeStorageTest = spvtest::TextToBinaryTest;

// It can assemble, but should not validate.  Validation checks for version
// and capability are in another test file.
TEST_F(OpTypePipeStorageTest, OpcodeAssemblesInV10) {
  EXPECT_THAT(
      CompiledInstructions("%res = OpTypePipeStorage", SPV_ENV_UNIVERSAL_1_0),
      Eq(MakeInstruction(SpvOpTypePipeStorage, {1})));
}

TEST_F(OpTypePipeStorageTest, ArgumentCount) {
  EXPECT_THAT(
      CompileFailure("OpTypePipeStorage", SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected <result-id> at the beginning of an instruction, found "
         "'OpTypePipeStorage'."));
  EXPECT_THAT(
      CompiledInstructions("%res = OpTypePipeStorage", SPV_ENV_UNIVERSAL_1_1),
      Eq(MakeInstruction(SpvOpTypePipeStorage, {1})));
  EXPECT_THAT(CompileFailure("%res = OpTypePipeStorage %1 %2 %3 %4 %5",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("'=' expected after result id."));
}

using OpConstantPipeStorageTest = spvtest::TextToBinaryTest;

TEST_F(OpConstantPipeStorageTest, OpcodeAssemblesInV10) {
  EXPECT_THAT(CompiledInstructions("%1 = OpConstantPipeStorage %2 3 4 5",
                                   SPV_ENV_UNIVERSAL_1_0),
              Eq(MakeInstruction(SpvOpConstantPipeStorage, {1, 2, 3, 4, 5})));
}

TEST_F(OpConstantPipeStorageTest, ArgumentCount) {
  EXPECT_THAT(
      CompileFailure("OpConstantPipeStorage", SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected <result-id> at the beginning of an instruction, found "
         "'OpConstantPipeStorage'."));
  EXPECT_THAT(
      CompileFailure("%1 = OpConstantPipeStorage", SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected operand, found end of stream."));
  EXPECT_THAT(CompileFailure("%1 = OpConstantPipeStorage %2 3 4",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected operand, found end of stream."));
  EXPECT_THAT(CompiledInstructions("%1 = OpConstantPipeStorage %2 3 4 5",
                                   SPV_ENV_UNIVERSAL_1_1),
              Eq(MakeInstruction(SpvOpConstantPipeStorage, {1, 2, 3, 4, 5})));
  EXPECT_THAT(CompileFailure("%1 = OpConstantPipeStorage %2 3 4 5 %6 %7",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("'=' expected after result id."));
}

TEST_F(OpConstantPipeStorageTest, ArgumentTypes) {
  EXPECT_THAT(CompileFailure("%1 = OpConstantPipeStorage %2 %3 4 5",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Invalid unsigned integer literal: %3"));
  EXPECT_THAT(CompileFailure("%1 = OpConstantPipeStorage %2 3 %4 5",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Invalid unsigned integer literal: %4"));
  EXPECT_THAT(CompileFailure("%1 = OpConstantPipeStorage 2 3 4 5",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected id to start with %."));
  EXPECT_THAT(CompileFailure("%1 = OpConstantPipeStorage %2 3 4 \"ab\"",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Invalid unsigned integer literal: \"ab\""));
}

using OpCreatePipeFromPipeStorageTest = spvtest::TextToBinaryTest;

TEST_F(OpCreatePipeFromPipeStorageTest, OpcodeAssemblesInV10) {
  EXPECT_THAT(CompiledInstructions("%1 = OpCreatePipeFromPipeStorage %2 %3",
                                   SPV_ENV_UNIVERSAL_1_0),
              Eq(MakeInstruction(SpvOpCreatePipeFromPipeStorage, {1, 2, 3})));
}

TEST_F(OpCreatePipeFromPipeStorageTest, ArgumentCount) {
  EXPECT_THAT(
      CompileFailure("OpCreatePipeFromPipeStorage", SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected <result-id> at the beginning of an instruction, found "
         "'OpCreatePipeFromPipeStorage'."));
  EXPECT_THAT(
      CompileFailure("%1 = OpCreatePipeFromPipeStorage", SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected operand, found end of stream."));
  EXPECT_THAT(CompileFailure("%1 = OpCreatePipeFromPipeStorage %2 OpNop",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected operand, found next instruction instead."));
  EXPECT_THAT(CompiledInstructions("%1 = OpCreatePipeFromPipeStorage %2 %3",
                                   SPV_ENV_UNIVERSAL_1_1),
              Eq(MakeInstruction(SpvOpCreatePipeFromPipeStorage, {1, 2, 3})));
  EXPECT_THAT(CompileFailure("%1 = OpCreatePipeFromPipeStorage %2 %3 %4 %5",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("'=' expected after result id."));
}

TEST_F(OpCreatePipeFromPipeStorageTest, ArgumentTypes) {
  EXPECT_THAT(CompileFailure("%1 = OpCreatePipeFromPipeStorage \"\" %3",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected id to start with %."));
  EXPECT_THAT(CompileFailure("%1 = OpCreatePipeFromPipeStorage %2 3",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected id to start with %."));
}

}  // namespace
}  // namespace spvtools
