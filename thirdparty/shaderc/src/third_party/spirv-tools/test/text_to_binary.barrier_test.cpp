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

// Assembler tests for instructions in the "Barrier Instructions" section
// of the SPIR-V spec.

#include <string>

#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::MakeInstruction;
using spvtest::TextToBinaryTest;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;

// Test OpMemoryBarrier

using OpMemoryBarrier = spvtest::TextToBinaryTest;

TEST_F(OpMemoryBarrier, Good) {
  const std::string input = "OpMemoryBarrier %1 %2\n";
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(SpvOpMemoryBarrier, {1, 2})));
  EXPECT_THAT(EncodeAndDecodeSuccessfully(input), Eq(input));
}

TEST_F(OpMemoryBarrier, BadMissingScopeId) {
  const std::string input = "OpMemoryBarrier\n";
  EXPECT_THAT(CompileFailure(input),
              Eq("Expected operand, found end of stream."));
}

TEST_F(OpMemoryBarrier, BadInvalidScopeId) {
  const std::string input = "OpMemoryBarrier 99\n";
  EXPECT_THAT(CompileFailure(input), Eq("Expected id to start with %."));
}

TEST_F(OpMemoryBarrier, BadMissingMemorySemanticsId) {
  const std::string input = "OpMemoryBarrier %scope\n";
  EXPECT_THAT(CompileFailure(input),
              Eq("Expected operand, found end of stream."));
}

TEST_F(OpMemoryBarrier, BadInvalidMemorySemanticsId) {
  const std::string input = "OpMemoryBarrier %scope 14\n";
  EXPECT_THAT(CompileFailure(input), Eq("Expected id to start with %."));
}

// TODO(dneto): OpControlBarrier
// TODO(dneto): OpGroupAsyncCopy
// TODO(dneto): OpGroupWaitEvents
// TODO(dneto): OpGroupAll
// TODO(dneto): OpGroupAny
// TODO(dneto): OpGroupBroadcast
// TODO(dneto): OpGroupIAdd
// TODO(dneto): OpGroupFAdd
// TODO(dneto): OpGroupFMin
// TODO(dneto): OpGroupUMin
// TODO(dneto): OpGroupSMin
// TODO(dneto): OpGroupFMax
// TODO(dneto): OpGroupUMax
// TODO(dneto): OpGroupSMax

using NamedMemoryBarrierTest = spvtest::TextToBinaryTest;

// OpMemoryNamedBarrier is not in 1.0, but it is enabled by a capability.
// We should be able to assemble it.  Validation checks are in another test
// file.
TEST_F(NamedMemoryBarrierTest, OpcodeAssemblesInV10) {
  EXPECT_THAT(
      CompiledInstructions("OpMemoryNamedBarrier %bar %scope %semantics",
                           SPV_ENV_UNIVERSAL_1_0),
      ElementsAre(spvOpcodeMake(4, SpvOpMemoryNamedBarrier), _, _, _));
}

TEST_F(NamedMemoryBarrierTest, ArgumentCount) {
  EXPECT_THAT(CompileFailure("OpMemoryNamedBarrier", SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected operand, found end of stream."));
  EXPECT_THAT(
      CompileFailure("OpMemoryNamedBarrier %bar", SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected operand, found end of stream."));
  EXPECT_THAT(
      CompileFailure("OpMemoryNamedBarrier %bar %scope", SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected operand, found end of stream."));
  EXPECT_THAT(
      CompiledInstructions("OpMemoryNamedBarrier %bar %scope %semantics",
                           SPV_ENV_UNIVERSAL_1_1),
      ElementsAre(spvOpcodeMake(4, SpvOpMemoryNamedBarrier), _, _, _));
  EXPECT_THAT(
      CompileFailure("OpMemoryNamedBarrier %bar %scope %semantics %extra",
                     SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected '=', found end of stream."));
}

TEST_F(NamedMemoryBarrierTest, ArgumentTypes) {
  EXPECT_THAT(CompileFailure("OpMemoryNamedBarrier 123 %scope %semantics",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected id to start with %."));
  EXPECT_THAT(CompileFailure("OpMemoryNamedBarrier %bar %scope \"semantics\"",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected id to start with %."));
}

using TypeNamedBarrierTest = spvtest::TextToBinaryTest;

TEST_F(TypeNamedBarrierTest, OpcodeAssemblesInV10) {
  EXPECT_THAT(
      CompiledInstructions("%t = OpTypeNamedBarrier", SPV_ENV_UNIVERSAL_1_0),
      ElementsAre(spvOpcodeMake(2, SpvOpTypeNamedBarrier), _));
}

TEST_F(TypeNamedBarrierTest, ArgumentCount) {
  EXPECT_THAT(CompileFailure("OpTypeNamedBarrier", SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected <result-id> at the beginning of an instruction, "
                 "found 'OpTypeNamedBarrier'."));
  EXPECT_THAT(
      CompiledInstructions("%t = OpTypeNamedBarrier", SPV_ENV_UNIVERSAL_1_1),
      ElementsAre(spvOpcodeMake(2, SpvOpTypeNamedBarrier), _));
  EXPECT_THAT(
      CompileFailure("%t = OpTypeNamedBarrier 1 2 3", SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected <opcode> or <result-id> at the beginning of an instruction, "
         "found '1'."));
}

using NamedBarrierInitializeTest = spvtest::TextToBinaryTest;

TEST_F(NamedBarrierInitializeTest, OpcodeAssemblesInV10) {
  EXPECT_THAT(
      CompiledInstructions("%bar = OpNamedBarrierInitialize %type %count",
                           SPV_ENV_UNIVERSAL_1_0),
      ElementsAre(spvOpcodeMake(4, SpvOpNamedBarrierInitialize), _, _, _));
}

TEST_F(NamedBarrierInitializeTest, ArgumentCount) {
  EXPECT_THAT(
      CompileFailure("%bar = OpNamedBarrierInitialize", SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected operand, found end of stream."));
  EXPECT_THAT(CompileFailure("%bar = OpNamedBarrierInitialize %ype",
                             SPV_ENV_UNIVERSAL_1_1),
              Eq("Expected operand, found end of stream."));
  EXPECT_THAT(
      CompiledInstructions("%bar = OpNamedBarrierInitialize %type %count",
                           SPV_ENV_UNIVERSAL_1_1),
      ElementsAre(spvOpcodeMake(4, SpvOpNamedBarrierInitialize), _, _, _));
  EXPECT_THAT(
      CompileFailure("%bar = OpNamedBarrierInitialize %type %count \"extra\"",
                     SPV_ENV_UNIVERSAL_1_1),
      Eq("Expected <opcode> or <result-id> at the beginning of an instruction, "
         "found '\"extra\"'."));
}

}  // namespace
}  // namespace spvtools
