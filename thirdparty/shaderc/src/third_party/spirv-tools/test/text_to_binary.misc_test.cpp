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

// Assembler tests for instructions in the "Miscellaneous" section of the
// SPIR-V spec.

#include "test/unit_spirv.h"

#include "gmock/gmock.h"
#include "test/test_fixture.h"

namespace spvtools {
namespace {

using SpirvVector = spvtest::TextToBinaryTest::SpirvVector;
using spvtest::MakeInstruction;
using ::testing::Eq;
using TextToBinaryMisc = spvtest::TextToBinaryTest;

TEST_F(TextToBinaryMisc, OpNop) {
  EXPECT_THAT(CompiledInstructions("OpNop"), Eq(MakeInstruction(SpvOpNop, {})));
}

TEST_F(TextToBinaryMisc, OpUndef) {
  const SpirvVector code = CompiledInstructions(R"(%f32 = OpTypeFloat 32
                                                   %u = OpUndef %f32)");
  const uint32_t typeID = 1;
  EXPECT_THAT(code[1], Eq(typeID));
  EXPECT_THAT(Subvector(code, 3), Eq(MakeInstruction(SpvOpUndef, {typeID, 2})));
}

TEST_F(TextToBinaryMisc, OpWrong) {
  EXPECT_THAT(CompileFailure(" OpWrong %1 %2"),
              Eq("Invalid Opcode name 'OpWrong'"));
}

TEST_F(TextToBinaryMisc, OpWrongAfterRight) {
  const auto assembly = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpXYZ
)";
  EXPECT_THAT(CompileFailure(assembly), Eq("Invalid Opcode name 'OpXYZ'"));
}

}  // namespace
}  // namespace spvtools
