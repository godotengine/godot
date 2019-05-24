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

#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::Concatenate;
using spvtest::MakeInstruction;
using spvtest::MakeVector;
using spvtest::TextToBinaryTest;
using testing::Eq;

TEST_F(TextToBinaryTest, Whitespace) {
  std::string input = R"(
; I'm a proud comment at the beginning of the file
; I hide:   OpCapability Shader
            OpMemoryModel Logical Simple ; comment after instruction
;;;;;;;; many ;'s
 %glsl450 = OpExtInstImport "GLSL.std.450"
            ; comment indented
)";

  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(Concatenate({MakeInstruction(SpvOpMemoryModel,
                                      {uint32_t(SpvAddressingModelLogical),
                                       uint32_t(SpvMemoryModelSimple)}),
                      MakeInstruction(SpvOpExtInstImport, {1},
                                      MakeVector("GLSL.std.450"))})));
}

}  // namespace
}  // namespace spvtools
