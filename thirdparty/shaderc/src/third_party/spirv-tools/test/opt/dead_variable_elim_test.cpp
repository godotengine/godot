// Copyright (c) 2017 Google Inc.
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

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using DeadVariableElimTest = PassTest<::testing::Test>;

// %dead is unused.  Make sure we remove it along with its name.
TEST_F(DeadVariableElimTest, RemoveUnreferenced) {
  const std::string before =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
OpName %dead "dead"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Private_float = OpTypePointer Private %float
%dead = OpVariable %_ptr_Private_float Private
%main = OpFunction %void None %5
%8 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
%void = OpTypeVoid
%5 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Private_float = OpTypePointer Private %float
%main = OpFunction %void None %5
%8 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<DeadVariableElimination>(before, after, true, true);
}

// Since %dead is exported, make sure we keep it.  It could be referenced
// somewhere else.
TEST_F(DeadVariableElimTest, KeepExported) {
  const std::string before =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
OpName %dead "dead"
OpDecorate %dead LinkageAttributes "dead" Export
%void = OpTypeVoid
%5 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Private_float = OpTypePointer Private %float
%dead = OpVariable %_ptr_Private_float Private
%main = OpFunction %void None %5
%8 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<DeadVariableElimination>(before, before, true, true);
}

// Delete %dead because it is unreferenced.  Then %initializer becomes
// unreferenced, so remove it as well.
TEST_F(DeadVariableElimTest, RemoveUnreferencedWithInit1) {
  const std::string before =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
OpName %dead "dead"
OpName %initializer "initializer"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Private_float = OpTypePointer Private %float
%initializer = OpVariable %_ptr_Private_float Private
%dead = OpVariable %_ptr_Private_float Private %initializer
%main = OpFunction %void None %6
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Private_float = OpTypePointer Private %float
%main = OpFunction %void None %6
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<DeadVariableElimination>(before, after, true, true);
}

// Delete %dead because it is unreferenced.  In this case, the initialized has
// another reference, and should not be removed.
TEST_F(DeadVariableElimTest, RemoveUnreferencedWithInit2) {
  const std::string before =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
OpName %dead "dead"
OpName %initializer "initializer"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Private_float = OpTypePointer Private %float
%initializer = OpVariable %_ptr_Private_float Private
%dead = OpVariable %_ptr_Private_float Private %initializer
%main = OpFunction %void None %6
%9 = OpLabel
%10 = OpLoad %float %initializer
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
OpName %initializer "initializer"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Private_float = OpTypePointer Private %float
%initializer = OpVariable %_ptr_Private_float Private
%main = OpFunction %void None %6
%9 = OpLabel
%10 = OpLoad %float %initializer
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<DeadVariableElimination>(before, after, true, true);
}

// Keep %live because it is used, and its initializer.
TEST_F(DeadVariableElimTest, KeepReferenced) {
  const std::string before =
      R"(OpCapability Shader
OpCapability Linkage
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 150
OpName %main "main"
OpName %live "live"
OpName %initializer "initializer"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Private_float = OpTypePointer Private %float
%initializer = OpVariable %_ptr_Private_float Private
%live = OpVariable %_ptr_Private_float Private %initializer
%main = OpFunction %void None %6
%9 = OpLabel
%10 = OpLoad %float %live
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<DeadVariableElimination>(before, before, true, true);
}

// This test that the decoration associated with a variable are removed when the
// variable is removed.
TEST_F(DeadVariableElimTest, RemoveVariableAndDecorations) {
  const std::string before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpSource GLSL 450
OpName %main "main"
OpName %B "B"
OpMemberName %B 0 "a"
OpName %Bdat "Bdat"
OpMemberDecorate %B 0 Offset 0
OpDecorate %B BufferBlock
OpDecorate %Bdat DescriptorSet 0
OpDecorate %Bdat Binding 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%B = OpTypeStruct %uint
%_ptr_Uniform_B = OpTypePointer Uniform %B
%Bdat = OpVariable %_ptr_Uniform_B Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%uint_1 = OpConstant %uint 1
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%main = OpFunction %void None %6
%13 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpSource GLSL 450
OpName %main "main"
OpName %B "B"
OpMemberName %B 0 "a"
OpMemberDecorate %B 0 Offset 0
OpDecorate %B BufferBlock
%void = OpTypeVoid
%6 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%B = OpTypeStruct %uint
%_ptr_Uniform_B = OpTypePointer Uniform %B
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%uint_1 = OpConstant %uint 1
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%main = OpFunction %void None %6
%13 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<DeadVariableElimination>(before, after, true, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
