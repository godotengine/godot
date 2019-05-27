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
#include <vector>

#include "gmock/gmock.h"
#include "test/link/linker_fixture.h"

namespace spvtools {
namespace {

using UniqueIds = spvtest::LinkerTest;

TEST_F(UniqueIds, UniquelyMerged) {
  std::vector<std::string> bodies(2);
  bodies[0] =
      // clang-format off
               "OpCapability Shader\n"
          "%1 = OpExtInstImport \"GLSL.std.450\"\n"
               "OpMemoryModel Logical GLSL450\n"
               "OpEntryPoint Vertex %main \"main\"\n"
               "OpSource ESSL 310\n"
               "OpName %main \"main\"\n"
               "OpName %f_ \"f(\"\n"
               "OpName %gv1 \"gv1\"\n"
               "OpName %gv2 \"gv2\"\n"
               "OpName %lv1 \"lv1\"\n"
               "OpName %lv2 \"lv2\"\n"
               "OpName %lv1_0 \"lv1\"\n"
       "%void = OpTypeVoid\n"
         "%10 = OpTypeFunction %void\n"
      "%float = OpTypeFloat 32\n"
         "%12 = OpTypeFunction %float\n"
 "%_ptr_Private_float = OpTypePointer Private %float\n"
        "%gv1 = OpVariable %_ptr_Private_float Private\n"
   "%float_10 = OpConstant %float 10\n"
        "%gv2 = OpVariable %_ptr_Private_float Private\n"
  "%float_100 = OpConstant %float 100\n"
 "%_ptr_Function_float = OpTypePointer Function %float\n"
       "%main = OpFunction %void None %10\n"
         "%17 = OpLabel\n"
      "%lv1_0 = OpVariable %_ptr_Function_float Function\n"
               "OpStore %gv1 %float_10\n"
               "OpStore %gv2 %float_100\n"
         "%18 = OpLoad %float %gv1\n"
         "%19 = OpLoad %float %gv2\n"
         "%20 = OpFSub %float %18 %19\n"
               "OpStore %lv1_0 %20\n"
               "OpReturn\n"
               "OpFunctionEnd\n"
         "%f_ = OpFunction %float None %12\n"
         "%21 = OpLabel\n"
        "%lv1 = OpVariable %_ptr_Function_float Function\n"
        "%lv2 = OpVariable %_ptr_Function_float Function\n"
         "%22 = OpLoad %float %gv1\n"
         "%23 = OpLoad %float %gv2\n"
         "%24 = OpFAdd %float %22 %23\n"
               "OpStore %lv1 %24\n"
         "%25 = OpLoad %float %gv1\n"
         "%26 = OpLoad %float %gv2\n"
         "%27 = OpFMul %float %25 %26\n"
               "OpStore %lv2 %27\n"
         "%28 = OpLoad %float %lv1\n"
         "%29 = OpLoad %float %lv2\n"
         "%30 = OpFDiv %float %28 %29\n"
               "OpReturnValue %30\n"
               "OpFunctionEnd\n";
  // clang-format on
  bodies[1] =
      // clang-format off
               "OpCapability Shader\n"
          "%1 = OpExtInstImport \"GLSL.std.450\"\n"
               "OpMemoryModel Logical GLSL450\n"
               "OpSource ESSL 310\n"
               "OpName %main \"main2\"\n"
               "OpName %f_ \"f(\"\n"
               "OpName %gv1 \"gv12\"\n"
               "OpName %gv2 \"gv22\"\n"
               "OpName %lv1 \"lv12\"\n"
               "OpName %lv2 \"lv22\"\n"
               "OpName %lv1_0 \"lv12\"\n"
       "%void = OpTypeVoid\n"
         "%10 = OpTypeFunction %void\n"
      "%float = OpTypeFloat 32\n"
         "%12 = OpTypeFunction %float\n"
 "%_ptr_Private_float = OpTypePointer Private %float\n"
        "%gv1 = OpVariable %_ptr_Private_float Private\n"
   "%float_10 = OpConstant %float 10\n"
        "%gv2 = OpVariable %_ptr_Private_float Private\n"
  "%float_100 = OpConstant %float 100\n"
 "%_ptr_Function_float = OpTypePointer Function %float\n"
       "%main = OpFunction %void None %10\n"
         "%17 = OpLabel\n"
      "%lv1_0 = OpVariable %_ptr_Function_float Function\n"
               "OpStore %gv1 %float_10\n"
               "OpStore %gv2 %float_100\n"
         "%18 = OpLoad %float %gv1\n"
         "%19 = OpLoad %float %gv2\n"
         "%20 = OpFSub %float %18 %19\n"
               "OpStore %lv1_0 %20\n"
               "OpReturn\n"
               "OpFunctionEnd\n"
         "%f_ = OpFunction %float None %12\n"
         "%21 = OpLabel\n"
        "%lv1 = OpVariable %_ptr_Function_float Function\n"
        "%lv2 = OpVariable %_ptr_Function_float Function\n"
         "%22 = OpLoad %float %gv1\n"
         "%23 = OpLoad %float %gv2\n"
         "%24 = OpFAdd %float %22 %23\n"
               "OpStore %lv1 %24\n"
         "%25 = OpLoad %float %gv1\n"
         "%26 = OpLoad %float %gv2\n"
         "%27 = OpFMul %float %25 %26\n"
               "OpStore %lv2 %27\n"
         "%28 = OpLoad %float %lv1\n"
         "%29 = OpLoad %float %lv2\n"
         "%30 = OpFDiv %float %28 %29\n"
               "OpReturnValue %30\n"
               "OpFunctionEnd\n";
  // clang-format on

  spvtest::Binary linked_binary;
  LinkerOptions options;
  options.SetVerifyIds(true);
  spv_result_t res = AssembleAndLink(bodies, &linked_binary, options);
  EXPECT_EQ(SPV_SUCCESS, res);
}

}  // namespace
}  // namespace spvtools
