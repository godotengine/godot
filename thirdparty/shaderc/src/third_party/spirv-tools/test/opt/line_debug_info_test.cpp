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

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

// A pass turning all none debug line instructions into Nop.
class NopifyPass : public Pass {
 public:
  const char* name() const override { return "NopifyPass"; }
  Status Process() override {
    bool modified = false;
    context()->module()->ForEachInst(
        [&modified](Instruction* inst) {
          inst->ToNop();
          modified = true;
        },
        /* run_on_debug_line_insts = */ false);
    return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
  }
};

using PassTestForLineDebugInfo = PassTest<::testing::Test>;

// This test's purpose to show our implementation choice: line debug info is
// preserved even if the following instruction is killed. It serves as a guard
// of potential behavior changes.
TEST_F(PassTestForLineDebugInfo, KeepLineDebugInfo) {
  // clang-format off
  const char* text =
               "OpCapability Shader "
          "%1 = OpExtInstImport \"GLSL.std.450\" "
               "OpMemoryModel Logical GLSL450 "
               "OpEntryPoint Vertex %2 \"main\" "
          "%3 = OpString \"minimal.vert\" "
               "OpNoLine "
               "OpLine %3 10 10 "
       "%void = OpTypeVoid "
               "OpLine %3 100 100 "
          "%5 = OpTypeFunction %void "
          "%2 = OpFunction %void None %5 "
               "OpLine %3 1 1 "
               "OpNoLine "
               "OpLine %3 2 2 "
               "OpLine %3 3 3 "
          "%6 = OpLabel "
               "OpLine %3 4 4 "
               "OpNoLine "
               "OpReturn "
               "OpLine %3 4 4 "
               "OpNoLine "
               "OpFunctionEnd ";
  // clang-format on

  const char* result_keep_nop =
      "OpNop\n"
      "OpNop\n"
      "OpNop\n"
      "OpNop\n"
      "OpNop\n"
      "OpNoLine\n"
      "OpLine %3 10 10\n"
      "OpNop\n"
      "OpLine %3 100 100\n"
      "OpNop\n"
      "OpNop\n"
      "OpLine %3 1 1\n"
      "OpNoLine\n"
      "OpLine %3 2 2\n"
      "OpLine %3 3 3\n"
      "OpNop\n"
      "OpLine %3 4 4\n"
      "OpNoLine\n"
      "OpNop\n"
      "OpLine %3 4 4\n"
      "OpNoLine\n"
      "OpNop\n";
  SinglePassRunAndCheck<NopifyPass>(text, result_keep_nop,
                                    /* skip_nop = */ false);
  const char* result_skip_nop =
      "OpNoLine\n"
      "OpLine %3 10 10\n"
      "OpLine %3 100 100\n"
      "OpLine %3 1 1\n"
      "OpNoLine\n"
      "OpLine %3 2 2\n"
      "OpLine %3 3 3\n"
      "OpLine %3 4 4\n"
      "OpNoLine\n"
      "OpLine %3 4 4\n"
      "OpNoLine\n";
  SinglePassRunAndCheck<NopifyPass>(text, result_skip_nop,
                                    /* skip_nop = */ true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
