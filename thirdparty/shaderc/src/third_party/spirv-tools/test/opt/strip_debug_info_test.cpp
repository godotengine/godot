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

#include <vector>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using StripLineDebugInfoTest = PassTest<::testing::Test>;

TEST_F(StripLineDebugInfoTest, LineNoLine) {
  std::vector<const char*> text = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Vertex %2 \"main\"",
          "%3 = OpString \"minimal.vert\"",
               "OpModuleProcessed \"42\"",
               "OpModuleProcessed \"43\"",
               "OpModuleProcessed \"44\"",
               "OpNoLine",
               "OpLine %3 10 10",
       "%void = OpTypeVoid",
               "OpLine %3 100 100",
          "%5 = OpTypeFunction %void",
          "%2 = OpFunction %void None %5",
               "OpLine %3 1 1",
               "OpNoLine",
               "OpLine %3 2 2",
               "OpLine %3 3 3",
          "%6 = OpLabel",
               "OpLine %3 4 4",
               "OpNoLine",
               "OpReturn",
               "OpLine %3 4 4",
               "OpNoLine",
               "OpFunctionEnd",
      // clang-format on
  };
  SinglePassRunAndCheck<StripDebugInfoPass>(JoinAllInsts(text),
                                            JoinNonDebugInsts(text),
                                            /* skip_nop = */ false);

  // Let's add more debug instruction before the "OpString" instruction.
  const std::vector<const char*> more_text = {
      "OpSourceContinued \"I'm a happy shader! Yay! ;)\"",
      "OpSourceContinued \"wahahaha\"",
      "OpSource ESSL 310",
      "OpSource ESSL 310",
      "OpSourceContinued \"wahahaha\"",
      "OpSourceContinued \"wahahaha\"",
      "OpSourceExtension \"save-the-world-extension\"",
      "OpName %2 \"main\"",
  };
  text.insert(text.begin() + 4, more_text.cbegin(), more_text.cend());
  SinglePassRunAndCheck<StripDebugInfoPass>(JoinAllInsts(text),
                                            JoinNonDebugInsts(text),
                                            /* skip_nop = */ false);
}

using StripDebugInfoTest = PassTest<::testing::TestWithParam<const char*>>;

TEST_P(StripDebugInfoTest, Kind) {
  std::vector<const char*> text = {
      "OpCapability Shader",
      "OpMemoryModel Logical GLSL450",
      GetParam(),
  };
  SinglePassRunAndCheck<StripDebugInfoPass>(JoinAllInsts(text),
                                            JoinNonDebugInsts(text),
                                            /* skip_nop = */ false);
}

// Test each possible non-line debug instruction.
// clang-format off
INSTANTIATE_TEST_SUITE_P(
    SingleKindDebugInst, StripDebugInfoTest,
    ::testing::ValuesIn(std::vector<const char*>({
        "OpSourceContinued \"I'm a happy shader! Yay! ;)\"",
        "OpSource ESSL 310",
        "OpSourceExtension \"save-the-world-extension\"",
        "OpName %main \"main\"",
        "OpMemberName %struct 0 \"field\"",
        "%1 = OpString \"name.vert\"",
        "OpModuleProcessed \"42\"",
    })));
// clang-format on

}  // namespace
}  // namespace opt
}  // namespace spvtools
