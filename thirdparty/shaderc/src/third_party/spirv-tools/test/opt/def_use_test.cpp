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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/build_module.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "spirv-tools/libspirv.hpp"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace analysis {
namespace {

using ::testing::Contains;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

// Returns the number of uses of |id|.
uint32_t NumUses(const std::unique_ptr<IRContext>& context, uint32_t id) {
  uint32_t count = 0;
  context->get_def_use_mgr()->ForEachUse(
      id, [&count](Instruction*, uint32_t) { ++count; });
  return count;
}

// Returns the opcode of each use of |id|.
//
// If |id| is used multiple times in a single instruction, that instruction's
// opcode will appear a corresponding number of times.
std::vector<SpvOp> GetUseOpcodes(const std::unique_ptr<IRContext>& context,
                                 uint32_t id) {
  std::vector<SpvOp> opcodes;
  context->get_def_use_mgr()->ForEachUse(
      id, [&opcodes](Instruction* user, uint32_t) {
        opcodes.push_back(user->opcode());
      });
  return opcodes;
}

// Disassembles the given |inst| and returns the disassembly.
std::string DisassembleInst(Instruction* inst) {
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_1);

  std::vector<uint32_t> binary;
  // We need this to generate the necessary header in the binary.
  tools.Assemble("", &binary);
  inst->ToBinaryWithoutAttachedDebugInsts(&binary);

  std::string text;
  // We'll need to check the underlying id numbers.
  // So turn off friendly names for ids.
  tools.Disassemble(binary, &text, SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  while (!text.empty() && text.back() == '\n') text.pop_back();
  return text;
}

// A struct for holding expected id defs and uses.
struct InstDefUse {
  using IdInstPair = std::pair<uint32_t, std::string>;
  using IdInstsPair = std::pair<uint32_t, std::vector<std::string>>;

  // Ids and their corresponding def instructions.
  std::vector<IdInstPair> defs;
  // Ids and their corresponding use instructions.
  std::vector<IdInstsPair> uses;
};

// Checks that the |actual_defs| and |actual_uses| are in accord with
// |expected_defs_uses|.
void CheckDef(const InstDefUse& expected_defs_uses,
              const DefUseManager::IdToDefMap& actual_defs) {
  // Check defs.
  ASSERT_EQ(expected_defs_uses.defs.size(), actual_defs.size());
  for (uint32_t i = 0; i < expected_defs_uses.defs.size(); ++i) {
    const auto id = expected_defs_uses.defs[i].first;
    const auto expected_def = expected_defs_uses.defs[i].second;
    ASSERT_EQ(1u, actual_defs.count(id)) << "expected to def id [" << id << "]";
    auto def = actual_defs.at(id);
    if (def->opcode() != SpvOpConstant) {
      // Constants don't disassemble properly without a full context.
      EXPECT_EQ(expected_def, DisassembleInst(actual_defs.at(id)));
    }
  }
}

using UserMap = std::unordered_map<uint32_t, std::vector<Instruction*>>;

// Creates a mapping of all definitions to their users (except OpConstant).
//
// OpConstants are skipped because they cannot be disassembled in isolation.
UserMap BuildAllUsers(const DefUseManager* mgr, uint32_t idBound) {
  UserMap userMap;
  for (uint32_t id = 0; id != idBound; ++id) {
    if (mgr->GetDef(id)) {
      mgr->ForEachUser(id, [id, &userMap](Instruction* user) {
        if (user->opcode() != SpvOpConstant) {
          userMap[id].push_back(user);
        }
      });
    }
  }
  return userMap;
}

// Constants don't disassemble properly without a full context, so skip them as
// checks.
void CheckUse(const InstDefUse& expected_defs_uses, const DefUseManager* mgr,
              uint32_t idBound) {
  UserMap actual_uses = BuildAllUsers(mgr, idBound);
  // Check uses.
  ASSERT_EQ(expected_defs_uses.uses.size(), actual_uses.size());
  for (uint32_t i = 0; i < expected_defs_uses.uses.size(); ++i) {
    const auto id = expected_defs_uses.uses[i].first;
    const auto& expected_uses = expected_defs_uses.uses[i].second;

    ASSERT_EQ(1u, actual_uses.count(id)) << "expected to use id [" << id << "]";
    const auto& uses = actual_uses.at(id);

    ASSERT_EQ(expected_uses.size(), uses.size())
        << "id [" << id << "] # uses: expected: " << expected_uses.size()
        << " actual: " << uses.size();

    std::vector<std::string> actual_uses_disassembled;
    for (const auto actual_use : uses) {
      actual_uses_disassembled.emplace_back(DisassembleInst(actual_use));
    }
    EXPECT_THAT(actual_uses_disassembled,
                UnorderedElementsAreArray(expected_uses));
  }
}

// The following test case mimics how LLVM handles induction variables.
// But, yeah, it's not very readable. However, we only care about the id
// defs and uses. So, no need to make sure this is valid OpPhi construct.
const char kOpPhiTestFunction[] =
    " %1 = OpTypeVoid "
    " %6 = OpTypeInt 32 0 "
    "%10 = OpTypeFloat 32 "
    "%16 = OpTypeBool "
    " %3 = OpTypeFunction %1 "
    " %8 = OpConstant %6 0 "
    "%18 = OpConstant %6 1 "
    "%12 = OpConstant %10 1.0 "
    " %2 = OpFunction %1 None %3 "
    " %4 = OpLabel "
    "      OpBranch %5 "

    " %5 = OpLabel "
    " %7 = OpPhi %6 %8 %4 %9 %5 "
    "%11 = OpPhi %10 %12 %4 %13 %5 "
    " %9 = OpIAdd %6 %7 %8 "
    "%13 = OpFAdd %10 %11 %12 "
    "%17 = OpSLessThan %16 %7 %18 "
    "      OpLoopMerge %19 %5 None "
    "      OpBranchConditional %17 %5 %19 "

    "%19 = OpLabel "
    "      OpReturn "
    "      OpFunctionEnd";

struct ParseDefUseCase {
  const char* text;
  InstDefUse du;
};

using ParseDefUseTest = ::testing::TestWithParam<ParseDefUseCase>;

TEST_P(ParseDefUseTest, Case) {
  const auto& tc = GetParam();

  // Build module.
  const std::vector<const char*> text = {tc.text};
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, JoinAllInsts(text),
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Analyze def and use.
  DefUseManager manager(context->module());

  CheckDef(tc.du, manager.id_to_defs());
  CheckUse(tc.du, &manager, context->module()->IdBound());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    TestCase, ParseDefUseTest,
    ::testing::ValuesIn(std::vector<ParseDefUseCase>{
        {"", {{}, {}}},                              // no instruction
        {"OpMemoryModel Logical GLSL450", {{}, {}}}, // no def and use
        { // single def, no use
          "%1 = OpString \"wow\"",
          {
            {{1, "%1 = OpString \"wow\""}}, // defs
            {}                              // uses
          }
        },
        { // multiple def, no use
          "%1 = OpString \"hello\" "
          "%2 = OpString \"world\" "
          "%3 = OpTypeVoid",
          {
            {  // defs
              {1, "%1 = OpString \"hello\""},
              {2, "%2 = OpString \"world\""},
              {3, "%3 = OpTypeVoid"},
            },
            {} // uses
          }
        },
        { // multiple def, multiple use
          "%1 = OpTypeBool "
          "%2 = OpTypeVector %1 3 "
          "%3 = OpTypeMatrix %2 3",
          {
            { // defs
              {1, "%1 = OpTypeBool"},
              {2, "%2 = OpTypeVector %1 3"},
              {3, "%3 = OpTypeMatrix %2 3"},
            },
            { // uses
              {1, {"%2 = OpTypeVector %1 3"}},
              {2, {"%3 = OpTypeMatrix %2 3"}},
            }
          }
        },
        { // multiple use of the same id
          "%1 = OpTypeBool "
          "%2 = OpTypeVector %1 2 "
          "%3 = OpTypeVector %1 3 "
          "%4 = OpTypeVector %1 4",
          {
            { // defs
              {1, "%1 = OpTypeBool"},
              {2, "%2 = OpTypeVector %1 2"},
              {3, "%3 = OpTypeVector %1 3"},
              {4, "%4 = OpTypeVector %1 4"},
            },
            { // uses
              {1,
                {
                  "%2 = OpTypeVector %1 2",
                  "%3 = OpTypeVector %1 3",
                  "%4 = OpTypeVector %1 4",
                }
              },
            }
          }
        },
        { // labels
          "%1 = OpTypeVoid "
          "%2 = OpTypeBool "
          "%3 = OpTypeFunction %1 "
          "%4 = OpConstantTrue %2 "
          "%5 = OpFunction %1 None %3 "

          "%6 = OpLabel "
          "OpBranchConditional %4 %7 %8 "

          "%7 = OpLabel "
          "OpBranch %7 "

          "%8 = OpLabel "
          "OpReturn "

          "OpFunctionEnd",
          {
            { // defs
              {1, "%1 = OpTypeVoid"},
              {2, "%2 = OpTypeBool"},
              {3, "%3 = OpTypeFunction %1"},
              {4, "%4 = OpConstantTrue %2"},
              {5, "%5 = OpFunction %1 None %3"},
              {6, "%6 = OpLabel"},
              {7, "%7 = OpLabel"},
              {8, "%8 = OpLabel"},
            },
            { // uses
              {1, {
                    "%3 = OpTypeFunction %1",
                    "%5 = OpFunction %1 None %3",
                  }
              },
              {2, {"%4 = OpConstantTrue %2"}},
              {3, {"%5 = OpFunction %1 None %3"}},
              {4, {"OpBranchConditional %4 %7 %8"}},
              {7,
                {
                  "OpBranchConditional %4 %7 %8",
                  "OpBranch %7",
                }
              },
              {8, {"OpBranchConditional %4 %7 %8"}},
            }
          }
        },
        { // cross function
          "%1 = OpTypeBool "
          "%3 = OpTypeFunction %1 "
          "%2 = OpFunction %1 None %3 "

          "%4 = OpLabel "
          "%5 = OpVariable %1 Function "
          "%6 = OpFunctionCall %1 %2 %5 "
          "OpReturnValue %6 "

          "OpFunctionEnd",
          {
            { // defs
              {1, "%1 = OpTypeBool"},
              {2, "%2 = OpFunction %1 None %3"},
              {3, "%3 = OpTypeFunction %1"},
              {4, "%4 = OpLabel"},
              {5, "%5 = OpVariable %1 Function"},
              {6, "%6 = OpFunctionCall %1 %2 %5"},
            },
            { // uses
              {1,
                {
                  "%2 = OpFunction %1 None %3",
                  "%3 = OpTypeFunction %1",
                  "%5 = OpVariable %1 Function",
                  "%6 = OpFunctionCall %1 %2 %5",
                }
              },
              {2, {"%6 = OpFunctionCall %1 %2 %5"}},
              {3, {"%2 = OpFunction %1 None %3"}},
              {5, {"%6 = OpFunctionCall %1 %2 %5"}},
              {6, {"OpReturnValue %6"}},
            }
          }
        },
        { // selection merge and loop merge
          "%1 = OpTypeVoid "
          "%3 = OpTypeFunction %1 "
          "%10 = OpTypeBool "
          "%8 = OpConstantTrue %10 "
          "%2 = OpFunction %1 None %3 "

          "%4 = OpLabel "
          "OpLoopMerge %5 %4 None "
          "OpBranch %6 "

          "%5 = OpLabel "
          "OpReturn "

          "%6 = OpLabel "
          "OpSelectionMerge %7 None "
          "OpBranchConditional %8 %9 %7 "

          "%7 = OpLabel "
          "OpReturn "

          "%9 = OpLabel "
          "OpReturn "

          "OpFunctionEnd",
          {
            { // defs
              {1, "%1 = OpTypeVoid"},
              {2, "%2 = OpFunction %1 None %3"},
              {3, "%3 = OpTypeFunction %1"},
              {4, "%4 = OpLabel"},
              {5, "%5 = OpLabel"},
              {6, "%6 = OpLabel"},
              {7, "%7 = OpLabel"},
              {8, "%8 = OpConstantTrue %10"},
              {9, "%9 = OpLabel"},
              {10, "%10 = OpTypeBool"},
            },
            { // uses
              {1,
                {
                  "%2 = OpFunction %1 None %3",
                  "%3 = OpTypeFunction %1",
                }
              },
              {3, {"%2 = OpFunction %1 None %3"}},
              {4, {"OpLoopMerge %5 %4 None"}},
              {5, {"OpLoopMerge %5 %4 None"}},
              {6, {"OpBranch %6"}},
              {7,
                {
                  "OpSelectionMerge %7 None",
                  "OpBranchConditional %8 %9 %7",
                }
              },
              {8, {"OpBranchConditional %8 %9 %7"}},
              {9, {"OpBranchConditional %8 %9 %7"}},
              {10, {"%8 = OpConstantTrue %10"}},
            }
          }
        },
        { // Forward reference
          "OpDecorate %1 Block "
          "OpTypeForwardPointer %2 Input "
          "%3 = OpTypeInt 32 0 "
          "%1 = OpTypeStruct %3 "
          "%2 = OpTypePointer Input %3",
          {
            { // defs
              {1, "%1 = OpTypeStruct %3"},
              {2, "%2 = OpTypePointer Input %3"},
              {3, "%3 = OpTypeInt 32 0"},
            },
            { // uses
              {1, {"OpDecorate %1 Block"}},
              {2, {"OpTypeForwardPointer %2 Input"}},
              {3,
                {
                  "%1 = OpTypeStruct %3",
                  "%2 = OpTypePointer Input %3",
                }
              }
            },
          },
        },
        { // OpPhi
          kOpPhiTestFunction,
          {
            { // defs
              {1, "%1 = OpTypeVoid"},
              {2, "%2 = OpFunction %1 None %3"},
              {3, "%3 = OpTypeFunction %1"},
              {4, "%4 = OpLabel"},
              {5, "%5 = OpLabel"},
              {6, "%6 = OpTypeInt 32 0"},
              {7, "%7 = OpPhi %6 %8 %4 %9 %5"},
              {8, "%8 = OpConstant %6 0"},
              {9, "%9 = OpIAdd %6 %7 %8"},
              {10, "%10 = OpTypeFloat 32"},
              {11, "%11 = OpPhi %10 %12 %4 %13 %5"},
              {12, "%12 = OpConstant %10 1.0"},
              {13, "%13 = OpFAdd %10 %11 %12"},
              {16, "%16 = OpTypeBool"},
              {17, "%17 = OpSLessThan %16 %7 %18"},
              {18, "%18 = OpConstant %6 1"},
              {19, "%19 = OpLabel"},
            },
            { // uses
              {1,
                {
                  "%2 = OpFunction %1 None %3",
                  "%3 = OpTypeFunction %1",
                }
              },
              {3, {"%2 = OpFunction %1 None %3"}},
              {4,
                {
                  "%7 = OpPhi %6 %8 %4 %9 %5",
                  "%11 = OpPhi %10 %12 %4 %13 %5",
                }
              },
              {5,
                {
                  "OpBranch %5",
                  "%7 = OpPhi %6 %8 %4 %9 %5",
                  "%11 = OpPhi %10 %12 %4 %13 %5",
                  "OpLoopMerge %19 %5 None",
                  "OpBranchConditional %17 %5 %19",
                }
              },
              {6,
                {
                  // Can't check constants properly
                  // "%8 = OpConstant %6 0",
                  // "%18 = OpConstant %6 1",
                  "%7 = OpPhi %6 %8 %4 %9 %5",
                  "%9 = OpIAdd %6 %7 %8",
                }
              },
              {7,
                {
                  "%9 = OpIAdd %6 %7 %8",
                  "%17 = OpSLessThan %16 %7 %18",
                }
              },
              {8,
                {
                  "%7 = OpPhi %6 %8 %4 %9 %5",
                  "%9 = OpIAdd %6 %7 %8",
                }
              },
              {9, {"%7 = OpPhi %6 %8 %4 %9 %5"}},
              {10,
                {
                  // "%12 = OpConstant %10 1.0",
                  "%11 = OpPhi %10 %12 %4 %13 %5",
                  "%13 = OpFAdd %10 %11 %12",
                }
              },
              {11, {"%13 = OpFAdd %10 %11 %12"}},
              {12,
                {
                  "%11 = OpPhi %10 %12 %4 %13 %5",
                  "%13 = OpFAdd %10 %11 %12",
                }
              },
              {13, {"%11 = OpPhi %10 %12 %4 %13 %5"}},
              {16, {"%17 = OpSLessThan %16 %7 %18"}},
              {17, {"OpBranchConditional %17 %5 %19"}},
              {18, {"%17 = OpSLessThan %16 %7 %18"}},
              {19,
                {
                  "OpLoopMerge %19 %5 None",
                  "OpBranchConditional %17 %5 %19",
                }
              },
            },
          },
        },
        { // OpPhi defining and referencing the same id.
          "%1 = OpTypeBool "
          "%3 = OpTypeFunction %1 "
          "%2 = OpConstantTrue %1 "
          "%4 = OpFunction %1 None %3 "
          "%6 = OpLabel "
          "     OpBranch %7 "
          "%7 = OpLabel "
          "%8 = OpPhi %1   %8 %7   %2 %6 " // both defines and uses %8
          "     OpBranch %7 "
          "     OpFunctionEnd",
          {
            { // defs
              {1, "%1 = OpTypeBool"},
              {2, "%2 = OpConstantTrue %1"},
              {3, "%3 = OpTypeFunction %1"},
              {4, "%4 = OpFunction %1 None %3"},
              {6, "%6 = OpLabel"},
              {7, "%7 = OpLabel"},
              {8, "%8 = OpPhi %1 %8 %7 %2 %6"},
            },
            { // uses
              {1,
                {
                  "%2 = OpConstantTrue %1",
                  "%3 = OpTypeFunction %1",
                  "%4 = OpFunction %1 None %3",
                  "%8 = OpPhi %1 %8 %7 %2 %6",
                }
              },
              {2, {"%8 = OpPhi %1 %8 %7 %2 %6"}},
              {3, {"%4 = OpFunction %1 None %3"}},
              {6, {"%8 = OpPhi %1 %8 %7 %2 %6"}},
              {7,
                {
                  "OpBranch %7",
                  "%8 = OpPhi %1 %8 %7 %2 %6",
                  "OpBranch %7",
                }
              },
              {8, {"%8 = OpPhi %1 %8 %7 %2 %6"}},
            },
          },
        },
    })
);
// clang-format on

struct ReplaceUseCase {
  const char* before;
  std::vector<std::pair<uint32_t, uint32_t>> candidates;
  const char* after;
  InstDefUse du;
};

using ReplaceUseTest = ::testing::TestWithParam<ReplaceUseCase>;

// Disassembles the given |module| and returns the disassembly.
std::string DisassembleModule(Module* module) {
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_1);

  std::vector<uint32_t> binary;
  module->ToBinary(&binary, /* skip_nop = */ false);

  std::string text;
  // We'll need to check the underlying id numbers.
  // So turn off friendly names for ids.
  tools.Disassemble(binary, &text, SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  while (!text.empty() && text.back() == '\n') text.pop_back();
  return text;
}

TEST_P(ReplaceUseTest, Case) {
  const auto& tc = GetParam();

  // Build module.
  const std::vector<const char*> text = {tc.before};
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, JoinAllInsts(text),
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Force a re-build of def-use manager.
  context->InvalidateAnalyses(IRContext::Analysis::kAnalysisDefUse);
  (void)context->get_def_use_mgr();

  // Do the substitution.
  for (const auto& candidate : tc.candidates) {
    context->ReplaceAllUsesWith(candidate.first, candidate.second);
  }

  EXPECT_EQ(tc.after, DisassembleModule(context->module()));
  CheckDef(tc.du, context->get_def_use_mgr()->id_to_defs());
  CheckUse(tc.du, context->get_def_use_mgr(), context->module()->IdBound());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    TestCase, ReplaceUseTest,
    ::testing::ValuesIn(std::vector<ReplaceUseCase>{
      { // no use, no replace request
        "", {}, "", {},
      },
      { // replace one use
        "%1 = OpTypeBool "
        "%2 = OpTypeVector %1 3 "
        "%3 = OpTypeInt 32 0 ",
        {{1, 3}},
        "%1 = OpTypeBool\n"
        "%2 = OpTypeVector %3 3\n"
        "%3 = OpTypeInt 32 0",
        {
          { // defs
            {1, "%1 = OpTypeBool"},
            {2, "%2 = OpTypeVector %3 3"},
            {3, "%3 = OpTypeInt 32 0"},
          },
          { // uses
            {3, {"%2 = OpTypeVector %3 3"}},
          },
        },
      },
      { // replace and then replace back
        "%1 = OpTypeBool "
        "%2 = OpTypeVector %1 3 "
        "%3 = OpTypeInt 32 0",
        {{1, 3}, {3, 1}},
        "%1 = OpTypeBool\n"
        "%2 = OpTypeVector %1 3\n"
        "%3 = OpTypeInt 32 0",
        {
          { // defs
            {1, "%1 = OpTypeBool"},
            {2, "%2 = OpTypeVector %1 3"},
            {3, "%3 = OpTypeInt 32 0"},
          },
          { // uses
            {1, {"%2 = OpTypeVector %1 3"}},
          },
        },
      },
      { // replace with the same id
        "%1 = OpTypeBool "
        "%2 = OpTypeVector %1 3",
        {{1, 1}, {2, 2}, {3, 3}},
        "%1 = OpTypeBool\n"
        "%2 = OpTypeVector %1 3",
        {
          { // defs
            {1, "%1 = OpTypeBool"},
            {2, "%2 = OpTypeVector %1 3"},
          },
          { // uses
            {1, {"%2 = OpTypeVector %1 3"}},
          },
        },
      },
      { // replace in sequence
        "%1 = OpTypeBool "
        "%2 = OpTypeVector %1 3 "
        "%3 = OpTypeInt 32 0 "
        "%4 = OpTypeInt 32 1 ",
        {{1, 3}, {3, 4}},
        "%1 = OpTypeBool\n"
        "%2 = OpTypeVector %4 3\n"
        "%3 = OpTypeInt 32 0\n"
        "%4 = OpTypeInt 32 1",
        {
          { // defs
            {1, "%1 = OpTypeBool"},
            {2, "%2 = OpTypeVector %4 3"},
            {3, "%3 = OpTypeInt 32 0"},
            {4, "%4 = OpTypeInt 32 1"},
          },
          { // uses
            {4, {"%2 = OpTypeVector %4 3"}},
          },
        },
      },
      { // replace multiple uses
        "%1 = OpTypeBool "
        "%2 = OpTypeVector %1 2 "
        "%3 = OpTypeVector %1 3 "
        "%4 = OpTypeVector %1 4 "
        "%5 = OpTypeMatrix %2 2 "
        "%6 = OpTypeMatrix %3 3 "
        "%7 = OpTypeMatrix %4 4 "
        "%8 = OpTypeInt 32 0 "
        "%9 = OpTypeInt 32 1 "
        "%10 = OpTypeInt 64 0",
        {{1, 8}, {2, 9}, {4, 10}},
        "%1 = OpTypeBool\n"
        "%2 = OpTypeVector %8 2\n"
        "%3 = OpTypeVector %8 3\n"
        "%4 = OpTypeVector %8 4\n"
        "%5 = OpTypeMatrix %9 2\n"
        "%6 = OpTypeMatrix %3 3\n"
        "%7 = OpTypeMatrix %10 4\n"
        "%8 = OpTypeInt 32 0\n"
        "%9 = OpTypeInt 32 1\n"
        "%10 = OpTypeInt 64 0",
        {
          { // defs
            {1, "%1 = OpTypeBool"},
            {2, "%2 = OpTypeVector %8 2"},
            {3, "%3 = OpTypeVector %8 3"},
            {4, "%4 = OpTypeVector %8 4"},
            {5, "%5 = OpTypeMatrix %9 2"},
            {6, "%6 = OpTypeMatrix %3 3"},
            {7, "%7 = OpTypeMatrix %10 4"},
            {8, "%8 = OpTypeInt 32 0"},
            {9, "%9 = OpTypeInt 32 1"},
            {10, "%10 = OpTypeInt 64 0"},
          },
          { // uses
            {8,
              {
                "%2 = OpTypeVector %8 2",
                "%3 = OpTypeVector %8 3",
                "%4 = OpTypeVector %8 4",
              }
            },
            {9, {"%5 = OpTypeMatrix %9 2"}},
            {3, {"%6 = OpTypeMatrix %3 3"}},
            {10, {"%7 = OpTypeMatrix %10 4"}},
          },
        },
      },
      { // OpPhi.
        kOpPhiTestFunction,
        // replace one id used by OpPhi, replace one id generated by OpPhi
        {{9, 13}, {11, 9}},
         "%1 = OpTypeVoid\n"
         "%6 = OpTypeInt 32 0\n"
         "%10 = OpTypeFloat 32\n"
         "%16 = OpTypeBool\n"
         "%3 = OpTypeFunction %1\n"
         "%8 = OpConstant %6 0\n"
         "%18 = OpConstant %6 1\n"
         "%12 = OpConstant %10 1\n"
         "%2 = OpFunction %1 None %3\n"
         "%4 = OpLabel\n"
               "OpBranch %5\n"

         "%5 = OpLabel\n"
         "%7 = OpPhi %6 %8 %4 %13 %5\n" // %9 -> %13
        "%11 = OpPhi %10 %12 %4 %13 %5\n"
         "%9 = OpIAdd %6 %7 %8\n"
        "%13 = OpFAdd %10 %9 %12\n"       // %11 -> %9
        "%17 = OpSLessThan %16 %7 %18\n"
              "OpLoopMerge %19 %5 None\n"
              "OpBranchConditional %17 %5 %19\n"

        "%19 = OpLabel\n"
              "OpReturn\n"
              "OpFunctionEnd",
        {
          { // defs.
            {1, "%1 = OpTypeVoid"},
            {2, "%2 = OpFunction %1 None %3"},
            {3, "%3 = OpTypeFunction %1"},
            {4, "%4 = OpLabel"},
            {5, "%5 = OpLabel"},
            {6, "%6 = OpTypeInt 32 0"},
            {7, "%7 = OpPhi %6 %8 %4 %13 %5"},
            {8, "%8 = OpConstant %6 0"},
            {9, "%9 = OpIAdd %6 %7 %8"},
            {10, "%10 = OpTypeFloat 32"},
            {11, "%11 = OpPhi %10 %12 %4 %13 %5"},
            {12, "%12 = OpConstant %10 1.0"},
            {13, "%13 = OpFAdd %10 %9 %12"},
            {16, "%16 = OpTypeBool"},
            {17, "%17 = OpSLessThan %16 %7 %18"},
            {18, "%18 = OpConstant %6 1"},
            {19, "%19 = OpLabel"},
          },
          { // uses
            {1,
              {
                "%2 = OpFunction %1 None %3",
                "%3 = OpTypeFunction %1",
              }
            },
            {3, {"%2 = OpFunction %1 None %3"}},
            {4,
              {
                "%7 = OpPhi %6 %8 %4 %13 %5",
                "%11 = OpPhi %10 %12 %4 %13 %5",
              }
            },
            {5,
              {
                "OpBranch %5",
                "%7 = OpPhi %6 %8 %4 %13 %5",
                "%11 = OpPhi %10 %12 %4 %13 %5",
                "OpLoopMerge %19 %5 None",
                "OpBranchConditional %17 %5 %19",
              }
            },
            {6,
              {
                // Can't properly check constants
                // "%8 = OpConstant %6 0",
                // "%18 = OpConstant %6 1",
                "%7 = OpPhi %6 %8 %4 %13 %5",
                "%9 = OpIAdd %6 %7 %8"
              }
            },
            {7,
              {
                "%9 = OpIAdd %6 %7 %8",
                "%17 = OpSLessThan %16 %7 %18",
              }
            },
            {8,
              {
                "%7 = OpPhi %6 %8 %4 %13 %5",
                "%9 = OpIAdd %6 %7 %8",
              }
            },
            {9, {"%13 = OpFAdd %10 %9 %12"}}, // uses of %9 changed from %7 to %13
            {10,
              {
                "%11 = OpPhi %10 %12 %4 %13 %5",
                // "%12 = OpConstant %10 1",
                "%13 = OpFAdd %10 %9 %12"
              }
            },
            // no more uses of %11
            {12,
              {
                "%11 = OpPhi %10 %12 %4 %13 %5",
                "%13 = OpFAdd %10 %9 %12"
              }
            },
            {13, {
                   "%7 = OpPhi %6 %8 %4 %13 %5",
                   "%11 = OpPhi %10 %12 %4 %13 %5",
                 }
            },
            {16, {"%17 = OpSLessThan %16 %7 %18"}},
            {17, {"OpBranchConditional %17 %5 %19"}},
            {18, {"%17 = OpSLessThan %16 %7 %18"}},
            {19,
              {
                "OpLoopMerge %19 %5 None",
                "OpBranchConditional %17 %5 %19",
              }
            },
          },
        },
      },
      { // OpPhi defining and referencing the same id.
        "%1 = OpTypeBool "
        "%3 = OpTypeFunction %1 "
        "%2 = OpConstantTrue %1 "

        "%4 = OpFunction %3 None %1 "
        "%6 = OpLabel "
        "     OpBranch %7 "
        "%7 = OpLabel "
        "%8 = OpPhi %1   %8 %7   %2 %6 " // both defines and uses %8
        "     OpBranch %7 "
        "     OpFunctionEnd",
        {{8, 2}},
        "%1 = OpTypeBool\n"
        "%3 = OpTypeFunction %1\n"
        "%2 = OpConstantTrue %1\n"

        "%4 = OpFunction %3 None %1\n"
        "%6 = OpLabel\n"
             "OpBranch %7\n"
        "%7 = OpLabel\n"
        "%8 = OpPhi %1 %2 %7 %2 %6\n" // use of %8 changed to %2
             "OpBranch %7\n"
             "OpFunctionEnd",
        {
          { // defs
            {1, "%1 = OpTypeBool"},
            {2, "%2 = OpConstantTrue %1"},
            {3, "%3 = OpTypeFunction %1"},
            {4, "%4 = OpFunction %3 None %1"},
            {6, "%6 = OpLabel"},
            {7, "%7 = OpLabel"},
            {8, "%8 = OpPhi %1 %2 %7 %2 %6"},
          },
          { // uses
            {1,
              {
                "%2 = OpConstantTrue %1",
                "%3 = OpTypeFunction %1",
                "%4 = OpFunction %3 None %1",
                "%8 = OpPhi %1 %2 %7 %2 %6",
              }
            },
            {2,
              {
                // Only checking users
                "%8 = OpPhi %1 %2 %7 %2 %6",
              }
            },
            {3, {"%4 = OpFunction %3 None %1"}},
            {6, {"%8 = OpPhi %1 %2 %7 %2 %6"}},
            {7,
              {
                "OpBranch %7",
                "%8 = OpPhi %1 %2 %7 %2 %6",
                "OpBranch %7",
              }
            },
            // {8, {"%8 = OpPhi %1 %8 %7 %2 %6"}},
          },
        },
      },
    })
);
// clang-format on

struct KillDefCase {
  const char* before;
  std::vector<uint32_t> ids_to_kill;
  const char* after;
  InstDefUse du;
};

using KillDefTest = ::testing::TestWithParam<KillDefCase>;

TEST_P(KillDefTest, Case) {
  const auto& tc = GetParam();

  // Build module.
  const std::vector<const char*> text = {tc.before};
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, JoinAllInsts(text),
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Analyze def and use.
  DefUseManager manager(context->module());

  // Do the substitution.
  for (const auto id : tc.ids_to_kill) context->KillDef(id);

  EXPECT_EQ(tc.after, DisassembleModule(context->module()));
  CheckDef(tc.du, context->get_def_use_mgr()->id_to_defs());
  CheckUse(tc.du, context->get_def_use_mgr(), context->module()->IdBound());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    TestCase, KillDefTest,
    ::testing::ValuesIn(std::vector<KillDefCase>{
      { // no def, no use, no kill
        "", {}, "", {}
      },
      { // kill nothing
        "%1 = OpTypeBool "
        "%2 = OpTypeVector %1 2 "
        "%3 = OpTypeVector %1 3 ",
        {},
        "%1 = OpTypeBool\n"
        "%2 = OpTypeVector %1 2\n"
        "%3 = OpTypeVector %1 3",
        {
          { // defs
            {1, "%1 = OpTypeBool"},
            {2, "%2 = OpTypeVector %1 2"},
            {3, "%3 = OpTypeVector %1 3"},
          },
          { // uses
            {1,
              {
                "%2 = OpTypeVector %1 2",
                "%3 = OpTypeVector %1 3",
              }
            },
          },
        },
      },
      { // kill id used, kill id not used, kill id not defined
        "%1 = OpTypeBool "
        "%2 = OpTypeVector %1 2 "
        "%3 = OpTypeVector %1 3 "
        "%4 = OpTypeVector %1 4 "
        "%5 = OpTypeMatrix %3 3 "
        "%6 = OpTypeMatrix %2 3",
        {1, 3, 5, 10}, // ids to kill
        "%2 = OpTypeVector %1 2\n"
        "%4 = OpTypeVector %1 4\n"
        "%6 = OpTypeMatrix %2 3",
        {
          { // defs
            {2, "%2 = OpTypeVector %1 2"},
            {4, "%4 = OpTypeVector %1 4"},
            {6, "%6 = OpTypeMatrix %2 3"},
          },
          { // uses. %1 and %3 are both killed, so no uses
            // recorded for them anymore.
            {2, {"%6 = OpTypeMatrix %2 3"}},
          }
        },
      },
      { // OpPhi.
        kOpPhiTestFunction,
        {9, 11}, // kill one id used by OpPhi, kill one id generated by OpPhi
         "%1 = OpTypeVoid\n"
         "%6 = OpTypeInt 32 0\n"
         "%10 = OpTypeFloat 32\n"
         "%16 = OpTypeBool\n"
         "%3 = OpTypeFunction %1\n"
         "%8 = OpConstant %6 0\n"
         "%18 = OpConstant %6 1\n"
         "%12 = OpConstant %10 1\n"
         "%2 = OpFunction %1 None %3\n"
         "%4 = OpLabel\n"
               "OpBranch %5\n"

         "%5 = OpLabel\n"
         "%7 = OpPhi %6 %8 %4 %9 %5\n"
        "%13 = OpFAdd %10 %11 %12\n"
        "%17 = OpSLessThan %16 %7 %18\n"
              "OpLoopMerge %19 %5 None\n"
              "OpBranchConditional %17 %5 %19\n"

        "%19 = OpLabel\n"
              "OpReturn\n"
              "OpFunctionEnd",
        {
          { // defs. %9 & %11 are killed.
            {1, "%1 = OpTypeVoid"},
            {2, "%2 = OpFunction %1 None %3"},
            {3, "%3 = OpTypeFunction %1"},
            {4, "%4 = OpLabel"},
            {5, "%5 = OpLabel"},
            {6, "%6 = OpTypeInt 32 0"},
            {7, "%7 = OpPhi %6 %8 %4 %9 %5"},
            {8, "%8 = OpConstant %6 0"},
            {10, "%10 = OpTypeFloat 32"},
            {12, "%12 = OpConstant %10 1.0"},
            {13, "%13 = OpFAdd %10 %11 %12"},
            {16, "%16 = OpTypeBool"},
            {17, "%17 = OpSLessThan %16 %7 %18"},
            {18, "%18 = OpConstant %6 1"},
            {19, "%19 = OpLabel"},
          },
          { // uses
            {1,
              {
                "%2 = OpFunction %1 None %3",
                "%3 = OpTypeFunction %1",
              }
            },
            {3, {"%2 = OpFunction %1 None %3"}},
            {4,
              {
                "%7 = OpPhi %6 %8 %4 %9 %5",
                // "%11 = OpPhi %10 %12 %4 %13 %5",
              }
            },
            {5,
              {
                "OpBranch %5",
                "%7 = OpPhi %6 %8 %4 %9 %5",
                // "%11 = OpPhi %10 %12 %4 %13 %5",
                "OpLoopMerge %19 %5 None",
                "OpBranchConditional %17 %5 %19",
              }
            },
            {6,
              {
                // Can't properly check constants
                // "%8 = OpConstant %6 0",
                // "%18 = OpConstant %6 1",
                "%7 = OpPhi %6 %8 %4 %9 %5",
                // "%9 = OpIAdd %6 %7 %8"
              }
            },
            {7, {"%17 = OpSLessThan %16 %7 %18"}},
            {8,
              {
                "%7 = OpPhi %6 %8 %4 %9 %5",
                // "%9 = OpIAdd %6 %7 %8",
              }
            },
            // {9, {"%7 = OpPhi %6 %8 %4 %13 %5"}},
            {10,
              {
                // "%11 = OpPhi %10 %12 %4 %13 %5",
                // "%12 = OpConstant %10 1",
                "%13 = OpFAdd %10 %11 %12"
              }
            },
            // {11, {"%13 = OpFAdd %10 %11 %12"}},
            {12,
              {
                // "%11 = OpPhi %10 %12 %4 %13 %5",
                "%13 = OpFAdd %10 %11 %12"
              }
            },
            // {13, {"%11 = OpPhi %10 %12 %4 %13 %5"}},
            {16, {"%17 = OpSLessThan %16 %7 %18"}},
            {17, {"OpBranchConditional %17 %5 %19"}},
            {18, {"%17 = OpSLessThan %16 %7 %18"}},
            {19,
              {
                "OpLoopMerge %19 %5 None",
                "OpBranchConditional %17 %5 %19",
              }
            },
          },
        },
      },
      { // OpPhi defining and referencing the same id.
        "%1 = OpTypeBool "
        "%3 = OpTypeFunction %1 "
        "%2 = OpConstantTrue %1 "
        "%4 = OpFunction %3 None %1 "
        "%6 = OpLabel "
        "     OpBranch %7 "
        "%7 = OpLabel "
        "%8 = OpPhi %1   %8 %7   %2 %6 " // both defines and uses %8
        "     OpBranch %7 "
        "     OpFunctionEnd",
        {8},
        "%1 = OpTypeBool\n"
        "%3 = OpTypeFunction %1\n"
        "%2 = OpConstantTrue %1\n"

        "%4 = OpFunction %3 None %1\n"
        "%6 = OpLabel\n"
             "OpBranch %7\n"
        "%7 = OpLabel\n"
             "OpBranch %7\n"
             "OpFunctionEnd",
        {
          { // defs
            {1, "%1 = OpTypeBool"},
            {2, "%2 = OpConstantTrue %1"},
            {3, "%3 = OpTypeFunction %1"},
            {4, "%4 = OpFunction %3 None %1"},
            {6, "%6 = OpLabel"},
            {7, "%7 = OpLabel"},
            // {8, "%8 = OpPhi %1 %8 %7 %2 %6"},
          },
          { // uses
            {1,
              {
                "%2 = OpConstantTrue %1",
                "%3 = OpTypeFunction %1",
                "%4 = OpFunction %3 None %1",
                // "%8 = OpPhi %1 %8 %7 %2 %6",
              }
            },
            // {2, {"%8 = OpPhi %1 %8 %7 %2 %6"}},
            {3, {"%4 = OpFunction %3 None %1"}},
            // {6, {"%8 = OpPhi %1 %8 %7 %2 %6"}},
            {7,
              {
                "OpBranch %7",
                // "%8 = OpPhi %1 %8 %7 %2 %6",
                "OpBranch %7",
              }
            },
            // {8, {"%8 = OpPhi %1 %8 %7 %2 %6"}},
          },
        },
      },
    })
);
// clang-format on

TEST(DefUseTest, OpSwitch) {
  // Because disassembler has basic type check for OpSwitch's selector, we
  // cannot use the DisassembleInst() in the above. Thus, this special spotcheck
  // test case.

  const char original_text[] =
      // int64 f(int64 v) {
      //   switch (v) {
      //     case 1:                   break;
      //     case -4294967296:         break;
      //     case 9223372036854775807: break;
      //     default:                  break;
      //   }
      //   return v;
      // }
      " %1 = OpTypeInt 64 1 "
      " %3 = OpTypePointer Input %1 "
      " %2 = OpFunction %1 None %3 "  // %3 is int64(int64)*
      " %4 = OpFunctionParameter %1 "
      " %5 = OpLabel "
      " %6 = OpLoad %1 %4 "  // selector value
      "      OpSelectionMerge %7 None "
      "      OpSwitch %6 %8 "
      "                  1                    %9 "  // 1
      "                  -4294967296         %10 "  // -2^32
      "                  9223372036854775807 %11 "  // 2^63-1
      " %8 = OpLabel "                              // default
      "      OpBranch %7 "
      " %9 = OpLabel "
      "      OpBranch %7 "
      "%10 = OpLabel "
      "      OpBranch %7 "
      "%11 = OpLabel "
      "      OpBranch %7 "
      " %7 = OpLabel "
      "      OpReturnValue %6 "
      "      OpFunctionEnd";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, original_text,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Force a re-build of def-use manager.
  context->InvalidateAnalyses(IRContext::Analysis::kAnalysisDefUse);
  (void)context->get_def_use_mgr();

  // Do a bunch replacements.
  context->ReplaceAllUsesWith(11, 7);   // to existing id
  context->ReplaceAllUsesWith(10, 11);  // to existing id
  context->ReplaceAllUsesWith(9, 10);   // to existing id

  // clang-format off
  const char modified_text[] =
       "%1 = OpTypeInt 64 1\n"
       "%3 = OpTypePointer Input %1\n"
       "%2 = OpFunction %1 None %3\n" // %3 is int64(int64)*
       "%4 = OpFunctionParameter %1\n"
       "%5 = OpLabel\n"
       "%6 = OpLoad %1 %4\n" // selector value
            "OpSelectionMerge %7 None\n"
            "OpSwitch %6 %8 1 %10 -4294967296 %11 9223372036854775807 %7\n" // changed!
       "%8 = OpLabel\n"      // default
            "OpBranch %7\n"
       "%9 = OpLabel\n"
            "OpBranch %7\n"
      "%10 = OpLabel\n"
            "OpBranch %7\n"
      "%11 = OpLabel\n"
            "OpBranch %7\n"
       "%7 = OpLabel\n"
            "OpReturnValue %6\n"
            "OpFunctionEnd";
  // clang-format on

  EXPECT_EQ(modified_text, DisassembleModule(context->module()));

  InstDefUse def_uses = {};
  def_uses.defs = {
      {1, "%1 = OpTypeInt 64 1"},
      {2, "%2 = OpFunction %1 None %3"},
      {3, "%3 = OpTypePointer Input %1"},
      {4, "%4 = OpFunctionParameter %1"},
      {5, "%5 = OpLabel"},
      {6, "%6 = OpLoad %1 %4"},
      {7, "%7 = OpLabel"},
      {8, "%8 = OpLabel"},
      {9, "%9 = OpLabel"},
      {10, "%10 = OpLabel"},
      {11, "%11 = OpLabel"},
  };
  CheckDef(def_uses, context->get_def_use_mgr()->id_to_defs());

  {
    EXPECT_EQ(2u, NumUses(context, 6));
    std::vector<SpvOp> opcodes = GetUseOpcodes(context, 6u);
    EXPECT_THAT(opcodes, UnorderedElementsAre(SpvOpSwitch, SpvOpReturnValue));
  }
  {
    EXPECT_EQ(6u, NumUses(context, 7));
    std::vector<SpvOp> opcodes = GetUseOpcodes(context, 7u);
    // OpSwitch is now a user of %7.
    EXPECT_THAT(opcodes, UnorderedElementsAre(SpvOpSelectionMerge, SpvOpBranch,
                                              SpvOpBranch, SpvOpBranch,
                                              SpvOpBranch, SpvOpSwitch));
  }
  // Check all ids only used by OpSwitch after replacement.
  for (const auto id : {8u, 10u, 11u}) {
    EXPECT_EQ(1u, NumUses(context, id));
    EXPECT_EQ(SpvOpSwitch, GetUseOpcodes(context, id).back());
  }
}

// Test case for analyzing individual instructions.
struct AnalyzeInstDefUseTestCase {
  const char* module_text;
  InstDefUse expected_define_use;
};

using AnalyzeInstDefUseTest =
    ::testing::TestWithParam<AnalyzeInstDefUseTestCase>;

// Test the analyzing result for individual instructions.
TEST_P(AnalyzeInstDefUseTest, Case) {
  auto tc = GetParam();

  // Build module.
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.module_text);
  ASSERT_NE(nullptr, context);

  // Analyze the instructions.
  DefUseManager manager(context->module());

  CheckDef(tc.expected_define_use, manager.id_to_defs());
  CheckUse(tc.expected_define_use, &manager, context->module()->IdBound());
  // CheckUse(tc.expected_define_use, manager.id_to_uses());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    TestCase, AnalyzeInstDefUseTest,
    ::testing::ValuesIn(std::vector<AnalyzeInstDefUseTestCase>{
      { // A type declaring instruction.
        "%1 = OpTypeInt 32 1",
        {
          // defs
          {{1, "%1 = OpTypeInt 32 1"}},
          {}, // no uses
        },
      },
      { // A type declaring instruction and a constant value.
        "%1 = OpTypeBool "
        "%2 = OpConstantTrue %1",
        {
          { // defs
            {1, "%1 = OpTypeBool"},
            {2, "%2 = OpConstantTrue %1"},
          },
          { // uses
            {1, {"%2 = OpConstantTrue %1"}},
          },
        },
      },
      }));
// clang-format on

using AnalyzeInstDefUse = ::testing::Test;

TEST(AnalyzeInstDefUse, UseWithNoResultId) {
  IRContext context(SPV_ENV_UNIVERSAL_1_2, nullptr);

  // Analyze the instructions.
  DefUseManager manager(context.module());

  Instruction label(&context, SpvOpLabel, 0, 2, {});
  manager.AnalyzeInstDefUse(&label);

  Instruction branch(&context, SpvOpBranch, 0, 0, {{SPV_OPERAND_TYPE_ID, {2}}});
  manager.AnalyzeInstDefUse(&branch);
  context.module()->SetIdBound(3);

  InstDefUse expected = {
      // defs
      {
          {2, "%2 = OpLabel"},
      },
      // uses
      {{2, {"OpBranch %2"}}},
  };

  CheckDef(expected, manager.id_to_defs());
  CheckUse(expected, &manager, context.module()->IdBound());
}

TEST(AnalyzeInstDefUse, AddNewInstruction) {
  const std::string input = "%1 = OpTypeBool";

  // Build module.
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, input);
  ASSERT_NE(nullptr, context);

  // Analyze the instructions.
  DefUseManager manager(context->module());

  Instruction newInst(context.get(), SpvOpConstantTrue, 1, 2, {});
  manager.AnalyzeInstDefUse(&newInst);

  InstDefUse expected = {
      {
          // defs
          {1, "%1 = OpTypeBool"},
          {2, "%2 = OpConstantTrue %1"},
      },
      {
          // uses
          {1, {"%2 = OpConstantTrue %1"}},
      },
  };

  CheckDef(expected, manager.id_to_defs());
  CheckUse(expected, &manager, context->module()->IdBound());
}

struct KillInstTestCase {
  const char* before;
  std::unordered_set<uint32_t> indices_for_inst_to_kill;
  const char* after;
  InstDefUse expected_define_use;
};

using KillInstTest = ::testing::TestWithParam<KillInstTestCase>;

TEST_P(KillInstTest, Case) {
  auto tc = GetParam();

  // Build module.
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.before,
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  // Force a re-build of the def-use manager.
  context->InvalidateAnalyses(IRContext::Analysis::kAnalysisDefUse);
  (void)context->get_def_use_mgr();

  // KillInst
  context->module()->ForEachInst([&tc, &context](Instruction* inst) {
    if (tc.indices_for_inst_to_kill.count(inst->result_id())) {
      context->KillInst(inst);
    }
  });

  EXPECT_EQ(tc.after, DisassembleModule(context->module()));
  CheckDef(tc.expected_define_use, context->get_def_use_mgr()->id_to_defs());
  CheckUse(tc.expected_define_use, context->get_def_use_mgr(),
           context->module()->IdBound());
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    TestCase, KillInstTest,
    ::testing::ValuesIn(std::vector<KillInstTestCase>{
      // Kill id defining instructions.
      {
        "%3 = OpTypeVoid "
        "%1 = OpTypeFunction %3 "
        "%2 = OpFunction %1 None %3 "
        "%4 = OpLabel "
        "     OpBranch %5 "
        "%5 = OpLabel "
        "     OpBranch %6 "
        "%6 = OpLabel "
        "     OpBranch %4 "
        "%7 = OpLabel "
        "     OpReturn "
        "     OpFunctionEnd",
        {3, 5, 7},
        "%1 = OpTypeFunction %3\n"
        "%2 = OpFunction %1 None %3\n"
        "%4 = OpLabel\n"
        "OpBranch %5\n"
        "OpNop\n"
        "OpBranch %6\n"
        "%6 = OpLabel\n"
        "OpBranch %4\n"
        "OpNop\n"
        "OpReturn\n"
        "OpFunctionEnd",
        {
          // defs
          {
            {1, "%1 = OpTypeFunction %3"},
            {2, "%2 = OpFunction %1 None %3"},
            {4, "%4 = OpLabel"},
            {6, "%6 = OpLabel"},
          },
          // uses
          {
            {1, {"%2 = OpFunction %1 None %3"}},
            {4, {"OpBranch %4"}},
            {6, {"OpBranch %6"}},
          }
        }
      },
      // Kill instructions that do not have result ids.
      {
        "%3 = OpTypeVoid "
        "%1 = OpTypeFunction %3 "
        "%2 = OpFunction %1 None %3 "
        "%4 = OpLabel "
        "     OpBranch %5 "
        "%5 = OpLabel "
        "     OpBranch %6 "
        "%6 = OpLabel "
        "     OpBranch %4 "
        "%7 = OpLabel "
        "     OpReturn "
        "     OpFunctionEnd",
        {2, 4},
        "%3 = OpTypeVoid\n"
        "%1 = OpTypeFunction %3\n"
             "OpNop\n"
             "OpNop\n"
             "OpBranch %5\n"
        "%5 = OpLabel\n"
             "OpBranch %6\n"
        "%6 = OpLabel\n"
             "OpBranch %4\n"
        "%7 = OpLabel\n"
             "OpReturn\n"
             "OpFunctionEnd",
        {
          // defs
          {
            {1, "%1 = OpTypeFunction %3"},
            {3, "%3 = OpTypeVoid"},
            {5, "%5 = OpLabel"},
            {6, "%6 = OpLabel"},
            {7, "%7 = OpLabel"},
          },
          // uses
          {
            {3, {"%1 = OpTypeFunction %3"}},
            {5, {"OpBranch %5"}},
            {6, {"OpBranch %6"}},
          }
        }
      },
      }));
// clang-format on

struct GetAnnotationsTestCase {
  const char* code;
  uint32_t id;
  std::vector<std::string> annotations;
};

using GetAnnotationsTest = ::testing::TestWithParam<GetAnnotationsTestCase>;

TEST_P(GetAnnotationsTest, Case) {
  const GetAnnotationsTestCase& tc = GetParam();

  // Build module.
  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, tc.code);
  ASSERT_NE(nullptr, context);

  // Get annotations
  DefUseManager manager(context->module());
  auto insts = manager.GetAnnotations(tc.id);

  // Check
  ASSERT_EQ(tc.annotations.size(), insts.size())
      << "wrong number of annotation instructions";
  auto inst_iter = insts.begin();
  for (const std::string& expected_anno_inst : tc.annotations) {
    EXPECT_EQ(expected_anno_inst, DisassembleInst(*inst_iter))
        << "annotation instruction mismatch";
    inst_iter++;
  }
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    TestCase, GetAnnotationsTest,
    ::testing::ValuesIn(std::vector<GetAnnotationsTestCase>{
      // empty
      {"", 0, {}},
      // basic
      {
        // code
        "OpDecorate %1 Block "
        "OpDecorate %1 RelaxedPrecision "
        "%3 = OpTypeInt 32 0 "
        "%1 = OpTypeStruct %3",
        // id
        1,
        // annotations
        {
          "OpDecorate %1 Block",
          "OpDecorate %1 RelaxedPrecision",
        },
      },
      // with debug instructions
      {
        // code
        "OpName %1 \"struct_type\" "
        "OpName %3 \"int_type\" "
        "OpDecorate %1 Block "
        "OpDecorate %1 RelaxedPrecision "
        "%3 = OpTypeInt 32 0 "
        "%1 = OpTypeStruct %3",
        // id
        1,
        // annotations
        {
          "OpDecorate %1 Block",
          "OpDecorate %1 RelaxedPrecision",
        },
      },
      // no annotations
      {
        // code
        "OpName %1 \"struct_type\" "
        "OpName %3 \"int_type\" "
        "OpDecorate %1 Block "
        "OpDecorate %1 RelaxedPrecision "
        "%3 = OpTypeInt 32 0 "
        "%1 = OpTypeStruct %3",
        // id
        3,
        // annotations
        {},
      },
      // decoration group
      {
        // code
        "OpDecorate %1 Block "
        "OpDecorate %1 RelaxedPrecision "
        "%1 = OpDecorationGroup "
        "OpGroupDecorate %1 %2 %3 "
        "%4 = OpTypeInt 32 0 "
        "%2 = OpTypeStruct %4 "
        "%3 = OpTypeStruct %4 %4",
        // id
        3,
        // annotations
        {
          "OpGroupDecorate %1 %2 %3",
        },
      },
      // memeber decorate
      {
        // code
        "OpMemberDecorate %1 0 RelaxedPrecision "
        "%2 = OpTypeInt 32 0 "
        "%1 = OpTypeStruct %2 %2",
        // id
        1,
        // annotations
        {
          "OpMemberDecorate %1 0 RelaxedPrecision",
        },
      },
      }));

using UpdateUsesTest = PassTest<::testing::Test>;

TEST_F(UpdateUsesTest, KeepOldUses) {
  const std::vector<const char*> text = {
      // clang-format off
      "OpCapability Shader",
      "%1 = OpExtInstImport \"GLSL.std.450\"",
      "OpMemoryModel Logical GLSL450",
      "OpEntryPoint Vertex %main \"main\"",
      "OpName %main \"main\"",
      "%void = OpTypeVoid",
      "%4 = OpTypeFunction %void",
      "%uint = OpTypeInt 32 0",
      "%uint_5 = OpConstant %uint 5",
      "%25 = OpConstant %uint 25",
      "%main = OpFunction %void None %4",
      "%8 = OpLabel",
      "%9 = OpIMul %uint %uint_5 %uint_5",
      "%10 = OpIMul %uint %9 %uint_5",
      "OpReturn",
      "OpFunctionEnd"
      // clang-format on
  };

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, JoinAllInsts(text),
                  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  ASSERT_NE(nullptr, context);

  DefUseManager* def_use_mgr = context->get_def_use_mgr();
  Instruction* def = def_use_mgr->GetDef(9);
  Instruction* use = def_use_mgr->GetDef(10);
  def->SetOpcode(SpvOpCopyObject);
  def->SetInOperands({{SPV_OPERAND_TYPE_ID, {25}}});
  context->UpdateDefUse(def);

  auto users = def_use_mgr->id_to_users();
  UserEntry entry = {def, use};
  EXPECT_THAT(users, Contains(entry));
}
// clang-format on

}  // namespace
}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
