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

// Validation tests for Control Flow Graph

#include <array>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"

#include "source/diagnostic.h"
#include "source/spirv_target_env.h"
#include "source/val/validate.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;
using ::testing::MatchesRegex;

using ValidateCFG = spvtest::ValidateBase<SpvCapability>;
using spvtest::ScopedContext;

std::string nameOps() { return ""; }

template <typename... Args>
std::string nameOps(std::pair<std::string, std::string> head, Args... names) {
  return "OpName %" + head.first + " \"" + head.second + "\"\n" +
         nameOps(names...);
}

template <typename... Args>
std::string nameOps(std::string head, Args... names) {
  return "OpName %" + head + " \"" + head + "\"\n" + nameOps(names...);
}

/// This class allows the easy creation of complex control flow without writing
/// SPIR-V. This class is used in the test cases below.
class Block {
  std::string label_;
  std::string body_;
  SpvOp type_;
  std::vector<Block> successors_;

 public:
  /// Creates a Block with a given label
  ///
  /// @param[in]: label the label id of the block
  /// @param[in]: type the branch instruciton that ends the block
  explicit Block(std::string label, SpvOp type = SpvOpBranch)
      : label_(label), body_(), type_(type), successors_() {}

  /// Sets the instructions which will appear in the body of the block
  Block& SetBody(std::string body) {
    body_ = body;
    return *this;
  }

  Block& AppendBody(std::string body) {
    body_ += body;
    return *this;
  }

  /// Converts the block into a SPIR-V string
  operator std::string() {
    std::stringstream out;
    out << std::setw(8) << "%" + label_ + "  = OpLabel \n";
    if (!body_.empty()) {
      out << body_;
    }

    switch (type_) {
      case SpvOpBranchConditional:
        out << "OpBranchConditional %cond ";
        for (Block& b : successors_) {
          out << "%" + b.label_ + " ";
        }
        break;
      case SpvOpSwitch: {
        out << "OpSwitch %one %" + successors_.front().label_;
        std::stringstream ss;
        for (size_t i = 1; i < successors_.size(); i++) {
          ss << " " << i << " %" << successors_[i].label_;
        }
        out << ss.str();
      } break;
      case SpvOpLoopMerge: {
        assert(successors_.size() == 2);
        out << "OpLoopMerge %" + successors_[0].label_ + " %" +
                   successors_[0].label_ + "None";
      } break;

      case SpvOpReturn:
        assert(successors_.size() == 0);
        out << "OpReturn\n";
        break;
      case SpvOpUnreachable:
        assert(successors_.size() == 0);
        out << "OpUnreachable\n";
        break;
      case SpvOpBranch:
        assert(successors_.size() == 1);
        out << "OpBranch %" + successors_.front().label_;
        break;
      case SpvOpKill:
        assert(successors_.size() == 0);
        out << "OpKill\n";
        break;
      default:
        assert(1 == 0 && "Unhandled");
    }
    out << "\n";

    return out.str();
  }
  friend Block& operator>>(Block& curr, std::vector<Block> successors);
  friend Block& operator>>(Block& lhs, Block& successor);
};

/// Assigns the successors for the Block on the lhs
Block& operator>>(Block& lhs, std::vector<Block> successors) {
  if (lhs.type_ == SpvOpBranchConditional) {
    assert(successors.size() == 2);
  } else if (lhs.type_ == SpvOpSwitch) {
    assert(successors.size() > 1);
  }
  lhs.successors_ = successors;
  return lhs;
}

/// Assigns the successor for the Block on the lhs
Block& operator>>(Block& lhs, Block& successor) {
  assert(lhs.type_ == SpvOpBranch);
  lhs.successors_.push_back(successor);
  return lhs;
}

const std::string& GetDefaultHeader(SpvCapability cap) {
  static const std::string shader_header =
      "OpCapability Shader\n"
      "OpCapability Linkage\n"
      "OpMemoryModel Logical GLSL450\n";

  static const std::string kernel_header =
      "OpCapability Kernel\n"
      "OpCapability Linkage\n"
      "OpMemoryModel Logical OpenCL\n";

  return (cap == SpvCapabilityShader) ? shader_header : kernel_header;
}

const std::string& GetWebGPUHeader() {
  static const std::string header =
      "OpCapability Shader\n"
      "OpCapability VulkanMemoryModelKHR\n"
      "OpExtension \"SPV_KHR_vulkan_memory_model\"\n"
      "OpMemoryModel Logical VulkanKHR\n";
  return header;
}

const std::string& types_consts() {
  static const std::string types =
      "%voidt   = OpTypeVoid\n"
      "%boolt   = OpTypeBool\n"
      "%intt    = OpTypeInt 32 0\n"
      "%one     = OpConstant %intt 1\n"
      "%two     = OpConstant %intt 2\n"
      "%ptrt    = OpTypePointer Function %intt\n"
      "%funct   = OpTypeFunction %voidt\n";
  return types;
}

INSTANTIATE_TEST_SUITE_P(StructuredControlFlow, ValidateCFG,
                         ::testing::Values(SpvCapabilityShader,
                                           SpvCapabilityKernel));

TEST_P(ValidateCFG, LoopReachableFromEntryButNeverLeadingToReturn) {
  // In this case, the loop is reachable from a node without a predecessor,
  // but never reaches a node with a return.
  //
  // This motivates the need for the pseudo-exit node to have a node
  // from a cycle in its predecessors list.  Otherwise the validator's
  // post-dominance calculation will go into an infinite loop.
  //
  // For more motivation, see
  // https://github.com/KhronosGroup/SPIRV-Tools/issues/279
  std::string str = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450

           OpName %entry "entry"
           OpName %loop "loop"
           OpName %exit "exit"

%voidt   = OpTypeVoid
%funct   = OpTypeFunction %voidt

%main    = OpFunction %voidt None %funct
%entry   = OpLabel
           OpBranch %loop
%loop    = OpLabel
           OpLoopMerge %exit %loop None
           OpBranch %loop
%exit    = OpLabel
           OpReturn
           OpFunctionEnd
  )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions()) << str;
}

TEST_P(ValidateCFG, LoopUnreachableFromEntryButLeadingToReturn) {
  // In this case, the loop is not reachable from a node without a
  // predecessor, but eventually reaches a node with a return.
  //
  // This motivates the need for the pseudo-entry node to have a node
  // from a cycle in its successors list.  Otherwise the validator's
  // dominance calculation will go into an infinite loop.
  //
  // For more motivation, see
  // https://github.com/KhronosGroup/SPIRV-Tools/issues/279
  // Before that fix, we'd have an infinite loop when calculating
  // post-dominators.
  std::string str = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450

           OpName %entry "entry"
           OpName %loop "loop"
           OpName %cont "cont"
           OpName %exit "exit"

%voidt   = OpTypeVoid
%funct   = OpTypeFunction %voidt
%boolt   = OpTypeBool
%false   = OpConstantFalse %boolt

%main    = OpFunction %voidt None %funct
%entry   = OpLabel
           OpReturn

%loop    = OpLabel
           OpLoopMerge %exit %cont None
           OpBranch %cont

%cont    = OpLabel
           OpBranchConditional %false %loop %exit

%exit    = OpLabel
           OpReturn
           OpFunctionEnd
  )";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions())
      << str << getDiagnosticString();
}

TEST_P(ValidateCFG, Simple) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop("loop", SpvOpBranchConditional);
  Block cont("cont");
  Block merge("merge", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) {
    loop.SetBody("OpLoopMerge %merge %cont None\n");
  }

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("loop", "entry", "cont", "merge",
                            std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> std::vector<Block>({cont, merge});
  str += cont >> loop;
  str += merge;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateCFG, Variable) {
  Block entry("entry");
  Block cont("cont");
  Block exit("exit", SpvOpReturn);

  entry.SetBody("%var = OpVariable %ptrt Function\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps(std::make_pair("func", "Main")) + types_consts() +
                    " %func    = OpFunction %voidt None %funct\n";
  str += entry >> cont;
  str += cont >> exit;
  str += exit;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateCFG, VariableNotInFirstBlockBad) {
  Block entry("entry");
  Block cont("cont");
  Block exit("exit", SpvOpReturn);

  // This operation should only be performed in the entry block
  cont.SetBody("%var = OpVariable %ptrt Function\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps(std::make_pair("func", "Main")) + types_consts() +
                    " %func    = OpFunction %voidt None %funct\n";

  str += entry >> cont;
  str += cont >> exit;
  str += exit;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Variables can only be defined in the first block of a function"));
}

TEST_P(ValidateCFG, BlockSelfLoopIsOk) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop("loop", SpvOpBranchConditional);
  Block merge("merge", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) loop.SetBody("OpLoopMerge %merge %loop None\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("loop", "merge", std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  // loop branches to itself, but does not trigger an error.
  str += loop >> std::vector<Block>({merge, loop});
  str += merge;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions()) << getDiagnosticString();
}

TEST_P(ValidateCFG, BlockAppearsBeforeDominatorBad) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block cont("cont");
  Block branch("branch", SpvOpBranchConditional);
  Block merge("merge", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) branch.SetBody("OpSelectionMerge %merge None\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("cont", "branch", std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> branch;
  str += cont >> merge;  // cont appears before its dominator
  str += branch >> std::vector<Block>({cont, merge});
  str += merge;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("Block .\\[%cont\\] appears in the binary "
                           "before its dominator .\\[%branch\\]\n"
                           "  %branch = OpLabel\n"));
}

TEST_P(ValidateCFG, MergeBlockTargetedByMultipleHeaderBlocksBad) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop("loop");
  Block selection("selection", SpvOpBranchConditional);
  Block merge("merge", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) loop.SetBody(" OpLoopMerge %merge %loop None\n");

  // cannot share the same merge
  if (is_shader) selection.SetBody("OpSelectionMerge %merge None\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("merge", std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> selection;
  str += selection >> std::vector<Block>({loop, merge});
  str += merge;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  if (is_shader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                MatchesRegex("Block .\\[%merge\\] is already a merge block "
                             "for another header\n"
                             "  %Main = OpFunction %void None %9\n"));
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_P(ValidateCFG, MergeBlockTargetedByMultipleHeaderBlocksSelectionBad) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop("loop", SpvOpBranchConditional);
  Block selection("selection", SpvOpBranchConditional);
  Block merge("merge", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) selection.SetBody(" OpSelectionMerge %merge None\n");

  // cannot share the same merge
  if (is_shader) loop.SetBody(" OpLoopMerge %merge %loop None\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("merge", std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> selection;
  str += selection >> std::vector<Block>({merge, loop});
  str += loop >> std::vector<Block>({loop, merge});
  str += merge;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  if (is_shader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                MatchesRegex("Block .\\[%merge\\] is already a merge block "
                             "for another header\n"
                             "  %Main = OpFunction %void None %9\n"));
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_P(ValidateCFG, BranchTargetFirstBlockBadSinceEntryBlock) {
  Block entry("entry");
  Block bad("bad");
  Block end("end", SpvOpReturn);
  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("entry", "bad", std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> entry;  // Cannot target entry block
  str += end;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("First block .\\[%entry\\] of function "
                           ".\\[%Main\\] is targeted by block .\\[%bad\\]\n"
                           "  %Main = OpFunction %void None %10\n"));
}

TEST_P(ValidateCFG, BranchTargetFirstBlockBadSinceValue) {
  Block entry("entry");
  entry.SetBody("%undef = OpUndef %voidt\n");
  Block bad("bad");
  Block end("end", SpvOpReturn);
  Block badvalue("undef");  // This referenes the OpUndef.
  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("entry", "bad", std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str +=
      bad >> badvalue;  // Check branch to a function value (it's not a block!)
  str += end;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("'Target Label' operands for OpBranch must "
                        "be the ID of an OpLabel instruction"));
}

TEST_P(ValidateCFG, BranchConditionalTrueTargetFirstBlockBad) {
  Block entry("entry");
  Block bad("bad", SpvOpBranchConditional);
  Block exit("exit", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  bad.SetBody(" OpLoopMerge %entry %exit None\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("entry", "bad", std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> std::vector<Block>({entry, exit});  // cannot target entry block
  str += exit;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("First block .\\[%entry\\] of function .\\[%Main\\] "
                           "is targeted by block .\\[%bad\\]\n"
                           "  %Main = OpFunction %void None %10\n"));
}

TEST_P(ValidateCFG, BranchConditionalFalseTargetFirstBlockBad) {
  Block entry("entry");
  Block bad("bad", SpvOpBranchConditional);
  Block t("t");
  Block merge("merge");
  Block end("end", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  bad.SetBody("OpLoopMerge %merge %cont None\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("entry", "bad", std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> std::vector<Block>({t, entry});
  str += merge >> end;
  str += end;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("First block .\\[%entry\\] of function .\\[%Main\\] "
                           "is targeted by block .\\[%bad\\]\n"
                           "  %Main = OpFunction %void None %10\n"));
}

TEST_P(ValidateCFG, SwitchTargetFirstBlockBad) {
  Block entry("entry");
  Block bad("bad", SpvOpSwitch);
  Block block1("block1");
  Block block2("block2");
  Block block3("block3");
  Block def("def");  // default block
  Block merge("merge");
  Block end("end", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  bad.SetBody("OpSelectionMerge %merge None\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("entry", "bad", std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> bad;
  str += bad >> std::vector<Block>({def, block1, block2, block3, entry});
  str += def >> merge;
  str += block1 >> merge;
  str += block2 >> merge;
  str += block3 >> merge;
  str += merge >> end;
  str += end;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              MatchesRegex("First block .\\[%entry\\] of function .\\[%Main\\] "
                           "is targeted by block .\\[%bad\\]\n"
                           "  %Main = OpFunction %void None %10\n"));
}

TEST_P(ValidateCFG, BranchToBlockInOtherFunctionBad) {
  Block entry("entry");
  Block middle("middle", SpvOpBranchConditional);
  Block end("end", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  middle.SetBody("OpSelectionMerge %end None\n");

  Block entry2("entry2");
  Block middle2("middle2");
  Block end2("end2", SpvOpReturn);

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("middle2", std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> middle;
  str += middle >> std::vector<Block>({end, middle2});
  str += end;
  str += "OpFunctionEnd\n";

  str += "%func2    = OpFunction %voidt None %funct\n";
  str += entry2 >> middle2;
  str += middle2 >> end2;
  str += end2;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      MatchesRegex("Block\\(s\\) \\{.\\[%middle2\\]\\} are referenced but not "
                   "defined in function .\\[%Main\\]\n"
                   "  %Main = OpFunction %void None %9\n"));
}

TEST_P(ValidateCFG, HeaderDoesntDominatesMergeBad) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block head("head", SpvOpBranchConditional);
  Block f("f");
  Block merge("merge", SpvOpReturn);

  head.SetBody("%cond = OpSLessThan %boolt %one %two\n");

  if (is_shader) head.AppendBody("OpSelectionMerge %merge None\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("head", "merge", std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> merge;
  str += head >> std::vector<Block>({merge, f});
  str += f >> merge;
  str += merge;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  if (is_shader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(
        getDiagnosticString(),
        MatchesRegex("The selection construct with the selection header "
                     ".\\[%head\\] does not dominate the merge block "
                     ".\\[%merge\\]\n  %merge = OpLabel\n"));
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_P(ValidateCFG, HeaderDoesntStrictlyDominateMergeBad) {
  // If a merge block is reachable, then it must be strictly dominated by
  // its header block.
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block head("head", SpvOpBranchConditional);
  Block exit("exit", SpvOpReturn);

  head.SetBody("%cond = OpSLessThan %boolt %one %two\n");

  if (is_shader) head.AppendBody("OpSelectionMerge %head None\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("head", "exit", std::make_pair("func", "Main")) +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += head >> std::vector<Block>({exit, exit});
  str += exit;
  str += "OpFunctionEnd\n";

  CompileSuccessfully(str);
  if (is_shader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(
        getDiagnosticString(),
        MatchesRegex("The selection construct with the selection header "
                     ".\\[%head\\] does not strictly dominate the merge block "
                     ".\\[%head\\]\n  %head = OpLabel\n"));
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions()) << str;
  }
}

std::string GetUnreachableMergeNoMergeInst(SpvCapability cap,
                                           spv_target_env env) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);
  Block entry("entry");
  Block branch("branch", SpvOpBranchConditional);
  Block t("t", SpvOpReturn);
  Block f("f", SpvOpReturn);
  Block merge("merge", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (!spvIsWebGPUEnv(env) && cap == SpvCapabilityShader)
    branch.AppendBody("OpSelectionMerge %merge None\n");

  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (!spvIsWebGPUEnv(env))
    str += nameOps("branch", "merge", std::make_pair("func", "Main"));
  str += types_consts() + "%func    = OpFunction %voidt None %funct\n";
  str += entry >> branch;
  str += branch >> std::vector<Block>({t, f});
  str += t;
  str += f;
  str += merge;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableMergeNoMergeInst) {
  CompileSuccessfully(
      GetUnreachableMergeNoMergeInst(GetParam(), SPV_ENV_UNIVERSAL_1_0));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, WebGPUUnreachableMergeNoMergeInst) {
  CompileSuccessfully(
      GetUnreachableMergeNoMergeInst(SpvCapabilityShader, SPV_ENV_WEBGPU_0));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("For WebGPU, all blocks must be reachable"));
}

std::string GetUnreachableMergeTerminatedBy(SpvCapability cap,
                                            spv_target_env env, SpvOp op) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block entry("entry");
  Block branch("branch", SpvOpBranchConditional);
  Block t("t", SpvOpReturn);
  Block f("f", SpvOpReturn);
  Block merge("merge", op);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (cap == SpvCapabilityShader)
    branch.AppendBody("OpSelectionMerge %merge None\n");
  if (!spvIsWebGPUEnv(env))
    str += nameOps("branch", "merge", std::make_pair("func", "Main"));

  str += types_consts();
  str += "%func    = OpFunction %voidt None %funct\n";
  str += entry >> branch;
  str += branch >> std::vector<Block>({t, f});
  str += t;
  str += f;
  str += merge;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableMergeTerminatedByOpUnreachable) {
  CompileSuccessfully(GetUnreachableMergeTerminatedBy(
      GetParam(), SPV_ENV_UNIVERSAL_1_0, SpvOpUnreachable));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, UnreachableMergeTerminatedByOpKill) {
  CompileSuccessfully(GetUnreachableMergeTerminatedBy(
      SpvCapabilityShader, SPV_ENV_UNIVERSAL_1_0, SpvOpKill));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateCFG, UnreachableMergeTerminatedByOpReturn) {
  CompileSuccessfully(GetUnreachableMergeTerminatedBy(
      GetParam(), SPV_ENV_UNIVERSAL_1_0, SpvOpReturn));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, WebGPUUnreachableMergeTerminatedByOpUnreachable) {
  CompileSuccessfully(GetUnreachableMergeTerminatedBy(
      SpvCapabilityShader, SPV_ENV_WEBGPU_0, SpvOpUnreachable));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

TEST_F(ValidateCFG, WebGPUUnreachableMergeTerminatedByOpKill) {
  CompileSuccessfully(GetUnreachableMergeTerminatedBy(
      SpvCapabilityShader, SPV_ENV_WEBGPU_0, SpvOpKill));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("must terminate with OpUnreachable"));
}

TEST_P(ValidateCFG, WebGPUUnreachableMergeTerminatedByOpReturn) {
  CompileSuccessfully(GetUnreachableMergeTerminatedBy(
      SpvCapabilityShader, SPV_ENV_WEBGPU_0, SpvOpReturn));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("must terminate with OpUnreachable"));
}

std::string GetUnreachableContinueTerminatedBy(SpvCapability cap,
                                               spv_target_env env, SpvOp op) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block entry("entry");
  Block branch("branch", SpvOpBranch);
  Block merge("merge", SpvOpReturn);
  Block target("target", op);

  if (op == SpvOpBranch) target >> branch;

  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (cap == SpvCapabilityShader)
    branch.AppendBody("OpLoopMerge %merge %target None\n");
  if (!spvIsWebGPUEnv(env))
    str += nameOps("branch", "merge", "target", std::make_pair("func", "Main"));

  str += types_consts();
  str += "%func    = OpFunction %voidt None %funct\n";
  str += entry >> branch;
  str += branch >> std::vector<Block>({merge});
  str += merge;
  str += target;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableContinueTerminatedBySpvOpUnreachable) {
  CompileSuccessfully(GetUnreachableContinueTerminatedBy(
      GetParam(), SPV_ENV_UNIVERSAL_1_0, SpvOpUnreachable));
  if (GetParam() == SpvCapabilityShader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("targeted by 0 back-edge blocks"));
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_F(ValidateCFG, UnreachableContinueTerminatedBySpvOpKill) {
  CompileSuccessfully(GetUnreachableContinueTerminatedBy(
      SpvCapabilityShader, SPV_ENV_UNIVERSAL_1_0, SpvOpKill));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("targeted by 0 back-edge blocks"));
}

TEST_P(ValidateCFG, UnreachableContinueTerminatedBySpvOpReturn) {
  CompileSuccessfully(GetUnreachableContinueTerminatedBy(
      GetParam(), SPV_ENV_UNIVERSAL_1_0, SpvOpReturn));
  if (GetParam() == SpvCapabilityShader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("targeted by 0 back-edge blocks"));
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_P(ValidateCFG, UnreachableContinueTerminatedBySpvOpBranch) {
  CompileSuccessfully(GetUnreachableContinueTerminatedBy(
      GetParam(), SPV_ENV_UNIVERSAL_1_0, SpvOpBranch));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, WebGPUUnreachableContinueTerminatedBySpvOpUnreachable) {
  CompileSuccessfully(GetUnreachableContinueTerminatedBy(
      SpvCapabilityShader, SPV_ENV_WEBGPU_0, SpvOpUnreachable));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("For WebGPU, unreachable continue-target must "
                        "terminate with OpBranch.\n  %12 = OpLabel\n"));
}

TEST_F(ValidateCFG, WebGPUUnreachableContinueTerminatedBySpvOpKill) {
  CompileSuccessfully(GetUnreachableContinueTerminatedBy(
      SpvCapabilityShader, SPV_ENV_WEBGPU_0, SpvOpKill));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("For WebGPU, unreachable continue-target must "
                        "terminate with OpBranch.\n  %12 = OpLabel\n"));
}

TEST_F(ValidateCFG, WebGPUUnreachableContinueTerminatedBySpvOpReturn) {
  CompileSuccessfully(GetUnreachableContinueTerminatedBy(
      SpvCapabilityShader, SPV_ENV_WEBGPU_0, SpvOpReturn));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("For WebGPU, unreachable continue-target must "
                        "terminate with OpBranch.\n  %12 = OpLabel\n"));
}

TEST_F(ValidateCFG, WebGPUUnreachableContinueTerminatedBySpvOpBranch) {
  CompileSuccessfully(GetUnreachableContinueTerminatedBy(
      SpvCapabilityShader, SPV_ENV_WEBGPU_0, SpvOpBranch));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

std::string GetUnreachableMergeUnreachableMergeInst(SpvCapability cap,
                                                    spv_target_env env) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block body("body", SpvOpReturn);
  Block entry("entry");
  Block branch("branch", SpvOpBranchConditional);
  Block t("t", SpvOpReturn);
  Block f("f", SpvOpReturn);
  Block merge("merge", SpvOpUnreachable);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (cap == SpvCapabilityShader)
    branch.AppendBody("OpSelectionMerge %merge None\n");
  if (!spvIsWebGPUEnv(env))
    str += nameOps("branch", "merge", std::make_pair("func", "Main"));

  str += types_consts();
  str += "%func    = OpFunction %voidt None %funct\n";
  str += body;
  str += merge;
  str += entry >> branch;
  str += branch >> std::vector<Block>({t, f});
  str += t;
  str += f;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableMergeUnreachableMergeInst) {
  CompileSuccessfully(GetUnreachableMergeUnreachableMergeInst(
      GetParam(), SPV_ENV_UNIVERSAL_1_0));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, WebGPUUnreachableMergeUnreachableMergeInst) {
  CompileSuccessfully(GetUnreachableMergeUnreachableMergeInst(
      SpvCapabilityShader, SPV_ENV_WEBGPU_0));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("must be referenced by a reachable merge instruction"));
}

std::string GetUnreachableContinueUnreachableLoopInst(SpvCapability cap,
                                                      spv_target_env env) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block body("body", SpvOpReturn);
  Block entry("entry");
  Block branch("branch", SpvOpBranch);
  Block merge("merge", SpvOpReturn);
  Block target("target", SpvOpBranch);

  target >> branch;

  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (cap == SpvCapabilityShader)
    branch.AppendBody("OpLoopMerge %merge %target None\n");
  if (!spvIsWebGPUEnv(env))
    str += nameOps("branch", "merge", "target", std::make_pair("func", "Main"));

  str += types_consts();
  str += "%func    = OpFunction %voidt None %funct\n";
  str += body;
  str += target;
  str += merge;
  str += entry >> branch;
  str += branch >> std::vector<Block>({merge});
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableContinueUnreachableLoopInst) {
  CompileSuccessfully(GetUnreachableContinueUnreachableLoopInst(
      GetParam(), SPV_ENV_UNIVERSAL_1_0));
  if (GetParam() == SpvCapabilityShader) {
    // Shader causes additional structured CFG checks that cause a failure.
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Back-edges (1[%branch] -> 3[%target]) can only be "
                          "formed between a block and a loop header."));

  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_F(ValidateCFG, WebGPUUnreachableContinueUnreachableLoopInst) {
  CompileSuccessfully(GetUnreachableContinueUnreachableLoopInst(
      SpvCapabilityShader, SPV_ENV_WEBGPU_0));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("must be referenced by a reachable loop instruction"));
}

std::string GetUnreachableMergeWithComplexBody(SpvCapability cap,
                                               spv_target_env env) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block entry("entry");
  Block branch("branch", SpvOpBranchConditional);
  Block t("t", SpvOpReturn);
  Block f("f", SpvOpReturn);
  Block merge("merge", SpvOpUnreachable);

  entry.AppendBody(spvIsWebGPUEnv(env)
                       ? "%dummy   = OpVariable %intptrt Function %two\n"
                       : "%dummy   = OpVariable %intptrt Function\n");
  entry.AppendBody("%cond    = OpSLessThan %boolt %one %two\n");
  merge.AppendBody("OpStore %dummy %one\n");

  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (cap == SpvCapabilityShader)
    branch.AppendBody("OpSelectionMerge %merge None\n");
  if (!spvIsWebGPUEnv(env))
    str += nameOps("branch", "merge", std::make_pair("func", "Main"));

  str += types_consts();
  str += "%intptrt = OpTypePointer Function %intt\n";
  str += "%func    = OpFunction %voidt None %funct\n";
  str += entry >> branch;
  str += branch >> std::vector<Block>({t, f});
  str += t;
  str += f;
  str += merge;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableMergeWithComplexBody) {
  CompileSuccessfully(
      GetUnreachableMergeWithComplexBody(GetParam(), SPV_ENV_UNIVERSAL_1_0));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, WebGPUUnreachableMergeWithComplexBody) {
  CompileSuccessfully(GetUnreachableMergeWithComplexBody(SpvCapabilityShader,
                                                         SPV_ENV_WEBGPU_0));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("must only contain an OpLabel and OpUnreachable instruction"));
}

std::string GetUnreachableContinueWithComplexBody(SpvCapability cap,
                                                  spv_target_env env) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block entry("entry");
  Block branch("branch", SpvOpBranch);
  Block merge("merge", SpvOpReturn);
  Block target("target", SpvOpBranch);

  target >> branch;

  entry.AppendBody(spvIsWebGPUEnv(env)
                       ? "%dummy   = OpVariable %intptrt Function %two\n"
                       : "%dummy   = OpVariable %intptrt Function\n");
  target.AppendBody("OpStore %dummy %one\n");

  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (cap == SpvCapabilityShader)
    branch.AppendBody("OpLoopMerge %merge %target None\n");
  if (!spvIsWebGPUEnv(env))
    str += nameOps("branch", "merge", "target", std::make_pair("func", "Main"));

  str += types_consts();
  str += "%intptrt = OpTypePointer Function %intt\n";
  str += "%func    = OpFunction %voidt None %funct\n";
  str += entry >> branch;
  str += branch >> std::vector<Block>({merge});
  str += merge;
  str += target;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableContinueWithComplexBody) {
  CompileSuccessfully(
      GetUnreachableContinueWithComplexBody(GetParam(), SPV_ENV_UNIVERSAL_1_0));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, WebGPUUnreachableContinueWithComplexBody) {
  CompileSuccessfully(GetUnreachableContinueWithComplexBody(SpvCapabilityShader,
                                                            SPV_ENV_WEBGPU_0));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("must only contain an OpLabel and an OpBranch instruction"));
}

std::string GetUnreachableMergeWithBranchUse(SpvCapability cap,
                                             spv_target_env env) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block entry("entry");
  Block branch("branch", SpvOpBranchConditional);
  Block t("t", SpvOpBranch);
  Block f("f", SpvOpReturn);
  Block merge("merge", SpvOpUnreachable);

  entry.AppendBody("%cond    = OpSLessThan %boolt %one %two\n");

  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (cap == SpvCapabilityShader)
    branch.AppendBody("OpSelectionMerge %merge None\n");
  if (!spvIsWebGPUEnv(env))
    str += nameOps("branch", "merge", std::make_pair("func", "Main"));

  str += types_consts();
  str += "%func    = OpFunction %voidt None %funct\n";
  str += entry >> branch;
  str += branch >> std::vector<Block>({t, f});
  str += t >> merge;
  str += f;
  str += merge;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableMergeWithBranchUse) {
  CompileSuccessfully(
      GetUnreachableMergeWithBranchUse(GetParam(), SPV_ENV_UNIVERSAL_1_0));
  EXPECT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Statically reachable blocks cannot be terminated by OpUnreachable"));
}

TEST_F(ValidateCFG, WebGPUUnreachableMergeWithBranchUse) {
  CompileSuccessfully(
      GetUnreachableMergeWithBranchUse(SpvCapabilityShader, SPV_ENV_WEBGPU_0));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("cannot be the target of a branch."));
}

std::string GetUnreachableMergeWithMultipleUses(SpvCapability cap,
                                                spv_target_env env) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block entry("entry");
  Block branch("branch", SpvOpBranchConditional);
  Block t("t", SpvOpReturn);
  Block f("f", SpvOpReturn);
  Block merge("merge", SpvOpUnreachable);
  Block duplicate("duplicate", SpvOpBranchConditional);

  entry.AppendBody("%cond    = OpSLessThan %boolt %one %two\n");

  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (cap == SpvCapabilityShader) {
    branch.AppendBody("OpSelectionMerge %merge None\n");
    duplicate.AppendBody("OpSelectionMerge %merge None\n");
  }
  if (!spvIsWebGPUEnv(env))
    str += nameOps("branch", "merge", std::make_pair("func", "Main"));

  str += types_consts();
  str += "%func    = OpFunction %voidt None %funct\n";
  str += entry >> branch;
  str += branch >> std::vector<Block>({t, f});
  str += duplicate >> std::vector<Block>({t, f});
  str += t;
  str += f;
  str += merge;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableMergeWithMultipleUses) {
  CompileSuccessfully(
      GetUnreachableMergeWithMultipleUses(GetParam(), SPV_ENV_UNIVERSAL_1_0));
  if (GetParam() == SpvCapabilityShader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("is already a merge block for another header"));
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_F(ValidateCFG, WebGPUUnreachableMergeWithMultipleUses) {
  CompileSuccessfully(GetUnreachableMergeWithMultipleUses(SpvCapabilityShader,
                                                          SPV_ENV_WEBGPU_0));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("is already a merge block for another header"));
}

std::string GetUnreachableContinueWithBranchUse(SpvCapability cap,
                                                spv_target_env env) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block entry("entry");
  Block foo("foo", SpvOpBranch);
  Block branch("branch", SpvOpBranch);
  Block merge("merge", SpvOpReturn);
  Block target("target", SpvOpBranch);

  foo >> target;
  target >> branch;

  entry.AppendBody(spvIsWebGPUEnv(env)
                       ? "%dummy   = OpVariable %intptrt Function %two\n"
                       : "%dummy   = OpVariable %intptrt Function\n");

  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (cap == SpvCapabilityShader)
    branch.AppendBody("OpLoopMerge %merge %target None\n");
  if (!spvIsWebGPUEnv(env))
    str += nameOps("branch", "merge", "target", std::make_pair("func", "Main"));

  str += types_consts();
  str += "%intptrt = OpTypePointer Function %intt\n";
  str += "%func    = OpFunction %voidt None %funct\n";
  str += entry >> branch;
  str += branch >> std::vector<Block>({merge});
  str += merge;
  str += target;
  str += foo;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableContinueWithBranchUse) {
  CompileSuccessfully(
      GetUnreachableContinueWithBranchUse(GetParam(), SPV_ENV_UNIVERSAL_1_0));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, WebGPUUnreachableContinueWithBranchUse) {
  CompileSuccessfully(GetUnreachableContinueWithBranchUse(SpvCapabilityShader,
                                                          SPV_ENV_WEBGPU_0));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("cannot be the target of a branch."));
}

std::string GetReachableMergeAndContinue(SpvCapability cap,
                                         spv_target_env env) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block entry("entry");
  Block branch("branch", SpvOpBranch);
  Block merge("merge", SpvOpReturn);
  Block target("target", SpvOpBranch);
  Block body("body", SpvOpBranchConditional);
  Block t("t", SpvOpBranch);
  Block f("f", SpvOpBranch);

  target >> branch;
  body.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  t >> merge;
  f >> target;

  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (cap == SpvCapabilityShader) {
    branch.AppendBody("OpLoopMerge %merge %target None\n");
    body.AppendBody("OpSelectionMerge %target None\n");
  }

  if (!spvIsWebGPUEnv(env))
    str += nameOps("branch", "merge", "target", "body", "t", "f",
                   std::make_pair("func", "Main"));

  str += types_consts();
  str += "%func    = OpFunction %voidt None %funct\n";
  str += entry >> branch;
  str += branch >> std::vector<Block>({body});
  str += body >> std::vector<Block>({t, f});
  str += t;
  str += f;
  str += merge;
  str += target;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, ReachableMergeAndContinue) {
  CompileSuccessfully(
      GetReachableMergeAndContinue(GetParam(), SPV_ENV_UNIVERSAL_1_0));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, WebGPUReachableMergeAndContinue) {
  CompileSuccessfully(
      GetReachableMergeAndContinue(SpvCapabilityShader, SPV_ENV_WEBGPU_0));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_WEBGPU_0));
}

std::string GetUnreachableMergeAndContinue(SpvCapability cap,
                                           spv_target_env env) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block entry("entry");
  Block branch("branch", SpvOpBranch);
  Block merge("merge", SpvOpReturn);
  Block target("target", SpvOpBranch);
  Block body("body", SpvOpBranchConditional);
  Block t("t", SpvOpReturn);
  Block f("f", SpvOpReturn);

  target >> branch;
  body.SetBody("%cond    = OpSLessThan %boolt %one %two\n");

  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (cap == SpvCapabilityShader) {
    branch.AppendBody("OpLoopMerge %merge %target None\n");
    body.AppendBody("OpSelectionMerge %target None\n");
  }

  if (!spvIsWebGPUEnv(env))
    str += nameOps("branch", "merge", "target", "body", "t", "f",
                   std::make_pair("func", "Main"));

  str += types_consts();
  str += "%func    = OpFunction %voidt None %funct\n";
  str += entry >> branch;
  str += branch >> std::vector<Block>({body});
  str += body >> std::vector<Block>({t, f});
  str += t;
  str += f;
  str += merge;
  str += target;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableMergeAndContinue) {
  CompileSuccessfully(
      GetUnreachableMergeAndContinue(GetParam(), SPV_ENV_UNIVERSAL_1_0));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, WebGPUUnreachableMergeAndContinue) {
  CompileSuccessfully(
      GetUnreachableMergeAndContinue(SpvCapabilityShader, SPV_ENV_WEBGPU_0));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("unreachable merge-blocks must terminate with OpUnreachable"));
}

std::string GetUnreachableBlock(SpvCapability cap, spv_target_env env) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block entry("entry");
  Block unreachable("unreachable");
  Block exit("exit", SpvOpReturn);

  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (!spvIsWebGPUEnv(env))
    str += nameOps("unreachable", "exit", std::make_pair("func", "Main"));
  str += types_consts();
  str += "%func    = OpFunction %voidt None %funct\n";
  str += entry >> exit;
  str += unreachable >> exit;
  str += exit;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableBlock) {
  CompileSuccessfully(GetUnreachableBlock(GetParam(), SPV_ENV_UNIVERSAL_1_0));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, WebGPUUnreachableBlock) {
  CompileSuccessfully(
      GetUnreachableBlock(SpvCapabilityShader, SPV_ENV_WEBGPU_0));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("all blocks must be reachable"));
}

std::string GetUnreachableBranch(SpvCapability cap, spv_target_env env) {
  std::string header =
      spvIsWebGPUEnv(env) ? GetWebGPUHeader() : GetDefaultHeader(cap);

  Block entry("entry");
  Block unreachable("unreachable", SpvOpBranchConditional);
  Block unreachablechildt("unreachablechildt");
  Block unreachablechildf("unreachablechildf");
  Block merge("merge");
  Block exit("exit", SpvOpReturn);

  unreachable.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (cap == SpvCapabilityShader)
    unreachable.AppendBody("OpSelectionMerge %merge None\n");

  std::string str = header;
  if (spvIsWebGPUEnv(env)) {
    str +=
        "OpEntryPoint Fragment %func \"func\"\n"
        "OpExecutionMode %func OriginUpperLeft\n";
  }
  if (!spvIsWebGPUEnv(env))
    str += nameOps("unreachable", "exit", std::make_pair("func", "Main"));
  str += types_consts();
  str += "%func    = OpFunction %voidt None %funct\n";

  str += entry >> exit;
  str +=
      unreachable >> std::vector<Block>({unreachablechildt, unreachablechildf});
  str += unreachablechildt >> merge;
  str += unreachablechildf >> merge;
  str += merge >> exit;
  str += exit;
  str += "OpFunctionEnd\n";

  return str;
}

TEST_P(ValidateCFG, UnreachableBranch) {
  CompileSuccessfully(GetUnreachableBranch(GetParam(), SPV_ENV_UNIVERSAL_1_0));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, WebGPUUnreachableBranch) {
  CompileSuccessfully(
      GetUnreachableBranch(SpvCapabilityShader, SPV_ENV_WEBGPU_0));
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("all blocks must be reachable"));
}

TEST_P(ValidateCFG, EmptyFunction) {
  std::string str = GetDefaultHeader(GetParam()) + std::string(types_consts()) +
                    R"(%func    = OpFunction %voidt None %funct
                  %l = OpLabel
                  OpReturn
                  OpFunctionEnd)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateCFG, SingleBlockLoop) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop("loop", SpvOpBranchConditional);
  Block exit("exit", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) loop.AppendBody("OpLoopMerge %exit %loop None\n");

  std::string str = GetDefaultHeader(GetParam()) + std::string(types_consts()) +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> std::vector<Block>({loop, exit});
  str += exit;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateCFG, NestedLoops) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop1("loop1");
  Block loop1_cont_break_block("loop1_cont_break_block",
                               SpvOpBranchConditional);
  Block loop2("loop2", SpvOpBranchConditional);
  Block loop2_merge("loop2_merge");
  Block loop1_merge("loop1_merge");
  Block exit("exit", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) {
    loop1.SetBody("OpLoopMerge %loop1_merge %loop2 None\n");
    loop2.SetBody("OpLoopMerge %loop2_merge %loop2 None\n");
  }

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("loop2", "loop2_merge") + types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop1;
  str += loop1 >> loop1_cont_break_block;
  str += loop1_cont_break_block >> std::vector<Block>({loop1_merge, loop2});
  str += loop2 >> std::vector<Block>({loop2, loop2_merge});
  str += loop2_merge >> loop1;
  str += loop1_merge >> exit;
  str += exit;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateCFG, NestedSelection) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  const int N = 256;
  std::vector<Block> if_blocks;
  std::vector<Block> merge_blocks;
  Block inner("inner");

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");

  if_blocks.emplace_back("if0", SpvOpBranchConditional);

  if (is_shader) if_blocks[0].SetBody("OpSelectionMerge %if_merge0 None\n");
  merge_blocks.emplace_back("if_merge0", SpvOpReturn);

  for (int i = 1; i < N; i++) {
    std::stringstream ss;
    ss << i;
    if_blocks.emplace_back("if" + ss.str(), SpvOpBranchConditional);
    if (is_shader)
      if_blocks[i].SetBody("OpSelectionMerge %if_merge" + ss.str() + " None\n");
    merge_blocks.emplace_back("if_merge" + ss.str(), SpvOpBranch);
  }
  std::string str = GetDefaultHeader(GetParam()) + std::string(types_consts()) +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> if_blocks[0];
  for (int i = 0; i < N - 1; i++) {
    str +=
        if_blocks[i] >> std::vector<Block>({if_blocks[i + 1], merge_blocks[i]});
  }
  str += if_blocks.back() >> std::vector<Block>({inner, merge_blocks.back()});
  str += inner >> merge_blocks.back();
  for (int i = N - 1; i > 0; i--) {
    str += merge_blocks[i] >> merge_blocks[i - 1];
  }
  str += merge_blocks[0];
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateCFG, BackEdgeBlockDoesntPostDominateContinueTargetBad) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop1("loop1", SpvOpBranchConditional);
  Block loop2("loop2", SpvOpBranchConditional);
  Block loop2_merge("loop2_merge", SpvOpBranchConditional);
  Block be_block("be_block");
  Block exit("exit", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) {
    loop1.SetBody("OpLoopMerge %exit %loop2_merge None\n");
    loop2.SetBody("OpLoopMerge %loop2_merge %loop2 None\n");
  }

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("loop1", "loop2", "be_block", "loop2_merge") +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop1;
  str += loop1 >> std::vector<Block>({loop2, exit});
  str += loop2 >> std::vector<Block>({loop2, loop2_merge});
  str += loop2_merge >> std::vector<Block>({be_block, exit});
  str += be_block >> loop1;
  str += exit;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  if (GetParam() == SpvCapabilityShader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                MatchesRegex("The continue construct with the continue target "
                             ".\\[%loop2_merge\\] is not post dominated by the "
                             "back-edge block .\\[%be_block\\]\n"
                             "  %be_block = OpLabel\n"));
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_P(ValidateCFG, BranchingToNonLoopHeaderBlockBad) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block split("split", SpvOpBranchConditional);
  Block t("t");
  Block f("f");
  Block exit("exit", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) split.SetBody("OpSelectionMerge %exit None\n");

  std::string str = GetDefaultHeader(GetParam()) + nameOps("split", "f") +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> split;
  str += split >> std::vector<Block>({t, f});
  str += t >> exit;
  str += f >> split;
  str += exit;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  if (is_shader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(
        getDiagnosticString(),
        MatchesRegex("Back-edges \\(.\\[%f\\] -> .\\[%split\\]\\) can only "
                     "be formed between a block and a loop header.\n"
                     "  %f = OpLabel\n"));
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_P(ValidateCFG, BranchingToSameNonLoopHeaderBlockBad) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block split("split", SpvOpBranchConditional);
  Block exit("exit", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) split.SetBody("OpSelectionMerge %exit None\n");

  std::string str = GetDefaultHeader(GetParam()) + nameOps("split") +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> split;
  str += split >> std::vector<Block>({split, exit});
  str += exit;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  if (is_shader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(
        getDiagnosticString(),
        MatchesRegex(
            "Back-edges \\(.\\[%split\\] -> .\\[%split\\]\\) can only be "
            "formed between a block and a loop header.\n  %split = OpLabel\n"));
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_P(ValidateCFG, MultipleBackEdgeBlocksToLoopHeaderBad) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop("loop", SpvOpBranchConditional);
  Block back0("back0");
  Block back1("back1");
  Block merge("merge", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) loop.SetBody("OpLoopMerge %merge %back0 None\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("loop", "back0", "back1") + types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> std::vector<Block>({back0, back1});
  str += back0 >> loop;
  str += back1 >> loop;
  str += merge;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  if (is_shader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(
        getDiagnosticString(),
        MatchesRegex(
            "Loop header .\\[%loop\\] is targeted by 2 back-edge blocks but "
            "the standard requires exactly one\n  %loop = OpLabel\n"))
        << str;
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_P(ValidateCFG, ContinueTargetMustBePostDominatedByBackEdge) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop("loop", SpvOpBranchConditional);
  Block cheader("cheader", SpvOpBranchConditional);
  Block be_block("be_block");
  Block merge("merge", SpvOpReturn);
  Block exit("exit", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) loop.SetBody("OpLoopMerge %merge %cheader None\n");

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("cheader", "be_block") + types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> std::vector<Block>({cheader, merge});
  str += cheader >> std::vector<Block>({exit, be_block});
  str += exit;  //  Branches out of a continue construct
  str += be_block >> loop;
  str += merge;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  if (is_shader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                MatchesRegex("The continue construct with the continue target "
                             ".\\[%cheader\\] is not post dominated by the "
                             "back-edge block .\\[%be_block\\]\n"
                             "  %be_block = OpLabel\n"));
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_P(ValidateCFG, BranchOutOfConstructToMergeBad) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop("loop", SpvOpBranchConditional);
  Block cont("cont", SpvOpBranchConditional);
  Block merge("merge", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) loop.SetBody("OpLoopMerge %merge %loop None\n");

  std::string str = GetDefaultHeader(GetParam()) + nameOps("cont", "loop") +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> std::vector<Block>({cont, merge});
  str += cont >> std::vector<Block>({loop, merge});
  str += merge;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  if (is_shader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                MatchesRegex("The continue construct with the continue target "
                             ".\\[%loop\\] is not post dominated by the "
                             "back-edge block .\\[%cont\\]\n"
                             "  %cont = OpLabel\n"))
        << str;
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_P(ValidateCFG, BranchOutOfConstructBad) {
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop("loop", SpvOpBranchConditional);
  Block cont("cont", SpvOpBranchConditional);
  Block merge("merge");
  Block exit("exit", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) loop.SetBody("OpLoopMerge %merge %loop None\n");

  std::string str = GetDefaultHeader(GetParam()) + nameOps("cont", "loop") +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> std::vector<Block>({cont, merge});
  str += cont >> std::vector<Block>({loop, exit});
  str += merge >> exit;
  str += exit;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  if (is_shader) {
    ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(),
                MatchesRegex("The continue construct with the continue target "
                             ".\\[%loop\\] is not post dominated by the "
                             "back-edge block .\\[%cont\\]\n"
                             "  %cont = OpLabel\n"));
  } else {
    ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
  }
}

TEST_F(ValidateCFG, OpSwitchToUnreachableBlock) {
  Block entry("entry", SpvOpSwitch);
  Block case0("case0");
  Block case1("case1");
  Block case2("case2");
  Block def("default", SpvOpUnreachable);
  Block phi("phi", SpvOpReturn);

  std::string str = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %id
OpExecutionMode %main LocalSize 1 1 1
OpSource GLSL 430
OpName %main "main"
OpDecorate %id BuiltIn GlobalInvocationId
%void      = OpTypeVoid
%voidf     = OpTypeFunction %void
%u32       = OpTypeInt 32 0
%f32       = OpTypeFloat 32
%uvec3     = OpTypeVector %u32 3
%fvec3     = OpTypeVector %f32 3
%uvec3ptr  = OpTypePointer Input %uvec3
%id        = OpVariable %uvec3ptr Input
%one       = OpConstant %u32 1
%three     = OpConstant %u32 3
%main      = OpFunction %void None %voidf
)";

  entry.SetBody(
      "%idval    = OpLoad %uvec3 %id\n"
      "%x        = OpCompositeExtract %u32 %idval 0\n"
      "%selector = OpUMod %u32 %x %three\n"
      "OpSelectionMerge %phi None\n");
  str += entry >> std::vector<Block>({def, case0, case1, case2});
  str += case1 >> phi;
  str += def;
  str += phi;
  str += case0 >> phi;
  str += case2 >> phi;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Statically reachable blocks cannot be terminated by "
                        "OpUnreachable"));
}

TEST_F(ValidateCFG, LoopWithZeroBackEdgesBad) {
  std::string str = R"(
           OpCapability Shader
           OpMemoryModel Logical GLSL450
           OpEntryPoint Fragment %main "main"
           OpExecutionMode %main OriginUpperLeft
           OpName %loop "loop"
%voidt   = OpTypeVoid
%funct   = OpTypeFunction %voidt
%main    = OpFunction %voidt None %funct
%loop    = OpLabel
           OpLoopMerge %exit %exit None
           OpBranch %exit
%exit    = OpLabel
           OpReturn
           OpFunctionEnd
)";
  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      MatchesRegex("Loop header .\\[%loop\\] is targeted by "
                   "0 back-edge blocks but the standard requires exactly "
                   "one\n  %loop = OpLabel\n"));
}

TEST_F(ValidateCFG, LoopWithBackEdgeFromUnreachableContinueConstructGood) {
  std::string str = R"(
           OpCapability Shader
           OpMemoryModel Logical GLSL450
           OpEntryPoint Fragment %main "main"
           OpExecutionMode %main OriginUpperLeft
           OpName %loop "loop"
%voidt   = OpTypeVoid
%funct   = OpTypeFunction %voidt
%floatt  = OpTypeFloat 32
%boolt   = OpTypeBool
%one     = OpConstant %floatt 1
%two     = OpConstant %floatt 2
%main    = OpFunction %voidt None %funct
%entry   = OpLabel
           OpBranch %loop
%loop    = OpLabel
           OpLoopMerge %exit %cont None
           OpBranch %16
%16      = OpLabel
%cond    = OpFOrdLessThan %boolt %one %two
           OpBranchConditional %cond %body %exit
%body    = OpLabel
           OpReturn
%cont    = OpLabel   ; Reachable only from OpLoopMerge ContinueTarget parameter
           OpBranch %loop ; Should be considered a back-edge
%exit    = OpLabel
           OpReturn
           OpFunctionEnd
)";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions()) << getDiagnosticString();
}

TEST_P(ValidateCFG,
       NestedConstructWithUnreachableMergeBlockBranchingToOuterMergeBlock) {
  // Test for https://github.com/KhronosGroup/SPIRV-Tools/issues/297
  // The nested construct has an unreachable merge block.  In the
  // augmented CFG that merge block
  // we still determine that the
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry", SpvOpBranchConditional);
  Block inner_head("inner_head", SpvOpBranchConditional);
  Block inner_true("inner_true", SpvOpReturn);
  Block inner_false("inner_false", SpvOpReturn);
  Block inner_merge("inner_merge");
  Block exit("exit", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) {
    entry.AppendBody("OpSelectionMerge %exit None\n");
    inner_head.SetBody("OpSelectionMerge %inner_merge None\n");
  }

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("entry", "inner_merge", "exit") + types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> std::vector<Block>({inner_head, exit});
  str += inner_head >> std::vector<Block>({inner_true, inner_false});
  str += inner_true;
  str += inner_false;
  str += inner_merge >> exit;
  str += exit;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions()) << getDiagnosticString();
}

TEST_P(ValidateCFG, ContinueTargetCanBeMergeBlockForNestedStructureGood) {
  // This example is valid.  It shows that the validator can't just add
  // an edge from the loop head to the continue target.  If that edge
  // is added, then the "if_merge" block is both the continue target
  // for the loop and also the merge block for the nested selection, but
  // then it wouldn't be dominated by "if_head", the header block for the
  // nested selection.
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop("loop");
  Block if_head("if_head", SpvOpBranchConditional);
  Block if_true("if_true");
  Block if_merge("if_merge", SpvOpBranchConditional);
  Block merge("merge", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) {
    loop.SetBody("OpLoopMerge %merge %if_merge None\n");
    if_head.SetBody("OpSelectionMerge %if_merge None\n");
  }

  std::string str =
      GetDefaultHeader(GetParam()) +
      nameOps("entry", "loop", "if_head", "if_true", "if_merge", "merge") +
      types_consts() + "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> if_head;
  str += if_head >> std::vector<Block>({if_true, if_merge});
  str += if_true >> if_merge;
  str += if_merge >> std::vector<Block>({loop, merge});
  str += merge;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions()) << getDiagnosticString();
}

TEST_P(ValidateCFG, SingleLatchBlockMultipleBranchesToLoopHeader) {
  // This test case ensures we allow both branches of a loop latch block
  // to go back to the loop header.  It still counts as a single back edge.
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop("loop", SpvOpBranchConditional);
  Block latch("latch", SpvOpBranchConditional);
  Block merge("merge", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) {
    loop.SetBody("OpLoopMerge %merge %latch None\n");
  }

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("entry", "loop", "latch", "merge") +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> std::vector<Block>({latch, merge});
  str += latch >> std::vector<Block>({loop, loop});  // This is the key
  str += merge;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions())
      << str << getDiagnosticString();
}

TEST_P(ValidateCFG, SingleLatchBlockHeaderContinueTargetIsItselfGood) {
  // This test case ensures we don't count a Continue Target from a loop
  // header to itself as a self-loop when computing back edges.
  // Also, it detects that there is an edge from %latch to the pseudo-exit
  // node, rather than from %loop.  In particular, it detects that we
  // have used the *reverse* textual order of blocks when computing
  // predecessor traversal roots.
  bool is_shader = GetParam() == SpvCapabilityShader;
  Block entry("entry");
  Block loop("loop");
  Block latch("latch");
  Block merge("merge", SpvOpReturn);

  entry.SetBody("%cond    = OpSLessThan %boolt %one %two\n");
  if (is_shader) {
    loop.SetBody("OpLoopMerge %merge %loop None\n");
  }

  std::string str = GetDefaultHeader(GetParam()) +
                    nameOps("entry", "loop", "latch", "merge") +
                    types_consts() +
                    "%func    = OpFunction %voidt None %funct\n";

  str += entry >> loop;
  str += loop >> latch;
  str += latch >> loop;
  str += merge;
  str += "OpFunctionEnd";

  CompileSuccessfully(str);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions())
      << str << getDiagnosticString();
}

// Unit test to check the case where a basic block is the entry block of 2
// different constructs. In this case, the basic block is the entry block of a
// continue construct as well as a selection construct. See issue# 517 for more
// details.
TEST_F(ValidateCFG, BasicBlockIsEntryBlockOfTwoConstructsGood) {
  std::string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
       %void = OpTypeVoid
       %bool = OpTypeBool
        %int = OpTypeInt 32 1
  %void_func = OpTypeFunction %void
      %int_0 = OpConstant %int 0
    %testfun = OpFunction %void None %void_func
    %label_1 = OpLabel
               OpBranch %start
      %start = OpLabel
       %cond = OpSLessThan %bool %int_0 %int_0
       ;
       ; Note: In this case, the "target" block is both the entry block of
       ;       the continue construct of the loop as well as the entry block of
       ;       the selection construct.
       ;
               OpLoopMerge %loop_merge %target None
               OpBranchConditional %cond %target %loop_merge
 %loop_merge = OpLabel
               OpReturn
     %target = OpLabel
               OpSelectionMerge %selection_merge None
               OpBranchConditional %cond %do_stuff %do_other_stuff
     %do_other_stuff = OpLabel
               OpBranch %selection_merge
     %selection_merge = OpLabel
               OpBranch %start
         %do_stuff = OpLabel
               OpBranch %selection_merge
               OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, OpReturnInNonVoidFunc) {
  std::string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
        %int = OpTypeInt 32 1
   %int_func = OpTypeFunction %int
    %testfun = OpFunction %int None %int_func
    %label_1 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "OpReturn can only be called from a function with void return type.\n"
          "  OpReturn"));
}

TEST_F(ValidateCFG, StructuredCFGBranchIntoSelectionBody) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpSelectionMerge %merge None
OpBranchConditional %true %then %merge
%merge = OpLabel
OpBranch %then
%then = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("branches to the selection construct, but not to the "
                        "selection header <ID> 6\n  %7 = OpLabel"));
}

TEST_F(ValidateCFG, SwitchDefaultOnly) {
  std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpTypeFunction %1
%5 = OpFunction %1 None %4
%6 = OpLabel
OpSelectionMerge %7 None
OpSwitch %3 %7
%7 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, SwitchSingleCase) {
  std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpTypeFunction %1
%5 = OpFunction %1 None %4
%6 = OpLabel
OpSelectionMerge %7 None
OpSwitch %3 %7 0 %8
%8 = OpLabel
OpBranch %7
%7 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, MultipleFallThroughBlocks) {
  std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpTypeFunction %1
%5 = OpTypeBool
%6 = OpConstantTrue %5
%7 = OpFunction %1 None %4
%8 = OpLabel
OpSelectionMerge %9 None
OpSwitch %3 %10 0 %11 1 %12
%10 = OpLabel
OpBranchConditional %6 %11 %12
%11 = OpLabel
OpBranch %9
%12 = OpLabel
OpBranch %9
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Case construct that targets 10[%10] has branches to multiple other "
          "case construct targets 12[%12] and 11[%11]\n  %10 = OpLabel"));
}

TEST_F(ValidateCFG, MultipleFallThroughToDefault) {
  std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpTypeFunction %1
%5 = OpTypeBool
%6 = OpConstantTrue %5
%7 = OpFunction %1 None %4
%8 = OpLabel
OpSelectionMerge %9 None
OpSwitch %3 %10 0 %11 1 %12
%10 = OpLabel
OpBranch %9
%11 = OpLabel
OpBranch %10
%12 = OpLabel
OpBranch %10
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Multiple case constructs have branches to the case construct "
                "that targets 10[%10]\n  %10 = OpLabel"));
}

TEST_F(ValidateCFG, MultipleFallThroughToNonDefault) {
  std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpTypeFunction %1
%5 = OpTypeBool
%6 = OpConstantTrue %5
%7 = OpFunction %1 None %4
%8 = OpLabel
OpSelectionMerge %9 None
OpSwitch %3 %10 0 %11 1 %12
%10 = OpLabel
OpBranch %12
%11 = OpLabel
OpBranch %12
%12 = OpLabel
OpBranch %9
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Multiple case constructs have branches to the case construct "
                "that targets 12[%12]\n  %12 = OpLabel"));
}

TEST_F(ValidateCFG, DuplicateTargetWithFallThrough) {
  std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpTypeFunction %1
%5 = OpTypeBool
%6 = OpConstantTrue %5
%7 = OpFunction %1 None %4
%8 = OpLabel
OpSelectionMerge %9 None
OpSwitch %3 %10 0 %10 1 %11
%10 = OpLabel
OpBranch %11
%11 = OpLabel
OpBranch %9
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, WrongOperandList) {
  std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpTypeFunction %1
%5 = OpTypeBool
%6 = OpConstantTrue %5
%7 = OpFunction %1 None %4
%8 = OpLabel
OpSelectionMerge %9 None
OpSwitch %3 %10 0 %11 1 %12
%10 = OpLabel
OpBranch %9
%12 = OpLabel
OpBranch %11
%11 = OpLabel
OpBranch %9
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Case construct that targets 12[%12] has branches to the case "
                "construct that targets 11[%11], but does not immediately "
                "precede it in the OpSwitch's target list\n"
                "  OpSwitch %uint_0 %10 0 %11 1 %12"));
}

TEST_F(ValidateCFG, WrongOperandListThroughDefault) {
  std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpTypeFunction %1
%5 = OpTypeBool
%6 = OpConstantTrue %5
%7 = OpFunction %1 None %4
%8 = OpLabel
OpSelectionMerge %9 None
OpSwitch %3 %10 0 %11 1 %12
%10 = OpLabel
OpBranch %11
%12 = OpLabel
OpBranch %10
%11 = OpLabel
OpBranch %9
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Case construct that targets 12[%12] has branches to the case "
                "construct that targets 11[%11], but does not immediately "
                "precede it in the OpSwitch's target list\n"
                "  OpSwitch %uint_0 %10 0 %11 1 %12"));
}

TEST_F(ValidateCFG, WrongOperandListNotLast) {
  std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpTypeFunction %1
%5 = OpTypeBool
%6 = OpConstantTrue %5
%7 = OpFunction %1 None %4
%8 = OpLabel
OpSelectionMerge %9 None
OpSwitch %3 %10 0 %11 1 %12 2 %13
%10 = OpLabel
OpBranch %9
%12 = OpLabel
OpBranch %11
%11 = OpLabel
OpBranch %9
%13 = OpLabel
OpBranch %9
%9 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Case construct that targets 12[%12] has branches to the case "
                "construct that targets 11[%11], but does not immediately "
                "precede it in the OpSwitch's target list\n"
                "  OpSwitch %uint_0 %10 0 %11 1 %12 2 %13"));
}

TEST_F(ValidateCFG, GoodUnreachableSwitch) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%5 = OpTypeBool
%6 = OpConstantTrue %5
%7 = OpTypeInt 32 1
%9 = OpConstant %7 0
%2 = OpFunction %3 None %4
%10 = OpLabel
OpSelectionMerge %11 None
OpBranchConditional %6 %12 %13
%12 = OpLabel
OpReturn
%13 = OpLabel
OpReturn
%11 = OpLabel
OpSelectionMerge %14 None
OpSwitch %9 %14 0 %15
%15 = OpLabel
OpBranch %14
%14 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_THAT(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, InvalidCaseExit) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "func"
OpExecutionMode %1 OriginUpperLeft
%2 = OpTypeVoid
%3 = OpTypeInt 32 0
%4 = OpTypeFunction %2
%5 = OpConstant %3 0
%1 = OpFunction %2 None %4
%6 = OpLabel
OpSelectionMerge %7 None
OpSwitch %5 %7 0 %8 1 %9
%8 = OpLabel
OpBranch %10
%9 = OpLabel
OpBranch %10
%10 = OpLabel
OpReturn
%7 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Case construct that targets 8[%8] has invalid branch "
                        "to block 10[%10] (not another case construct, "
                        "corresponding merge, outer loop merge or outer loop "
                        "continue)"));
}

TEST_F(ValidateCFG, GoodCaseExitsToOuterConstructs) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %7 %6 None
OpBranch %3
%3 = OpLabel
OpSelectionMerge %5 None
OpSwitch %int0 %5 0 %4
%4 = OpLabel
OpBranchConditional %true %6 %7
%5 = OpLabel
OpBranchConditional %true %6 %7
%6 = OpLabel
OpBranch %2
%7 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, GoodUnreachableSelection) {
  const std::string text = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%8 = OpTypeFunction %void
%bool = OpTypeBool
%false = OpConstantFalse %bool
%main = OpFunction %void None %8
%15 = OpLabel
OpBranch %16
%16 = OpLabel
OpLoopMerge %17 %18 None
OpBranch %19
%19 = OpLabel
OpBranchConditional %false %21 %17
%21 = OpLabel
OpSelectionMerge %22 None
OpBranchConditional %false %23 %22
%23 = OpLabel
OpBranch %24
%24 = OpLabel
OpLoopMerge %25 %26 None
OpBranch %27
%27 = OpLabel
OpReturn
%26 = OpLabel
OpBranchConditional %false %24 %25
%25 = OpLabel
OpSelectionMerge %28 None
OpBranchConditional %false %18 %28
%28 = OpLabel
OpBranch %22
%22 = OpLabel
OpBranch %18
%18 = OpLabel
OpBranch %16
%17 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, ShaderWithPhiPtr) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
               OpExecutionMode %1 LocalSize 1 1 1
               OpSource HLSL 600
       %bool = OpTypeBool
%_ptr_Function_bool = OpTypePointer Function %bool
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
          %1 = OpFunction %void None %5
          %6 = OpLabel
          %7 = OpVariable %_ptr_Function_bool Function
          %8 = OpVariable %_ptr_Function_bool Function
          %9 = OpUndef %bool
               OpSelectionMerge %10 None
               OpBranchConditional %9 %11 %10
         %11 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %12 = OpPhi %_ptr_Function_bool %7 %6 %8 %11
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Using pointers with OpPhi requires capability "
                        "VariablePointers or VariablePointersStorageBuffer"));
}

TEST_F(ValidateCFG, VarPtrShaderWithPhiPtr) {
  const std::string text = R"(
               OpCapability Shader
               OpCapability VariablePointers
               OpExtension "SPV_KHR_variable_pointers"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
               OpExecutionMode %1 LocalSize 1 1 1
               OpSource HLSL 600
       %bool = OpTypeBool
%_ptr_Function_bool = OpTypePointer Function %bool
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
          %1 = OpFunction %void None %5
          %6 = OpLabel
          %7 = OpVariable %_ptr_Function_bool Function
          %8 = OpVariable %_ptr_Function_bool Function
          %9 = OpUndef %bool
               OpSelectionMerge %10 None
               OpBranchConditional %9 %11 %10
         %11 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %12 = OpPhi %_ptr_Function_bool %7 %6 %8 %11
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, VarPtrStgBufShaderWithPhiStgBufPtr) {
  const std::string text = R"(
               OpCapability Shader
               OpCapability VariablePointersStorageBuffer
               OpExtension "SPV_KHR_variable_pointers"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "main"
               OpExecutionMode %1 LocalSize 1 1 1
               OpSource HLSL 600
       %bool = OpTypeBool
       %float = OpTypeFloat 32
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
          %7 = OpVariable %_ptr_StorageBuffer_float StorageBuffer
          %8 = OpVariable %_ptr_StorageBuffer_float StorageBuffer
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
          %1 = OpFunction %void None %5
          %6 = OpLabel
          %9 = OpUndef %bool
               OpSelectionMerge %10 None
               OpBranchConditional %9 %11 %10
         %11 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %12 = OpPhi %_ptr_StorageBuffer_float %7 %6 %8 %11
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, KernelWithPhiPtr) {
  const std::string text = R"(
               OpCapability Kernel
               OpCapability Addresses
               OpMemoryModel Physical32 OpenCL
               OpEntryPoint Kernel %1 "main"
               OpExecutionMode %1 LocalSize 1 1 1
               OpSource HLSL 600
       %bool = OpTypeBool
%_ptr_Function_bool = OpTypePointer Function %bool
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
          %1 = OpFunction %void None %5
          %6 = OpLabel
          %7 = OpVariable %_ptr_Function_bool Function
          %8 = OpVariable %_ptr_Function_bool Function
          %9 = OpUndef %bool
               OpSelectionMerge %10 None
               OpBranchConditional %9 %11 %10
         %11 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %12 = OpPhi %_ptr_Function_bool %7 %6 %8 %11
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, SwitchTargetMustBeLabel) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "foo"
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
          %1 = OpFunction %void None %5
          %6 = OpLabel
          %7 = OpCopyObject %uint %uint_0
               OpSelectionMerge %8 None
               OpSwitch %uint_0 %8 0 %7
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("'Target Label' operands for OpSwitch must "
                        "be IDs of an OpLabel instruction"));
}

TEST_F(ValidateCFG, BranchTargetMustBeLabel) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "foo"
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
       %void = OpTypeVoid
          %5 = OpTypeFunction %void
          %1 = OpFunction %void None %5
          %2 = OpLabel
          %7 = OpCopyObject %uint %uint_0
               OpBranch %7
          %8 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("'Target Label' operands for OpBranch must "
                        "be the ID of an OpLabel instruction"));
}

TEST_F(ValidateCFG, ReachableOpUnreachableOneBlock) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%func = OpFunction %void None %void_fn
%entry = OpLabel
OpUnreachable
OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Statically reachable blocks cannot be terminated by OpUnreachable"));
}

TEST_F(ValidateCFG, ReachableOpUnreachableOpBranch) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%func = OpFunction %void None %void_fn
%entry = OpLabel
OpBranch %block
%block = OpLabel
OpUnreachable
OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Statically reachable blocks cannot be terminated by OpUnreachable"));
}

TEST_F(ValidateCFG, ReachableOpUnreachableOpBranchConditional) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%bool = OpTypeBool
%undef = OpUndef %bool
%func = OpFunction %void None %void_fn
%entry = OpLabel
OpBranchConditional %undef %block %unreachable
%block = OpLabel
OpReturn
%unreachable = OpLabel
OpUnreachable
OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Statically reachable blocks cannot be terminated by OpUnreachable"));
}

TEST_F(ValidateCFG, ReachableOpUnreachableOpSwitch) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%int = OpTypeInt 32 0
%undef = OpUndef %int
%func = OpFunction %void None %void_fn
%entry = OpLabel
OpSwitch %undef %block1 0 %unreachable 1 %block2
%block1 = OpLabel
OpReturn
%unreachable = OpLabel
OpUnreachable
%block2 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Statically reachable blocks cannot be terminated by OpUnreachable"));
}

TEST_F(ValidateCFG, ReachableOpUnreachableLoop) {
  const std::string text = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%bool = OpTypeBool
%undef = OpUndef %bool
%func = OpFunction %void None %void_fn
%entry = OpLabel
OpBranch %loop
%loop = OpLabel
OpLoopMerge %unreachable %loop None
OpBranchConditional %undef %loop %unreachable
%unreachable = OpLabel
OpUnreachable
OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Statically reachable blocks cannot be terminated by OpUnreachable"));
}

TEST_F(ValidateCFG, UnreachableLoopBadBackedge) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "main"
OpExecutionMode %2 OriginUpperLeft
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%8 = OpTypeBool
%13 = OpConstantTrue %8
%2 = OpFunction %4 None %5
%14 = OpLabel
OpSelectionMerge %15 None
OpBranchConditional %13 %15 %15
%16 = OpLabel
OpLoopMerge %17 %18 None
OpBranch %17
%18 = OpLabel
OpBranch %17
%17 = OpLabel
OpBranch %15
%15 = OpLabel
OpReturn
OpFunctionEnd
)";

  // The back-edge in this test is bad, but the validator fails to identify it
  // because it is in an entirely unreachable section of code. Prior to #2488
  // this code failed an assert in Construct::blocks().
  CompileSuccessfully(text);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateCFG, OneContinueTwoBackedges) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "main"
OpExecutionMode %1 LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%5 = OpTypeFunction %void
%1 = OpFunction %void None %5
%6 = OpLabel
OpBranch %7
%7 = OpLabel
OpLoopMerge %8 %9 None
OpBranch %10
%10 = OpLabel
OpLoopMerge %11 %9 None
OpBranchConditional %true %11 %9
%9 = OpLabel
OpBranchConditional %true %10 %7
%11 = OpLabel
OpBranch %8
%8 = OpLabel
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(text);
  EXPECT_EQ(SPV_ERROR_INVALID_CFG, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("block <ID> 9 branches to the loop construct, but not "
                        "to the loop header <ID> 7"));
}

/// TODO(umar): Nested CFG constructs

}  // namespace
}  // namespace val
}  // namespace spvtools
