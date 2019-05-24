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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "source/opt/build_module.h"
#include "source/opt/cfg.h"
#include "source/opt/ir_context.h"
#include "source/opt/pass.h"
#include "source/opt/propagator.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::UnorderedElementsAre;

class PropagatorTest : public testing::Test {
 protected:
  virtual void TearDown() {
    ctx_.reset(nullptr);
    values_.clear();
    values_vec_.clear();
  }

  void Assemble(const std::string& input) {
    ctx_ = BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, input);
    ASSERT_NE(nullptr, ctx_) << "Assembling failed for shader:\n"
                             << input << "\n";
  }

  bool Propagate(const SSAPropagator::VisitFunction& visit_fn) {
    SSAPropagator propagator(ctx_.get(), visit_fn);
    bool retval = false;
    for (auto& fn : *ctx_->module()) {
      retval |= propagator.Run(&fn);
    }
    return retval;
  }

  const std::vector<uint32_t>& GetValues() {
    values_vec_.clear();
    for (const auto& it : values_) {
      values_vec_.push_back(it.second);
    }
    return values_vec_;
  }

  std::unique_ptr<IRContext> ctx_;
  std::map<uint32_t, uint32_t> values_;
  std::vector<uint32_t> values_vec_;
};

TEST_F(PropagatorTest, LocalPropagate) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outparm
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %x "x"
               OpName %y "y"
               OpName %z "z"
               OpName %outparm "outparm"
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_4 = OpConstant %int 4
      %int_3 = OpConstant %int 3
      %int_1 = OpConstant %int 1
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %3
          %5 = OpLabel
          %x = OpVariable %_ptr_Function_int Function
          %y = OpVariable %_ptr_Function_int Function
          %z = OpVariable %_ptr_Function_int Function
               OpStore %x %int_4
               OpStore %y %int_3
               OpStore %z %int_1
         %20 = OpLoad %int %z
               OpStore %outparm %20
               OpReturn
               OpFunctionEnd
               )";
  Assemble(spv_asm);

  const auto visit_fn = [this](Instruction* instr, BasicBlock** dest_bb) {
    *dest_bb = nullptr;
    if (instr->opcode() == SpvOpStore) {
      uint32_t lhs_id = instr->GetSingleWordOperand(0);
      uint32_t rhs_id = instr->GetSingleWordOperand(1);
      Instruction* rhs_def = ctx_->get_def_use_mgr()->GetDef(rhs_id);
      if (rhs_def->opcode() == SpvOpConstant) {
        uint32_t val = rhs_def->GetSingleWordOperand(2);
        values_[lhs_id] = val;
        return SSAPropagator::kInteresting;
      }
    }
    return SSAPropagator::kVarying;
  };

  EXPECT_TRUE(Propagate(visit_fn));
  EXPECT_THAT(GetValues(), UnorderedElementsAre(4, 3, 1));
}

TEST_F(PropagatorTest, PropagateThroughPhis) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %x %outparm
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %x "x"
               OpName %outparm "outparm"
               OpDecorate %x Flat
               OpDecorate %x Location 0
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
       %bool = OpTypeBool
%_ptr_Function_int = OpTypePointer Function %int
      %int_4 = OpConstant %int 4
      %int_3 = OpConstant %int 3
      %int_1 = OpConstant %int 1
%_ptr_Input_int = OpTypePointer Input %int
          %x = OpVariable %_ptr_Input_int Input
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %3
          %4 = OpLabel
          %5 = OpLoad %int %x
          %6 = OpSGreaterThan %bool %5 %int_3
               OpSelectionMerge %25 None
               OpBranchConditional %6 %22 %23
         %22 = OpLabel
          %7 = OpLoad %int %int_4
               OpBranch %25
         %23 = OpLabel
          %8 = OpLoad %int %int_4
               OpBranch %25
         %25 = OpLabel
         %35 = OpPhi %int %7 %22 %8 %23
               OpStore %outparm %35
               OpReturn
               OpFunctionEnd
               )";

  Assemble(spv_asm);

  Instruction* phi_instr = nullptr;
  const auto visit_fn = [this, &phi_instr](Instruction* instr,
                                           BasicBlock** dest_bb) {
    *dest_bb = nullptr;
    if (instr->opcode() == SpvOpLoad) {
      uint32_t rhs_id = instr->GetSingleWordOperand(2);
      Instruction* rhs_def = ctx_->get_def_use_mgr()->GetDef(rhs_id);
      if (rhs_def->opcode() == SpvOpConstant) {
        uint32_t val = rhs_def->GetSingleWordOperand(2);
        values_[instr->result_id()] = val;
        return SSAPropagator::kInteresting;
      }
    } else if (instr->opcode() == SpvOpPhi) {
      phi_instr = instr;
      SSAPropagator::PropStatus retval;
      for (uint32_t i = 2; i < instr->NumOperands(); i += 2) {
        uint32_t phi_arg_id = instr->GetSingleWordOperand(i);
        auto it = values_.find(phi_arg_id);
        if (it != values_.end()) {
          EXPECT_EQ(it->second, 4u);
          retval = SSAPropagator::kInteresting;
          values_[instr->result_id()] = it->second;
        } else {
          retval = SSAPropagator::kNotInteresting;
          break;
        }
      }
      return retval;
    }

    return SSAPropagator::kVarying;
  };

  EXPECT_TRUE(Propagate(visit_fn));

  // The propagator should've concluded that the Phi instruction has a constant
  // value of 4.
  EXPECT_NE(phi_instr, nullptr);
  EXPECT_EQ(values_[phi_instr->result_id()], 4u);

  EXPECT_THAT(GetValues(), UnorderedElementsAre(4u, 4u, 4u));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
