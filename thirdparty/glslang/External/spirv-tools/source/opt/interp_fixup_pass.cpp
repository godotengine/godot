// Copyright (c) 2021 The Khronos Group Inc.
// Copyright (c) 2021 Valve Corporation
// Copyright (c) 2021 LunarG Inc.
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

#include "source/opt/interp_fixup_pass.h"

#include <set>
#include <string>

#include "source/opt/ir_context.h"
#include "type_manager.h"

namespace spvtools {
namespace opt {
namespace {

// Input Operand Indices
constexpr int kSpvVariableStorageClassInIdx = 0;

// Folding rule function which attempts to replace |op(OpLoad(a),...)|
// by |op(a,...)|, where |op| is one of the GLSLstd450 InterpolateAt*
// instructions. Returns true if replaced, false otherwise.
bool ReplaceInternalInterpolate(IRContext* ctx, Instruction* inst,
                                const std::vector<const analysis::Constant*>&) {
  uint32_t glsl450_ext_inst_id =
      ctx->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  assert(glsl450_ext_inst_id != 0);

  uint32_t ext_opcode = inst->GetSingleWordInOperand(1);

  uint32_t op1_id = inst->GetSingleWordInOperand(2);

  Instruction* load_inst = ctx->get_def_use_mgr()->GetDef(op1_id);
  if (load_inst->opcode() != spv::Op::OpLoad) return false;

  Instruction* base_inst = load_inst->GetBaseAddress();
  USE_ASSERT(base_inst->opcode() == spv::Op::OpVariable &&
             spv::StorageClass(base_inst->GetSingleWordInOperand(
                 kSpvVariableStorageClassInIdx)) == spv::StorageClass::Input &&
             "unexpected interpolant in InterpolateAt*");

  uint32_t ptr_id = load_inst->GetSingleWordInOperand(0);
  uint32_t op2_id = (ext_opcode != GLSLstd450InterpolateAtCentroid)
                        ? inst->GetSingleWordInOperand(3)
                        : 0;

  Instruction::OperandList new_operands;
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {glsl450_ext_inst_id}});
  new_operands.push_back(
      {SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER, {ext_opcode}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {ptr_id}});
  if (op2_id != 0) new_operands.push_back({SPV_OPERAND_TYPE_ID, {op2_id}});

  inst->SetInOperands(std::move(new_operands));
  ctx->UpdateDefUse(inst);
  return true;
}

class InterpFoldingRules : public FoldingRules {
 public:
  explicit InterpFoldingRules(IRContext* ctx) : FoldingRules(ctx) {}

 protected:
  virtual void AddFoldingRules() override {
    uint32_t extension_id =
        context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();

    if (extension_id != 0) {
      ext_rules_[{extension_id, GLSLstd450InterpolateAtCentroid}].push_back(
          ReplaceInternalInterpolate);
      ext_rules_[{extension_id, GLSLstd450InterpolateAtSample}].push_back(
          ReplaceInternalInterpolate);
      ext_rules_[{extension_id, GLSLstd450InterpolateAtOffset}].push_back(
          ReplaceInternalInterpolate);
    }
  }
};

class InterpConstFoldingRules : public ConstantFoldingRules {
 public:
  InterpConstFoldingRules(IRContext* ctx) : ConstantFoldingRules(ctx) {}

 protected:
  virtual void AddFoldingRules() override {}
};

}  // namespace

Pass::Status InterpFixupPass::Process() {
  bool changed = false;

  // Traverse the body of the functions to replace instructions that require
  // the extensions.
  InstructionFolder folder(
      context(),
      std::unique_ptr<InterpFoldingRules>(new InterpFoldingRules(context())),
      MakeUnique<InterpConstFoldingRules>(context()));
  for (Function& func : *get_module()) {
    func.ForEachInst([&changed, &folder](Instruction* inst) {
      if (folder.FoldInstruction(inst)) {
        changed = true;
      }
    });
  }

  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
