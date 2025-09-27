// Copyright (c) 2018 Google LLC
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

#include "source/opt/if_conversion.h"

#include <memory>
#include <vector>

#include "source/opt/value_number_table.h"

namespace spvtools {
namespace opt {

Pass::Status IfConversion::Process() {
  if (!context()->get_feature_mgr()->HasCapability(spv::Capability::Shader)) {
    return Status::SuccessWithoutChange;
  }

  const ValueNumberTable& vn_table = *context()->GetValueNumberTable();
  bool modified = false;
  std::vector<Instruction*> to_kill;
  for (auto& func : *get_module()) {
    DominatorAnalysis* dominators = context()->GetDominatorAnalysis(&func);
    for (auto& block : func) {
      // Check if it is possible for |block| to have phis that can be
      // transformed.
      BasicBlock* common = nullptr;
      if (!CheckBlock(&block, dominators, &common)) continue;

      // Get an insertion point.
      auto iter = block.begin();
      while (iter != block.end() && iter->opcode() == spv::Op::OpPhi) {
        ++iter;
      }

      InstructionBuilder builder(
          context(), &*iter,
          IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
      block.ForEachPhiInst([this, &builder, &modified, &common, &to_kill,
                            dominators, &block, &vn_table](Instruction* phi) {
        // This phi is not compatible, but subsequent phis might be.
        if (!CheckType(phi->type_id())) return;

        // We cannot transform cases where the phi is used by another phi in the
        // same block due to instruction ordering restrictions.
        // TODO(alan-baker): If all inappropriate uses could also be
        // transformed, we could still remove this phi.
        if (!CheckPhiUsers(phi, &block)) return;

        // Identify the incoming values associated with the true and false
        // branches. If |then_block| dominates |inc0| or if the true edge
        // branches straight to this block and |common| is |inc0|, then |inc0|
        // is on the true branch. Otherwise the |inc1| is on the true branch.
        BasicBlock* inc0 = GetIncomingBlock(phi, 0u);
        Instruction* branch = common->terminator();
        uint32_t condition = branch->GetSingleWordInOperand(0u);
        BasicBlock* then_block = GetBlock(branch->GetSingleWordInOperand(1u));
        Instruction* true_value = nullptr;
        Instruction* false_value = nullptr;
        if ((then_block == &block && inc0 == common) ||
            dominators->Dominates(then_block, inc0)) {
          true_value = GetIncomingValue(phi, 0u);
          false_value = GetIncomingValue(phi, 1u);
        } else {
          true_value = GetIncomingValue(phi, 1u);
          false_value = GetIncomingValue(phi, 0u);
        }

        BasicBlock* true_def_block = context()->get_instr_block(true_value);
        BasicBlock* false_def_block = context()->get_instr_block(false_value);

        uint32_t true_vn = vn_table.GetValueNumber(true_value);
        uint32_t false_vn = vn_table.GetValueNumber(false_value);
        if (true_vn != 0 && true_vn == false_vn) {
          Instruction* inst_to_use = nullptr;

          // Try to pick an instruction that is not in a side node.  If we can't
          // pick either the true for false branch as long as they can be
          // legally moved.
          if (!true_def_block ||
              dominators->Dominates(true_def_block, &block)) {
            inst_to_use = true_value;
          } else if (!false_def_block ||
                     dominators->Dominates(false_def_block, &block)) {
            inst_to_use = false_value;
          } else if (CanHoistInstruction(true_value, common, dominators)) {
            inst_to_use = true_value;
          } else if (CanHoistInstruction(false_value, common, dominators)) {
            inst_to_use = false_value;
          }

          if (inst_to_use != nullptr) {
            modified = true;
            HoistInstruction(inst_to_use, common, dominators);
            context()->KillNamesAndDecorates(phi);
            context()->ReplaceAllUsesWith(phi->result_id(),
                                          inst_to_use->result_id());
          }
          return;
        }

        // If either incoming value is defined in a block that does not dominate
        // this phi, then we cannot eliminate the phi with a select.
        // TODO(alan-baker): Perform code motion where it makes sense to enable
        // the transform in this case.
        if (true_def_block && !dominators->Dominates(true_def_block, &block))
          return;

        if (false_def_block && !dominators->Dominates(false_def_block, &block))
          return;

        analysis::Type* data_ty =
            context()->get_type_mgr()->GetType(true_value->type_id());
        if (analysis::Vector* vec_data_ty = data_ty->AsVector()) {
          condition = SplatCondition(vec_data_ty, condition, &builder);
        }

        Instruction* select = builder.AddSelect(phi->type_id(), condition,
                                                true_value->result_id(),
                                                false_value->result_id());
        context()->get_def_use_mgr()->AnalyzeInstDefUse(select);
        select->UpdateDebugInfoFrom(phi);
        context()->ReplaceAllUsesWith(phi->result_id(), select->result_id());
        to_kill.push_back(phi);
        modified = true;

        return;
      });
    }
  }

  for (auto inst : to_kill) {
    context()->KillInst(inst);
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool IfConversion::CheckBlock(BasicBlock* block, DominatorAnalysis* dominators,
                              BasicBlock** common) {
  const std::vector<uint32_t>& preds = cfg()->preds(block->id());

  // TODO(alan-baker): Extend to more than two predecessors
  if (preds.size() != 2) return false;

  BasicBlock* inc0 = context()->get_instr_block(preds[0]);
  if (dominators->Dominates(block, inc0)) return false;

  BasicBlock* inc1 = context()->get_instr_block(preds[1]);
  if (dominators->Dominates(block, inc1)) return false;

  if (inc0 == inc1) {
    // If the predecessor blocks are the same, then there is only 1 value for
    // the OpPhi.  Other transformation should be able to simplify that.
    return false;
  }
  // All phis will have the same common dominator, so cache the result
  // for this block. If there is no common dominator, then we cannot transform
  // any phi in this basic block.
  *common = dominators->CommonDominator(inc0, inc1);
  if (!*common || cfg()->IsPseudoEntryBlock(*common)) return false;
  Instruction* branch = (*common)->terminator();
  if (branch->opcode() != spv::Op::OpBranchConditional) return false;
  auto merge = (*common)->GetMergeInst();
  if (!merge || merge->opcode() != spv::Op::OpSelectionMerge) return false;
  if (spv::SelectionControlMask(merge->GetSingleWordInOperand(1)) ==
      spv::SelectionControlMask::DontFlatten) {
    return false;
  }
  if ((*common)->MergeBlockIdIfAny() != block->id()) return false;

  return true;
}

bool IfConversion::CheckPhiUsers(Instruction* phi, BasicBlock* block) {
  return get_def_use_mgr()->WhileEachUser(
      phi, [block, this](Instruction* user) {
        if (user->opcode() == spv::Op::OpPhi &&
            context()->get_instr_block(user) == block)
          return false;
        return true;
      });
}

uint32_t IfConversion::SplatCondition(analysis::Vector* vec_data_ty,
                                      uint32_t cond,
                                      InstructionBuilder* builder) {
  // If the data inputs to OpSelect are vectors, the condition for
  // OpSelect must be a boolean vector with the same number of
  // components. So splat the condition for the branch into a vector
  // type.
  analysis::Bool bool_ty;
  analysis::Vector bool_vec_ty(&bool_ty, vec_data_ty->element_count());
  uint32_t bool_vec_id =
      context()->get_type_mgr()->GetTypeInstruction(&bool_vec_ty);
  std::vector<uint32_t> ids(vec_data_ty->element_count(), cond);
  return builder->AddCompositeConstruct(bool_vec_id, ids)->result_id();
}

bool IfConversion::CheckType(uint32_t id) {
  Instruction* type = get_def_use_mgr()->GetDef(id);
  spv::Op op = type->opcode();
  if (spvOpcodeIsScalarType(op) || op == spv::Op::OpTypePointer ||
      op == spv::Op::OpTypeVector)
    return true;
  return false;
}

BasicBlock* IfConversion::GetBlock(uint32_t id) {
  return context()->get_instr_block(get_def_use_mgr()->GetDef(id));
}

BasicBlock* IfConversion::GetIncomingBlock(Instruction* phi,
                                           uint32_t predecessor) {
  uint32_t in_index = 2 * predecessor + 1;
  return GetBlock(phi->GetSingleWordInOperand(in_index));
}

Instruction* IfConversion::GetIncomingValue(Instruction* phi,
                                            uint32_t predecessor) {
  uint32_t in_index = 2 * predecessor;
  return get_def_use_mgr()->GetDef(phi->GetSingleWordInOperand(in_index));
}

void IfConversion::HoistInstruction(Instruction* inst, BasicBlock* target_block,
                                    DominatorAnalysis* dominators) {
  BasicBlock* inst_block = context()->get_instr_block(inst);
  if (!inst_block) {
    // This is in the header, and dominates everything.
    return;
  }

  if (dominators->Dominates(inst_block, target_block)) {
    // Already in position.  No work to do.
    return;
  }

  assert(inst->IsOpcodeCodeMotionSafe() &&
         "Trying to move an instruction that is not safe to move.");

  // First hoist all instructions it depends on.
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  inst->ForEachInId(
      [this, target_block, def_use_mgr, dominators](uint32_t* id) {
        Instruction* operand_inst = def_use_mgr->GetDef(*id);
        HoistInstruction(operand_inst, target_block, dominators);
      });

  Instruction* insertion_pos = target_block->terminator();
  if ((insertion_pos)->PreviousNode()->opcode() == spv::Op::OpSelectionMerge) {
    insertion_pos = insertion_pos->PreviousNode();
  }
  inst->RemoveFromList();
  insertion_pos->InsertBefore(std::unique_ptr<Instruction>(inst));
  context()->set_instr_block(inst, target_block);
}

bool IfConversion::CanHoistInstruction(Instruction* inst,
                                       BasicBlock* target_block,
                                       DominatorAnalysis* dominators) {
  BasicBlock* inst_block = context()->get_instr_block(inst);
  if (!inst_block) {
    // This is in the header, and dominates everything.
    return true;
  }

  if (dominators->Dominates(inst_block, target_block)) {
    // Already in position.  No work to do.
    return true;
  }

  if (!inst->IsOpcodeCodeMotionSafe()) {
    return false;
  }

  // Check all instruction |inst| depends on.
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  return inst->WhileEachInId(
      [this, target_block, def_use_mgr, dominators](uint32_t* id) {
        Instruction* operand_inst = def_use_mgr->GetDef(*id);
        return CanHoistInstruction(operand_inst, target_block, dominators);
      });
}

}  // namespace opt
}  // namespace spvtools
