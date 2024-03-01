// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
// Copyright (c) 2019 Google LLC
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

#include "block_merge_util.h"

namespace spvtools {
namespace opt {
namespace blockmergeutil {
namespace {

// Returns true if |block| contains a merge instruction.
bool IsHeader(BasicBlock* block) { return block->GetMergeInst() != nullptr; }

// Returns true if |id| contains a merge instruction.
bool IsHeader(IRContext* context, uint32_t id) {
  return IsHeader(
      context->get_instr_block(context->get_def_use_mgr()->GetDef(id)));
}

// Returns true if |id| is the merge target of a merge instruction.
bool IsMerge(IRContext* context, uint32_t id) {
  return !context->get_def_use_mgr()->WhileEachUse(
      id, [](Instruction* user, uint32_t index) {
        spv::Op op = user->opcode();
        if ((op == spv::Op::OpLoopMerge || op == spv::Op::OpSelectionMerge) &&
            index == 0u) {
          return false;
        }
        return true;
      });
}

// Returns true if |block| is the merge target of a merge instruction.
bool IsMerge(IRContext* context, BasicBlock* block) {
  return IsMerge(context, block->id());
}

// Returns true if |id| is the continue target of a merge instruction.
bool IsContinue(IRContext* context, uint32_t id) {
  return !context->get_def_use_mgr()->WhileEachUse(
      id, [](Instruction* user, uint32_t index) {
        spv::Op op = user->opcode();
        if (op == spv::Op::OpLoopMerge && index == 1u) {
          return false;
        }
        return true;
      });
}

// Removes any OpPhi instructions in |block|, which should have exactly one
// predecessor, replacing uses of OpPhi ids with the ids associated with the
// predecessor.
void EliminateOpPhiInstructions(IRContext* context, BasicBlock* block) {
  block->ForEachPhiInst([context](Instruction* phi) {
    assert(2 == phi->NumInOperands() &&
           "A block can only have one predecessor for block merging to make "
           "sense.");
    context->ReplaceAllUsesWith(phi->result_id(),
                                phi->GetSingleWordInOperand(0));
    context->KillInst(phi);
  });
}

}  // Anonymous namespace

bool CanMergeWithSuccessor(IRContext* context, BasicBlock* block) {
  // Find block with single successor which has no other predecessors.
  auto ii = block->end();
  --ii;
  Instruction* br = &*ii;
  if (br->opcode() != spv::Op::OpBranch) {
    return false;
  }

  const uint32_t lab_id = br->GetSingleWordInOperand(0);
  if (context->cfg()->preds(lab_id).size() != 1) {
    return false;
  }

  bool pred_is_merge = IsMerge(context, block);
  bool succ_is_merge = IsMerge(context, lab_id);
  if (pred_is_merge && succ_is_merge) {
    // Cannot merge two merges together.
    return false;
  }

  // Note: This means that the instructions in a break block will execute as if
  // they were still diverged according to the loop iteration. This restricts
  // potential transformations an implementation may perform on the IR to match
  // shader author expectations. Similarly, instructions in the loop construct
  // cannot be moved into the continue construct unless it can be proven that
  // invocations are always converged.
  if (succ_is_merge && context->get_feature_mgr()->HasExtension(
                           kSPV_KHR_maximal_reconvergence)) {
    return false;
  }

  if (pred_is_merge && IsContinue(context, lab_id)) {
    // Cannot merge a continue target with a merge block.
    return false;
  }

  Instruction* merge_inst = block->GetMergeInst();
  const bool pred_is_header = IsHeader(block);
  if (pred_is_header && lab_id != merge_inst->GetSingleWordInOperand(0u)) {
    bool succ_is_header = IsHeader(context, lab_id);
    if (pred_is_header && succ_is_header) {
      // Cannot merge two headers together when the successor is not the merge
      // block of the predecessor.
      return false;
    }

    // If this is a header block and the successor is not its merge, we must
    // be careful about which blocks we are willing to merge together.
    // OpLoopMerge must be followed by a conditional or unconditional branch.
    // The merge must be a loop merge because a selection merge cannot be
    // followed by an unconditional branch.
    BasicBlock* succ_block = context->get_instr_block(lab_id);
    spv::Op succ_term_op = succ_block->terminator()->opcode();
    assert(merge_inst->opcode() == spv::Op::OpLoopMerge);
    if (succ_term_op != spv::Op::OpBranch &&
        succ_term_op != spv::Op::OpBranchConditional) {
      return false;
    }
  }

  if (succ_is_merge || IsContinue(context, lab_id)) {
    auto* struct_cfg = context->GetStructuredCFGAnalysis();
    auto switch_block_id = struct_cfg->ContainingSwitch(block->id());
    if (switch_block_id) {
      auto switch_merge_id = struct_cfg->SwitchMergeBlock(switch_block_id);
      const auto* switch_inst =
          &*block->GetParent()->FindBlock(switch_block_id)->tail();
      for (uint32_t i = 1; i < switch_inst->NumInOperands(); i += 2) {
        auto target_id = switch_inst->GetSingleWordInOperand(i);
        if (target_id == block->id() && target_id != switch_merge_id) {
          // Case constructs must be structurally dominated by the OpSwitch.
          // Since the successor is the merge/continue for another construct,
          // merging the blocks would break that requirement.
          return false;
        }
      }
    }
  }

  return true;
}

void MergeWithSuccessor(IRContext* context, Function* func,
                        Function::iterator bi) {
  assert(CanMergeWithSuccessor(context, &*bi) &&
         "Precondition failure for MergeWithSuccessor: it must be legal to "
         "merge the block and its successor.");

  auto ii = bi->end();
  --ii;
  Instruction* br = &*ii;
  const uint32_t lab_id = br->GetSingleWordInOperand(0);
  Instruction* merge_inst = bi->GetMergeInst();
  bool pred_is_header = IsHeader(&*bi);

  // Merge blocks.
  context->KillInst(br);
  auto sbi = bi;
  for (; sbi != func->end(); ++sbi)
    if (sbi->id() == lab_id) break;
  // If bi is sbi's only predecessor, it dominates sbi and thus
  // sbi must follow bi in func's ordering.
  assert(sbi != func->end());

  if (sbi->tail()->opcode() == spv::Op::OpSwitch &&
      sbi->MergeBlockIdIfAny() != 0) {
    context->InvalidateAnalyses(IRContext::Analysis::kAnalysisStructuredCFG);
  }

  // Update the inst-to-block mapping for the instructions in sbi.
  for (auto& inst : *sbi) {
    context->set_instr_block(&inst, &*bi);
  }

  EliminateOpPhiInstructions(context, &*sbi);

  // Now actually move the instructions.
  bi->AddInstructions(&*sbi);

  if (merge_inst) {
    if (pred_is_header && lab_id == merge_inst->GetSingleWordInOperand(0u)) {
      // Merging the header and merge blocks, so remove the structured control
      // flow declaration.
      context->KillInst(merge_inst);
    } else {
      // Move OpLine/OpNoLine information to merge_inst. This solves
      // the validation error that OpLine is placed between OpLoopMerge
      // and OpBranchConditional.
      auto terminator = bi->terminator();
      auto& vec = terminator->dbg_line_insts();
      if (vec.size() > 0) {
        merge_inst->ClearDbgLineInsts();
        auto& new_vec = merge_inst->dbg_line_insts();
        new_vec.insert(new_vec.end(), vec.begin(), vec.end());
        terminator->ClearDbgLineInsts();
        for (auto& l_inst : new_vec)
          context->get_def_use_mgr()->AnalyzeInstDefUse(&l_inst);
      }
      // Clear debug scope of terminator to avoid DebugScope
      // emitted between terminator and merge.
      terminator->SetDebugScope(DebugScope(kNoDebugScope, kNoInlinedAt));
      // Move the merge instruction to just before the terminator.
      merge_inst->InsertBefore(terminator);
    }
  }
  context->ReplaceAllUsesWith(lab_id, bi->id());
  context->KillInst(sbi->GetLabelInst());
  (void)sbi.Erase();
}

}  // namespace blockmergeutil
}  // namespace opt
}  // namespace spvtools
