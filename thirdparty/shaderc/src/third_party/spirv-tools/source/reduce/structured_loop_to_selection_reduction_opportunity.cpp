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

#include "source/reduce/structured_loop_to_selection_reduction_opportunity.h"

#include "source/opt/aggressive_dead_code_elim_pass.h"
#include "source/opt/ir_context.h"
#include "source/reduce/reduction_util.h"

namespace spvtools {
namespace reduce {

using opt::BasicBlock;
using opt::IRContext;
using opt::Instruction;
using opt::Operand;

namespace {
const uint32_t kMergeNodeIndex = 0;
}  // namespace

bool StructuredLoopToSelectionReductionOpportunity::PreconditionHolds() {
  // Is the loop header reachable?
  return loop_construct_header_->GetLabel()
      ->context()
      ->GetDominatorAnalysis(enclosing_function_)
      ->IsReachable(loop_construct_header_);
}

void StructuredLoopToSelectionReductionOpportunity::Apply() {
  // Force computation of dominator analysis, CFG and structured CFG analysis
  // before we start to mess with edges in the function.
  context_->GetDominatorAnalysis(enclosing_function_);
  context_->cfg();
  context_->GetStructuredCFGAnalysis();

  // (1) Redirect edges that point to the loop's continue target to their
  // closest merge block.
  RedirectToClosestMergeBlock(loop_construct_header_->ContinueBlockId());

  // (2) Redirect edges that point to the loop's merge block to their closest
  // merge block (which might be that of an enclosing selection, for instance).
  RedirectToClosestMergeBlock(loop_construct_header_->MergeBlockId());

  // (3) Turn the loop construct header into a selection.
  ChangeLoopToSelection();

  // We have made control flow changes that do not preserve the analyses that
  // were performed.
  context_->InvalidateAnalysesExceptFor(IRContext::Analysis::kAnalysisNone);

  // (4) By changing CFG edges we may have created scenarios where ids are used
  // without being dominated; we fix instances of this.
  FixNonDominatedIdUses();

  // Invalidate the analyses we just used.
  context_->InvalidateAnalysesExceptFor(IRContext::Analysis::kAnalysisNone);
}

void StructuredLoopToSelectionReductionOpportunity::RedirectToClosestMergeBlock(
    uint32_t original_target_id) {
  // Consider every predecessor of the node with respect to which edges should
  // be redirected.
  std::set<uint32_t> already_seen;
  for (auto pred : context_->cfg()->preds(original_target_id)) {
    if (already_seen.find(pred) != already_seen.end()) {
      // We have already handled this predecessor (this scenario can arise if
      // there are multiple edges from a block b to original_target_id).
      continue;
    }
    already_seen.insert(pred);

    if (!context_->GetDominatorAnalysis(enclosing_function_)
             ->IsReachable(pred)) {
      // We do not care about unreachable predecessors (and dominance
      // information, and thus the notion of structured control flow, makes
      // little sense for unreachable blocks).
      continue;
    }
    // Find the merge block of the structured control construct that most
    // tightly encloses the predecessor.
    uint32_t new_merge_target;
    // The structured CFG analysis deliberately does not regard a header as
    // belonging to the structure that it heads. We want it to, so handle this
    // case specially.
    if (context_->cfg()->block(pred)->MergeBlockIdIfAny()) {
      new_merge_target = context_->cfg()->block(pred)->MergeBlockIdIfAny();
    } else {
      new_merge_target = context_->GetStructuredCFGAnalysis()->MergeBlock(pred);
    }
    assert(new_merge_target != pred);

    if (!new_merge_target) {
      // If the loop being transformed is outermost, and the predecessor is
      // part of that loop's continue construct, there will be no such
      // enclosing control construct.  In this case, the continue construct
      // will become unreachable anyway, so it is fine not to redirect the
      // edge.
      continue;
    }

    if (new_merge_target != original_target_id) {
      // Redirect the edge if it doesn't already point to the desired block.
      RedirectEdge(pred, original_target_id, new_merge_target);
    }
  }
}

void StructuredLoopToSelectionReductionOpportunity::RedirectEdge(
    uint32_t source_id, uint32_t original_target_id, uint32_t new_target_id) {
  // Redirect edge source_id->original_target_id to edge
  // source_id->new_target_id, where the blocks involved are all different.
  assert(source_id != original_target_id);
  assert(source_id != new_target_id);
  assert(original_target_id != new_target_id);

  // original_target_id must either be the merge target or continue construct
  // for the loop being operated on.
  assert(original_target_id == loop_construct_header_->MergeBlockId() ||
         original_target_id == loop_construct_header_->ContinueBlockId());

  auto terminator = context_->cfg()->block(source_id)->terminator();

  // Figure out which operands of the terminator need to be considered for
  // redirection.
  std::vector<uint32_t> operand_indices;
  if (terminator->opcode() == SpvOpBranch) {
    operand_indices = {0};
  } else if (terminator->opcode() == SpvOpBranchConditional) {
    operand_indices = {1, 2};
  } else {
    assert(terminator->opcode() == SpvOpSwitch);
    for (uint32_t label_index = 1; label_index < terminator->NumOperands();
         label_index += 2) {
      operand_indices.push_back(label_index);
    }
  }

  // Redirect the relevant operands, asserting that at least one redirection is
  // made.
  bool redirected = false;
  for (auto operand_index : operand_indices) {
    if (terminator->GetSingleWordOperand(operand_index) == original_target_id) {
      terminator->SetOperand(operand_index, {new_target_id});
      redirected = true;
    }
  }
  (void)(redirected);
  assert(redirected);

  // The old and new targets may have phi instructions; these will need to
  // respect the change in edges.
  AdaptPhiInstructionsForRemovedEdge(
      source_id, context_->cfg()->block(original_target_id));
  AdaptPhiInstructionsForAddedEdge(source_id,
                                   context_->cfg()->block(new_target_id));
}

void StructuredLoopToSelectionReductionOpportunity::
    AdaptPhiInstructionsForRemovedEdge(uint32_t from_id, BasicBlock* to_block) {
  to_block->ForEachPhiInst([&from_id](Instruction* phi_inst) {
    Instruction::OperandList new_in_operands;
    // Go through the OpPhi's input operands in (variable, parent) pairs.
    for (uint32_t index = 0; index < phi_inst->NumInOperands(); index += 2) {
      // Keep all pairs where the parent is not the block from which the edge
      // is being removed.
      if (phi_inst->GetInOperand(index + 1).words[0] != from_id) {
        new_in_operands.push_back(phi_inst->GetInOperand(index));
        new_in_operands.push_back(phi_inst->GetInOperand(index + 1));
      }
    }
    phi_inst->SetInOperands(std::move(new_in_operands));
  });
}

void StructuredLoopToSelectionReductionOpportunity::
    AdaptPhiInstructionsForAddedEdge(uint32_t from_id, BasicBlock* to_block) {
  to_block->ForEachPhiInst([this, &from_id](Instruction* phi_inst) {
    // Add to the phi operand an (undef, from_id) pair to reflect the added
    // edge.
    auto undef_id = FindOrCreateGlobalUndef(context_, phi_inst->type_id());
    phi_inst->AddOperand(Operand(SPV_OPERAND_TYPE_ID, {undef_id}));
    phi_inst->AddOperand(Operand(SPV_OPERAND_TYPE_ID, {from_id}));
  });
}

void StructuredLoopToSelectionReductionOpportunity::ChangeLoopToSelection() {
  // Change the merge instruction from OpLoopMerge to OpSelectionMerge, with
  // the same merge block.
  auto loop_merge_inst = loop_construct_header_->GetLoopMergeInst();
  auto const loop_merge_block_id =
      loop_merge_inst->GetSingleWordOperand(kMergeNodeIndex);
  loop_merge_inst->SetOpcode(SpvOpSelectionMerge);
  loop_merge_inst->ReplaceOperands(
      {{loop_merge_inst->GetOperand(kMergeNodeIndex).type,
        {loop_merge_block_id}},
       {SPV_OPERAND_TYPE_SELECTION_CONTROL, {SpvSelectionControlMaskNone}}});

  // The loop header either finishes with OpBranch or OpBranchConditional.
  // The latter is fine for a selection.  In the former case we need to turn
  // it into OpBranchConditional.  We use "true" as the condition, and make
  // the "else" branch be the merge block.
  auto terminator = loop_construct_header_->terminator();
  if (terminator->opcode() == SpvOpBranch) {
    opt::analysis::Bool temp;
    const opt::analysis::Bool* bool_type =
        context_->get_type_mgr()->GetRegisteredType(&temp)->AsBool();
    auto const_mgr = context_->get_constant_mgr();
    auto true_const = const_mgr->GetConstant(bool_type, {1});
    auto true_const_result_id =
        const_mgr->GetDefiningInstruction(true_const)->result_id();
    auto original_branch_id = terminator->GetSingleWordOperand(0);
    terminator->SetOpcode(SpvOpBranchConditional);
    terminator->ReplaceOperands({{SPV_OPERAND_TYPE_ID, {true_const_result_id}},
                                 {SPV_OPERAND_TYPE_ID, {original_branch_id}},
                                 {SPV_OPERAND_TYPE_ID, {loop_merge_block_id}}});
    if (original_branch_id != loop_merge_block_id) {
      AdaptPhiInstructionsForAddedEdge(
          loop_construct_header_->id(),
          context_->cfg()->block(loop_merge_block_id));
    }
  }
}

void StructuredLoopToSelectionReductionOpportunity::FixNonDominatedIdUses() {
  // Consider each instruction in the function.
  for (auto& block : *enclosing_function_) {
    for (auto& def : block) {
      if (def.opcode() == SpvOpVariable) {
        // Variables are defined at the start of the function, and can be
        // accessed by all blocks, even by unreachable blocks that have no
        // dominators, so we do not need to worry about them.
        continue;
      }
      context_->get_def_use_mgr()->ForEachUse(&def, [this, &block, &def](
                                                        Instruction* use,
                                                        uint32_t index) {
        // Ignore uses outside of blocks, such as in OpDecorate.
        if (context_->get_instr_block(use) == nullptr) {
          return;
        }
        // If a use is not appropriately dominated by its definition,
        // replace the use with an OpUndef, unless the definition is an
        // access chain, in which case replace it with some (possibly fresh)
        // variable (as we cannot load from / store to OpUndef).
        if (!DefinitionSufficientlyDominatesUse(&def, use, index, block)) {
          if (def.opcode() == SpvOpAccessChain) {
            auto pointer_type =
                context_->get_type_mgr()->GetType(def.type_id())->AsPointer();
            switch (pointer_type->storage_class()) {
              case SpvStorageClassFunction:
                use->SetOperand(
                    index, {FindOrCreateFunctionVariable(
                               context_->get_type_mgr()->GetId(pointer_type))});
                break;
              default:
                // TODO(2183) Need to think carefully about whether it makes
                // sense to add new variables for all storage classes; it's fine
                // for Private but might not be OK for input/output storage
                // classes for example.
                use->SetOperand(
                    index, {FindOrCreateGlobalVariable(
                               context_->get_type_mgr()->GetId(pointer_type))});
                break;
            }
          } else {
            use->SetOperand(index,
                            {FindOrCreateGlobalUndef(context_, def.type_id())});
          }
        }
      });
    }
  }
}

bool StructuredLoopToSelectionReductionOpportunity::
    DefinitionSufficientlyDominatesUse(Instruction* def, Instruction* use,
                                       uint32_t use_index,
                                       BasicBlock& def_block) {
  if (use->opcode() == SpvOpPhi) {
    // A use in a phi doesn't need to be dominated by its definition, but the
    // associated parent block does need to be dominated by the definition.
    return context_->GetDominatorAnalysis(enclosing_function_)
        ->Dominates(def_block.id(), use->GetSingleWordOperand(use_index + 1));
  }
  // In non-phi cases, a use needs to be dominated by its definition.
  return context_->GetDominatorAnalysis(enclosing_function_)
      ->Dominates(def, use);
}

uint32_t
StructuredLoopToSelectionReductionOpportunity::FindOrCreateGlobalVariable(
    uint32_t pointer_type_id) {
  for (auto& inst : context_->module()->types_values()) {
    if (inst.opcode() != SpvOpVariable) {
      continue;
    }
    if (inst.type_id() == pointer_type_id) {
      return inst.result_id();
    }
  }
  const uint32_t variable_id = context_->TakeNextId();
  std::unique_ptr<Instruction> variable_inst(
      new Instruction(context_, SpvOpVariable, pointer_type_id, variable_id,
                      {{SPV_OPERAND_TYPE_STORAGE_CLASS,
                        {(uint32_t)context_->get_type_mgr()
                             ->GetType(pointer_type_id)
                             ->AsPointer()
                             ->storage_class()}}}));
  context_->module()->AddGlobalValue(std::move(variable_inst));
  return variable_id;
}

uint32_t
StructuredLoopToSelectionReductionOpportunity::FindOrCreateFunctionVariable(
    uint32_t pointer_type_id) {
  // The pointer type of a function variable must have Function storage class.
  assert(context_->get_type_mgr()
             ->GetType(pointer_type_id)
             ->AsPointer()
             ->storage_class() == SpvStorageClassFunction);

  // Go through the instructions in the function's first block until we find a
  // suitable variable, or go past all the variables.
  BasicBlock::iterator iter = enclosing_function_->begin()->begin();
  for (;; ++iter) {
    // We will either find a suitable variable, or find a non-variable
    // instruction; we won't exhaust all instructions.
    assert(iter != enclosing_function_->begin()->end());
    if (iter->opcode() != SpvOpVariable) {
      // If we see a non-variable, we have gone through all the variables.
      break;
    }
    if (iter->type_id() == pointer_type_id) {
      return iter->result_id();
    }
  }
  // At this point, iter refers to the first non-function instruction of the
  // function's entry block.
  const uint32_t variable_id = context_->TakeNextId();
  std::unique_ptr<Instruction> variable_inst(new Instruction(
      context_, SpvOpVariable, pointer_type_id, variable_id,
      {{SPV_OPERAND_TYPE_STORAGE_CLASS, {SpvStorageClassFunction}}}));
  iter->InsertBefore(std::move(variable_inst));
  return variable_id;
}

}  // namespace reduce
}  // namespace spvtools
