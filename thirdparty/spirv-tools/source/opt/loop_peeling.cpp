// Copyright (c) 2018 Google LLC.
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

#include <algorithm>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/loop_peeling.h"
#include "source/opt/loop_utils.h"
#include "source/opt/scalar_analysis.h"
#include "source/opt/scalar_analysis_nodes.h"

namespace spvtools {
namespace opt {
namespace {
// Gather the set of blocks for all the path from |entry| to |root|.
void GetBlocksInPath(uint32_t block, uint32_t entry,
                     std::unordered_set<uint32_t>* blocks_in_path,
                     const CFG& cfg) {
  for (uint32_t pid : cfg.preds(block)) {
    if (blocks_in_path->insert(pid).second) {
      if (pid != entry) {
        GetBlocksInPath(pid, entry, blocks_in_path, cfg);
      }
    }
  }
}
}  // namespace

size_t LoopPeelingPass::code_grow_threshold_ = 1000;

void LoopPeeling::DuplicateAndConnectLoop(
    LoopUtils::LoopCloningResult* clone_results) {
  CFG& cfg = *context_->cfg();
  analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

  assert(CanPeelLoop() && "Cannot peel loop!");

  std::vector<BasicBlock*> ordered_loop_blocks;
  // TODO(1841): Handle failure to create pre-header.
  BasicBlock* pre_header = loop_->GetOrCreatePreHeaderBlock();

  loop_->ComputeLoopStructuredOrder(&ordered_loop_blocks);

  cloned_loop_ = loop_utils_.CloneLoop(clone_results, ordered_loop_blocks);

  // Add the basic block to the function.
  Function::iterator it =
      loop_utils_.GetFunction()->FindBlock(pre_header->id());
  assert(it != loop_utils_.GetFunction()->end() &&
         "Pre-header not found in the function.");
  loop_utils_.GetFunction()->AddBasicBlocks(
      clone_results->cloned_bb_.begin(), clone_results->cloned_bb_.end(), ++it);

  // Make the |loop_|'s preheader the |cloned_loop_| one.
  BasicBlock* cloned_header = cloned_loop_->GetHeaderBlock();
  pre_header->ForEachSuccessorLabel(
      [cloned_header](uint32_t* succ) { *succ = cloned_header->id(); });

  // Update cfg.
  cfg.RemoveEdge(pre_header->id(), loop_->GetHeaderBlock()->id());
  cloned_loop_->SetPreHeaderBlock(pre_header);
  loop_->SetPreHeaderBlock(nullptr);

  // When cloning the loop, we didn't cloned the merge block, so currently
  // |cloned_loop_| shares the same block as |loop_|.
  // We mutate all branches from |cloned_loop_| block to |loop_|'s merge into a
  // branch to |loop_|'s header (so header will also be the merge of
  // |cloned_loop_|).
  uint32_t cloned_loop_exit = 0;
  for (uint32_t pred_id : cfg.preds(loop_->GetMergeBlock()->id())) {
    if (loop_->IsInsideLoop(pred_id)) continue;
    BasicBlock* bb = cfg.block(pred_id);
    assert(cloned_loop_exit == 0 && "The loop has multiple exits.");
    cloned_loop_exit = bb->id();
    bb->ForEachSuccessorLabel([this](uint32_t* succ) {
      if (*succ == loop_->GetMergeBlock()->id())
        *succ = loop_->GetHeaderBlock()->id();
    });
  }

  // Update cfg.
  cfg.RemoveNonExistingEdges(loop_->GetMergeBlock()->id());
  cfg.AddEdge(cloned_loop_exit, loop_->GetHeaderBlock()->id());

  // Patch the phi of the original loop header:
  //  - Set the loop entry branch to come from the cloned loop exit block;
  //  - Set the initial value of the phi using the corresponding cloned loop
  //    exit values.
  //
  // We patch the iterating value initializers of the original loop using the
  // corresponding cloned loop exit values. Connects the cloned loop iterating
  // values to the original loop. This make sure that the initial value of the
  // second loop starts with the last value of the first loop.
  //
  // For example, loops like:
  //
  // int z = 0;
  // for (int i = 0; i++ < M; i += cst1) {
  //   if (cond)
  //     z += cst2;
  // }
  //
  // Will become:
  //
  // int z = 0;
  // int i = 0;
  // for (; i++ < M; i += cst1) {
  //   if (cond)
  //     z += cst2;
  // }
  // for (; i++ < M; i += cst1) {
  //   if (cond)
  //     z += cst2;
  // }
  loop_->GetHeaderBlock()->ForEachPhiInst([cloned_loop_exit, def_use_mgr,
                                           clone_results,
                                           this](Instruction* phi) {
    for (uint32_t i = 0; i < phi->NumInOperands(); i += 2) {
      if (!loop_->IsInsideLoop(phi->GetSingleWordInOperand(i + 1))) {
        phi->SetInOperand(i,
                          {clone_results->value_map_.at(
                              exit_value_.at(phi->result_id())->result_id())});
        phi->SetInOperand(i + 1, {cloned_loop_exit});
        def_use_mgr->AnalyzeInstUse(phi);
        return;
      }
    }
  });

  // Force the creation of a new preheader for the original loop and set it as
  // the merge block for the cloned loop.
  // TODO(1841): Handle failure to create pre-header.
  cloned_loop_->SetMergeBlock(loop_->GetOrCreatePreHeaderBlock());
}

void LoopPeeling::InsertCanonicalInductionVariable(
    LoopUtils::LoopCloningResult* clone_results) {
  if (original_loop_canonical_induction_variable_) {
    canonical_induction_variable_ =
        context_->get_def_use_mgr()->GetDef(clone_results->value_map_.at(
            original_loop_canonical_induction_variable_->result_id()));
    return;
  }

  BasicBlock::iterator insert_point = GetClonedLoop()->GetLatchBlock()->tail();
  if (GetClonedLoop()->GetLatchBlock()->GetMergeInst()) {
    --insert_point;
  }
  InstructionBuilder builder(
      context_, &*insert_point,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* uint_1_cst =
      builder.GetIntConstant<uint32_t>(1, int_type_->IsSigned());
  // Create the increment.
  // Note that we do "1 + 1" here, one of the operand should the phi
  // value but we don't have it yet. The operand will be set latter.
  Instruction* iv_inc = builder.AddIAdd(
      uint_1_cst->type_id(), uint_1_cst->result_id(), uint_1_cst->result_id());

  builder.SetInsertPoint(&*GetClonedLoop()->GetHeaderBlock()->begin());

  canonical_induction_variable_ = builder.AddPhi(
      uint_1_cst->type_id(),
      {builder.GetIntConstant<uint32_t>(0, int_type_->IsSigned())->result_id(),
       GetClonedLoop()->GetPreHeaderBlock()->id(), iv_inc->result_id(),
       GetClonedLoop()->GetLatchBlock()->id()});
  // Connect everything.
  iv_inc->SetInOperand(0, {canonical_induction_variable_->result_id()});

  // Update def/use manager.
  context_->get_def_use_mgr()->AnalyzeInstUse(iv_inc);

  // If do-while form, use the incremented value.
  if (do_while_form_) {
    canonical_induction_variable_ = iv_inc;
  }
}

void LoopPeeling::GetIteratorUpdateOperations(
    const Loop* loop, Instruction* iterator,
    std::unordered_set<Instruction*>* operations) {
  analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();
  operations->insert(iterator);
  iterator->ForEachInId([def_use_mgr, loop, operations, this](uint32_t* id) {
    Instruction* insn = def_use_mgr->GetDef(*id);
    if (insn->opcode() == spv::Op::OpLabel) {
      return;
    }
    if (operations->count(insn)) {
      return;
    }
    if (!loop->IsInsideLoop(insn)) {
      return;
    }
    GetIteratorUpdateOperations(loop, insn, operations);
  });
}

bool LoopPeeling::IsConditionCheckSideEffectFree() const {
  CFG& cfg = *context_->cfg();

  // The "do-while" form does not cause issues, the algorithm takes into account
  // the first iteration.
  if (!do_while_form_) {
    uint32_t condition_block_id = cfg.preds(loop_->GetMergeBlock()->id())[0];

    std::unordered_set<uint32_t> blocks_in_path;

    blocks_in_path.insert(condition_block_id);
    GetBlocksInPath(condition_block_id, loop_->GetHeaderBlock()->id(),
                    &blocks_in_path, cfg);

    for (uint32_t bb_id : blocks_in_path) {
      BasicBlock* bb = cfg.block(bb_id);
      if (!bb->WhileEachInst([this](Instruction* insn) {
            if (insn->IsBranch()) return true;
            switch (insn->opcode()) {
              case spv::Op::OpLabel:
              case spv::Op::OpSelectionMerge:
              case spv::Op::OpLoopMerge:
                return true;
              default:
                break;
            }
            return context_->IsCombinatorInstruction(insn);
          })) {
        return false;
      }
    }
  }

  return true;
}

void LoopPeeling::GetIteratingExitValues() {
  CFG& cfg = *context_->cfg();

  loop_->GetHeaderBlock()->ForEachPhiInst(
      [this](Instruction* phi) { exit_value_[phi->result_id()] = nullptr; });

  if (!loop_->GetMergeBlock()) {
    return;
  }
  if (cfg.preds(loop_->GetMergeBlock()->id()).size() != 1) {
    return;
  }
  analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

  uint32_t condition_block_id = cfg.preds(loop_->GetMergeBlock()->id())[0];

  auto& header_pred = cfg.preds(loop_->GetHeaderBlock()->id());
  do_while_form_ = std::find(header_pred.begin(), header_pred.end(),
                             condition_block_id) != header_pred.end();
  if (do_while_form_) {
    loop_->GetHeaderBlock()->ForEachPhiInst(
        [condition_block_id, def_use_mgr, this](Instruction* phi) {
          std::unordered_set<Instruction*> operations;

          for (uint32_t i = 0; i < phi->NumInOperands(); i += 2) {
            if (condition_block_id == phi->GetSingleWordInOperand(i + 1)) {
              exit_value_[phi->result_id()] =
                  def_use_mgr->GetDef(phi->GetSingleWordInOperand(i));
            }
          }
        });
  } else {
    DominatorTree* dom_tree =
        &context_->GetDominatorAnalysis(loop_utils_.GetFunction())
             ->GetDomTree();
    BasicBlock* condition_block = cfg.block(condition_block_id);

    loop_->GetHeaderBlock()->ForEachPhiInst(
        [dom_tree, condition_block, this](Instruction* phi) {
          std::unordered_set<Instruction*> operations;

          // Not the back-edge value, check if the phi instruction is the only
          // possible candidate.
          GetIteratorUpdateOperations(loop_, phi, &operations);

          for (Instruction* insn : operations) {
            if (insn == phi) {
              continue;
            }
            if (dom_tree->Dominates(context_->get_instr_block(insn),
                                    condition_block)) {
              return;
            }
          }
          exit_value_[phi->result_id()] = phi;
        });
  }
}

void LoopPeeling::FixExitCondition(
    const std::function<uint32_t(Instruction*)>& condition_builder) {
  CFG& cfg = *context_->cfg();

  uint32_t condition_block_id = 0;
  for (uint32_t id : cfg.preds(GetClonedLoop()->GetMergeBlock()->id())) {
    if (GetClonedLoop()->IsInsideLoop(id)) {
      condition_block_id = id;
      break;
    }
  }
  assert(condition_block_id != 0 && "2nd loop in improperly connected");

  BasicBlock* condition_block = cfg.block(condition_block_id);
  Instruction* exit_condition = condition_block->terminator();
  assert(exit_condition->opcode() == spv::Op::OpBranchConditional);
  BasicBlock::iterator insert_point = condition_block->tail();
  if (condition_block->GetMergeInst()) {
    --insert_point;
  }

  exit_condition->SetInOperand(0, {condition_builder(&*insert_point)});

  uint32_t to_continue_block_idx =
      GetClonedLoop()->IsInsideLoop(exit_condition->GetSingleWordInOperand(1))
          ? 1
          : 2;
  exit_condition->SetInOperand(
      1, {exit_condition->GetSingleWordInOperand(to_continue_block_idx)});
  exit_condition->SetInOperand(2, {GetClonedLoop()->GetMergeBlock()->id()});

  // Update def/use manager.
  context_->get_def_use_mgr()->AnalyzeInstUse(exit_condition);
}

BasicBlock* LoopPeeling::CreateBlockBefore(BasicBlock* bb) {
  analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();
  CFG& cfg = *context_->cfg();
  assert(cfg.preds(bb->id()).size() == 1 && "More than one predecessor");

  // TODO(1841): Handle id overflow.
  std::unique_ptr<BasicBlock> new_bb =
      MakeUnique<BasicBlock>(std::unique_ptr<Instruction>(new Instruction(
          context_, spv::Op::OpLabel, 0, context_->TakeNextId(), {})));
  // Update the loop descriptor.
  Loop* in_loop = (*loop_utils_.GetLoopDescriptor())[bb];
  if (in_loop) {
    in_loop->AddBasicBlock(new_bb.get());
    loop_utils_.GetLoopDescriptor()->SetBasicBlockToLoop(new_bb->id(), in_loop);
  }

  context_->set_instr_block(new_bb->GetLabelInst(), new_bb.get());
  def_use_mgr->AnalyzeInstDefUse(new_bb->GetLabelInst());

  BasicBlock* bb_pred = cfg.block(cfg.preds(bb->id())[0]);
  bb_pred->tail()->ForEachInId([bb, &new_bb](uint32_t* id) {
    if (*id == bb->id()) {
      *id = new_bb->id();
    }
  });
  cfg.RemoveEdge(bb_pred->id(), bb->id());
  cfg.AddEdge(bb_pred->id(), new_bb->id());
  def_use_mgr->AnalyzeInstUse(&*bb_pred->tail());

  // Update the incoming branch.
  bb->ForEachPhiInst([&new_bb, def_use_mgr](Instruction* phi) {
    phi->SetInOperand(1, {new_bb->id()});
    def_use_mgr->AnalyzeInstUse(phi);
  });
  InstructionBuilder(
      context_, new_bb.get(),
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping)
      .AddBranch(bb->id());
  cfg.RegisterBlock(new_bb.get());

  // Add the basic block to the function.
  Function::iterator it = loop_utils_.GetFunction()->FindBlock(bb->id());
  assert(it != loop_utils_.GetFunction()->end() &&
         "Basic block not found in the function.");
  BasicBlock* ret = new_bb.get();
  loop_utils_.GetFunction()->AddBasicBlock(std::move(new_bb), it);
  return ret;
}

BasicBlock* LoopPeeling::ProtectLoop(Loop* loop, Instruction* condition,
                                     BasicBlock* if_merge) {
  // TODO(1841): Handle failure to create pre-header.
  BasicBlock* if_block = loop->GetOrCreatePreHeaderBlock();
  // Will no longer be a pre-header because of the if.
  loop->SetPreHeaderBlock(nullptr);
  // Kill the branch to the header.
  context_->KillInst(&*if_block->tail());

  InstructionBuilder builder(
      context_, if_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  builder.AddConditionalBranch(condition->result_id(),
                               loop->GetHeaderBlock()->id(), if_merge->id(),
                               if_merge->id());

  return if_block;
}

void LoopPeeling::PeelBefore(uint32_t peel_factor) {
  assert(CanPeelLoop() && "Cannot peel loop");
  LoopUtils::LoopCloningResult clone_results;

  // Clone the loop and insert the cloned one before the loop.
  DuplicateAndConnectLoop(&clone_results);

  // Add a canonical induction variable "canonical_induction_variable_".
  InsertCanonicalInductionVariable(&clone_results);

  InstructionBuilder builder(
      context_, &*cloned_loop_->GetPreHeaderBlock()->tail(),
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* factor =
      builder.GetIntConstant(peel_factor, int_type_->IsSigned());

  Instruction* has_remaining_iteration = builder.AddLessThan(
      factor->result_id(), loop_iteration_count_->result_id());
  Instruction* max_iteration = builder.AddSelect(
      factor->type_id(), has_remaining_iteration->result_id(),
      factor->result_id(), loop_iteration_count_->result_id());

  // Change the exit condition of the cloned loop to be (exit when become
  // false):
  //  "canonical_induction_variable_" < min("factor", "loop_iteration_count_")
  FixExitCondition([max_iteration, this](Instruction* insert_before_point) {
    return InstructionBuilder(context_, insert_before_point,
                              IRContext::kAnalysisDefUse |
                                  IRContext::kAnalysisInstrToBlockMapping)
        .AddLessThan(canonical_induction_variable_->result_id(),
                     max_iteration->result_id())
        ->result_id();
  });

  // "Protect" the second loop: the second loop can only be executed if
  // |has_remaining_iteration| is true (i.e. factor < loop_iteration_count_).
  BasicBlock* if_merge_block = loop_->GetMergeBlock();
  loop_->SetMergeBlock(CreateBlockBefore(loop_->GetMergeBlock()));
  // Prevent the second loop from being executed if we already executed all the
  // required iterations.
  BasicBlock* if_block =
      ProtectLoop(loop_, has_remaining_iteration, if_merge_block);
  // Patch the phi of the merge block.
  if_merge_block->ForEachPhiInst(
      [&clone_results, if_block, this](Instruction* phi) {
        // if_merge_block had previously only 1 predecessor.
        uint32_t incoming_value = phi->GetSingleWordInOperand(0);
        auto def_in_loop = clone_results.value_map_.find(incoming_value);
        if (def_in_loop != clone_results.value_map_.end())
          incoming_value = def_in_loop->second;
        phi->AddOperand(
            {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {incoming_value}});
        phi->AddOperand(
            {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {if_block->id()}});
        context_->get_def_use_mgr()->AnalyzeInstUse(phi);
      });

  context_->InvalidateAnalysesExceptFor(
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping |
      IRContext::kAnalysisLoopAnalysis | IRContext::kAnalysisCFG);
}

void LoopPeeling::PeelAfter(uint32_t peel_factor) {
  assert(CanPeelLoop() && "Cannot peel loop");
  LoopUtils::LoopCloningResult clone_results;

  // Clone the loop and insert the cloned one before the loop.
  DuplicateAndConnectLoop(&clone_results);

  // Add a canonical induction variable "canonical_induction_variable_".
  InsertCanonicalInductionVariable(&clone_results);

  InstructionBuilder builder(
      context_, &*cloned_loop_->GetPreHeaderBlock()->tail(),
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* factor =
      builder.GetIntConstant(peel_factor, int_type_->IsSigned());

  Instruction* has_remaining_iteration = builder.AddLessThan(
      factor->result_id(), loop_iteration_count_->result_id());

  // Change the exit condition of the cloned loop to be (exit when become
  // false):
  //  "canonical_induction_variable_" + "factor" < "loop_iteration_count_"
  FixExitCondition([factor, this](Instruction* insert_before_point) {
    InstructionBuilder cond_builder(
        context_, insert_before_point,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    // Build the following check: canonical_induction_variable_ + factor <
    // iteration_count
    return cond_builder
        .AddLessThan(cond_builder
                         .AddIAdd(canonical_induction_variable_->type_id(),
                                  canonical_induction_variable_->result_id(),
                                  factor->result_id())
                         ->result_id(),
                     loop_iteration_count_->result_id())
        ->result_id();
  });

  // "Protect" the first loop: the first loop can only be executed if
  // factor < loop_iteration_count_.

  // The original loop's pre-header was the cloned loop merge block.
  GetClonedLoop()->SetMergeBlock(
      CreateBlockBefore(GetOriginalLoop()->GetPreHeaderBlock()));
  // Use the second loop preheader as if merge block.

  // Prevent the first loop if only the peeled loop needs it.
  BasicBlock* if_block = ProtectLoop(cloned_loop_, has_remaining_iteration,
                                     GetOriginalLoop()->GetPreHeaderBlock());

  // Patch the phi of the header block.
  // We added an if to enclose the first loop and because the phi node are
  // connected to the exit value of the first loop, the definition no longer
  // dominate the preheader.
  // We had to the preheader (our if merge block) the required phi instruction
  // and patch the header phi.
  GetOriginalLoop()->GetHeaderBlock()->ForEachPhiInst(
      [&clone_results, if_block, this](Instruction* phi) {
        analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

        auto find_value_idx = [](Instruction* phi_inst, Loop* loop) {
          uint32_t preheader_value_idx =
              !loop->IsInsideLoop(phi_inst->GetSingleWordInOperand(1)) ? 0 : 2;
          return preheader_value_idx;
        };

        Instruction* cloned_phi =
            def_use_mgr->GetDef(clone_results.value_map_.at(phi->result_id()));
        uint32_t cloned_preheader_value = cloned_phi->GetSingleWordInOperand(
            find_value_idx(cloned_phi, GetClonedLoop()));

        Instruction* new_phi =
            InstructionBuilder(context_,
                               &*GetOriginalLoop()->GetPreHeaderBlock()->tail(),
                               IRContext::kAnalysisDefUse |
                                   IRContext::kAnalysisInstrToBlockMapping)
                .AddPhi(phi->type_id(),
                        {phi->GetSingleWordInOperand(
                             find_value_idx(phi, GetOriginalLoop())),
                         GetClonedLoop()->GetMergeBlock()->id(),
                         cloned_preheader_value, if_block->id()});

        phi->SetInOperand(find_value_idx(phi, GetOriginalLoop()),
                          {new_phi->result_id()});
        def_use_mgr->AnalyzeInstUse(phi);
      });

  context_->InvalidateAnalysesExceptFor(
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping |
      IRContext::kAnalysisLoopAnalysis | IRContext::kAnalysisCFG);
}

Pass::Status LoopPeelingPass::Process() {
  bool modified = false;
  Module* module = context()->module();

  // Process each function in the module
  for (Function& f : *module) {
    modified |= ProcessFunction(&f);
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool LoopPeelingPass::ProcessFunction(Function* f) {
  bool modified = false;
  LoopDescriptor& loop_descriptor = *context()->GetLoopDescriptor(f);

  std::vector<Loop*> to_process_loop;
  to_process_loop.reserve(loop_descriptor.NumLoops());
  for (Loop& l : loop_descriptor) {
    to_process_loop.push_back(&l);
  }

  ScalarEvolutionAnalysis scev_analysis(context());

  for (Loop* loop : to_process_loop) {
    CodeMetrics loop_size;
    loop_size.Analyze(*loop);

    auto try_peel = [&loop_size, &modified, this](Loop* loop_to_peel) -> Loop* {
      if (!loop_to_peel->IsLCSSA()) {
        LoopUtils(context(), loop_to_peel).MakeLoopClosedSSA();
      }

      bool peeled_loop;
      Loop* still_peelable_loop;
      std::tie(peeled_loop, still_peelable_loop) =
          ProcessLoop(loop_to_peel, &loop_size);

      if (peeled_loop) {
        modified = true;
      }

      return still_peelable_loop;
    };

    Loop* still_peelable_loop = try_peel(loop);
    // The pass is working out the maximum factor by which a loop can be peeled.
    // If the loop can potentially be peeled again, then there is only one
    // possible direction, so only one call is still needed.
    if (still_peelable_loop) {
      try_peel(loop);
    }
  }

  return modified;
}

std::pair<bool, Loop*> LoopPeelingPass::ProcessLoop(Loop* loop,
                                                    CodeMetrics* loop_size) {
  ScalarEvolutionAnalysis* scev_analysis =
      context()->GetScalarEvolutionAnalysis();
  // Default values for bailing out.
  std::pair<bool, Loop*> bail_out{false, nullptr};

  BasicBlock* exit_block = loop->FindConditionBlock();
  if (!exit_block) {
    return bail_out;
  }

  Instruction* exiting_iv = loop->FindConditionVariable(exit_block);
  if (!exiting_iv) {
    return bail_out;
  }
  size_t iterations = 0;
  if (!loop->FindNumberOfIterations(exiting_iv, &*exit_block->tail(),
                                    &iterations)) {
    return bail_out;
  }
  if (!iterations) {
    return bail_out;
  }

  Instruction* canonical_induction_variable = nullptr;

  loop->GetHeaderBlock()->WhileEachPhiInst([&canonical_induction_variable,
                                            scev_analysis,
                                            this](Instruction* insn) {
    if (const SERecurrentNode* iv =
            scev_analysis->AnalyzeInstruction(insn)->AsSERecurrentNode()) {
      const SEConstantNode* offset = iv->GetOffset()->AsSEConstantNode();
      const SEConstantNode* coeff = iv->GetCoefficient()->AsSEConstantNode();
      if (offset && coeff && offset->FoldToSingleValue() == 0 &&
          coeff->FoldToSingleValue() == 1) {
        if (context()->get_type_mgr()->GetType(insn->type_id())->AsInteger()) {
          canonical_induction_variable = insn;
          return false;
        }
      }
    }
    return true;
  });

  bool is_signed = canonical_induction_variable
                       ? context()
                             ->get_type_mgr()
                             ->GetType(canonical_induction_variable->type_id())
                             ->AsInteger()
                             ->IsSigned()
                       : false;

  LoopPeeling peeler(
      loop,
      InstructionBuilder(
          context(), loop->GetHeaderBlock(),
          IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping)
          .GetIntConstant<uint32_t>(static_cast<uint32_t>(iterations),
                                    is_signed),
      canonical_induction_variable);

  if (!peeler.CanPeelLoop()) {
    return bail_out;
  }

  // For each basic block in the loop, check if it can be peeled. If it
  // can, get the direction (before/after) and by which factor.
  LoopPeelingInfo peel_info(loop, iterations, scev_analysis);

  uint32_t peel_before_factor = 0;
  uint32_t peel_after_factor = 0;

  for (uint32_t block : loop->GetBlocks()) {
    if (block == exit_block->id()) {
      continue;
    }
    BasicBlock* bb = cfg()->block(block);
    PeelDirection direction;
    uint32_t factor;
    std::tie(direction, factor) = peel_info.GetPeelingInfo(bb);

    if (direction == PeelDirection::kNone) {
      continue;
    }
    if (direction == PeelDirection::kBefore) {
      peel_before_factor = std::max(peel_before_factor, factor);
    } else {
      assert(direction == PeelDirection::kAfter);
      peel_after_factor = std::max(peel_after_factor, factor);
    }
  }
  PeelDirection direction = PeelDirection::kNone;
  uint32_t factor = 0;

  // Find which direction we should peel.
  if (peel_before_factor) {
    factor = peel_before_factor;
    direction = PeelDirection::kBefore;
  }
  if (peel_after_factor) {
    if (peel_before_factor < peel_after_factor) {
      // Favor a peel after here and give the peel before another shot later.
      factor = peel_after_factor;
      direction = PeelDirection::kAfter;
    }
  }

  // Do the peel if we can.
  if (direction == PeelDirection::kNone) return bail_out;

  // This does not take into account branch elimination opportunities and
  // the unrolling. It assumes the peeled loop will be unrolled as well.
  if (factor * loop_size->roi_size_ > code_grow_threshold_) {
    return bail_out;
  }
  loop_size->roi_size_ *= factor;

  // Find if a loop should be peeled again.
  Loop* extra_opportunity = nullptr;

  if (direction == PeelDirection::kBefore) {
    peeler.PeelBefore(factor);
    if (stats_) {
      stats_->peeled_loops_.emplace_back(loop, PeelDirection::kBefore, factor);
    }
    if (peel_after_factor) {
      // We could have peeled after, give it another try.
      extra_opportunity = peeler.GetOriginalLoop();
    }
  } else {
    peeler.PeelAfter(factor);
    if (stats_) {
      stats_->peeled_loops_.emplace_back(loop, PeelDirection::kAfter, factor);
    }
    if (peel_before_factor) {
      // We could have peeled before, give it another try.
      extra_opportunity = peeler.GetClonedLoop();
    }
  }

  return {true, extra_opportunity};
}

uint32_t LoopPeelingPass::LoopPeelingInfo::GetFirstLoopInvariantOperand(
    Instruction* condition) const {
  for (uint32_t i = 0; i < condition->NumInOperands(); i++) {
    BasicBlock* bb =
        context_->get_instr_block(condition->GetSingleWordInOperand(i));
    if (bb && loop_->IsInsideLoop(bb)) {
      return condition->GetSingleWordInOperand(i);
    }
  }

  return 0;
}

uint32_t LoopPeelingPass::LoopPeelingInfo::GetFirstNonLoopInvariantOperand(
    Instruction* condition) const {
  for (uint32_t i = 0; i < condition->NumInOperands(); i++) {
    BasicBlock* bb =
        context_->get_instr_block(condition->GetSingleWordInOperand(i));
    if (!bb || !loop_->IsInsideLoop(bb)) {
      return condition->GetSingleWordInOperand(i);
    }
  }

  return 0;
}

static bool IsHandledCondition(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpIEqual:
    case spv::Op::OpINotEqual:
    case spv::Op::OpUGreaterThan:
    case spv::Op::OpSGreaterThan:
    case spv::Op::OpUGreaterThanEqual:
    case spv::Op::OpSGreaterThanEqual:
    case spv::Op::OpULessThan:
    case spv::Op::OpSLessThan:
    case spv::Op::OpULessThanEqual:
    case spv::Op::OpSLessThanEqual:
      return true;
    default:
      return false;
  }
}

LoopPeelingPass::LoopPeelingInfo::Direction
LoopPeelingPass::LoopPeelingInfo::GetPeelingInfo(BasicBlock* bb) const {
  if (bb->terminator()->opcode() != spv::Op::OpBranchConditional) {
    return GetNoneDirection();
  }

  analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

  Instruction* condition =
      def_use_mgr->GetDef(bb->terminator()->GetSingleWordInOperand(0));

  if (!IsHandledCondition(condition->opcode())) {
    return GetNoneDirection();
  }

  if (!GetFirstLoopInvariantOperand(condition)) {
    // No loop invariant, it cannot be peeled by this pass.
    return GetNoneDirection();
  }
  if (!GetFirstNonLoopInvariantOperand(condition)) {
    // Seems to be a job for the unswitch pass.
    return GetNoneDirection();
  }

  // Left hand-side.
  SExpression lhs = scev_analysis_->AnalyzeInstruction(
      def_use_mgr->GetDef(condition->GetSingleWordInOperand(0)));
  if (lhs->GetType() == SENode::CanNotCompute) {
    // Can't make any conclusion.
    return GetNoneDirection();
  }

  // Right hand-side.
  SExpression rhs = scev_analysis_->AnalyzeInstruction(
      def_use_mgr->GetDef(condition->GetSingleWordInOperand(1)));
  if (rhs->GetType() == SENode::CanNotCompute) {
    // Can't make any conclusion.
    return GetNoneDirection();
  }

  // Only take into account recurrent expression over the current loop.
  bool is_lhs_rec = !scev_analysis_->IsLoopInvariant(loop_, lhs);
  bool is_rhs_rec = !scev_analysis_->IsLoopInvariant(loop_, rhs);

  if ((is_lhs_rec && is_rhs_rec) || (!is_lhs_rec && !is_rhs_rec)) {
    return GetNoneDirection();
  }

  if (is_lhs_rec) {
    if (!lhs->AsSERecurrentNode() ||
        lhs->AsSERecurrentNode()->GetLoop() != loop_) {
      return GetNoneDirection();
    }
  }
  if (is_rhs_rec) {
    if (!rhs->AsSERecurrentNode() ||
        rhs->AsSERecurrentNode()->GetLoop() != loop_) {
      return GetNoneDirection();
    }
  }

  // If the op code is ==, then we try a peel before or after.
  // If opcode is not <, >, <= or >=, we bail out.
  //
  // For the remaining cases, we canonicalize the expression so that the
  // constant expression is on the left hand side and the recurring expression
  // is on the right hand side. If we swap hand side, then < becomes >, <=
  // becomes >= etc.
  // If the opcode is <=, then we add 1 to the right hand side and do the peel
  // check on <.
  // If the opcode is >=, then we add 1 to the left hand side and do the peel
  // check on >.

  CmpOperator cmp_operator;
  switch (condition->opcode()) {
    default:
      return GetNoneDirection();
    case spv::Op::OpIEqual:
    case spv::Op::OpINotEqual:
      return HandleEquality(lhs, rhs);
    case spv::Op::OpUGreaterThan:
    case spv::Op::OpSGreaterThan: {
      cmp_operator = CmpOperator::kGT;
      break;
    }
    case spv::Op::OpULessThan:
    case spv::Op::OpSLessThan: {
      cmp_operator = CmpOperator::kLT;
      break;
    }
    // We add one to transform >= into > and <= into <.
    case spv::Op::OpUGreaterThanEqual:
    case spv::Op::OpSGreaterThanEqual: {
      cmp_operator = CmpOperator::kGE;
      break;
    }
    case spv::Op::OpULessThanEqual:
    case spv::Op::OpSLessThanEqual: {
      cmp_operator = CmpOperator::kLE;
      break;
    }
  }

  // Force the left hand side to be the non recurring expression.
  if (is_lhs_rec) {
    std::swap(lhs, rhs);
    switch (cmp_operator) {
      case CmpOperator::kLT: {
        cmp_operator = CmpOperator::kGT;
        break;
      }
      case CmpOperator::kGT: {
        cmp_operator = CmpOperator::kLT;
        break;
      }
      case CmpOperator::kLE: {
        cmp_operator = CmpOperator::kGE;
        break;
      }
      case CmpOperator::kGE: {
        cmp_operator = CmpOperator::kLE;
        break;
      }
    }
  }
  return HandleInequality(cmp_operator, lhs, rhs->AsSERecurrentNode());
}

SExpression LoopPeelingPass::LoopPeelingInfo::GetValueAtFirstIteration(
    SERecurrentNode* rec) const {
  return rec->GetOffset();
}

SExpression LoopPeelingPass::LoopPeelingInfo::GetValueAtIteration(
    SERecurrentNode* rec, int64_t iteration) const {
  SExpression coeff = rec->GetCoefficient();
  SExpression offset = rec->GetOffset();

  return (coeff * iteration) + offset;
}

SExpression LoopPeelingPass::LoopPeelingInfo::GetValueAtLastIteration(
    SERecurrentNode* rec) const {
  return GetValueAtIteration(rec, loop_max_iterations_ - 1);
}

bool LoopPeelingPass::LoopPeelingInfo::EvalOperator(CmpOperator cmp_op,
                                                    SExpression lhs,
                                                    SExpression rhs,
                                                    bool* result) const {
  assert(scev_analysis_->IsLoopInvariant(loop_, lhs));
  assert(scev_analysis_->IsLoopInvariant(loop_, rhs));
  // We perform the test: 0 cmp_op rhs - lhs
  // What is left is then to determine the sign of the expression.
  switch (cmp_op) {
    case CmpOperator::kLT: {
      return scev_analysis_->IsAlwaysGreaterThanZero(rhs - lhs, result);
    }
    case CmpOperator::kGT: {
      return scev_analysis_->IsAlwaysGreaterThanZero(lhs - rhs, result);
    }
    case CmpOperator::kLE: {
      return scev_analysis_->IsAlwaysGreaterOrEqualToZero(rhs - lhs, result);
    }
    case CmpOperator::kGE: {
      return scev_analysis_->IsAlwaysGreaterOrEqualToZero(lhs - rhs, result);
    }
  }
  return false;
}

LoopPeelingPass::LoopPeelingInfo::Direction
LoopPeelingPass::LoopPeelingInfo::HandleEquality(SExpression lhs,
                                                 SExpression rhs) const {
  {
    // Try peel before opportunity.
    SExpression lhs_cst = lhs;
    if (SERecurrentNode* rec_node = lhs->AsSERecurrentNode()) {
      lhs_cst = rec_node->GetOffset();
    }
    SExpression rhs_cst = rhs;
    if (SERecurrentNode* rec_node = rhs->AsSERecurrentNode()) {
      rhs_cst = rec_node->GetOffset();
    }

    if (lhs_cst == rhs_cst) {
      return Direction{LoopPeelingPass::PeelDirection::kBefore, 1};
    }
  }

  {
    // Try peel after opportunity.
    SExpression lhs_cst = lhs;
    if (SERecurrentNode* rec_node = lhs->AsSERecurrentNode()) {
      // rec_node(x) = a * x + b
      // assign to lhs: a * (loop_max_iterations_ - 1) + b
      lhs_cst = GetValueAtLastIteration(rec_node);
    }
    SExpression rhs_cst = rhs;
    if (SERecurrentNode* rec_node = rhs->AsSERecurrentNode()) {
      // rec_node(x) = a * x + b
      // assign to lhs: a * (loop_max_iterations_ - 1) + b
      rhs_cst = GetValueAtLastIteration(rec_node);
    }

    if (lhs_cst == rhs_cst) {
      return Direction{LoopPeelingPass::PeelDirection::kAfter, 1};
    }
  }

  return GetNoneDirection();
}

LoopPeelingPass::LoopPeelingInfo::Direction
LoopPeelingPass::LoopPeelingInfo::HandleInequality(CmpOperator cmp_op,
                                                   SExpression lhs,
                                                   SERecurrentNode* rhs) const {
  SExpression offset = rhs->GetOffset();
  SExpression coefficient = rhs->GetCoefficient();
  // Compute (cst - B) / A.
  std::pair<SExpression, int64_t> flip_iteration = (lhs - offset) / coefficient;
  if (!flip_iteration.first->AsSEConstantNode()) {
    return GetNoneDirection();
  }
  // note: !!flip_iteration.second normalize to 0/1 (via bool cast).
  int64_t iteration =
      flip_iteration.first->AsSEConstantNode()->FoldToSingleValue() +
      !!flip_iteration.second;
  if (iteration <= 0 ||
      loop_max_iterations_ <= static_cast<uint64_t>(iteration)) {
    // Always true or false within the loop bounds.
    return GetNoneDirection();
  }
  // If this is a <= or >= operator and the iteration, make sure |iteration| is
  // the one flipping the condition.
  // If (cst - B) and A are not divisible, this equivalent to a < or > check, so
  // we skip this test.
  if (!flip_iteration.second &&
      (cmp_op == CmpOperator::kLE || cmp_op == CmpOperator::kGE)) {
    bool first_iteration;
    bool current_iteration;
    if (!EvalOperator(cmp_op, lhs, offset, &first_iteration) ||
        !EvalOperator(cmp_op, lhs, GetValueAtIteration(rhs, iteration),
                      &current_iteration)) {
      return GetNoneDirection();
    }
    // If the condition did not flip the next will.
    if (first_iteration == current_iteration) {
      iteration++;
    }
  }

  uint32_t cast_iteration = 0;
  // Integrity check: can we fit |iteration| in a uint32_t ?
  if (static_cast<uint64_t>(iteration) < std::numeric_limits<uint32_t>::max()) {
    cast_iteration = static_cast<uint32_t>(iteration);
  }

  if (cast_iteration) {
    // Peel before if we are closer to the start, after if closer to the end.
    if (loop_max_iterations_ / 2 > cast_iteration) {
      return Direction{LoopPeelingPass::PeelDirection::kBefore, cast_iteration};
    } else {
      return Direction{
          LoopPeelingPass::PeelDirection::kAfter,
          static_cast<uint32_t>(loop_max_iterations_ - cast_iteration)};
    }
  }

  return GetNoneDirection();
}

}  // namespace opt
}  // namespace spvtools
