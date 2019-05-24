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

#include "source/opt/loop_unswitch_pass.h"

#include <functional>
#include <list>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/dominator_tree.h"
#include "source/opt/fold.h"
#include "source/opt/function.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/opt/loop_descriptor.h"

#include "source/opt/loop_utils.h"

namespace spvtools {
namespace opt {
namespace {

static const uint32_t kTypePointerStorageClassInIdx = 0;

}  // anonymous namespace

namespace {

// This class handle the unswitch procedure for a given loop.
// The unswitch will not happen if:
//  - The loop has any instruction that will prevent it;
//  - The loop invariant condition is not uniform.
class LoopUnswitch {
 public:
  LoopUnswitch(IRContext* context, Function* function, Loop* loop,
               LoopDescriptor* loop_desc)
      : function_(function),
        loop_(loop),
        loop_desc_(*loop_desc),
        context_(context),
        switch_block_(nullptr) {}

  // Returns true if the loop can be unswitched.
  // Can be unswitch if:
  //  - The loop has no instructions that prevents it (such as barrier);
  //  - The loop has one conditional branch or switch that do not depends on the
  //  loop;
  //  - The loop invariant condition is uniform;
  bool CanUnswitchLoop() {
    if (switch_block_) return true;
    if (loop_->IsSafeToClone()) return false;

    CFG& cfg = *context_->cfg();

    for (uint32_t bb_id : loop_->GetBlocks()) {
      BasicBlock* bb = cfg.block(bb_id);
      if (loop_->GetLatchBlock() == bb) {
        continue;
      }

      if (bb->terminator()->IsBranch() &&
          bb->terminator()->opcode() != SpvOpBranch) {
        if (IsConditionNonConstantLoopInvariant(bb->terminator())) {
          switch_block_ = bb;
          break;
        }
      }
    }

    return switch_block_;
  }

  // Return the iterator to the basic block |bb|.
  Function::iterator FindBasicBlockPosition(BasicBlock* bb_to_find) {
    Function::iterator it = function_->FindBlock(bb_to_find->id());
    assert(it != function_->end() && "Basic Block not found");
    return it;
  }

  // Creates a new basic block and insert it into the function |fn| at the
  // position |ip|. This function preserves the def/use and instr to block
  // managers.
  BasicBlock* CreateBasicBlock(Function::iterator ip) {
    analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

    // TODO(1841): Handle id overflow.
    BasicBlock* bb = &*ip.InsertBefore(std::unique_ptr<BasicBlock>(
        new BasicBlock(std::unique_ptr<Instruction>(new Instruction(
            context_, SpvOpLabel, 0, context_->TakeNextId(), {})))));
    bb->SetParent(function_);
    def_use_mgr->AnalyzeInstDef(bb->GetLabelInst());
    context_->set_instr_block(bb->GetLabelInst(), bb);

    return bb;
  }

  Instruction* GetValueForDefaultPathForSwitch(Instruction* switch_inst) {
    assert(switch_inst->opcode() == SpvOpSwitch &&
           "The given instructoin must be an OpSwitch.");

    // Find a value that can be used to select the default path.
    // If none are possible, then it will just use 0.  The value does not matter
    // because this path will never be taken becaues the new switch outside of
    // the loop cannot select this path either.
    std::vector<uint32_t> existing_values;
    for (uint32_t i = 2; i < switch_inst->NumInOperands(); i += 2) {
      existing_values.push_back(switch_inst->GetSingleWordInOperand(i));
    }
    std::sort(existing_values.begin(), existing_values.end());
    uint32_t value_for_default_path = 0;
    if (existing_values.size() < std::numeric_limits<uint32_t>::max()) {
      for (value_for_default_path = 0;
           value_for_default_path < existing_values.size();
           value_for_default_path++) {
        if (existing_values[value_for_default_path] != value_for_default_path) {
          break;
        }
      }
    }
    InstructionBuilder builder(
        context_, static_cast<Instruction*>(nullptr),
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    return builder.GetUintConstant(value_for_default_path);
  }

  // Unswitches |loop_|.
  void PerformUnswitch() {
    assert(CanUnswitchLoop() &&
           "Cannot unswitch if there is not constant condition");
    assert(loop_->GetPreHeaderBlock() && "This loop has no pre-header block");
    assert(loop_->IsLCSSA() && "This loop is not in LCSSA form");

    CFG& cfg = *context_->cfg();
    DominatorTree* dom_tree =
        &context_->GetDominatorAnalysis(function_)->GetDomTree();
    analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();
    LoopUtils loop_utils(context_, loop_);

    //////////////////////////////////////////////////////////////////////////////
    // Step 1: Create the if merge block for structured modules.
    //    To do so, the |loop_| merge block will become the if's one and we
    //    create a merge for the loop. This will limit the amount of duplicated
    //    code the structured control flow imposes.
    //    For non structured program, the new loop will be connected to
    //    the old loop's exit blocks.
    //////////////////////////////////////////////////////////////////////////////

    // Get the merge block if it exists.
    BasicBlock* if_merge_block = loop_->GetMergeBlock();
    // The merge block is only created if the loop has a unique exit block. We
    // have this guarantee for structured loops, for compute loop it will
    // trivially help maintain both a structured-like form and LCSAA.
    BasicBlock* loop_merge_block =
        if_merge_block
            ? CreateBasicBlock(FindBasicBlockPosition(if_merge_block))
            : nullptr;
    if (loop_merge_block) {
      // Add the instruction and update managers.
      InstructionBuilder builder(
          context_, loop_merge_block,
          IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
      builder.AddBranch(if_merge_block->id());
      builder.SetInsertPoint(&*loop_merge_block->begin());
      cfg.RegisterBlock(loop_merge_block);
      def_use_mgr->AnalyzeInstDef(loop_merge_block->GetLabelInst());
      // Update CFG.
      if_merge_block->ForEachPhiInst(
          [loop_merge_block, &builder, this](Instruction* phi) {
            Instruction* cloned = phi->Clone(context_);
            cloned->SetResultId(TakeNextId());
            builder.AddInstruction(std::unique_ptr<Instruction>(cloned));
            phi->SetInOperand(0, {cloned->result_id()});
            phi->SetInOperand(1, {loop_merge_block->id()});
            for (uint32_t j = phi->NumInOperands() - 1; j > 1; j--)
              phi->RemoveInOperand(j);
          });
      // Copy the predecessor list (will get invalidated otherwise).
      std::vector<uint32_t> preds = cfg.preds(if_merge_block->id());
      for (uint32_t pid : preds) {
        if (pid == loop_merge_block->id()) continue;
        BasicBlock* p_bb = cfg.block(pid);
        p_bb->ForEachSuccessorLabel(
            [if_merge_block, loop_merge_block](uint32_t* id) {
              if (*id == if_merge_block->id()) *id = loop_merge_block->id();
            });
        cfg.AddEdge(pid, loop_merge_block->id());
      }
      cfg.RemoveNonExistingEdges(if_merge_block->id());
      // Update loop descriptor.
      if (Loop* ploop = loop_->GetParent()) {
        ploop->AddBasicBlock(loop_merge_block);
        loop_desc_.SetBasicBlockToLoop(loop_merge_block->id(), ploop);
      }
      // Update the dominator tree.
      DominatorTreeNode* loop_merge_dtn =
          dom_tree->GetOrInsertNode(loop_merge_block);
      DominatorTreeNode* if_merge_block_dtn =
          dom_tree->GetOrInsertNode(if_merge_block);
      loop_merge_dtn->parent_ = if_merge_block_dtn->parent_;
      loop_merge_dtn->children_.push_back(if_merge_block_dtn);
      loop_merge_dtn->parent_->children_.push_back(loop_merge_dtn);
      if_merge_block_dtn->parent_->children_.erase(std::find(
          if_merge_block_dtn->parent_->children_.begin(),
          if_merge_block_dtn->parent_->children_.end(), if_merge_block_dtn));

      loop_->SetMergeBlock(loop_merge_block);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Step 2: Build a new preheader for |loop_|, use the old one
    //         for the invariant branch.
    ////////////////////////////////////////////////////////////////////////////

    BasicBlock* if_block = loop_->GetPreHeaderBlock();
    // If this preheader is the parent loop header,
    // we need to create a dedicated block for the if.
    BasicBlock* loop_pre_header =
        CreateBasicBlock(++FindBasicBlockPosition(if_block));
    InstructionBuilder(
        context_, loop_pre_header,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping)
        .AddBranch(loop_->GetHeaderBlock()->id());

    if_block->tail()->SetInOperand(0, {loop_pre_header->id()});

    // Update loop descriptor.
    if (Loop* ploop = loop_desc_[if_block]) {
      ploop->AddBasicBlock(loop_pre_header);
      loop_desc_.SetBasicBlockToLoop(loop_pre_header->id(), ploop);
    }

    // Update the CFG.
    cfg.RegisterBlock(loop_pre_header);
    def_use_mgr->AnalyzeInstDef(loop_pre_header->GetLabelInst());
    cfg.AddEdge(if_block->id(), loop_pre_header->id());
    cfg.RemoveNonExistingEdges(loop_->GetHeaderBlock()->id());

    loop_->GetHeaderBlock()->ForEachPhiInst(
        [loop_pre_header, if_block](Instruction* phi) {
          phi->ForEachInId([loop_pre_header, if_block](uint32_t* id) {
            if (*id == if_block->id()) {
              *id = loop_pre_header->id();
            }
          });
        });
    loop_->SetPreHeaderBlock(loop_pre_header);

    // Update the dominator tree.
    DominatorTreeNode* loop_pre_header_dtn =
        dom_tree->GetOrInsertNode(loop_pre_header);
    DominatorTreeNode* if_block_dtn = dom_tree->GetTreeNode(if_block);
    loop_pre_header_dtn->parent_ = if_block_dtn;
    assert(
        if_block_dtn->children_.size() == 1 &&
        "A loop preheader should only have the header block as a child in the "
        "dominator tree");
    loop_pre_header_dtn->children_.push_back(if_block_dtn->children_[0]);
    if_block_dtn->children_.clear();
    if_block_dtn->children_.push_back(loop_pre_header_dtn);

    // Make domination queries valid.
    dom_tree->ResetDFNumbering();

    // Compute an ordered list of basic block to clone: loop blocks + pre-header
    // + merge block.
    loop_->ComputeLoopStructuredOrder(&ordered_loop_blocks_, true, true);

    /////////////////////////////
    // Do the actual unswitch: //
    //   - Clone the loop      //
    //   - Connect exits       //
    //   - Specialize the loop //
    /////////////////////////////

    Instruction* iv_condition = &*switch_block_->tail();
    SpvOp iv_opcode = iv_condition->opcode();
    Instruction* condition =
        def_use_mgr->GetDef(iv_condition->GetOperand(0).words[0]);

    analysis::ConstantManager* cst_mgr = context_->get_constant_mgr();
    const analysis::Type* cond_type =
        context_->get_type_mgr()->GetType(condition->type_id());

    // Build the list of value for which we need to clone and specialize the
    // loop.
    std::vector<std::pair<Instruction*, BasicBlock*>> constant_branch;
    // Special case for the original loop
    Instruction* original_loop_constant_value;
    if (iv_opcode == SpvOpBranchConditional) {
      constant_branch.emplace_back(
          cst_mgr->GetDefiningInstruction(cst_mgr->GetConstant(cond_type, {0})),
          nullptr);
      original_loop_constant_value =
          cst_mgr->GetDefiningInstruction(cst_mgr->GetConstant(cond_type, {1}));
    } else {
      // We are looking to take the default branch, so we can't provide a
      // specific value.
      original_loop_constant_value =
          GetValueForDefaultPathForSwitch(iv_condition);

      for (uint32_t i = 2; i < iv_condition->NumInOperands(); i += 2) {
        constant_branch.emplace_back(
            cst_mgr->GetDefiningInstruction(cst_mgr->GetConstant(
                cond_type, iv_condition->GetInOperand(i).words)),
            nullptr);
      }
    }

    // Get the loop landing pads.
    std::unordered_set<uint32_t> if_merging_blocks;
    std::function<bool(uint32_t)> is_from_original_loop;
    if (loop_->GetHeaderBlock()->GetLoopMergeInst()) {
      if_merging_blocks.insert(if_merge_block->id());
      is_from_original_loop = [this](uint32_t id) {
        return loop_->IsInsideLoop(id) || loop_->GetMergeBlock()->id() == id;
      };
    } else {
      loop_->GetExitBlocks(&if_merging_blocks);
      is_from_original_loop = [this](uint32_t id) {
        return loop_->IsInsideLoop(id);
      };
    }

    for (auto& specialisation_pair : constant_branch) {
      Instruction* specialisation_value = specialisation_pair.first;
      //////////////////////////////////////////////////////////
      // Step 3: Duplicate |loop_|.
      //////////////////////////////////////////////////////////
      LoopUtils::LoopCloningResult clone_result;

      Loop* cloned_loop =
          loop_utils.CloneLoop(&clone_result, ordered_loop_blocks_);
      specialisation_pair.second = cloned_loop->GetPreHeaderBlock();

      ////////////////////////////////////
      // Step 4: Specialize the loop.   //
      ////////////////////////////////////

      {
        SpecializeLoop(cloned_loop, condition, specialisation_value);

        ///////////////////////////////////////////////////////////
        // Step 5: Connect convergent edges to the landing pads. //
        ///////////////////////////////////////////////////////////

        for (uint32_t merge_bb_id : if_merging_blocks) {
          BasicBlock* merge = context_->cfg()->block(merge_bb_id);
          // We are in LCSSA so we only care about phi instructions.
          merge->ForEachPhiInst(
              [is_from_original_loop, &clone_result](Instruction* phi) {
                uint32_t num_in_operands = phi->NumInOperands();
                for (uint32_t i = 0; i < num_in_operands; i += 2) {
                  uint32_t pred = phi->GetSingleWordInOperand(i + 1);
                  if (is_from_original_loop(pred)) {
                    pred = clone_result.value_map_.at(pred);
                    uint32_t incoming_value_id = phi->GetSingleWordInOperand(i);
                    // Not all the incoming values are coming from the loop.
                    ValueMapTy::iterator new_value =
                        clone_result.value_map_.find(incoming_value_id);
                    if (new_value != clone_result.value_map_.end()) {
                      incoming_value_id = new_value->second;
                    }
                    phi->AddOperand({SPV_OPERAND_TYPE_ID, {incoming_value_id}});
                    phi->AddOperand({SPV_OPERAND_TYPE_ID, {pred}});
                  }
                }
              });
        }
      }
      function_->AddBasicBlocks(clone_result.cloned_bb_.begin(),
                                clone_result.cloned_bb_.end(),
                                ++FindBasicBlockPosition(if_block));
    }

    // Specialize the existing loop.
    SpecializeLoop(loop_, condition, original_loop_constant_value);
    BasicBlock* original_loop_target = loop_->GetPreHeaderBlock();

    /////////////////////////////////////
    // Finally: connect the new loops. //
    /////////////////////////////////////

    // Delete the old jump
    context_->KillInst(&*if_block->tail());
    InstructionBuilder builder(context_, if_block);
    if (iv_opcode == SpvOpBranchConditional) {
      assert(constant_branch.size() == 1);
      builder.AddConditionalBranch(
          condition->result_id(), original_loop_target->id(),
          constant_branch[0].second->id(),
          if_merge_block ? if_merge_block->id() : kInvalidId);
    } else {
      std::vector<std::pair<Operand::OperandData, uint32_t>> targets;
      for (auto& t : constant_branch) {
        targets.emplace_back(t.first->GetInOperand(0).words, t.second->id());
      }

      builder.AddSwitch(condition->result_id(), original_loop_target->id(),
                        targets,
                        if_merge_block ? if_merge_block->id() : kInvalidId);
    }

    switch_block_ = nullptr;
    ordered_loop_blocks_.clear();

    context_->InvalidateAnalysesExceptFor(
        IRContext::Analysis::kAnalysisLoopAnalysis);
  }

 private:
  using ValueMapTy = std::unordered_map<uint32_t, uint32_t>;
  using BlockMapTy = std::unordered_map<uint32_t, BasicBlock*>;

  Function* function_;
  Loop* loop_;
  LoopDescriptor& loop_desc_;
  IRContext* context_;

  BasicBlock* switch_block_;
  // Map between instructions and if they are dynamically uniform.
  std::unordered_map<uint32_t, bool> dynamically_uniform_;
  // The loop basic blocks in structured order.
  std::vector<BasicBlock*> ordered_loop_blocks_;

  // Returns the next usable id for the context.
  uint32_t TakeNextId() {
    // TODO(1841): Handle id overflow.
    return context_->TakeNextId();
  }

  // Simplifies |loop| assuming the instruction |to_version_insn| takes the
  // value |cst_value|. |block_range| is an iterator range returning the loop
  // basic blocks in a structured order (dominator first).
  // The function will ignore basic blocks returned by |block_range| if they
  // does not belong to the loop.
  // The set |dead_blocks| will contain all the dead basic blocks.
  //
  // Requirements:
  //   - |loop| must be in the LCSSA form;
  //   - |cst_value| must be constant.
  void SpecializeLoop(Loop* loop, Instruction* to_version_insn,
                      Instruction* cst_value) {
    analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

    std::function<bool(uint32_t)> ignore_node;
    ignore_node = [loop](uint32_t bb_id) { return !loop->IsInsideLoop(bb_id); };

    std::vector<std::pair<Instruction*, uint32_t>> use_list;
    def_use_mgr->ForEachUse(to_version_insn,
                            [&use_list, &ignore_node, this](
                                Instruction* inst, uint32_t operand_index) {
                              BasicBlock* bb = context_->get_instr_block(inst);

                              if (!bb || ignore_node(bb->id())) {
                                // Out of the loop, the specialization does not
                                // apply any more.
                                return;
                              }
                              use_list.emplace_back(inst, operand_index);
                            });

    // First pass: inject the specialized value into the loop (and only the
    // loop).
    for (auto use : use_list) {
      Instruction* inst = use.first;
      uint32_t operand_index = use.second;

      // To also handle switch, cst_value can be nullptr: this case
      // means that we are looking to branch to the default target of
      // the switch. We don't actually know its value so we don't touch
      // it if it not a switch.
      assert(cst_value && "We do not have a value to use.");
      inst->SetOperand(operand_index, {cst_value->result_id()});
      def_use_mgr->AnalyzeInstUse(inst);
    }
  }

  // Returns true if |var| is dynamically uniform.
  // Note: this is currently approximated as uniform.
  bool IsDynamicallyUniform(Instruction* var, const BasicBlock* entry,
                            const DominatorTree& post_dom_tree) {
    assert(post_dom_tree.IsPostDominator());
    analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

    auto it = dynamically_uniform_.find(var->result_id());

    if (it != dynamically_uniform_.end()) return it->second;

    analysis::DecorationManager* dec_mgr = context_->get_decoration_mgr();

    bool& is_uniform = dynamically_uniform_[var->result_id()];
    is_uniform = false;

    dec_mgr->WhileEachDecoration(var->result_id(), SpvDecorationUniform,
                                 [&is_uniform](const Instruction&) {
                                   is_uniform = true;
                                   return false;
                                 });
    if (is_uniform) {
      return is_uniform;
    }

    BasicBlock* parent = context_->get_instr_block(var);
    if (!parent) {
      return is_uniform = true;
    }

    if (!post_dom_tree.Dominates(parent->id(), entry->id())) {
      return is_uniform = false;
    }
    if (var->opcode() == SpvOpLoad) {
      const uint32_t PtrTypeId =
          def_use_mgr->GetDef(var->GetSingleWordInOperand(0))->type_id();
      const Instruction* PtrTypeInst = def_use_mgr->GetDef(PtrTypeId);
      uint32_t storage_class =
          PtrTypeInst->GetSingleWordInOperand(kTypePointerStorageClassInIdx);
      if (storage_class != SpvStorageClassUniform &&
          storage_class != SpvStorageClassUniformConstant) {
        return is_uniform = false;
      }
    } else {
      if (!context_->IsCombinatorInstruction(var)) {
        return is_uniform = false;
      }
    }

    return is_uniform = var->WhileEachInId([entry, &post_dom_tree,
                                            this](const uint32_t* id) {
      return IsDynamicallyUniform(context_->get_def_use_mgr()->GetDef(*id),
                                  entry, post_dom_tree);
    });
  }

  // Returns true if |insn| is not a constant, but is loop invariant and
  // dynamically uniform.
  bool IsConditionNonConstantLoopInvariant(Instruction* insn) {
    assert(insn->IsBranch());
    assert(insn->opcode() != SpvOpBranch);
    analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

    Instruction* condition = def_use_mgr->GetDef(insn->GetOperand(0).words[0]);
    if (condition->IsConstant()) {
      return false;
    }

    if (loop_->IsInsideLoop(condition)) {
      return false;
    }

    return IsDynamicallyUniform(
        condition, function_->entry().get(),
        context_->GetPostDominatorAnalysis(function_)->GetDomTree());
  }
};

}  // namespace

Pass::Status LoopUnswitchPass::Process() {
  bool modified = false;
  Module* module = context()->module();

  // Process each function in the module
  for (Function& f : *module) {
    modified |= ProcessFunction(&f);
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool LoopUnswitchPass::ProcessFunction(Function* f) {
  bool modified = false;
  std::unordered_set<Loop*> processed_loop;

  LoopDescriptor& loop_descriptor = *context()->GetLoopDescriptor(f);

  bool loop_changed = true;
  while (loop_changed) {
    loop_changed = false;
    for (Loop& loop :
         make_range(++TreeDFIterator<Loop>(loop_descriptor.GetDummyRootLoop()),
                    TreeDFIterator<Loop>())) {
      if (processed_loop.count(&loop)) continue;
      processed_loop.insert(&loop);

      LoopUnswitch unswitcher(context(), f, &loop, &loop_descriptor);
      while (unswitcher.CanUnswitchLoop()) {
        if (!loop.IsLCSSA()) {
          LoopUtils(context(), &loop).MakeLoopClosedSSA();
        }
        modified = true;
        loop_changed = true;
        unswitcher.PerformUnswitch();
      }
      if (loop_changed) break;
    }
  }

  return modified;
}

}  // namespace opt
}  // namespace spvtools
