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
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/cfa.h"
#include "source/opt/cfg.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/loop_utils.h"

namespace spvtools {
namespace opt {

namespace {
// Return true if |bb| is dominated by at least one block in |exits|
static inline bool DominatesAnExit(BasicBlock* bb,
                                   const std::unordered_set<BasicBlock*>& exits,
                                   const DominatorTree& dom_tree) {
  for (BasicBlock* e_bb : exits)
    if (dom_tree.Dominates(bb, e_bb)) return true;
  return false;
}

// Utility class to rewrite out-of-loop uses of an in-loop definition in terms
// of phi instructions to achieve a LCSSA form.
// For a given definition, the class user registers phi instructions using that
// definition in all loop exit blocks by which the definition escapes.
// Then, when rewriting a use of the definition, the rewriter walks the
// paths from the use the loop exits. At each step, it will insert a phi
// instruction to merge the incoming value according to exit blocks definition.
class LCSSARewriter {
 public:
  LCSSARewriter(IRContext* context, const DominatorTree& dom_tree,
                const std::unordered_set<BasicBlock*>& exit_bb,
                BasicBlock* merge_block)
      : context_(context),
        cfg_(context_->cfg()),
        dom_tree_(dom_tree),
        exit_bb_(exit_bb),
        merge_block_id_(merge_block ? merge_block->id() : 0) {}

  struct UseRewriter {
    explicit UseRewriter(LCSSARewriter* base, const Instruction& def_insn)
        : base_(base), def_insn_(def_insn) {}
    // Rewrites the use of |def_insn_| by the instruction |user| at the index
    // |operand_index| in terms of phi instruction. This recursively builds new
    // phi instructions from |user| to the loop exit blocks' phis. The use of
    // |def_insn_| in |user| is replaced by the relevant phi instruction at the
    // end of the operation.
    // It is assumed that |user| does not dominates any of the loop exit basic
    // block. This operation does not update the def/use manager, instead it
    // records what needs to be updated. The actual update is performed by
    // UpdateManagers.
    void RewriteUse(BasicBlock* bb, Instruction* user, uint32_t operand_index) {
      assert(
          (user->opcode() != SpvOpPhi || bb != GetParent(user)) &&
          "The root basic block must be the incoming edge if |user| is a phi "
          "instruction");
      assert((user->opcode() == SpvOpPhi || bb == GetParent(user)) &&
             "The root basic block must be the instruction parent if |user| is "
             "not "
             "phi instruction");

      Instruction* new_def = GetOrBuildIncoming(bb->id());

      user->SetOperand(operand_index, {new_def->result_id()});
      rewritten_.insert(user);
    }

    // In-place update of some managers (avoid full invalidation).
    inline void UpdateManagers() {
      analysis::DefUseManager* def_use_mgr = base_->context_->get_def_use_mgr();
      // Register all new definitions.
      for (Instruction* insn : rewritten_) {
        def_use_mgr->AnalyzeInstDef(insn);
      }
      // Register all new uses.
      for (Instruction* insn : rewritten_) {
        def_use_mgr->AnalyzeInstUse(insn);
      }
    }

   private:
    // Return the basic block that |instr| belongs to.
    BasicBlock* GetParent(Instruction* instr) {
      return base_->context_->get_instr_block(instr);
    }

    // Builds a phi instruction for the basic block |bb|. The function assumes
    // that |defining_blocks| contains the list of basic block that define the
    // usable value for each predecessor of |bb|.
    inline Instruction* CreatePhiInstruction(
        BasicBlock* bb, const std::vector<uint32_t>& defining_blocks) {
      std::vector<uint32_t> incomings;
      const std::vector<uint32_t>& bb_preds = base_->cfg_->preds(bb->id());
      assert(bb_preds.size() == defining_blocks.size());
      for (size_t i = 0; i < bb_preds.size(); i++) {
        incomings.push_back(
            GetOrBuildIncoming(defining_blocks[i])->result_id());
        incomings.push_back(bb_preds[i]);
      }
      InstructionBuilder builder(base_->context_, &*bb->begin(),
                                 IRContext::kAnalysisInstrToBlockMapping);
      Instruction* incoming_phi =
          builder.AddPhi(def_insn_.type_id(), incomings);

      rewritten_.insert(incoming_phi);
      return incoming_phi;
    }

    // Builds a phi instruction for the basic block |bb|, all incoming values
    // will be |value|.
    inline Instruction* CreatePhiInstruction(BasicBlock* bb,
                                             const Instruction& value) {
      std::vector<uint32_t> incomings;
      const std::vector<uint32_t>& bb_preds = base_->cfg_->preds(bb->id());
      for (size_t i = 0; i < bb_preds.size(); i++) {
        incomings.push_back(value.result_id());
        incomings.push_back(bb_preds[i]);
      }
      InstructionBuilder builder(base_->context_, &*bb->begin(),
                                 IRContext::kAnalysisInstrToBlockMapping);
      Instruction* incoming_phi =
          builder.AddPhi(def_insn_.type_id(), incomings);

      rewritten_.insert(incoming_phi);
      return incoming_phi;
    }

    // Return the new def to use for the basic block |bb_id|.
    // If |bb_id| does not have a suitable def to use then we:
    //   - return the common def used by all predecessors;
    //   - if there is no common def, then we build a new phi instr at the
    //     beginning of |bb_id| and return this new instruction.
    Instruction* GetOrBuildIncoming(uint32_t bb_id) {
      assert(base_->cfg_->block(bb_id) != nullptr && "Unknown basic block");

      Instruction*& incoming_phi = bb_to_phi_[bb_id];
      if (incoming_phi) {
        return incoming_phi;
      }

      BasicBlock* bb = &*base_->cfg_->block(bb_id);
      // If this is an exit basic block, look if there already is an eligible
      // phi instruction. An eligible phi has |def_insn_| as all incoming
      // values.
      if (base_->exit_bb_.count(bb)) {
        // Look if there is an eligible phi in this block.
        if (!bb->WhileEachPhiInst([&incoming_phi, this](Instruction* phi) {
              for (uint32_t i = 0; i < phi->NumInOperands(); i += 2) {
                if (phi->GetSingleWordInOperand(i) != def_insn_.result_id())
                  return true;
              }
              incoming_phi = phi;
              rewritten_.insert(incoming_phi);
              return false;
            })) {
          return incoming_phi;
        }
        incoming_phi = CreatePhiInstruction(bb, def_insn_);
        return incoming_phi;
      }

      // Get the block that defines the value to use for each predecessor.
      // If the vector has 1 value, then it means that this block does not need
      // to build a phi instruction unless |bb_id| is the loop merge block.
      const std::vector<uint32_t>& defining_blocks =
          base_->GetDefiningBlocks(bb_id);

      // Special case for structured loops: merge block might be different from
      // the exit block set. To maintain structured properties it will ease
      // transformations if the merge block also holds a phi instruction like
      // the exit ones.
      if (defining_blocks.size() > 1 || bb_id == base_->merge_block_id_) {
        if (defining_blocks.size() > 1) {
          incoming_phi = CreatePhiInstruction(bb, defining_blocks);
        } else {
          assert(bb_id == base_->merge_block_id_);
          incoming_phi =
              CreatePhiInstruction(bb, *GetOrBuildIncoming(defining_blocks[0]));
        }
      } else {
        incoming_phi = GetOrBuildIncoming(defining_blocks[0]);
      }

      return incoming_phi;
    }

    LCSSARewriter* base_;
    const Instruction& def_insn_;
    std::unordered_map<uint32_t, Instruction*> bb_to_phi_;
    std::unordered_set<Instruction*> rewritten_;
  };

 private:
  // Return the new def to use for the basic block |bb_id|.
  // If |bb_id| does not have a suitable def to use then we:
  //   - return the common def used by all predecessors;
  //   - if there is no common def, then we build a new phi instr at the
  //     beginning of |bb_id| and return this new instruction.
  const std::vector<uint32_t>& GetDefiningBlocks(uint32_t bb_id) {
    assert(cfg_->block(bb_id) != nullptr && "Unknown basic block");
    std::vector<uint32_t>& defining_blocks = bb_to_defining_blocks_[bb_id];

    if (defining_blocks.size()) return defining_blocks;

    // Check if one of the loop exit basic block dominates |bb_id|.
    for (const BasicBlock* e_bb : exit_bb_) {
      if (dom_tree_.Dominates(e_bb->id(), bb_id)) {
        defining_blocks.push_back(e_bb->id());
        return defining_blocks;
      }
    }

    // Process parents, they will returns their suitable blocks.
    // If they are all the same, this means this basic block is dominated by a
    // common block, so we won't need to build a phi instruction.
    for (uint32_t pred_id : cfg_->preds(bb_id)) {
      const std::vector<uint32_t>& pred_blocks = GetDefiningBlocks(pred_id);
      if (pred_blocks.size() == 1)
        defining_blocks.push_back(pred_blocks[0]);
      else
        defining_blocks.push_back(pred_id);
    }
    assert(defining_blocks.size());
    if (std::all_of(defining_blocks.begin(), defining_blocks.end(),
                    [&defining_blocks](uint32_t id) {
                      return id == defining_blocks[0];
                    })) {
      // No need for a phi.
      defining_blocks.resize(1);
    }

    return defining_blocks;
  }

  IRContext* context_;
  CFG* cfg_;
  const DominatorTree& dom_tree_;
  const std::unordered_set<BasicBlock*>& exit_bb_;
  uint32_t merge_block_id_;
  // This map represent the set of known paths. For each key, the vector
  // represent the set of blocks holding the definition to be used to build the
  // phi instruction.
  // If the vector has 0 value, then the path is unknown yet, and must be built.
  // If the vector has 1 value, then the value defined by that basic block
  //   should be used.
  // If the vector has more than 1 value, then a phi node must be created, the
  //   basic block ordering is the same as the predecessor ordering.
  std::unordered_map<uint32_t, std::vector<uint32_t>> bb_to_defining_blocks_;
};

// Make the set |blocks| closed SSA. The set is closed SSA if all the uses
// outside the set are phi instructions in exiting basic block set (hold by
// |lcssa_rewriter|).
inline void MakeSetClosedSSA(IRContext* context, Function* function,
                             const std::unordered_set<uint32_t>& blocks,
                             const std::unordered_set<BasicBlock*>& exit_bb,
                             LCSSARewriter* lcssa_rewriter) {
  CFG& cfg = *context->cfg();
  DominatorTree& dom_tree =
      context->GetDominatorAnalysis(function)->GetDomTree();
  analysis::DefUseManager* def_use_manager = context->get_def_use_mgr();

  for (uint32_t bb_id : blocks) {
    BasicBlock* bb = cfg.block(bb_id);
    // If bb does not dominate an exit block, then it cannot have escaping defs.
    if (!DominatesAnExit(bb, exit_bb, dom_tree)) continue;
    for (Instruction& inst : *bb) {
      LCSSARewriter::UseRewriter rewriter(lcssa_rewriter, inst);
      def_use_manager->ForEachUse(
          &inst, [&blocks, &rewriter, &exit_bb, context](
                     Instruction* use, uint32_t operand_index) {
            BasicBlock* use_parent = context->get_instr_block(use);
            assert(use_parent);
            if (blocks.count(use_parent->id())) return;

            if (use->opcode() == SpvOpPhi) {
              // If the use is a Phi instruction and the incoming block is
              // coming from the loop, then that's consistent with LCSSA form.
              if (exit_bb.count(use_parent)) {
                return;
              } else {
                // That's not an exit block, but the user is a phi instruction.
                // Consider the incoming branch only.
                use_parent = context->get_instr_block(
                    use->GetSingleWordOperand(operand_index + 1));
              }
            }
            // Rewrite the use. Note that this call does not invalidate the
            // def/use manager. So this operation is safe.
            rewriter.RewriteUse(use_parent, use, operand_index);
          });
      rewriter.UpdateManagers();
    }
  }
}

}  // namespace

void LoopUtils::CreateLoopDedicatedExits() {
  Function* function = loop_->GetHeaderBlock()->GetParent();
  LoopDescriptor& loop_desc = *context_->GetLoopDescriptor(function);
  CFG& cfg = *context_->cfg();
  analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

  const IRContext::Analysis PreservedAnalyses =
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping;

  // Gathers the set of basic block that are not in this loop and have at least
  // one predecessor in the loop and one not in the loop.
  std::unordered_set<uint32_t> exit_bb_set;
  loop_->GetExitBlocks(&exit_bb_set);

  std::unordered_set<BasicBlock*> new_loop_exits;
  bool made_change = false;
  // For each block, we create a new one that gathers all branches from
  // the loop and fall into the block.
  for (uint32_t non_dedicate_id : exit_bb_set) {
    BasicBlock* non_dedicate = cfg.block(non_dedicate_id);
    const std::vector<uint32_t>& bb_pred = cfg.preds(non_dedicate_id);
    // Ignore the block if all the predecessors are in the loop.
    if (std::all_of(bb_pred.begin(), bb_pred.end(),
                    [this](uint32_t id) { return loop_->IsInsideLoop(id); })) {
      new_loop_exits.insert(non_dedicate);
      continue;
    }

    made_change = true;
    Function::iterator insert_pt = function->begin();
    for (; insert_pt != function->end() && &*insert_pt != non_dedicate;
         ++insert_pt) {
    }
    assert(insert_pt != function->end() && "Basic Block not found");

    // Create the dedicate exit basic block.
    // TODO(1841): Handle id overflow.
    BasicBlock& exit = *insert_pt.InsertBefore(std::unique_ptr<BasicBlock>(
        new BasicBlock(std::unique_ptr<Instruction>(new Instruction(
            context_, SpvOpLabel, 0, context_->TakeNextId(), {})))));
    exit.SetParent(function);

    // Redirect in loop predecessors to |exit| block.
    for (uint32_t exit_pred_id : bb_pred) {
      if (loop_->IsInsideLoop(exit_pred_id)) {
        BasicBlock* pred_block = cfg.block(exit_pred_id);
        pred_block->ForEachSuccessorLabel([non_dedicate, &exit](uint32_t* id) {
          if (*id == non_dedicate->id()) *id = exit.id();
        });
        // Update the CFG.
        // |non_dedicate|'s predecessor list will be updated at the end of the
        // loop.
        cfg.RegisterBlock(pred_block);
      }
    }

    // Register the label to the def/use manager, requires for the phi patching.
    def_use_mgr->AnalyzeInstDefUse(exit.GetLabelInst());
    context_->set_instr_block(exit.GetLabelInst(), &exit);

    InstructionBuilder builder(context_, &exit, PreservedAnalyses);
    // Now jump from our dedicate basic block to the old exit.
    // We also reset the insert point so all instructions are inserted before
    // the branch.
    builder.SetInsertPoint(builder.AddBranch(non_dedicate->id()));
    non_dedicate->ForEachPhiInst(
        [&builder, &exit, def_use_mgr, this](Instruction* phi) {
          // New phi operands for this instruction.
          std::vector<uint32_t> new_phi_op;
          // Phi operands for the dedicated exit block.
          std::vector<uint32_t> exit_phi_op;
          for (uint32_t i = 0; i < phi->NumInOperands(); i += 2) {
            uint32_t def_id = phi->GetSingleWordInOperand(i);
            uint32_t incoming_id = phi->GetSingleWordInOperand(i + 1);
            if (loop_->IsInsideLoop(incoming_id)) {
              exit_phi_op.push_back(def_id);
              exit_phi_op.push_back(incoming_id);
            } else {
              new_phi_op.push_back(def_id);
              new_phi_op.push_back(incoming_id);
            }
          }

          // Build the new phi instruction dedicated exit block.
          Instruction* exit_phi = builder.AddPhi(phi->type_id(), exit_phi_op);
          // Build the new incoming branch.
          new_phi_op.push_back(exit_phi->result_id());
          new_phi_op.push_back(exit.id());
          // Rewrite operands.
          uint32_t idx = 0;
          for (; idx < new_phi_op.size(); idx++)
            phi->SetInOperand(idx, {new_phi_op[idx]});
          // Remove extra operands, from last to first (more efficient).
          for (uint32_t j = phi->NumInOperands() - 1; j >= idx; j--)
            phi->RemoveInOperand(j);
          // Update the def/use manager for this |phi|.
          def_use_mgr->AnalyzeInstUse(phi);
        });
    // Update the CFG.
    cfg.RegisterBlock(&exit);
    cfg.RemoveNonExistingEdges(non_dedicate->id());
    new_loop_exits.insert(&exit);
    // If non_dedicate is in a loop, add the new dedicated exit in that loop.
    if (Loop* parent_loop = loop_desc[non_dedicate])
      parent_loop->AddBasicBlock(&exit);
  }

  if (new_loop_exits.size() == 1) {
    loop_->SetMergeBlock(*new_loop_exits.begin());
  }

  if (made_change) {
    context_->InvalidateAnalysesExceptFor(
        PreservedAnalyses | IRContext::kAnalysisCFG |
        IRContext::Analysis::kAnalysisLoopAnalysis);
  }
}

void LoopUtils::MakeLoopClosedSSA() {
  CreateLoopDedicatedExits();

  Function* function = loop_->GetHeaderBlock()->GetParent();
  CFG& cfg = *context_->cfg();
  DominatorTree& dom_tree =
      context_->GetDominatorAnalysis(function)->GetDomTree();

  std::unordered_set<BasicBlock*> exit_bb;
  {
    std::unordered_set<uint32_t> exit_bb_id;
    loop_->GetExitBlocks(&exit_bb_id);
    for (uint32_t bb_id : exit_bb_id) {
      exit_bb.insert(cfg.block(bb_id));
    }
  }

  LCSSARewriter lcssa_rewriter(context_, dom_tree, exit_bb,
                               loop_->GetMergeBlock());
  MakeSetClosedSSA(context_, function, loop_->GetBlocks(), exit_bb,
                   &lcssa_rewriter);

  // Make sure all defs post-dominated by the merge block have their last use no
  // further than the merge block.
  if (loop_->GetMergeBlock()) {
    std::unordered_set<uint32_t> merging_bb_id;
    loop_->GetMergingBlocks(&merging_bb_id);
    merging_bb_id.erase(loop_->GetMergeBlock()->id());
    // Reset the exit set, now only the merge block is the exit.
    exit_bb.clear();
    exit_bb.insert(loop_->GetMergeBlock());
    // LCSSARewriter is reusable here only because it forces the creation of a
    // phi instruction in the merge block.
    MakeSetClosedSSA(context_, function, merging_bb_id, exit_bb,
                     &lcssa_rewriter);
  }

  context_->InvalidateAnalysesExceptFor(
      IRContext::Analysis::kAnalysisCFG |
      IRContext::Analysis::kAnalysisDominatorAnalysis |
      IRContext::Analysis::kAnalysisLoopAnalysis);
}

Loop* LoopUtils::CloneLoop(LoopCloningResult* cloning_result) const {
  // Compute the structured order of the loop basic blocks and store it in the
  // vector ordered_loop_blocks.
  std::vector<BasicBlock*> ordered_loop_blocks;
  loop_->ComputeLoopStructuredOrder(&ordered_loop_blocks);

  // Clone the loop.
  return CloneLoop(cloning_result, ordered_loop_blocks);
}

Loop* LoopUtils::CloneAndAttachLoopToHeader(LoopCloningResult* cloning_result) {
  // Clone the loop.
  Loop* new_loop = CloneLoop(cloning_result);

  // Create a new exit block/label for the new loop.
  // TODO(1841): Handle id overflow.
  std::unique_ptr<Instruction> new_label{new Instruction(
      context_, SpvOp::SpvOpLabel, 0, context_->TakeNextId(), {})};
  std::unique_ptr<BasicBlock> new_exit_bb{new BasicBlock(std::move(new_label))};
  new_exit_bb->SetParent(loop_->GetMergeBlock()->GetParent());

  // Create an unconditional branch to the header block.
  InstructionBuilder builder{context_, new_exit_bb.get()};
  builder.AddBranch(loop_->GetHeaderBlock()->id());

  // Save the ids of the new and old merge block.
  const uint32_t old_merge_block = loop_->GetMergeBlock()->id();
  const uint32_t new_merge_block = new_exit_bb->id();

  // Replace the uses of the old merge block in the new loop with the new merge
  // block.
  for (std::unique_ptr<BasicBlock>& basic_block : cloning_result->cloned_bb_) {
    for (Instruction& inst : *basic_block) {
      // For each operand in each instruction check if it is using the old merge
      // block and change it to be the new merge block.
      auto replace_merge_use = [old_merge_block,
                                new_merge_block](uint32_t* id) {
        if (*id == old_merge_block) *id = new_merge_block;
      };
      inst.ForEachInOperand(replace_merge_use);
    }
  }

  const uint32_t old_header = loop_->GetHeaderBlock()->id();
  const uint32_t new_header = new_loop->GetHeaderBlock()->id();
  analysis::DefUseManager* def_use = context_->get_def_use_mgr();

  def_use->ForEachUse(old_header,
                      [new_header, this](Instruction* inst, uint32_t operand) {
                        if (!this->loop_->IsInsideLoop(inst))
                          inst->SetOperand(operand, {new_header});
                      });

  // TODO(1841): Handle failure to create pre-header.
  def_use->ForEachUse(
      loop_->GetOrCreatePreHeaderBlock()->id(),
      [new_merge_block, this](Instruction* inst, uint32_t operand) {
        if (this->loop_->IsInsideLoop(inst))
          inst->SetOperand(operand, {new_merge_block});

      });
  new_loop->SetMergeBlock(new_exit_bb.get());

  new_loop->SetPreHeaderBlock(loop_->GetPreHeaderBlock());

  // Add the new block into the cloned instructions.
  cloning_result->cloned_bb_.push_back(std::move(new_exit_bb));

  return new_loop;
}

Loop* LoopUtils::CloneLoop(
    LoopCloningResult* cloning_result,
    const std::vector<BasicBlock*>& ordered_loop_blocks) const {
  analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

  std::unique_ptr<Loop> new_loop = MakeUnique<Loop>(context_);

  CFG& cfg = *context_->cfg();

  // Clone and place blocks in a SPIR-V compliant order (dominators first).
  for (BasicBlock* old_bb : ordered_loop_blocks) {
    // For each basic block in the loop, we clone it and register the mapping
    // between old and new ids.
    BasicBlock* new_bb = old_bb->Clone(context_);
    new_bb->SetParent(&function_);
    // TODO(1841): Handle id overflow.
    new_bb->GetLabelInst()->SetResultId(context_->TakeNextId());
    def_use_mgr->AnalyzeInstDef(new_bb->GetLabelInst());
    context_->set_instr_block(new_bb->GetLabelInst(), new_bb);
    cloning_result->cloned_bb_.emplace_back(new_bb);

    cloning_result->old_to_new_bb_[old_bb->id()] = new_bb;
    cloning_result->new_to_old_bb_[new_bb->id()] = old_bb;
    cloning_result->value_map_[old_bb->id()] = new_bb->id();

    if (loop_->IsInsideLoop(old_bb)) new_loop->AddBasicBlock(new_bb);

    for (auto new_inst = new_bb->begin(), old_inst = old_bb->begin();
         new_inst != new_bb->end(); ++new_inst, ++old_inst) {
      cloning_result->ptr_map_[&*new_inst] = &*old_inst;
      if (new_inst->HasResultId()) {
        // TODO(1841): Handle id overflow.
        new_inst->SetResultId(context_->TakeNextId());
        cloning_result->value_map_[old_inst->result_id()] =
            new_inst->result_id();

        // Only look at the defs for now, uses are not updated yet.
        def_use_mgr->AnalyzeInstDef(&*new_inst);
      }
    }
  }

  // All instructions (including all labels) have been cloned,
  // remap instruction operands id with the new ones.
  for (std::unique_ptr<BasicBlock>& bb_ref : cloning_result->cloned_bb_) {
    BasicBlock* bb = bb_ref.get();

    for (Instruction& insn : *bb) {
      insn.ForEachInId([cloning_result](uint32_t* old_id) {
        // If the operand is defined in the loop, remap the id.
        auto id_it = cloning_result->value_map_.find(*old_id);
        if (id_it != cloning_result->value_map_.end()) {
          *old_id = id_it->second;
        }
      });
      // Only look at what the instruction uses. All defs are register, so all
      // should be fine now.
      def_use_mgr->AnalyzeInstUse(&insn);
      context_->set_instr_block(&insn, bb);
    }
    cfg.RegisterBlock(bb);
  }

  PopulateLoopNest(new_loop.get(), *cloning_result);

  return new_loop.release();
}

void LoopUtils::PopulateLoopNest(
    Loop* new_loop, const LoopCloningResult& cloning_result) const {
  std::unordered_map<Loop*, Loop*> loop_mapping;
  loop_mapping[loop_] = new_loop;

  if (loop_->HasParent()) loop_->GetParent()->AddNestedLoop(new_loop);
  PopulateLoopDesc(new_loop, loop_, cloning_result);

  for (Loop& sub_loop :
       make_range(++TreeDFIterator<Loop>(loop_), TreeDFIterator<Loop>())) {
    Loop* cloned = new Loop(context_);
    if (Loop* parent = loop_mapping[sub_loop.GetParent()])
      parent->AddNestedLoop(cloned);
    loop_mapping[&sub_loop] = cloned;
    PopulateLoopDesc(cloned, &sub_loop, cloning_result);
  }

  loop_desc_->AddLoopNest(std::unique_ptr<Loop>(new_loop));
}

// Populates |new_loop| descriptor according to |old_loop|'s one.
void LoopUtils::PopulateLoopDesc(
    Loop* new_loop, Loop* old_loop,
    const LoopCloningResult& cloning_result) const {
  for (uint32_t bb_id : old_loop->GetBlocks()) {
    BasicBlock* bb = cloning_result.old_to_new_bb_.at(bb_id);
    new_loop->AddBasicBlock(bb);
  }
  new_loop->SetHeaderBlock(
      cloning_result.old_to_new_bb_.at(old_loop->GetHeaderBlock()->id()));
  if (old_loop->GetLatchBlock())
    new_loop->SetLatchBlock(
        cloning_result.old_to_new_bb_.at(old_loop->GetLatchBlock()->id()));
  if (old_loop->GetContinueBlock())
    new_loop->SetContinueBlock(
        cloning_result.old_to_new_bb_.at(old_loop->GetContinueBlock()->id()));
  if (old_loop->GetMergeBlock()) {
    auto it =
        cloning_result.old_to_new_bb_.find(old_loop->GetMergeBlock()->id());
    BasicBlock* bb = it != cloning_result.old_to_new_bb_.end()
                         ? it->second
                         : old_loop->GetMergeBlock();
    new_loop->SetMergeBlock(bb);
  }
  if (old_loop->GetPreHeaderBlock()) {
    auto it =
        cloning_result.old_to_new_bb_.find(old_loop->GetPreHeaderBlock()->id());
    if (it != cloning_result.old_to_new_bb_.end()) {
      new_loop->SetPreHeaderBlock(it->second);
    }
  }
}

// Class to gather some metrics about a region of interest.
void CodeMetrics::Analyze(const Loop& loop) {
  CFG& cfg = *loop.GetContext()->cfg();

  roi_size_ = 0;
  block_sizes_.clear();

  for (uint32_t id : loop.GetBlocks()) {
    const BasicBlock* bb = cfg.block(id);
    size_t bb_size = 0;
    bb->ForEachInst([&bb_size](const Instruction* insn) {
      if (insn->opcode() == SpvOpLabel) return;
      if (insn->IsNop()) return;
      if (insn->opcode() == SpvOpPhi) return;
      bb_size++;
    });
    block_sizes_[bb->id()] = bb_size;
    roi_size_ += bb_size;
  }
}

}  // namespace opt
}  // namespace spvtools
