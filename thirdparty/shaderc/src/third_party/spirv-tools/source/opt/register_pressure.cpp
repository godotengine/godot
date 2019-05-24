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

#include "source/opt/register_pressure.h"

#include <algorithm>
#include <iterator>

#include "source/opt/cfg.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/dominator_tree.h"
#include "source/opt/function.h"
#include "source/opt/ir_context.h"
#include "source/opt/iterator.h"

namespace spvtools {
namespace opt {

namespace {
// Predicate for the FilterIterator to only consider instructions that are not
// phi instructions defined in the basic block |bb|.
class ExcludePhiDefinedInBlock {
 public:
  ExcludePhiDefinedInBlock(IRContext* context, const BasicBlock* bb)
      : context_(context), bb_(bb) {}

  bool operator()(Instruction* insn) const {
    return !(insn->opcode() == SpvOpPhi &&
             context_->get_instr_block(insn) == bb_);
  }

 private:
  IRContext* context_;
  const BasicBlock* bb_;
};

// Returns true if |insn| generates a SSA register that is likely to require a
// physical register.
bool CreatesRegisterUsage(Instruction* insn) {
  if (!insn->HasResultId()) return false;
  if (insn->opcode() == SpvOpUndef) return false;
  if (IsConstantInst(insn->opcode())) return false;
  if (insn->opcode() == SpvOpLabel) return false;
  return true;
}

// Compute the register liveness for each basic block of a function. This also
// fill-up some information about the pick register usage and a break down of
// register usage. This implements: "A non-iterative data-flow algorithm for
// computing liveness sets in strict ssa programs" from Boissinot et al.
class ComputeRegisterLiveness {
 public:
  ComputeRegisterLiveness(RegisterLiveness* reg_pressure, Function* f)
      : reg_pressure_(reg_pressure),
        context_(reg_pressure->GetContext()),
        function_(f),
        cfg_(*reg_pressure->GetContext()->cfg()),
        def_use_manager_(*reg_pressure->GetContext()->get_def_use_mgr()),
        dom_tree_(
            reg_pressure->GetContext()->GetDominatorAnalysis(f)->GetDomTree()),
        loop_desc_(*reg_pressure->GetContext()->GetLoopDescriptor(f)) {}

  // Computes the register liveness for |function_| and then estimate the
  // register usage. The liveness algorithm works in 2 steps:
  //   - First, compute the liveness for each basic blocks, but will ignore any
  //   back-edge;
  //   - Second, walk loop forest to propagate registers crossing back-edges
  //   (add iterative values into the liveness set).
  void Compute() {
    cfg_.ForEachBlockInPostOrder(&*function_->begin(), [this](BasicBlock* bb) {
      ComputePartialLiveness(bb);
    });
    DoLoopLivenessUnification();
    EvaluateRegisterRequirements();
  }

 private:
  // Registers all SSA register used by successors of |bb| in their phi
  // instructions.
  void ComputePhiUses(const BasicBlock& bb,
                      RegisterLiveness::RegionRegisterLiveness::LiveSet* live) {
    uint32_t bb_id = bb.id();
    bb.ForEachSuccessorLabel([live, bb_id, this](uint32_t sid) {
      BasicBlock* succ_bb = cfg_.block(sid);
      succ_bb->ForEachPhiInst([live, bb_id, this](const Instruction* phi) {
        for (uint32_t i = 0; i < phi->NumInOperands(); i += 2) {
          if (phi->GetSingleWordInOperand(i + 1) == bb_id) {
            Instruction* insn_op =
                def_use_manager_.GetDef(phi->GetSingleWordInOperand(i));
            if (CreatesRegisterUsage(insn_op)) {
              live->insert(insn_op);
              break;
            }
          }
        }
      });
    });
  }

  // Computes register liveness for each basic blocks but ignores all
  // back-edges.
  void ComputePartialLiveness(BasicBlock* bb) {
    assert(reg_pressure_->Get(bb) == nullptr &&
           "Basic block already processed");

    RegisterLiveness::RegionRegisterLiveness* live_inout =
        reg_pressure_->GetOrInsert(bb->id());
    ComputePhiUses(*bb, &live_inout->live_out_);

    const BasicBlock* cbb = bb;
    cbb->ForEachSuccessorLabel([&live_inout, bb, this](uint32_t sid) {
      // Skip back edges.
      if (dom_tree_.Dominates(sid, bb->id())) {
        return;
      }

      BasicBlock* succ_bb = cfg_.block(sid);
      RegisterLiveness::RegionRegisterLiveness* succ_live_inout =
          reg_pressure_->Get(succ_bb);
      assert(succ_live_inout &&
             "Successor liveness analysis was not performed");

      ExcludePhiDefinedInBlock predicate(context_, succ_bb);
      auto filter =
          MakeFilterIteratorRange(succ_live_inout->live_in_.begin(),
                                  succ_live_inout->live_in_.end(), predicate);
      live_inout->live_out_.insert(filter.begin(), filter.end());
    });

    live_inout->live_in_ = live_inout->live_out_;
    for (Instruction& insn : make_range(bb->rbegin(), bb->rend())) {
      if (insn.opcode() == SpvOpPhi) {
        live_inout->live_in_.insert(&insn);
        break;
      }
      live_inout->live_in_.erase(&insn);
      insn.ForEachInId([live_inout, this](uint32_t* id) {
        Instruction* insn_op = def_use_manager_.GetDef(*id);
        if (CreatesRegisterUsage(insn_op)) {
          live_inout->live_in_.insert(insn_op);
        }
      });
    }
  }

  // Propagates the register liveness information of each loop iterators.
  void DoLoopLivenessUnification() {
    for (const Loop* loop : *loop_desc_.GetDummyRootLoop()) {
      DoLoopLivenessUnification(*loop);
    }
  }

  // Propagates the register liveness information of loop iterators trough-out
  // the loop body.
  void DoLoopLivenessUnification(const Loop& loop) {
    auto blocks_in_loop = MakeFilterIteratorRange(
        loop.GetBlocks().begin(), loop.GetBlocks().end(),
        [&loop, this](uint32_t bb_id) {
          return bb_id != loop.GetHeaderBlock()->id() &&
                 loop_desc_[bb_id] == &loop;
        });

    RegisterLiveness::RegionRegisterLiveness* header_live_inout =
        reg_pressure_->Get(loop.GetHeaderBlock());
    assert(header_live_inout &&
           "Liveness analysis was not performed for the current block");

    ExcludePhiDefinedInBlock predicate(context_, loop.GetHeaderBlock());
    auto live_loop =
        MakeFilterIteratorRange(header_live_inout->live_in_.begin(),
                                header_live_inout->live_in_.end(), predicate);

    for (uint32_t bb_id : blocks_in_loop) {
      BasicBlock* bb = cfg_.block(bb_id);

      RegisterLiveness::RegionRegisterLiveness* live_inout =
          reg_pressure_->Get(bb);
      live_inout->live_in_.insert(live_loop.begin(), live_loop.end());
      live_inout->live_out_.insert(live_loop.begin(), live_loop.end());
    }

    for (const Loop* inner_loop : loop) {
      RegisterLiveness::RegionRegisterLiveness* live_inout =
          reg_pressure_->Get(inner_loop->GetHeaderBlock());
      live_inout->live_in_.insert(live_loop.begin(), live_loop.end());
      live_inout->live_out_.insert(live_loop.begin(), live_loop.end());

      DoLoopLivenessUnification(*inner_loop);
    }
  }

  // Get the number of required registers for this each basic block.
  void EvaluateRegisterRequirements() {
    for (BasicBlock& bb : *function_) {
      RegisterLiveness::RegionRegisterLiveness* live_inout =
          reg_pressure_->Get(bb.id());
      assert(live_inout != nullptr && "Basic block not processed");

      size_t reg_count = live_inout->live_out_.size();
      for (Instruction* insn : live_inout->live_out_) {
        live_inout->AddRegisterClass(insn);
      }
      live_inout->used_registers_ = reg_count;

      std::unordered_set<uint32_t> die_in_block;
      for (Instruction& insn : make_range(bb.rbegin(), bb.rend())) {
        // If it is a phi instruction, the register pressure will not change
        // anymore.
        if (insn.opcode() == SpvOpPhi) {
          break;
        }

        insn.ForEachInId(
            [live_inout, &die_in_block, &reg_count, this](uint32_t* id) {
              Instruction* op_insn = def_use_manager_.GetDef(*id);
              if (!CreatesRegisterUsage(op_insn) ||
                  live_inout->live_out_.count(op_insn)) {
                // already taken into account.
                return;
              }
              if (!die_in_block.count(*id)) {
                live_inout->AddRegisterClass(def_use_manager_.GetDef(*id));
                reg_count++;
                die_in_block.insert(*id);
              }
            });
        live_inout->used_registers_ =
            std::max(live_inout->used_registers_, reg_count);
        if (CreatesRegisterUsage(&insn)) {
          reg_count--;
        }
      }
    }
  }

  RegisterLiveness* reg_pressure_;
  IRContext* context_;
  Function* function_;
  CFG& cfg_;
  analysis::DefUseManager& def_use_manager_;
  DominatorTree& dom_tree_;
  LoopDescriptor& loop_desc_;
};
}  // namespace

// Get the number of required registers for each basic block.
void RegisterLiveness::RegionRegisterLiveness::AddRegisterClass(
    Instruction* insn) {
  assert(CreatesRegisterUsage(insn) && "Instruction does not use a register");
  analysis::Type* type =
      insn->context()->get_type_mgr()->GetType(insn->type_id());

  RegisterLiveness::RegisterClass reg_class{type, false};

  insn->context()->get_decoration_mgr()->WhileEachDecoration(
      insn->result_id(), SpvDecorationUniform,
      [&reg_class](const Instruction&) {
        reg_class.is_uniform_ = true;
        return false;
      });

  AddRegisterClass(reg_class);
}

void RegisterLiveness::Analyze(Function* f) {
  block_pressure_.clear();
  ComputeRegisterLiveness(this, f).Compute();
}

void RegisterLiveness::ComputeLoopRegisterPressure(
    const Loop& loop, RegionRegisterLiveness* loop_reg_pressure) const {
  loop_reg_pressure->Clear();

  const RegionRegisterLiveness* header_live_inout = Get(loop.GetHeaderBlock());
  loop_reg_pressure->live_in_ = header_live_inout->live_in_;

  std::unordered_set<uint32_t> exit_blocks;
  loop.GetExitBlocks(&exit_blocks);

  for (uint32_t bb_id : exit_blocks) {
    const RegionRegisterLiveness* live_inout = Get(bb_id);
    loop_reg_pressure->live_out_.insert(live_inout->live_in_.begin(),
                                        live_inout->live_in_.end());
  }

  std::unordered_set<uint32_t> seen_insn;
  for (Instruction* insn : loop_reg_pressure->live_out_) {
    loop_reg_pressure->AddRegisterClass(insn);
    seen_insn.insert(insn->result_id());
  }
  for (Instruction* insn : loop_reg_pressure->live_in_) {
    if (!seen_insn.count(insn->result_id())) {
      continue;
    }
    loop_reg_pressure->AddRegisterClass(insn);
    seen_insn.insert(insn->result_id());
  }

  loop_reg_pressure->used_registers_ = 0;

  for (uint32_t bb_id : loop.GetBlocks()) {
    BasicBlock* bb = context_->cfg()->block(bb_id);

    const RegionRegisterLiveness* live_inout = Get(bb_id);
    assert(live_inout != nullptr && "Basic block not processed");
    loop_reg_pressure->used_registers_ = std::max(
        loop_reg_pressure->used_registers_, live_inout->used_registers_);

    for (Instruction& insn : *bb) {
      if (insn.opcode() == SpvOpPhi || !CreatesRegisterUsage(&insn) ||
          seen_insn.count(insn.result_id())) {
        continue;
      }
      loop_reg_pressure->AddRegisterClass(&insn);
    }
  }
}

void RegisterLiveness::SimulateFusion(
    const Loop& l1, const Loop& l2, RegionRegisterLiveness* sim_result) const {
  sim_result->Clear();

  // Compute the live-in state:
  //   sim_result.live_in = l1.live_in U l2.live_in
  // This assumes that |l1| does not generated register that is live-out for
  // |l1|.
  const RegionRegisterLiveness* l1_header_live_inout = Get(l1.GetHeaderBlock());
  sim_result->live_in_ = l1_header_live_inout->live_in_;

  const RegionRegisterLiveness* l2_header_live_inout = Get(l2.GetHeaderBlock());
  sim_result->live_in_.insert(l2_header_live_inout->live_in_.begin(),
                              l2_header_live_inout->live_in_.end());

  // The live-out set of the fused loop is the l2 live-out set.
  std::unordered_set<uint32_t> exit_blocks;
  l2.GetExitBlocks(&exit_blocks);

  for (uint32_t bb_id : exit_blocks) {
    const RegionRegisterLiveness* live_inout = Get(bb_id);
    sim_result->live_out_.insert(live_inout->live_in_.begin(),
                                 live_inout->live_in_.end());
  }

  // Compute the register usage information.
  std::unordered_set<uint32_t> seen_insn;
  for (Instruction* insn : sim_result->live_out_) {
    sim_result->AddRegisterClass(insn);
    seen_insn.insert(insn->result_id());
  }
  for (Instruction* insn : sim_result->live_in_) {
    if (!seen_insn.count(insn->result_id())) {
      continue;
    }
    sim_result->AddRegisterClass(insn);
    seen_insn.insert(insn->result_id());
  }

  sim_result->used_registers_ = 0;

  // The loop fusion is injecting the l1 before the l2, the latch of l1 will be
  // connected to the header of l2.
  // To compute the register usage, we inject the loop live-in (union of l1 and
  // l2 live-in header blocks) into the the live in/out of each basic block of
  // l1 to get the peak register usage. We then repeat the operation to for l2
  // basic blocks but in this case we inject the live-out of the latch of l1.
  auto live_loop = MakeFilterIteratorRange(
      sim_result->live_in_.begin(), sim_result->live_in_.end(),
      [&l1, &l2](Instruction* insn) {
        BasicBlock* bb = insn->context()->get_instr_block(insn);
        return insn->HasResultId() &&
               !(insn->opcode() == SpvOpPhi &&
                 (bb == l1.GetHeaderBlock() || bb == l2.GetHeaderBlock()));
      });

  for (uint32_t bb_id : l1.GetBlocks()) {
    BasicBlock* bb = context_->cfg()->block(bb_id);

    const RegionRegisterLiveness* live_inout_info = Get(bb_id);
    assert(live_inout_info != nullptr && "Basic block not processed");
    RegionRegisterLiveness::LiveSet live_out = live_inout_info->live_out_;
    live_out.insert(live_loop.begin(), live_loop.end());
    sim_result->used_registers_ =
        std::max(sim_result->used_registers_,
                 live_inout_info->used_registers_ + live_out.size() -
                     live_inout_info->live_out_.size());

    for (Instruction& insn : *bb) {
      if (insn.opcode() == SpvOpPhi || !CreatesRegisterUsage(&insn) ||
          seen_insn.count(insn.result_id())) {
        continue;
      }
      sim_result->AddRegisterClass(&insn);
    }
  }

  const RegionRegisterLiveness* l1_latch_live_inout_info =
      Get(l1.GetLatchBlock()->id());
  assert(l1_latch_live_inout_info != nullptr && "Basic block not processed");
  RegionRegisterLiveness::LiveSet l1_latch_live_out =
      l1_latch_live_inout_info->live_out_;
  l1_latch_live_out.insert(live_loop.begin(), live_loop.end());

  auto live_loop_l2 =
      make_range(l1_latch_live_out.begin(), l1_latch_live_out.end());

  for (uint32_t bb_id : l2.GetBlocks()) {
    BasicBlock* bb = context_->cfg()->block(bb_id);

    const RegionRegisterLiveness* live_inout_info = Get(bb_id);
    assert(live_inout_info != nullptr && "Basic block not processed");
    RegionRegisterLiveness::LiveSet live_out = live_inout_info->live_out_;
    live_out.insert(live_loop_l2.begin(), live_loop_l2.end());
    sim_result->used_registers_ =
        std::max(sim_result->used_registers_,
                 live_inout_info->used_registers_ + live_out.size() -
                     live_inout_info->live_out_.size());

    for (Instruction& insn : *bb) {
      if (insn.opcode() == SpvOpPhi || !CreatesRegisterUsage(&insn) ||
          seen_insn.count(insn.result_id())) {
        continue;
      }
      sim_result->AddRegisterClass(&insn);
    }
  }
}

void RegisterLiveness::SimulateFission(
    const Loop& loop, const std::unordered_set<Instruction*>& moved_inst,
    const std::unordered_set<Instruction*>& copied_inst,
    RegionRegisterLiveness* l1_sim_result,
    RegionRegisterLiveness* l2_sim_result) const {
  l1_sim_result->Clear();
  l2_sim_result->Clear();

  // Filter predicates: consider instructions that only belong to the first and
  // second loop.
  auto belong_to_loop1 = [&moved_inst, &copied_inst, &loop](Instruction* insn) {
    return moved_inst.count(insn) || copied_inst.count(insn) ||
           !loop.IsInsideLoop(insn);
  };
  auto belong_to_loop2 = [&moved_inst](Instruction* insn) {
    return !moved_inst.count(insn);
  };

  const RegionRegisterLiveness* header_live_inout = Get(loop.GetHeaderBlock());
  // l1 live-in
  {
    auto live_loop = MakeFilterIteratorRange(
        header_live_inout->live_in_.begin(), header_live_inout->live_in_.end(),
        belong_to_loop1);
    l1_sim_result->live_in_.insert(live_loop.begin(), live_loop.end());
  }
  // l2 live-in
  {
    auto live_loop = MakeFilterIteratorRange(
        header_live_inout->live_in_.begin(), header_live_inout->live_in_.end(),
        belong_to_loop2);
    l2_sim_result->live_in_.insert(live_loop.begin(), live_loop.end());
  }

  std::unordered_set<uint32_t> exit_blocks;
  loop.GetExitBlocks(&exit_blocks);

  // l2 live-out.
  for (uint32_t bb_id : exit_blocks) {
    const RegionRegisterLiveness* live_inout = Get(bb_id);
    l2_sim_result->live_out_.insert(live_inout->live_in_.begin(),
                                    live_inout->live_in_.end());
  }
  // l1 live-out.
  {
    auto live_out = MakeFilterIteratorRange(l2_sim_result->live_out_.begin(),
                                            l2_sim_result->live_out_.end(),
                                            belong_to_loop1);
    l1_sim_result->live_out_.insert(live_out.begin(), live_out.end());
  }
  {
    auto live_out =
        MakeFilterIteratorRange(l2_sim_result->live_in_.begin(),
                                l2_sim_result->live_in_.end(), belong_to_loop1);
    l1_sim_result->live_out_.insert(live_out.begin(), live_out.end());
  }
  // Lives out of l1 are live out of l2 so are live in of l2 as well.
  l2_sim_result->live_in_.insert(l1_sim_result->live_out_.begin(),
                                 l1_sim_result->live_out_.end());

  for (Instruction* insn : l1_sim_result->live_in_) {
    l1_sim_result->AddRegisterClass(insn);
  }
  for (Instruction* insn : l2_sim_result->live_in_) {
    l2_sim_result->AddRegisterClass(insn);
  }

  l1_sim_result->used_registers_ = 0;
  l2_sim_result->used_registers_ = 0;

  for (uint32_t bb_id : loop.GetBlocks()) {
    BasicBlock* bb = context_->cfg()->block(bb_id);

    const RegisterLiveness::RegionRegisterLiveness* live_inout = Get(bb_id);
    assert(live_inout != nullptr && "Basic block not processed");
    auto l1_block_live_out =
        MakeFilterIteratorRange(live_inout->live_out_.begin(),
                                live_inout->live_out_.end(), belong_to_loop1);
    auto l2_block_live_out =
        MakeFilterIteratorRange(live_inout->live_out_.begin(),
                                live_inout->live_out_.end(), belong_to_loop2);

    size_t l1_reg_count =
        std::distance(l1_block_live_out.begin(), l1_block_live_out.end());
    size_t l2_reg_count =
        std::distance(l2_block_live_out.begin(), l2_block_live_out.end());

    std::unordered_set<uint32_t> die_in_block;
    for (Instruction& insn : make_range(bb->rbegin(), bb->rend())) {
      if (insn.opcode() == SpvOpPhi) {
        break;
      }

      bool does_belong_to_loop1 = belong_to_loop1(&insn);
      bool does_belong_to_loop2 = belong_to_loop2(&insn);
      insn.ForEachInId([live_inout, &die_in_block, &l1_reg_count, &l2_reg_count,
                        does_belong_to_loop1, does_belong_to_loop2,
                        this](uint32_t* id) {
        Instruction* op_insn = context_->get_def_use_mgr()->GetDef(*id);
        if (!CreatesRegisterUsage(op_insn) ||
            live_inout->live_out_.count(op_insn)) {
          // already taken into account.
          return;
        }
        if (!die_in_block.count(*id)) {
          if (does_belong_to_loop1) {
            l1_reg_count++;
          }
          if (does_belong_to_loop2) {
            l2_reg_count++;
          }
          die_in_block.insert(*id);
        }
      });
      l1_sim_result->used_registers_ =
          std::max(l1_sim_result->used_registers_, l1_reg_count);
      l2_sim_result->used_registers_ =
          std::max(l2_sim_result->used_registers_, l2_reg_count);
      if (CreatesRegisterUsage(&insn)) {
        if (does_belong_to_loop1) {
          if (!l1_sim_result->live_in_.count(&insn)) {
            l1_sim_result->AddRegisterClass(&insn);
          }
          l1_reg_count--;
        }
        if (does_belong_to_loop2) {
          if (!l2_sim_result->live_in_.count(&insn)) {
            l2_sim_result->AddRegisterClass(&insn);
          }
          l2_reg_count--;
        }
      }
    }
  }
}

}  // namespace opt
}  // namespace spvtools
