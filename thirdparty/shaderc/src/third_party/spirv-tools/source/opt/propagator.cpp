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

#include "source/opt/propagator.h"

namespace spvtools {
namespace opt {

void SSAPropagator::AddControlEdge(const Edge& edge) {
  BasicBlock* dest_bb = edge.dest;

  // Refuse to add the exit block to the work list.
  if (dest_bb == ctx_->cfg()->pseudo_exit_block()) {
    return;
  }

  // Try to mark the edge executable.  If it was already in the set of
  // executable edges, do nothing.
  if (!MarkEdgeExecutable(edge)) {
    return;
  }

  // If the edge had not already been marked executable, add the destination
  // basic block to the work list.
  blocks_.push(dest_bb);
}

void SSAPropagator::AddSSAEdges(Instruction* instr) {
  // Ignore instructions that produce no result.
  if (instr->result_id() == 0) {
    return;
  }

  get_def_use_mgr()->ForEachUser(
      instr->result_id(), [this](Instruction* use_instr) {
        // If the basic block for |use_instr| has not been simulated yet, do
        // nothing.  The instruction |use_instr| will be simulated next time the
        // block is scheduled.
        if (!BlockHasBeenSimulated(ctx_->get_instr_block(use_instr))) {
          return;
        }

        if (ShouldSimulateAgain(use_instr)) {
          ssa_edge_uses_.push(use_instr);
        }
      });
}

bool SSAPropagator::IsPhiArgExecutable(Instruction* phi, uint32_t i) const {
  BasicBlock* phi_bb = ctx_->get_instr_block(phi);

  uint32_t in_label_id = phi->GetSingleWordOperand(i + 1);
  Instruction* in_label_instr = get_def_use_mgr()->GetDef(in_label_id);
  BasicBlock* in_bb = ctx_->get_instr_block(in_label_instr);

  return IsEdgeExecutable(Edge(in_bb, phi_bb));
}

bool SSAPropagator::SetStatus(Instruction* inst, PropStatus status) {
  bool has_old_status = false;
  PropStatus old_status = kVarying;
  if (HasStatus(inst)) {
    has_old_status = true;
    old_status = Status(inst);
  }

  assert((!has_old_status || old_status <= status) &&
         "Invalid lattice transition");

  bool status_changed = !has_old_status || (old_status != status);
  if (status_changed) statuses_[inst] = status;

  return status_changed;
}

bool SSAPropagator::Simulate(Instruction* instr) {
  bool changed = false;

  // Don't bother visiting instructions that should not be simulated again.
  if (!ShouldSimulateAgain(instr)) {
    return changed;
  }

  BasicBlock* dest_bb = nullptr;
  PropStatus status = visit_fn_(instr, &dest_bb);
  bool status_changed = SetStatus(instr, status);

  if (status == kVarying) {
    // The statement produces a varying result, add it to the list of statements
    // not to simulate anymore and add its SSA def-use edges for simulation.
    DontSimulateAgain(instr);
    if (status_changed) {
      AddSSAEdges(instr);
    }

    // If |instr| is a block terminator, add all the control edges out of its
    // block.
    if (instr->IsBlockTerminator()) {
      BasicBlock* block = ctx_->get_instr_block(instr);
      for (const auto& e : bb_succs_.at(block)) {
        AddControlEdge(e);
      }
    }
    return false;
  } else if (status == kInteresting) {
    // Add the SSA edges coming out of this instruction if the propagation
    // status has changed.
    if (status_changed) {
      AddSSAEdges(instr);
    }

    // If there are multiple outgoing control flow edges and we know which one
    // will be taken, add the destination block to the CFG work list.
    if (dest_bb) {
      AddControlEdge(Edge(ctx_->get_instr_block(instr), dest_bb));
    }
    changed = true;
  }

  // At this point, we are dealing with instructions that are in status
  // kInteresting or kNotInteresting.  To decide whether this instruction should
  // be simulated again, we examine its operands.  If at least one operand O is
  // defined at an instruction D that should be simulated again, then the output
  // of D might affect |instr|, so we should simulate |instr| again.
  bool has_operands_to_simulate = false;
  if (instr->opcode() == SpvOpPhi) {
    // For Phi instructions, an operand causes the Phi to be simulated again if
    // the operand comes from an edge that has not yet been traversed or if its
    // definition should be simulated again.
    for (uint32_t i = 2; i < instr->NumOperands(); i += 2) {
      // Phi arguments come in pairs. Index 'i' contains the
      // variable id, index 'i + 1' is the originating block id.
      assert(i % 2 == 0 && i < instr->NumOperands() - 1 &&
             "malformed Phi arguments");

      uint32_t arg_id = instr->GetSingleWordOperand(i);
      Instruction* arg_def_instr = get_def_use_mgr()->GetDef(arg_id);
      if (!IsPhiArgExecutable(instr, i) || ShouldSimulateAgain(arg_def_instr)) {
        has_operands_to_simulate = true;
        break;
      }
    }
  } else {
    // For regular instructions, check if the defining instruction of each
    // operand needs to be simulated again.  If so, then this instruction should
    // also be simulated again.
    has_operands_to_simulate =
        !instr->WhileEachInId([this](const uint32_t* use) {
          Instruction* def_instr = get_def_use_mgr()->GetDef(*use);
          if (ShouldSimulateAgain(def_instr)) {
            return false;
          }
          return true;
        });
  }

  if (!has_operands_to_simulate) {
    DontSimulateAgain(instr);
  }

  return changed;
}

bool SSAPropagator::Simulate(BasicBlock* block) {
  if (block == ctx_->cfg()->pseudo_exit_block()) {
    return false;
  }

  // Always simulate Phi instructions, even if we have simulated this block
  // before. We do this because Phi instructions receive their inputs from
  // incoming edges. When those edges are marked executable, the corresponding
  // operand can be simulated.
  bool changed = false;
  block->ForEachPhiInst(
      [&changed, this](Instruction* instr) { changed |= Simulate(instr); });

  // If this is the first time this block is being simulated, simulate every
  // statement in it.
  if (!BlockHasBeenSimulated(block)) {
    block->ForEachInst([this, &changed](Instruction* instr) {
      if (instr->opcode() != SpvOpPhi) {
        changed |= Simulate(instr);
      }
    });

    MarkBlockSimulated(block);

    // If this block has exactly one successor, mark the edge to its successor
    // as executable.
    if (bb_succs_.at(block).size() == 1) {
      AddControlEdge(bb_succs_.at(block).at(0));
    }
  }

  return changed;
}

void SSAPropagator::Initialize(Function* fn) {
  // Compute predecessor and successor blocks for every block in |fn|'s CFG.
  // TODO(dnovillo): Move this to CFG and always build them. Alternately,
  // move it to IRContext and build CFG preds/succs on-demand.
  bb_succs_[ctx_->cfg()->pseudo_entry_block()].push_back(
      Edge(ctx_->cfg()->pseudo_entry_block(), fn->entry().get()));

  for (auto& block : *fn) {
    const auto& const_block = block;
    const_block.ForEachSuccessorLabel([this, &block](const uint32_t label_id) {
      BasicBlock* succ_bb =
          ctx_->get_instr_block(get_def_use_mgr()->GetDef(label_id));
      bb_succs_[&block].push_back(Edge(&block, succ_bb));
      bb_preds_[succ_bb].push_back(Edge(succ_bb, &block));
    });
    if (block.IsReturnOrAbort()) {
      bb_succs_[&block].push_back(
          Edge(&block, ctx_->cfg()->pseudo_exit_block()));
      bb_preds_[ctx_->cfg()->pseudo_exit_block()].push_back(
          Edge(ctx_->cfg()->pseudo_exit_block(), &block));
    }
  }

  // Add the edges out of the entry block to seed the propagator.
  const auto& entry_succs = bb_succs_[ctx_->cfg()->pseudo_entry_block()];
  for (const auto& e : entry_succs) {
    AddControlEdge(e);
  }
}

bool SSAPropagator::Run(Function* fn) {
  Initialize(fn);

  bool changed = false;
  while (!blocks_.empty() || !ssa_edge_uses_.empty()) {
    // Simulate all blocks first. Simulating blocks will add SSA edges to
    // follow after all the blocks have been simulated.
    if (!blocks_.empty()) {
      auto block = blocks_.front();
      changed |= Simulate(block);
      blocks_.pop();
      continue;
    }

    // Simulate edges from the SSA queue.
    if (!ssa_edge_uses_.empty()) {
      Instruction* instr = ssa_edge_uses_.front();
      changed |= Simulate(instr);
      ssa_edge_uses_.pop();
    }
  }

#ifndef NDEBUG
  // Verify all visited values have settled. No value that has been simulated
  // should end on not interesting.
  fn->ForEachInst([this](Instruction* inst) {
    assert(
        (!HasStatus(inst) || Status(inst) != SSAPropagator::kNotInteresting) &&
        "Unsettled value");
  });
#endif

  return changed;
}

std::ostream& operator<<(std::ostream& str,
                         const SSAPropagator::PropStatus& status) {
  switch (status) {
    case SSAPropagator::kVarying:
      str << "Varying";
      break;
    case SSAPropagator::kInteresting:
      str << "Interesting";
      break;
    default:
      str << "Not interesting";
      break;
  }
  return str;
}

}  // namespace opt
}  // namespace spvtools
