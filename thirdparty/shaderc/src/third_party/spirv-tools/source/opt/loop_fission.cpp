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

#include "source/opt/loop_fission.h"

#include <set>

#include "source/opt/register_pressure.h"

// Implement loop fission with an optional parameter to split only
// if the register pressure in a given loop meets a certain criteria. This is
// controlled via the constructors of LoopFissionPass.
//
// 1 - Build a list of loops to be split, these are top level loops (loops
// without child loops themselves) which meet the register pressure criteria, as
// determined by the ShouldSplitLoop method of LoopFissionPass.
//
// 2 - For each loop in the list, group each instruction into a set of related
// instructions by traversing each instructions users and operands recursively.
// We stop if we encounter an instruction we have seen before or an instruction
// which we don't consider relevent (i.e OpLoopMerge). We then group these
// groups into two different sets, one for the first loop and one for the
// second.
//
// 3 - We then run CanPerformSplit to check that it would be legal to split a
// loop using those two sets. We check that we haven't altered the relative
// order load/stores appear in the binary and that we aren't breaking any
// dependency between load/stores by splitting them into two loops. We also
// check that none of the OpBranch instructions are dependent on a load as we
// leave control flow structure intact and move only instructions in the body so
// we want to avoid any loads with side affects or aliasing.
//
// 4 - We then split the loop by calling SplitLoop. This function clones the
// loop and attaches it to the preheader and connects the new loops merge block
// to the current loop header block. We then use the two sets built in step 2 to
// remove instructions from each loop. If an instruction appears in the first
// set it is removed from the second loop and vice versa.
//
// 5 - If the multiple split passes flag is set we check if each of the loops
// still meet the register pressure criteria. If they do then we add them to the
// list of loops to be split (created in step one) to allow for loops to be
// split multiple times.
//

namespace spvtools {
namespace opt {

class LoopFissionImpl {
 public:
  LoopFissionImpl(IRContext* context, Loop* loop)
      : context_(context), loop_(loop), load_used_in_condition_(false) {}

  // Group each instruction in the loop into sets of instructions related by
  // their usedef chains. An instruction which uses another will appear in the
  // same set. Then merge those sets into just two sets. Returns false if there
  // was one or less sets created.
  bool GroupInstructionsByUseDef();

  // Check if the sets built by GroupInstructionsByUseDef violate any data
  // dependence rules.
  bool CanPerformSplit();

  // Split the loop and return a pointer to the new loop.
  Loop* SplitLoop();

  // Checks if |inst| is safe to move. We can only move instructions which don't
  // have any side effects and OpLoads and OpStores.
  bool MovableInstruction(const Instruction& inst) const;

 private:
  // Traverse the def use chain of |inst| and add the users and uses of |inst|
  // which are in the same loop to the |returned_set|.
  void TraverseUseDef(Instruction* inst, std::set<Instruction*>* returned_set,
                      bool ignore_phi_users = false, bool report_loads = false);

  // We group the instructions in the block into two different groups, the
  // instructions to be kept in the original loop and the ones to be cloned into
  // the new loop. As the cloned loop is attached to the preheader it will be
  // the first loop and the second loop will be the original.
  std::set<Instruction*> cloned_loop_instructions_;
  std::set<Instruction*> original_loop_instructions_;

  // We need a set of all the instructions to be seen so we can break any
  // recursion and also so we can ignore certain instructions by preemptively
  // adding them to this set.
  std::set<Instruction*> seen_instructions_;

  // A map of instructions to their relative position in the function.
  std::map<Instruction*, size_t> instruction_order_;

  IRContext* context_;

  Loop* loop_;

  // This is set to true by TraverseUseDef when traversing the instructions
  // related to the loop condition and any if conditions should any of those
  // instructions be a load.
  bool load_used_in_condition_;
};

bool LoopFissionImpl::MovableInstruction(const Instruction& inst) const {
  return inst.opcode() == SpvOp::SpvOpLoad ||
         inst.opcode() == SpvOp::SpvOpStore ||
         inst.opcode() == SpvOp::SpvOpSelectionMerge ||
         inst.opcode() == SpvOp::SpvOpPhi || inst.IsOpcodeCodeMotionSafe();
}

void LoopFissionImpl::TraverseUseDef(Instruction* inst,
                                     std::set<Instruction*>* returned_set,
                                     bool ignore_phi_users, bool report_loads) {
  assert(returned_set && "Set to be returned cannot be null.");

  analysis::DefUseManager* def_use = context_->get_def_use_mgr();
  std::set<Instruction*>& inst_set = *returned_set;

  // We create this functor to traverse the use def chain to build the
  // grouping of related instructions. The lambda captures the std::function
  // to allow it to recurse.
  std::function<void(Instruction*)> traverser_functor;
  traverser_functor = [this, def_use, &inst_set, &traverser_functor,
                       ignore_phi_users, report_loads](Instruction* user) {
    // If we've seen the instruction before or it is not inside the loop end the
    // traversal.
    if (!user || seen_instructions_.count(user) != 0 ||
        !context_->get_instr_block(user) ||
        !loop_->IsInsideLoop(context_->get_instr_block(user))) {
      return;
    }

    // Don't include labels or loop merge instructions in the instruction sets.
    // Including them would mean we group instructions related only by using the
    // same labels (i.e phis). We already preempt the inclusion of
    // OpSelectionMerge by adding related instructions to the seen_instructions_
    // set.
    if (user->opcode() == SpvOp::SpvOpLoopMerge ||
        user->opcode() == SpvOp::SpvOpLabel)
      return;

    // If the |report_loads| flag is set, set the class field
    // load_used_in_condition_ to false. This is used to check that none of the
    // condition checks in the loop rely on loads.
    if (user->opcode() == SpvOp::SpvOpLoad && report_loads) {
      load_used_in_condition_ = true;
    }

    // Add the instruction to the set of instructions already seen, this breaks
    // recursion and allows us to ignore certain instructions.
    seen_instructions_.insert(user);

    inst_set.insert(user);

    // Wrapper functor to traverse the operands of each instruction.
    auto traverse_operand = [&traverser_functor, def_use](const uint32_t* id) {
      traverser_functor(def_use->GetDef(*id));
    };
    user->ForEachInOperand(traverse_operand);

    // For the first traversal we want to ignore the users of the phi.
    if (ignore_phi_users && user->opcode() == SpvOp::SpvOpPhi) return;

    // Traverse each user with this lambda.
    def_use->ForEachUser(user, traverser_functor);

    // Wrapper functor for the use traversal.
    auto traverse_use = [&traverser_functor](Instruction* use, uint32_t) {
      traverser_functor(use);
    };
    def_use->ForEachUse(user, traverse_use);

  };

  // We start the traversal of the use def graph by invoking the above
  // lambda with the |inst| parameter.
  traverser_functor(inst);
}

bool LoopFissionImpl::GroupInstructionsByUseDef() {
  std::vector<std::set<Instruction*>> sets{};

  // We want to ignore all the instructions stemming from the loop condition
  // instruction.
  BasicBlock* condition_block = loop_->FindConditionBlock();

  if (!condition_block) return false;
  Instruction* condition = &*condition_block->tail();

  // We iterate over the blocks via iterating over all the blocks in the
  // function, we do this so we are iterating in the same order which the blocks
  // appear in the binary.
  Function& function = *loop_->GetHeaderBlock()->GetParent();

  // Create a temporary set to ignore certain groups of instructions within the
  // loop. We don't want any instructions related to control flow to be removed
  // from either loop only instructions within the control flow bodies.
  std::set<Instruction*> instructions_to_ignore{};
  TraverseUseDef(condition, &instructions_to_ignore, true, true);

  // Traverse control flow instructions to ensure they are added to the
  // seen_instructions_ set and will be ignored when it it called with actual
  // sets.
  for (BasicBlock& block : function) {
    if (!loop_->IsInsideLoop(block.id())) continue;

    for (Instruction& inst : block) {
      // Ignore all instructions related to control flow.
      if (inst.opcode() == SpvOp::SpvOpSelectionMerge || inst.IsBranch()) {
        TraverseUseDef(&inst, &instructions_to_ignore, true, true);
      }
    }
  }

  // Traverse the instructions and generate the sets, automatically ignoring any
  // instructions in instructions_to_ignore.
  for (BasicBlock& block : function) {
    if (!loop_->IsInsideLoop(block.id()) ||
        loop_->GetHeaderBlock()->id() == block.id())
      continue;

    for (Instruction& inst : block) {
      // Record the order that each load/store is seen.
      if (inst.opcode() == SpvOp::SpvOpLoad ||
          inst.opcode() == SpvOp::SpvOpStore) {
        instruction_order_[&inst] = instruction_order_.size();
      }

      // Ignore instructions already seen in a traversal.
      if (seen_instructions_.count(&inst) != 0) {
        continue;
      }

      // Build the set.
      std::set<Instruction*> inst_set{};
      TraverseUseDef(&inst, &inst_set);
      if (!inst_set.empty()) sets.push_back(std::move(inst_set));
    }
  }

  // If we have one or zero sets return false to indicate that due to
  // insufficient instructions we couldn't split the loop into two groups and
  // thus the loop can't be split any further.
  if (sets.size() < 2) {
    return false;
  }

  // Merge the loop sets into two different sets. In CanPerformSplit we will
  // validate that we don't break the relative ordering of loads/stores by doing
  // this.
  for (size_t index = 0; index < sets.size() / 2; ++index) {
    cloned_loop_instructions_.insert(sets[index].begin(), sets[index].end());
  }
  for (size_t index = sets.size() / 2; index < sets.size(); ++index) {
    original_loop_instructions_.insert(sets[index].begin(), sets[index].end());
  }

  return true;
}

bool LoopFissionImpl::CanPerformSplit() {
  // Return false if any of the condition instructions in the loop depend on a
  // load.
  if (load_used_in_condition_) {
    return false;
  }

  // Build a list of all parent loops of this loop. Loop dependence analysis
  // needs this structure.
  std::vector<const Loop*> loops;
  Loop* parent_loop = loop_;
  while (parent_loop) {
    loops.push_back(parent_loop);
    parent_loop = parent_loop->GetParent();
  }

  LoopDependenceAnalysis analysis{context_, loops};

  // A list of all the stores in the cloned loop.
  std::vector<Instruction*> set_one_stores{};

  // A list of all the loads in the cloned loop.
  std::vector<Instruction*> set_one_loads{};

  // Populate the above lists.
  for (Instruction* inst : cloned_loop_instructions_) {
    if (inst->opcode() == SpvOp::SpvOpStore) {
      set_one_stores.push_back(inst);
    } else if (inst->opcode() == SpvOp::SpvOpLoad) {
      set_one_loads.push_back(inst);
    }

    // If we find any instruction which we can't move (such as a barrier),
    // return false.
    if (!MovableInstruction(*inst)) return false;
  }

  // We need to calculate the depth of the loop to create the loop dependency
  // distance vectors.
  const size_t loop_depth = loop_->GetDepth();

  // Check the dependencies between loads in the cloned loop and stores in the
  // original and vice versa.
  for (Instruction* inst : original_loop_instructions_) {
    // If we find any instruction which we can't move (such as a barrier),
    // return false.
    if (!MovableInstruction(*inst)) return false;

    // Look at the dependency between the loads in the original and stores in
    // the cloned loops.
    if (inst->opcode() == SpvOp::SpvOpLoad) {
      for (Instruction* store : set_one_stores) {
        DistanceVector vec{loop_depth};

        // If the store actually should appear after the load, return false.
        // This means the store has been placed in the wrong grouping.
        if (instruction_order_[store] > instruction_order_[inst]) {
          return false;
        }
        // If not independent check the distance vector.
        if (!analysis.GetDependence(store, inst, &vec)) {
          for (DistanceEntry& entry : vec.GetEntries()) {
            // A distance greater than zero means that the store in the cloned
            // loop has a dependency on the load in the original loop.
            if (entry.distance > 0) return false;
          }
        }
      }
    } else if (inst->opcode() == SpvOp::SpvOpStore) {
      for (Instruction* load : set_one_loads) {
        DistanceVector vec{loop_depth};

        // If the load actually should appear after the store, return false.
        if (instruction_order_[load] > instruction_order_[inst]) {
          return false;
        }

        // If not independent check the distance vector.
        if (!analysis.GetDependence(inst, load, &vec)) {
          for (DistanceEntry& entry : vec.GetEntries()) {
            // A distance less than zero means the load in the cloned loop is
            // dependent on the store instruction in the original loop.
            if (entry.distance < 0) return false;
          }
        }
      }
    }
  }
  return true;
}

Loop* LoopFissionImpl::SplitLoop() {
  // Clone the loop.
  LoopUtils util{context_, loop_};
  LoopUtils::LoopCloningResult clone_results;
  Loop* cloned_loop = util.CloneAndAttachLoopToHeader(&clone_results);

  // Update the OpLoopMerge in the cloned loop.
  cloned_loop->UpdateLoopMergeInst();

  // Add the loop_ to the module.
  // TODO(1841): Handle failure to create pre-header.
  Function::iterator it =
      util.GetFunction()->FindBlock(loop_->GetOrCreatePreHeaderBlock()->id());
  util.GetFunction()->AddBasicBlocks(clone_results.cloned_bb_.begin(),
                                     clone_results.cloned_bb_.end(), ++it);
  loop_->SetPreHeaderBlock(cloned_loop->GetMergeBlock());

  std::vector<Instruction*> instructions_to_kill{};

  // Kill all the instructions which should appear in the cloned loop but not in
  // the original loop.
  for (uint32_t id : loop_->GetBlocks()) {
    BasicBlock* block = context_->cfg()->block(id);

    for (Instruction& inst : *block) {
      // If the instruction appears in the cloned loop instruction group, kill
      // it.
      if (cloned_loop_instructions_.count(&inst) == 1 &&
          original_loop_instructions_.count(&inst) == 0) {
        instructions_to_kill.push_back(&inst);
        if (inst.opcode() == SpvOp::SpvOpPhi) {
          context_->ReplaceAllUsesWith(
              inst.result_id(), clone_results.value_map_[inst.result_id()]);
        }
      }
    }
  }

  // Kill all instructions which should appear in the original loop and not in
  // the cloned loop.
  for (uint32_t id : cloned_loop->GetBlocks()) {
    BasicBlock* block = context_->cfg()->block(id);
    for (Instruction& inst : *block) {
      Instruction* old_inst = clone_results.ptr_map_[&inst];
      // If the instruction belongs to the original loop instruction group, kill
      // it.
      if (cloned_loop_instructions_.count(old_inst) == 0 &&
          original_loop_instructions_.count(old_inst) == 1) {
        instructions_to_kill.push_back(&inst);
      }
    }
  }

  for (Instruction* i : instructions_to_kill) {
    context_->KillInst(i);
  }

  return cloned_loop;
}

LoopFissionPass::LoopFissionPass(const size_t register_threshold_to_split,
                                 bool split_multiple_times)
    : split_multiple_times_(split_multiple_times) {
  // Split if the number of registers in the loop exceeds
  // |register_threshold_to_split|.
  split_criteria_ =
      [register_threshold_to_split](
          const RegisterLiveness::RegionRegisterLiveness& liveness) {
        return liveness.used_registers_ > register_threshold_to_split;
      };
}

LoopFissionPass::LoopFissionPass() : split_multiple_times_(false) {
  // Split by default.
  split_criteria_ = [](const RegisterLiveness::RegionRegisterLiveness&) {
    return true;
  };
}

bool LoopFissionPass::ShouldSplitLoop(const Loop& loop, IRContext* c) {
  LivenessAnalysis* analysis = c->GetLivenessAnalysis();

  RegisterLiveness::RegionRegisterLiveness liveness{};

  Function* function = loop.GetHeaderBlock()->GetParent();
  analysis->Get(function)->ComputeLoopRegisterPressure(loop, &liveness);

  return split_criteria_(liveness);
}

Pass::Status LoopFissionPass::Process() {
  bool changed = false;

  for (Function& f : *context()->module()) {
    // We collect all the inner most loops in the function and run the loop
    // splitting util on each. The reason we do this is to allow us to iterate
    // over each, as creating new loops will invalidate the the loop iterator.
    std::vector<Loop*> inner_most_loops{};
    LoopDescriptor& loop_descriptor = *context()->GetLoopDescriptor(&f);
    for (Loop& loop : loop_descriptor) {
      if (!loop.HasChildren() && ShouldSplitLoop(loop, context())) {
        inner_most_loops.push_back(&loop);
      }
    }

    // List of new loops which meet the criteria to be split again.
    std::vector<Loop*> new_loops_to_split{};

    while (!inner_most_loops.empty()) {
      for (Loop* loop : inner_most_loops) {
        LoopFissionImpl impl{context(), loop};

        // Group the instructions in the loop into two different sets of related
        // instructions. If we can't group the instructions into the two sets
        // then we can't split the loop any further.
        if (!impl.GroupInstructionsByUseDef()) {
          continue;
        }

        if (impl.CanPerformSplit()) {
          Loop* second_loop = impl.SplitLoop();
          changed = true;
          context()->InvalidateAnalysesExceptFor(
              IRContext::kAnalysisLoopAnalysis);

          // If the newly created loop meets the criteria to be split, split it
          // again.
          if (ShouldSplitLoop(*second_loop, context()))
            new_loops_to_split.push_back(second_loop);

          // If the original loop (now split) still meets the criteria to be
          // split, split it again.
          if (ShouldSplitLoop(*loop, context()))
            new_loops_to_split.push_back(loop);
        }
      }

      // If the split multiple times flag has been set add the new loops which
      // meet the splitting criteria into the list of loops to be split on the
      // next iteration.
      if (split_multiple_times_) {
        inner_most_loops = std::move(new_loops_to_split);
      } else {
        break;
      }
    }
  }

  return changed ? Pass::Status::SuccessWithChange
                 : Pass::Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
