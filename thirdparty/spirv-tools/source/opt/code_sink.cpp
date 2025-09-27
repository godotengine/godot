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

#include "code_sink.h"

#include <set>
#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/util/bit_vector.h"

namespace spvtools {
namespace opt {

Pass::Status CodeSinkingPass::Process() {
  bool modified = false;
  for (Function& function : *get_module()) {
    cfg()->ForEachBlockInPostOrder(function.entry().get(),
                                   [&modified, this](BasicBlock* bb) {
                                     if (SinkInstructionsInBB(bb)) {
                                       modified = true;
                                     }
                                   });
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool CodeSinkingPass::SinkInstructionsInBB(BasicBlock* bb) {
  bool modified = false;
  for (auto inst = bb->rbegin(); inst != bb->rend(); ++inst) {
    if (SinkInstruction(&*inst)) {
      inst = bb->rbegin();
      modified = true;
    }
  }
  return modified;
}

bool CodeSinkingPass::SinkInstruction(Instruction* inst) {
  if (inst->opcode() != spv::Op::OpLoad &&
      inst->opcode() != spv::Op::OpAccessChain) {
    return false;
  }

  if (ReferencesMutableMemory(inst)) {
    return false;
  }

  if (BasicBlock* target_bb = FindNewBasicBlockFor(inst)) {
    Instruction* pos = &*target_bb->begin();
    while (pos->opcode() == spv::Op::OpPhi) {
      pos = pos->NextNode();
    }

    inst->InsertBefore(pos);
    context()->set_instr_block(inst, target_bb);
    return true;
  }
  return false;
}

BasicBlock* CodeSinkingPass::FindNewBasicBlockFor(Instruction* inst) {
  assert(inst->result_id() != 0 && "Instruction should have a result.");
  BasicBlock* original_bb = context()->get_instr_block(inst);
  BasicBlock* bb = original_bb;

  std::unordered_set<uint32_t> bbs_with_uses;
  get_def_use_mgr()->ForEachUse(
      inst, [&bbs_with_uses, this](Instruction* use, uint32_t idx) {
        if (use->opcode() != spv::Op::OpPhi) {
          BasicBlock* use_bb = context()->get_instr_block(use);
          if (use_bb) {
            bbs_with_uses.insert(use_bb->id());
          }
        } else {
          bbs_with_uses.insert(use->GetSingleWordOperand(idx + 1));
        }
      });

  while (true) {
    // If |inst| is used in |bb|, then |inst| cannot be moved any further.
    if (bbs_with_uses.count(bb->id())) {
      break;
    }

    // If |bb| has one successor (succ_bb), and |bb| is the only predecessor
    // of succ_bb, then |inst| can be moved to succ_bb.  If succ_bb, has move
    // then one predecessor, then moving |inst| into succ_bb could cause it to
    // be executed more often, so the search has to stop.
    if (bb->terminator()->opcode() == spv::Op::OpBranch) {
      uint32_t succ_bb_id = bb->terminator()->GetSingleWordInOperand(0);
      if (cfg()->preds(succ_bb_id).size() == 1) {
        bb = context()->get_instr_block(succ_bb_id);
        continue;
      } else {
        break;
      }
    }

    // The remaining checks need to know the merge node.  If there is no merge
    // instruction or an OpLoopMerge, then it is a break or continue.  We could
    // figure it out, but not worth doing it now.
    Instruction* merge_inst = bb->GetMergeInst();
    if (merge_inst == nullptr ||
        merge_inst->opcode() != spv::Op::OpSelectionMerge) {
      break;
    }

    // Check all of the successors of |bb| it see which lead to a use of |inst|
    // before reaching the merge node.
    bool used_in_multiple_blocks = false;
    uint32_t bb_used_in = 0;
    bb->ForEachSuccessorLabel([this, bb, &bb_used_in, &used_in_multiple_blocks,
                               &bbs_with_uses](uint32_t* succ_bb_id) {
      if (IntersectsPath(*succ_bb_id, bb->MergeBlockIdIfAny(), bbs_with_uses)) {
        if (bb_used_in == 0) {
          bb_used_in = *succ_bb_id;
        } else {
          used_in_multiple_blocks = true;
        }
      }
    });

    // If more than one successor, which is not the merge block, uses |inst|
    // then we have to leave |inst| in bb because there is none of the
    // successors dominate all uses of |inst|.
    if (used_in_multiple_blocks) {
      break;
    }

    if (bb_used_in == 0) {
      // If |inst| is not used before reaching the merge node, then we can move
      // |inst| to the merge node.
      bb = context()->get_instr_block(bb->MergeBlockIdIfAny());
    } else {
      // If the only successor that leads to a used of |inst| has more than 1
      // predecessor, then moving |inst| could cause it to be executed more
      // often, so we cannot move it.
      if (cfg()->preds(bb_used_in).size() != 1) {
        break;
      }

      // If |inst| is used after the merge block, then |bb_used_in| does not
      // dominate all of the uses.  So we cannot move |inst| any further.
      if (IntersectsPath(bb->MergeBlockIdIfAny(), original_bb->id(),
                         bbs_with_uses)) {
        break;
      }

      // Otherwise, |bb_used_in| dominates all uses, so move |inst| into that
      // block.
      bb = context()->get_instr_block(bb_used_in);
    }
    continue;
  }
  return (bb != original_bb ? bb : nullptr);
}

bool CodeSinkingPass::ReferencesMutableMemory(Instruction* inst) {
  if (!inst->IsLoad()) {
    return false;
  }

  Instruction* base_ptr = inst->GetBaseAddress();
  if (base_ptr->opcode() != spv::Op::OpVariable) {
    return true;
  }

  if (base_ptr->IsReadOnlyPointer()) {
    return false;
  }

  if (HasUniformMemorySync()) {
    return true;
  }

  if (spv::StorageClass(base_ptr->GetSingleWordInOperand(0)) !=
      spv::StorageClass::Uniform) {
    return true;
  }

  return HasPossibleStore(base_ptr);
}

bool CodeSinkingPass::HasUniformMemorySync() {
  if (checked_for_uniform_sync_) {
    return has_uniform_sync_;
  }

  bool has_sync = false;
  get_module()->ForEachInst([this, &has_sync](Instruction* inst) {
    switch (inst->opcode()) {
      case spv::Op::OpMemoryBarrier: {
        uint32_t mem_semantics_id = inst->GetSingleWordInOperand(1);
        if (IsSyncOnUniform(mem_semantics_id)) {
          has_sync = true;
        }
        break;
      }
      case spv::Op::OpControlBarrier:
      case spv::Op::OpAtomicLoad:
      case spv::Op::OpAtomicStore:
      case spv::Op::OpAtomicExchange:
      case spv::Op::OpAtomicIIncrement:
      case spv::Op::OpAtomicIDecrement:
      case spv::Op::OpAtomicIAdd:
      case spv::Op::OpAtomicFAddEXT:
      case spv::Op::OpAtomicISub:
      case spv::Op::OpAtomicSMin:
      case spv::Op::OpAtomicUMin:
      case spv::Op::OpAtomicFMinEXT:
      case spv::Op::OpAtomicSMax:
      case spv::Op::OpAtomicUMax:
      case spv::Op::OpAtomicFMaxEXT:
      case spv::Op::OpAtomicAnd:
      case spv::Op::OpAtomicOr:
      case spv::Op::OpAtomicXor:
      case spv::Op::OpAtomicFlagTestAndSet:
      case spv::Op::OpAtomicFlagClear: {
        uint32_t mem_semantics_id = inst->GetSingleWordInOperand(2);
        if (IsSyncOnUniform(mem_semantics_id)) {
          has_sync = true;
        }
        break;
      }
      case spv::Op::OpAtomicCompareExchange:
      case spv::Op::OpAtomicCompareExchangeWeak:
        if (IsSyncOnUniform(inst->GetSingleWordInOperand(2)) ||
            IsSyncOnUniform(inst->GetSingleWordInOperand(3))) {
          has_sync = true;
        }
        break;
      default:
        break;
    }
  });
  has_uniform_sync_ = has_sync;
  return has_sync;
}

bool CodeSinkingPass::IsSyncOnUniform(uint32_t mem_semantics_id) const {
  const analysis::Constant* mem_semantics_const =
      context()->get_constant_mgr()->FindDeclaredConstant(mem_semantics_id);
  assert(mem_semantics_const != nullptr &&
         "Expecting memory semantics id to be a constant.");
  assert(mem_semantics_const->AsIntConstant() &&
         "Memory semantics should be an integer.");
  uint32_t mem_semantics_int = mem_semantics_const->GetU32();

  // If it does not affect uniform memory, then it is does not apply to uniform
  // memory.
  if ((mem_semantics_int & uint32_t(spv::MemorySemanticsMask::UniformMemory)) ==
      0) {
    return false;
  }

  // Check if there is an acquire or release.  If so not, this it does not add
  // any memory constraints.
  return (mem_semantics_int &
          uint32_t(spv::MemorySemanticsMask::Acquire |
                   spv::MemorySemanticsMask::AcquireRelease |
                   spv::MemorySemanticsMask::Release)) != 0;
}

bool CodeSinkingPass::HasPossibleStore(Instruction* var_inst) {
  assert(var_inst->opcode() == spv::Op::OpVariable ||
         var_inst->opcode() == spv::Op::OpAccessChain ||
         var_inst->opcode() == spv::Op::OpPtrAccessChain);

  return get_def_use_mgr()->WhileEachUser(var_inst, [this](Instruction* use) {
    switch (use->opcode()) {
      case spv::Op::OpStore:
        return true;
      case spv::Op::OpAccessChain:
      case spv::Op::OpPtrAccessChain:
        return HasPossibleStore(use);
      default:
        return false;
    }
  });
}

bool CodeSinkingPass::IntersectsPath(uint32_t start, uint32_t end,
                                     const std::unordered_set<uint32_t>& set) {
  std::vector<uint32_t> worklist;
  worklist.push_back(start);
  std::unordered_set<uint32_t> already_done;
  already_done.insert(start);

  while (!worklist.empty()) {
    BasicBlock* bb = context()->get_instr_block(worklist.back());
    worklist.pop_back();

    if (bb->id() == end) {
      continue;
    }

    if (set.count(bb->id())) {
      return true;
    }

    bb->ForEachSuccessorLabel([&already_done, &worklist](uint32_t* succ_bb_id) {
      if (already_done.insert(*succ_bb_id).second) {
        worklist.push_back(*succ_bb_id);
      }
    });
  }
  return false;
}

// namespace opt

}  // namespace opt
}  // namespace spvtools
