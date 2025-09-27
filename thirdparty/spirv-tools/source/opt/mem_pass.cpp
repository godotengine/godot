// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "source/opt/mem_pass.h"

#include <memory>
#include <set>
#include <vector>

#include "source/cfa.h"
#include "source/opt/basic_block.h"
#include "source/opt/dominator_analysis.h"
#include "source/opt/ir_context.h"
#include "source/opt/iterator.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kCopyObjectOperandInIdx = 0;
constexpr uint32_t kTypePointerStorageClassInIdx = 0;
constexpr uint32_t kTypePointerTypeIdInIdx = 1;
}  // namespace

bool MemPass::IsBaseTargetType(const Instruction* typeInst) const {
  switch (typeInst->opcode()) {
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeBool:
    case spv::Op::OpTypeVector:
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeImage:
    case spv::Op::OpTypeSampler:
    case spv::Op::OpTypeSampledImage:
    case spv::Op::OpTypePointer:
      return true;
    default:
      break;
  }
  return false;
}

bool MemPass::IsTargetType(const Instruction* typeInst) const {
  if (IsBaseTargetType(typeInst)) return true;
  if (typeInst->opcode() == spv::Op::OpTypeArray) {
    if (!IsTargetType(
            get_def_use_mgr()->GetDef(typeInst->GetSingleWordOperand(1)))) {
      return false;
    }
    return true;
  }
  if (typeInst->opcode() != spv::Op::OpTypeStruct) return false;
  // All struct members must be math type
  return typeInst->WhileEachInId([this](const uint32_t* tid) {
    Instruction* compTypeInst = get_def_use_mgr()->GetDef(*tid);
    if (!IsTargetType(compTypeInst)) return false;
    return true;
  });
}

bool MemPass::IsNonPtrAccessChain(const spv::Op opcode) const {
  return opcode == spv::Op::OpAccessChain ||
         opcode == spv::Op::OpInBoundsAccessChain;
}

bool MemPass::IsPtr(uint32_t ptrId) {
  uint32_t varId = ptrId;
  Instruction* ptrInst = get_def_use_mgr()->GetDef(varId);
  while (ptrInst->opcode() == spv::Op::OpCopyObject) {
    varId = ptrInst->GetSingleWordInOperand(kCopyObjectOperandInIdx);
    ptrInst = get_def_use_mgr()->GetDef(varId);
  }
  const spv::Op op = ptrInst->opcode();
  if (op == spv::Op::OpVariable || IsNonPtrAccessChain(op)) return true;
  const uint32_t varTypeId = ptrInst->type_id();
  if (varTypeId == 0) return false;
  const Instruction* varTypeInst = get_def_use_mgr()->GetDef(varTypeId);
  return varTypeInst->opcode() == spv::Op::OpTypePointer;
}

Instruction* MemPass::GetPtr(uint32_t ptrId, uint32_t* varId) {
  *varId = ptrId;
  Instruction* ptrInst = get_def_use_mgr()->GetDef(*varId);
  Instruction* varInst;

  if (ptrInst->opcode() == spv::Op::OpConstantNull) {
    *varId = 0;
    return ptrInst;
  }

  if (ptrInst->opcode() != spv::Op::OpVariable &&
      ptrInst->opcode() != spv::Op::OpFunctionParameter) {
    varInst = ptrInst->GetBaseAddress();
  } else {
    varInst = ptrInst;
  }
  if (varInst->opcode() == spv::Op::OpVariable) {
    *varId = varInst->result_id();
  } else {
    *varId = 0;
  }

  while (ptrInst->opcode() == spv::Op::OpCopyObject) {
    uint32_t temp = ptrInst->GetSingleWordInOperand(0);
    ptrInst = get_def_use_mgr()->GetDef(temp);
  }

  return ptrInst;
}

Instruction* MemPass::GetPtr(Instruction* ip, uint32_t* varId) {
  assert(ip->opcode() == spv::Op::OpStore || ip->opcode() == spv::Op::OpLoad ||
         ip->opcode() == spv::Op::OpImageTexelPointer ||
         ip->IsAtomicWithLoad());

  // All of these opcode place the pointer in position 0.
  const uint32_t ptrId = ip->GetSingleWordInOperand(0);
  return GetPtr(ptrId, varId);
}

bool MemPass::HasOnlyNamesAndDecorates(uint32_t id) const {
  return get_def_use_mgr()->WhileEachUser(id, [this](Instruction* user) {
    spv::Op op = user->opcode();
    if (op != spv::Op::OpName && !IsNonTypeDecorate(op)) {
      return false;
    }
    return true;
  });
}

void MemPass::KillAllInsts(BasicBlock* bp, bool killLabel) {
  bp->KillAllInsts(killLabel);
}

bool MemPass::HasLoads(uint32_t varId) const {
  return !get_def_use_mgr()->WhileEachUser(varId, [this](Instruction* user) {
    spv::Op op = user->opcode();
    // TODO(): The following is slightly conservative. Could be
    // better handling of non-store/name.
    if (IsNonPtrAccessChain(op) || op == spv::Op::OpCopyObject) {
      if (HasLoads(user->result_id())) {
        return false;
      }
    } else if (op != spv::Op::OpStore && op != spv::Op::OpName &&
               !IsNonTypeDecorate(op)) {
      return false;
    }
    return true;
  });
}

bool MemPass::IsLiveVar(uint32_t varId) const {
  const Instruction* varInst = get_def_use_mgr()->GetDef(varId);
  // assume live if not a variable eg. function parameter
  if (varInst->opcode() != spv::Op::OpVariable) return true;
  // non-function scope vars are live
  const uint32_t varTypeId = varInst->type_id();
  const Instruction* varTypeInst = get_def_use_mgr()->GetDef(varTypeId);
  if (spv::StorageClass(varTypeInst->GetSingleWordInOperand(
          kTypePointerStorageClassInIdx)) != spv::StorageClass::Function)
    return true;
  // test if variable is loaded from
  return HasLoads(varId);
}

void MemPass::AddStores(uint32_t ptr_id, std::queue<Instruction*>* insts) {
  get_def_use_mgr()->ForEachUser(ptr_id, [this, insts](Instruction* user) {
    spv::Op op = user->opcode();
    if (IsNonPtrAccessChain(op)) {
      AddStores(user->result_id(), insts);
    } else if (op == spv::Op::OpStore) {
      insts->push(user);
    }
  });
}

void MemPass::DCEInst(Instruction* inst,
                      const std::function<void(Instruction*)>& call_back) {
  std::queue<Instruction*> deadInsts;
  deadInsts.push(inst);
  while (!deadInsts.empty()) {
    Instruction* di = deadInsts.front();
    // Don't delete labels
    if (di->opcode() == spv::Op::OpLabel) {
      deadInsts.pop();
      continue;
    }
    // Remember operands
    std::set<uint32_t> ids;
    di->ForEachInId([&ids](uint32_t* iid) { ids.insert(*iid); });
    uint32_t varId = 0;
    // Remember variable if dead load
    if (di->opcode() == spv::Op::OpLoad) (void)GetPtr(di, &varId);
    if (call_back) {
      call_back(di);
    }
    context()->KillInst(di);
    // For all operands with no remaining uses, add their instruction
    // to the dead instruction queue.
    for (auto id : ids)
      if (HasOnlyNamesAndDecorates(id)) {
        Instruction* odi = get_def_use_mgr()->GetDef(id);
        if (context()->IsCombinatorInstruction(odi)) deadInsts.push(odi);
      }
    // if a load was deleted and it was the variable's
    // last load, add all its stores to dead queue
    if (varId != 0 && !IsLiveVar(varId)) AddStores(varId, &deadInsts);
    deadInsts.pop();
  }
}

MemPass::MemPass() {}

bool MemPass::HasOnlySupportedRefs(uint32_t varId) {
  return get_def_use_mgr()->WhileEachUser(varId, [this](Instruction* user) {
    auto dbg_op = user->GetCommonDebugOpcode();
    if (dbg_op == CommonDebugInfoDebugDeclare ||
        dbg_op == CommonDebugInfoDebugValue) {
      return true;
    }
    spv::Op op = user->opcode();
    if (op != spv::Op::OpStore && op != spv::Op::OpLoad &&
        op != spv::Op::OpName && !IsNonTypeDecorate(op)) {
      return false;
    }
    return true;
  });
}

uint32_t MemPass::Type2Undef(uint32_t type_id) {
  const auto uitr = type2undefs_.find(type_id);
  if (uitr != type2undefs_.end()) return uitr->second;
  const uint32_t undefId = TakeNextId();
  if (undefId == 0) {
    return 0;
  }

  std::unique_ptr<Instruction> undef_inst(
      new Instruction(context(), spv::Op::OpUndef, type_id, undefId, {}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*undef_inst);
  get_module()->AddGlobalValue(std::move(undef_inst));
  type2undefs_[type_id] = undefId;
  return undefId;
}

bool MemPass::IsTargetVar(uint32_t varId) {
  if (varId == 0) {
    return false;
  }

  if (seen_non_target_vars_.find(varId) != seen_non_target_vars_.end())
    return false;
  if (seen_target_vars_.find(varId) != seen_target_vars_.end()) return true;
  const Instruction* varInst = get_def_use_mgr()->GetDef(varId);
  if (varInst->opcode() != spv::Op::OpVariable) return false;
  const uint32_t varTypeId = varInst->type_id();
  const Instruction* varTypeInst = get_def_use_mgr()->GetDef(varTypeId);
  if (spv::StorageClass(varTypeInst->GetSingleWordInOperand(
          kTypePointerStorageClassInIdx)) != spv::StorageClass::Function) {
    seen_non_target_vars_.insert(varId);
    return false;
  }
  const uint32_t varPteTypeId =
      varTypeInst->GetSingleWordInOperand(kTypePointerTypeIdInIdx);
  Instruction* varPteTypeInst = get_def_use_mgr()->GetDef(varPteTypeId);
  if (!IsTargetType(varPteTypeInst)) {
    seen_non_target_vars_.insert(varId);
    return false;
  }
  seen_target_vars_.insert(varId);
  return true;
}

// Remove all |phi| operands coming from unreachable blocks (i.e., blocks not in
// |reachable_blocks|).  There are two types of removal that this function can
// perform:
//
// 1- Any operand that comes directly from an unreachable block is completely
//    removed.  Since the block is unreachable, the edge between the unreachable
//    block and the block holding |phi| has been removed.
//
// 2- Any operand that comes via a live block and was defined at an unreachable
//    block gets its value replaced with an OpUndef value. Since the argument
//    was generated in an unreachable block, it no longer exists, so it cannot
//    be referenced.  However, since the value does not reach |phi| directly
//    from the unreachable block, the operand cannot be removed from |phi|.
//    Therefore, we replace the argument value with OpUndef.
//
// For example, in the switch() below, assume that we want to remove the
// argument with value %11 coming from block %41.
//
//          [ ... ]
//          %41 = OpLabel                    <--- Unreachable block
//          %11 = OpLoad %int %y
//          [ ... ]
//                OpSelectionMerge %16 None
//                OpSwitch %12 %16 10 %13 13 %14 18 %15
//          %13 = OpLabel
//                OpBranch %16
//          %14 = OpLabel
//                OpStore %outparm %int_14
//                OpBranch %16
//          %15 = OpLabel
//                OpStore %outparm %int_15
//                OpBranch %16
//          %16 = OpLabel
//          %30 = OpPhi %int %11 %41 %int_42 %13 %11 %14 %11 %15
//
// Since %41 is now an unreachable block, the first operand of |phi| needs to
// be removed completely.  But the operands (%11 %14) and (%11 %15) cannot be
// removed because %14 and %15 are reachable blocks.  Since %11 no longer exist,
// in those arguments, we replace all references to %11 with an OpUndef value.
// This results in |phi| looking like:
//
//           %50 = OpUndef %int
//           [ ... ]
//           %30 = OpPhi %int %int_42 %13 %50 %14 %50 %15
void MemPass::RemovePhiOperands(
    Instruction* phi, const std::unordered_set<BasicBlock*>& reachable_blocks) {
  std::vector<Operand> keep_operands;
  uint32_t type_id = 0;
  // The id of an undefined value we've generated.
  uint32_t undef_id = 0;

  // Traverse all the operands in |phi|. Build the new operand vector by adding
  // all the original operands from |phi| except the unwanted ones.
  for (uint32_t i = 0; i < phi->NumOperands();) {
    if (i < 2) {
      // The first two arguments are always preserved.
      keep_operands.push_back(phi->GetOperand(i));
      ++i;
      continue;
    }

    // The remaining Phi arguments come in pairs. Index 'i' contains the
    // variable id, index 'i + 1' is the originating block id.
    assert(i % 2 == 0 && i < phi->NumOperands() - 1 &&
           "malformed Phi arguments");

    BasicBlock* in_block = cfg()->block(phi->GetSingleWordOperand(i + 1));
    if (reachable_blocks.find(in_block) == reachable_blocks.end()) {
      // If the incoming block is unreachable, remove both operands as this
      // means that the |phi| has lost an incoming edge.
      i += 2;
      continue;
    }

    // In all other cases, the operand must be kept but may need to be changed.
    uint32_t arg_id = phi->GetSingleWordOperand(i);
    Instruction* arg_def_instr = get_def_use_mgr()->GetDef(arg_id);
    BasicBlock* def_block = context()->get_instr_block(arg_def_instr);
    if (def_block &&
        reachable_blocks.find(def_block) == reachable_blocks.end()) {
      // If the current |phi| argument was defined in an unreachable block, it
      // means that this |phi| argument is no longer defined. Replace it with
      // |undef_id|.
      if (!undef_id) {
        type_id = arg_def_instr->type_id();
        undef_id = Type2Undef(type_id);
      }
      keep_operands.push_back(
          Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID, {undef_id}));
    } else {
      // Otherwise, the argument comes from a reachable block or from no block
      // at all (meaning that it was defined in the global section of the
      // program).  In both cases, keep the argument intact.
      keep_operands.push_back(phi->GetOperand(i));
    }

    keep_operands.push_back(phi->GetOperand(i + 1));

    i += 2;
  }

  context()->ForgetUses(phi);
  phi->ReplaceOperands(keep_operands);
  context()->AnalyzeUses(phi);
}

void MemPass::RemoveBlock(Function::iterator* bi) {
  auto& rm_block = **bi;

  // Remove instructions from the block.
  rm_block.ForEachInst([&rm_block, this](Instruction* inst) {
    // Note that we do not kill the block label instruction here. The label
    // instruction is needed to identify the block, which is needed by the
    // removal of phi operands.
    if (inst != rm_block.GetLabelInst()) {
      context()->KillInst(inst);
    }
  });

  // Remove the label instruction last.
  auto label = rm_block.GetLabelInst();
  context()->KillInst(label);

  *bi = bi->Erase();
}

bool MemPass::RemoveUnreachableBlocks(Function* func) {
  bool modified = false;

  // Mark reachable all blocks reachable from the function's entry block.
  std::unordered_set<BasicBlock*> reachable_blocks;
  std::unordered_set<BasicBlock*> visited_blocks;
  std::queue<BasicBlock*> worklist;
  reachable_blocks.insert(func->entry().get());

  // Initially mark the function entry point as reachable.
  worklist.push(func->entry().get());

  auto mark_reachable = [&reachable_blocks, &visited_blocks, &worklist,
                         this](uint32_t label_id) {
    auto successor = cfg()->block(label_id);
    if (visited_blocks.count(successor) == 0) {
      reachable_blocks.insert(successor);
      worklist.push(successor);
      visited_blocks.insert(successor);
    }
  };

  // Transitively mark all blocks reachable from the entry as reachable.
  while (!worklist.empty()) {
    BasicBlock* block = worklist.front();
    worklist.pop();

    // All the successors of a live block are also live.
    static_cast<const BasicBlock*>(block)->ForEachSuccessorLabel(
        mark_reachable);

    // All the Merge and ContinueTarget blocks of a live block are also live.
    block->ForMergeAndContinueLabel(mark_reachable);
  }

  // Update operands of Phi nodes that reference unreachable blocks.
  for (auto& block : *func) {
    // If the block is about to be removed, don't bother updating its
    // Phi instructions.
    if (reachable_blocks.count(&block) == 0) {
      continue;
    }

    // If the block is reachable and has Phi instructions, remove all
    // operands from its Phi instructions that reference unreachable blocks.
    // If the block has no Phi instructions, this is a no-op.
    block.ForEachPhiInst([&reachable_blocks, this](Instruction* phi) {
      RemovePhiOperands(phi, reachable_blocks);
    });
  }

  // Erase unreachable blocks.
  for (auto ebi = func->begin(); ebi != func->end();) {
    if (reachable_blocks.count(&*ebi) == 0) {
      RemoveBlock(&ebi);
      modified = true;
    } else {
      ++ebi;
    }
  }

  return modified;
}

bool MemPass::CFGCleanup(Function* func) {
  bool modified = false;
  modified |= RemoveUnreachableBlocks(func);
  return modified;
}

void MemPass::CollectTargetVars(Function* func) {
  seen_target_vars_.clear();
  seen_non_target_vars_.clear();
  type2undefs_.clear();

  // Collect target (and non-) variable sets. Remove variables with
  // non-load/store refs from target variable set
  for (auto& blk : *func) {
    for (auto& inst : blk) {
      switch (inst.opcode()) {
        case spv::Op::OpStore:
        case spv::Op::OpLoad: {
          uint32_t varId;
          (void)GetPtr(&inst, &varId);
          if (!IsTargetVar(varId)) break;
          if (HasOnlySupportedRefs(varId)) break;
          seen_non_target_vars_.insert(varId);
          seen_target_vars_.erase(varId);
        } break;
        default:
          break;
      }
    }
  }
}

}  // namespace opt
}  // namespace spvtools
