// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "source/opt/aggressive_dead_code_elim_pass.h"

#include <memory>
#include <stack>

#include "source/cfa.h"
#include "source/latest_version_glsl_std_450_header.h"
#include "source/opt/iterator.h"
#include "source/opt/reflect.h"

namespace spvtools {
namespace opt {

namespace {

const uint32_t kTypePointerStorageClassInIdx = 0;
const uint32_t kEntryPointFunctionIdInIdx = 1;
const uint32_t kSelectionMergeMergeBlockIdInIdx = 0;
const uint32_t kLoopMergeMergeBlockIdInIdx = 0;
const uint32_t kLoopMergeContinueBlockIdInIdx = 1;
const uint32_t kCopyMemoryTargetAddrInIdx = 0;
const uint32_t kCopyMemorySourceAddrInIdx = 1;

// Sorting functor to present annotation instructions in an easy-to-process
// order. The functor orders by opcode first and falls back on unique id
// ordering if both instructions have the same opcode.
//
// Desired priority:
// SpvOpGroupDecorate
// SpvOpGroupMemberDecorate
// SpvOpDecorate
// SpvOpMemberDecorate
// SpvOpDecorateId
// SpvOpDecorateStringGOOGLE
// SpvOpDecorationGroup
struct DecorationLess {
  bool operator()(const Instruction* lhs, const Instruction* rhs) const {
    assert(lhs && rhs);
    SpvOp lhsOp = lhs->opcode();
    SpvOp rhsOp = rhs->opcode();
    if (lhsOp != rhsOp) {
#define PRIORITY_CASE(opcode)                          \
  if (lhsOp == opcode && rhsOp != opcode) return true; \
  if (rhsOp == opcode && lhsOp != opcode) return false;
      // OpGroupDecorate and OpGroupMember decorate are highest priority to
      // eliminate dead targets early and simplify subsequent checks.
      PRIORITY_CASE(SpvOpGroupDecorate)
      PRIORITY_CASE(SpvOpGroupMemberDecorate)
      PRIORITY_CASE(SpvOpDecorate)
      PRIORITY_CASE(SpvOpMemberDecorate)
      PRIORITY_CASE(SpvOpDecorateId)
      PRIORITY_CASE(SpvOpDecorateStringGOOGLE)
      // OpDecorationGroup is lowest priority to ensure use/def chains remain
      // usable for instructions that target this group.
      PRIORITY_CASE(SpvOpDecorationGroup)
#undef PRIORITY_CASE
    }

    // Fall back to maintain total ordering (compare unique ids).
    return *lhs < *rhs;
  }
};

}  // namespace

bool AggressiveDCEPass::IsVarOfStorage(uint32_t varId, uint32_t storageClass) {
  if (varId == 0) return false;
  const Instruction* varInst = get_def_use_mgr()->GetDef(varId);
  const SpvOp op = varInst->opcode();
  if (op != SpvOpVariable) return false;
  const uint32_t varTypeId = varInst->type_id();
  const Instruction* varTypeInst = get_def_use_mgr()->GetDef(varTypeId);
  if (varTypeInst->opcode() != SpvOpTypePointer) return false;
  return varTypeInst->GetSingleWordInOperand(kTypePointerStorageClassInIdx) ==
         storageClass;
}

bool AggressiveDCEPass::IsLocalVar(uint32_t varId) {
  if (IsVarOfStorage(varId, SpvStorageClassFunction)) {
    return true;
  }
  if (!private_like_local_) {
    return false;
  }

  return IsVarOfStorage(varId, SpvStorageClassPrivate) ||
         IsVarOfStorage(varId, SpvStorageClassWorkgroup);
}

void AggressiveDCEPass::AddStores(uint32_t ptrId) {
  get_def_use_mgr()->ForEachUser(ptrId, [this, ptrId](Instruction* user) {
    switch (user->opcode()) {
      case SpvOpAccessChain:
      case SpvOpInBoundsAccessChain:
      case SpvOpCopyObject:
        this->AddStores(user->result_id());
        break;
      case SpvOpLoad:
        break;
      case SpvOpCopyMemory:
      case SpvOpCopyMemorySized:
        if (user->GetSingleWordInOperand(kCopyMemoryTargetAddrInIdx) == ptrId) {
          AddToWorklist(user);
        }
        break;
      // If default, assume it stores e.g. frexp, modf, function call
      case SpvOpStore:
      default:
        AddToWorklist(user);
        break;
    }
  });
}

bool AggressiveDCEPass::AllExtensionsSupported() const {
  // If any extension not in whitelist, return false
  for (auto& ei : get_module()->extensions()) {
    const char* extName =
        reinterpret_cast<const char*>(&ei.GetInOperand(0).words[0]);
    if (extensions_whitelist_.find(extName) == extensions_whitelist_.end())
      return false;
  }
  return true;
}

bool AggressiveDCEPass::IsDead(Instruction* inst) {
  if (IsLive(inst)) return false;
  if ((inst->IsBranch() || inst->opcode() == SpvOpUnreachable) &&
      !IsStructuredHeader(context()->get_instr_block(inst), nullptr, nullptr,
                          nullptr))
    return false;
  return true;
}

bool AggressiveDCEPass::IsTargetDead(Instruction* inst) {
  const uint32_t tId = inst->GetSingleWordInOperand(0);
  Instruction* tInst = get_def_use_mgr()->GetDef(tId);
  if (IsAnnotationInst(tInst->opcode())) {
    // This must be a decoration group. We go through annotations in a specific
    // order. So if this is not used by any group or group member decorates, it
    // is dead.
    assert(tInst->opcode() == SpvOpDecorationGroup);
    bool dead = true;
    get_def_use_mgr()->ForEachUser(tInst, [&dead](Instruction* user) {
      if (user->opcode() == SpvOpGroupDecorate ||
          user->opcode() == SpvOpGroupMemberDecorate)
        dead = false;
    });
    return dead;
  }
  return IsDead(tInst);
}

void AggressiveDCEPass::ProcessLoad(uint32_t varId) {
  // Only process locals
  if (!IsLocalVar(varId)) return;
  // Return if already processed
  if (live_local_vars_.find(varId) != live_local_vars_.end()) return;
  // Mark all stores to varId as live
  AddStores(varId);
  // Cache varId as processed
  live_local_vars_.insert(varId);
}

bool AggressiveDCEPass::IsStructuredHeader(BasicBlock* bp,
                                           Instruction** mergeInst,
                                           Instruction** branchInst,
                                           uint32_t* mergeBlockId) {
  if (!bp) return false;
  Instruction* mi = bp->GetMergeInst();
  if (mi == nullptr) return false;
  Instruction* bri = &*bp->tail();
  if (branchInst != nullptr) *branchInst = bri;
  if (mergeInst != nullptr) *mergeInst = mi;
  if (mergeBlockId != nullptr) *mergeBlockId = mi->GetSingleWordInOperand(0);
  return true;
}

void AggressiveDCEPass::ComputeBlock2HeaderMaps(
    std::list<BasicBlock*>& structuredOrder) {
  block2headerBranch_.clear();
  header2nextHeaderBranch_.clear();
  branch2merge_.clear();
  structured_order_index_.clear();
  std::stack<Instruction*> currentHeaderBranch;
  currentHeaderBranch.push(nullptr);
  uint32_t currentMergeBlockId = 0;
  uint32_t index = 0;
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end();
       ++bi, ++index) {
    structured_order_index_[*bi] = index;
    // If this block is the merge block of the current control construct,
    // we are leaving the current construct so we must update state
    if ((*bi)->id() == currentMergeBlockId) {
      currentHeaderBranch.pop();
      Instruction* chb = currentHeaderBranch.top();
      if (chb != nullptr)
        currentMergeBlockId = branch2merge_[chb]->GetSingleWordInOperand(0);
    }
    Instruction* mergeInst;
    Instruction* branchInst;
    uint32_t mergeBlockId;
    bool is_header =
        IsStructuredHeader(*bi, &mergeInst, &branchInst, &mergeBlockId);
    // Map header block to next enclosing header.
    if (is_header) header2nextHeaderBranch_[*bi] = currentHeaderBranch.top();
    // If this is a loop header, update state first so the block will map to
    // itself.
    if (is_header && mergeInst->opcode() == SpvOpLoopMerge) {
      currentHeaderBranch.push(branchInst);
      branch2merge_[branchInst] = mergeInst;
      currentMergeBlockId = mergeBlockId;
    }
    // Map the block to the current construct.
    block2headerBranch_[*bi] = currentHeaderBranch.top();
    // If this is an if header, update state so following blocks map to the if.
    if (is_header && mergeInst->opcode() == SpvOpSelectionMerge) {
      currentHeaderBranch.push(branchInst);
      branch2merge_[branchInst] = mergeInst;
      currentMergeBlockId = mergeBlockId;
    }
  }
}

void AggressiveDCEPass::AddBranch(uint32_t labelId, BasicBlock* bp) {
  std::unique_ptr<Instruction> newBranch(
      new Instruction(context(), SpvOpBranch, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {labelId}}}));
  context()->AnalyzeDefUse(&*newBranch);
  context()->set_instr_block(&*newBranch, bp);
  bp->AddInstruction(std::move(newBranch));
}

void AggressiveDCEPass::AddBreaksAndContinuesToWorklist(
    Instruction* mergeInst) {
  assert(mergeInst->opcode() == SpvOpSelectionMerge ||
         mergeInst->opcode() == SpvOpLoopMerge);

  BasicBlock* header = context()->get_instr_block(mergeInst);
  uint32_t headerIndex = structured_order_index_[header];
  const uint32_t mergeId = mergeInst->GetSingleWordInOperand(0);
  BasicBlock* merge = context()->get_instr_block(mergeId);
  uint32_t mergeIndex = structured_order_index_[merge];
  get_def_use_mgr()->ForEachUser(
      mergeId, [headerIndex, mergeIndex, this](Instruction* user) {
        if (!user->IsBranch()) return;
        BasicBlock* block = context()->get_instr_block(user);
        uint32_t index = structured_order_index_[block];
        if (headerIndex < index && index < mergeIndex) {
          // This is a break from the loop.
          AddToWorklist(user);
          // Add branch's merge if there is one.
          Instruction* userMerge = branch2merge_[user];
          if (userMerge != nullptr) AddToWorklist(userMerge);
        }
      });

  if (mergeInst->opcode() != SpvOpLoopMerge) {
    return;
  }

  // For loops we need to find the continues as well.
  const uint32_t contId =
      mergeInst->GetSingleWordInOperand(kLoopMergeContinueBlockIdInIdx);
  get_def_use_mgr()->ForEachUser(contId, [&contId, this](Instruction* user) {
    SpvOp op = user->opcode();
    if (op == SpvOpBranchConditional || op == SpvOpSwitch) {
      // A conditional branch or switch can only be a continue if it does not
      // have a merge instruction or its merge block is not the continue block.
      Instruction* hdrMerge = branch2merge_[user];
      if (hdrMerge != nullptr && hdrMerge->opcode() == SpvOpSelectionMerge) {
        uint32_t hdrMergeId =
            hdrMerge->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx);
        if (hdrMergeId == contId) return;
        // Need to mark merge instruction too
        AddToWorklist(hdrMerge);
      }
    } else if (op == SpvOpBranch) {
      // An unconditional branch can only be a continue if it is not
      // branching to its own merge block.
      BasicBlock* blk = context()->get_instr_block(user);
      Instruction* hdrBranch = block2headerBranch_[blk];
      if (hdrBranch == nullptr) return;
      Instruction* hdrMerge = branch2merge_[hdrBranch];
      if (hdrMerge->opcode() == SpvOpLoopMerge) return;
      uint32_t hdrMergeId =
          hdrMerge->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx);
      if (contId == hdrMergeId) return;
    } else {
      return;
    }
    AddToWorklist(user);
  });
}

bool AggressiveDCEPass::AggressiveDCE(Function* func) {
  // Mark function parameters as live.
  AddToWorklist(&func->DefInst());
  func->ForEachParam(
      [this](const Instruction* param) {
        AddToWorklist(const_cast<Instruction*>(param));
      },
      false);

  // Compute map from block to controlling conditional branch
  std::list<BasicBlock*> structuredOrder;
  cfg()->ComputeStructuredOrder(func, &*func->begin(), &structuredOrder);
  ComputeBlock2HeaderMaps(structuredOrder);
  bool modified = false;
  // Add instructions with external side effects to worklist. Also add branches
  // EXCEPT those immediately contained in an "if" selection construct or a loop
  // or continue construct.
  // TODO(greg-lunarg): Handle Frexp, Modf more optimally
  call_in_func_ = false;
  func_is_entry_point_ = false;
  private_stores_.clear();
  // Stacks to keep track of when we are inside an if- or loop-construct.
  // When immediately inside an if- or loop-construct, we do not initially
  // mark branches live. All other branches must be marked live.
  std::stack<bool> assume_branches_live;
  std::stack<uint32_t> currentMergeBlockId;
  // Push sentinel values on stack for when outside of any control flow.
  assume_branches_live.push(true);
  currentMergeBlockId.push(0);
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end(); ++bi) {
    // If exiting if or loop, update stacks
    if ((*bi)->id() == currentMergeBlockId.top()) {
      assume_branches_live.pop();
      currentMergeBlockId.pop();
    }
    for (auto ii = (*bi)->begin(); ii != (*bi)->end(); ++ii) {
      SpvOp op = ii->opcode();
      switch (op) {
        case SpvOpStore: {
          uint32_t varId;
          (void)GetPtr(&*ii, &varId);
          // Mark stores as live if their variable is not function scope
          // and is not private scope. Remember private stores for possible
          // later inclusion.  We cannot call IsLocalVar at this point because
          // private_like_local_ has not been set yet.
          if (IsVarOfStorage(varId, SpvStorageClassPrivate) ||
              IsVarOfStorage(varId, SpvStorageClassWorkgroup))
            private_stores_.push_back(&*ii);
          else if (!IsVarOfStorage(varId, SpvStorageClassFunction))
            AddToWorklist(&*ii);
        } break;
        case SpvOpCopyMemory:
        case SpvOpCopyMemorySized: {
          uint32_t varId;
          (void)GetPtr(ii->GetSingleWordInOperand(kCopyMemoryTargetAddrInIdx),
                       &varId);
          if (IsVarOfStorage(varId, SpvStorageClassPrivate) ||
              IsVarOfStorage(varId, SpvStorageClassWorkgroup))
            private_stores_.push_back(&*ii);
          else if (!IsVarOfStorage(varId, SpvStorageClassFunction))
            AddToWorklist(&*ii);
        } break;
        case SpvOpLoopMerge: {
          assume_branches_live.push(false);
          currentMergeBlockId.push(
              ii->GetSingleWordInOperand(kLoopMergeMergeBlockIdInIdx));
        } break;
        case SpvOpSelectionMerge: {
          assume_branches_live.push(false);
          currentMergeBlockId.push(
              ii->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx));
        } break;
        case SpvOpSwitch:
        case SpvOpBranch:
        case SpvOpBranchConditional:
        case SpvOpUnreachable: {
          if (assume_branches_live.top()) {
            AddToWorklist(&*ii);
          }
        } break;
        default: {
          // Function calls, atomics, function params, function returns, etc.
          // TODO(greg-lunarg): function calls live only if write to non-local
          if (!ii->IsOpcodeSafeToDelete()) {
            AddToWorklist(&*ii);
          }
          // Remember function calls
          if (op == SpvOpFunctionCall) call_in_func_ = true;
        } break;
      }
    }
  }
  // See if current function is an entry point
  for (auto& ei : get_module()->entry_points()) {
    if (ei.GetSingleWordInOperand(kEntryPointFunctionIdInIdx) ==
        func->result_id()) {
      func_is_entry_point_ = true;
      break;
    }
  }
  // If the current function is an entry point and has no function calls,
  // we can optimize private variables as locals
  private_like_local_ = func_is_entry_point_ && !call_in_func_;
  // If privates are not like local, add their stores to worklist
  if (!private_like_local_)
    for (auto& ps : private_stores_) AddToWorklist(ps);
  // Perform closure on live instruction set.
  while (!worklist_.empty()) {
    Instruction* liveInst = worklist_.front();
    // Add all operand instructions if not already live
    liveInst->ForEachInId([&liveInst, this](const uint32_t* iid) {
      Instruction* inInst = get_def_use_mgr()->GetDef(*iid);
      // Do not add label if an operand of a branch. This is not needed
      // as part of live code discovery and can create false live code,
      // for example, the branch to a header of a loop.
      if (inInst->opcode() == SpvOpLabel && liveInst->IsBranch()) return;
      AddToWorklist(inInst);
    });
    if (liveInst->type_id() != 0) {
      AddToWorklist(get_def_use_mgr()->GetDef(liveInst->type_id()));
    }
    // If in a structured if or loop construct, add the controlling
    // conditional branch and its merge.
    BasicBlock* blk = context()->get_instr_block(liveInst);
    Instruction* branchInst = block2headerBranch_[blk];
    if (branchInst != nullptr) {
      AddToWorklist(branchInst);
      Instruction* mergeInst = branch2merge_[branchInst];
      AddToWorklist(mergeInst);
    }
    // If the block is a header, add the next outermost controlling
    // conditional branch and its merge.
    Instruction* nextBranchInst = header2nextHeaderBranch_[blk];
    if (nextBranchInst != nullptr) {
      AddToWorklist(nextBranchInst);
      Instruction* mergeInst = branch2merge_[nextBranchInst];
      AddToWorklist(mergeInst);
    }
    // If local load, add all variable's stores if variable not already live
    if (liveInst->opcode() == SpvOpLoad || liveInst->IsAtomicWithLoad()) {
      uint32_t varId;
      (void)GetPtr(liveInst, &varId);
      if (varId != 0) {
        ProcessLoad(varId);
      }
      // Process memory copies like loads
    } else if (liveInst->opcode() == SpvOpCopyMemory ||
               liveInst->opcode() == SpvOpCopyMemorySized) {
      uint32_t varId;
      (void)GetPtr(liveInst->GetSingleWordInOperand(kCopyMemorySourceAddrInIdx),
                   &varId);
      if (varId != 0) {
        ProcessLoad(varId);
      }
      // If merge, add other branches that are part of its control structure
    } else if (liveInst->opcode() == SpvOpLoopMerge ||
               liveInst->opcode() == SpvOpSelectionMerge) {
      AddBreaksAndContinuesToWorklist(liveInst);
      // If function call, treat as if it loads from all pointer arguments
    } else if (liveInst->opcode() == SpvOpFunctionCall) {
      liveInst->ForEachInId([this](const uint32_t* iid) {
        // Skip non-ptr args
        if (!IsPtr(*iid)) return;
        uint32_t varId;
        (void)GetPtr(*iid, &varId);
        ProcessLoad(varId);
      });
      // If function parameter, treat as if it's result id is loaded from
    } else if (liveInst->opcode() == SpvOpFunctionParameter) {
      ProcessLoad(liveInst->result_id());
      // We treat an OpImageTexelPointer as a load of the pointer, and
      // that value is manipulated to get the result.
    } else if (liveInst->opcode() == SpvOpImageTexelPointer) {
      uint32_t varId;
      (void)GetPtr(liveInst, &varId);
      if (varId != 0) {
        ProcessLoad(varId);
      }
    }
    worklist_.pop();
  }

  // Kill dead instructions and remember dead blocks
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end();) {
    uint32_t mergeBlockId = 0;
    (*bi)->ForEachInst([this, &modified, &mergeBlockId](Instruction* inst) {
      if (!IsDead(inst)) return;
      if (inst->opcode() == SpvOpLabel) return;
      // If dead instruction is selection merge, remember merge block
      // for new branch at end of block
      if (inst->opcode() == SpvOpSelectionMerge ||
          inst->opcode() == SpvOpLoopMerge)
        mergeBlockId = inst->GetSingleWordInOperand(0);
      to_kill_.push_back(inst);
      modified = true;
    });
    // If a structured if or loop was deleted, add a branch to its merge
    // block, and traverse to the merge block and continue processing there.
    // We know the block still exists because the label is not deleted.
    if (mergeBlockId != 0) {
      AddBranch(mergeBlockId, *bi);
      for (++bi; (*bi)->id() != mergeBlockId; ++bi) {
      }

      auto merge_terminator = (*bi)->terminator();
      if (merge_terminator->opcode() == SpvOpUnreachable) {
        // The merge was unreachable. This is undefined behaviour so just
        // return (or return an undef). Then mark the new return as live.
        auto func_ret_type_inst = get_def_use_mgr()->GetDef(func->type_id());
        if (func_ret_type_inst->opcode() == SpvOpTypeVoid) {
          merge_terminator->SetOpcode(SpvOpReturn);
        } else {
          // Find an undef for the return value and make sure it gets kept by
          // the pass.
          auto undef_id = Type2Undef(func->type_id());
          auto undef = get_def_use_mgr()->GetDef(undef_id);
          live_insts_.Set(undef->unique_id());
          merge_terminator->SetOpcode(SpvOpReturnValue);
          merge_terminator->SetInOperands({{SPV_OPERAND_TYPE_ID, {undef_id}}});
          get_def_use_mgr()->AnalyzeInstUse(merge_terminator);
        }
        live_insts_.Set(merge_terminator->unique_id());
      }
    } else {
      ++bi;
    }
  }

  return modified;
}

void AggressiveDCEPass::InitializeModuleScopeLiveInstructions() {
  // Keep all execution modes.
  for (auto& exec : get_module()->execution_modes()) {
    AddToWorklist(&exec);
  }
  // Keep all entry points.
  for (auto& entry : get_module()->entry_points()) {
    AddToWorklist(&entry);
  }
  // Keep workgroup size.
  for (auto& anno : get_module()->annotations()) {
    if (anno.opcode() == SpvOpDecorate) {
      if (anno.GetSingleWordInOperand(1u) == SpvDecorationBuiltIn &&
          anno.GetSingleWordInOperand(2u) == SpvBuiltInWorkgroupSize) {
        AddToWorklist(&anno);
      }
    }
  }
}

Pass::Status AggressiveDCEPass::ProcessImpl() {
  // Current functionality assumes shader capability
  // TODO(greg-lunarg): Handle additional capabilities
  if (!context()->get_feature_mgr()->HasCapability(SpvCapabilityShader))
    return Status::SuccessWithoutChange;

  // Current functionality assumes relaxed logical addressing (see
  // instruction.h)
  // TODO(greg-lunarg): Handle non-logical addressing
  if (context()->get_feature_mgr()->HasCapability(SpvCapabilityAddresses))
    return Status::SuccessWithoutChange;

  // The variable pointer extension is no longer needed to use the capability,
  // so we have to look for the capability.
  if (context()->get_feature_mgr()->HasCapability(
          SpvCapabilityVariablePointersStorageBuffer))
    return Status::SuccessWithoutChange;

  // If any extensions in the module are not explicitly supported,
  // return unmodified.
  if (!AllExtensionsSupported()) return Status::SuccessWithoutChange;

  // If the decoration manager is kept live then the context will try to keep it
  // up to date.  ADCE deals with group decorations by changing the operands in
  // |OpGroupDecorate| instruction directly without informing the decoration
  // manager.  This can put it in an invalid state which will cause an error
  // when the context tries to update it.  To avoid this problem invalidate
  // the decoration manager upfront.
  context()->InvalidateAnalyses(IRContext::Analysis::kAnalysisDecorations);

  // Eliminate Dead functions.
  bool modified = EliminateDeadFunctions();

  InitializeModuleScopeLiveInstructions();

  // Process all entry point functions.
  ProcessFunction pfn = [this](Function* fp) { return AggressiveDCE(fp); };
  modified |= context()->ProcessEntryPointCallTree(pfn);

  // Process module-level instructions. Now that all live instructions have
  // been marked, it is safe to remove dead global values.
  modified |= ProcessGlobalValues();

  // Kill all dead instructions.
  for (auto inst : to_kill_) {
    context()->KillInst(inst);
  }

  // Cleanup all CFG including all unreachable blocks.
  ProcessFunction cleanup = [this](Function* f) { return CFGCleanup(f); };
  modified |= context()->ProcessEntryPointCallTree(cleanup);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool AggressiveDCEPass::EliminateDeadFunctions() {
  // Identify live functions first. Those that are not live
  // are dead. ADCE is disabled for non-shaders so we do not check for exported
  // functions here.
  std::unordered_set<const Function*> live_function_set;
  ProcessFunction mark_live = [&live_function_set](Function* fp) {
    live_function_set.insert(fp);
    return false;
  };
  context()->ProcessEntryPointCallTree(mark_live);

  bool modified = false;
  for (auto funcIter = get_module()->begin();
       funcIter != get_module()->end();) {
    if (live_function_set.count(&*funcIter) == 0) {
      modified = true;
      EliminateFunction(&*funcIter);
      funcIter = funcIter.Erase();
    } else {
      ++funcIter;
    }
  }

  return modified;
}

void AggressiveDCEPass::EliminateFunction(Function* func) {
  // Remove all of the instruction in the function body
  func->ForEachInst([this](Instruction* inst) { context()->KillInst(inst); },
                    true);
}

bool AggressiveDCEPass::ProcessGlobalValues() {
  // Remove debug and annotation statements referencing dead instructions.
  // This must be done before killing the instructions, otherwise there are
  // dead objects in the def/use database.
  bool modified = false;
  Instruction* instruction = &*get_module()->debug2_begin();
  while (instruction) {
    if (instruction->opcode() != SpvOpName) {
      instruction = instruction->NextNode();
      continue;
    }

    if (IsTargetDead(instruction)) {
      instruction = context()->KillInst(instruction);
      modified = true;
    } else {
      instruction = instruction->NextNode();
    }
  }

  // This code removes all unnecessary decorations safely (see #1174). It also
  // does so in a more efficient manner than deleting them only as the targets
  // are deleted.
  std::vector<Instruction*> annotations;
  for (auto& inst : get_module()->annotations()) annotations.push_back(&inst);
  std::sort(annotations.begin(), annotations.end(), DecorationLess());
  for (auto annotation : annotations) {
    switch (annotation->opcode()) {
      case SpvOpDecorate:
      case SpvOpMemberDecorate:
      case SpvOpDecorateStringGOOGLE:
      case SpvOpMemberDecorateStringGOOGLE:
        if (IsTargetDead(annotation)) {
          context()->KillInst(annotation);
          modified = true;
        }
        break;
      case SpvOpDecorateId:
        if (IsTargetDead(annotation)) {
          context()->KillInst(annotation);
          modified = true;
        } else {
          if (annotation->GetSingleWordInOperand(1) ==
              SpvDecorationHlslCounterBufferGOOGLE) {
            // HlslCounterBuffer will reference an id other than the target.
            // If that id is dead, then the decoration can be removed as well.
            uint32_t counter_buffer_id = annotation->GetSingleWordInOperand(2);
            Instruction* counter_buffer_inst =
                get_def_use_mgr()->GetDef(counter_buffer_id);
            if (IsDead(counter_buffer_inst)) {
              context()->KillInst(annotation);
              modified = true;
            }
          }
        }
        break;
      case SpvOpGroupDecorate: {
        // Go through the targets of this group decorate. Remove each dead
        // target. If all targets are dead, remove this decoration.
        bool dead = true;
        bool removed_operand = false;
        for (uint32_t i = 1; i < annotation->NumOperands();) {
          Instruction* opInst =
              get_def_use_mgr()->GetDef(annotation->GetSingleWordOperand(i));
          if (IsDead(opInst)) {
            // Don't increment |i|.
            annotation->RemoveOperand(i);
            modified = true;
            removed_operand = true;
          } else {
            i++;
            dead = false;
          }
        }
        if (dead) {
          context()->KillInst(annotation);
          modified = true;
        } else if (removed_operand) {
          context()->UpdateDefUse(annotation);
        }
        break;
      }
      case SpvOpGroupMemberDecorate: {
        // Go through the targets of this group member decorate. Remove each
        // dead target (and member index). If all targets are dead, remove this
        // decoration.
        bool dead = true;
        bool removed_operand = false;
        for (uint32_t i = 1; i < annotation->NumOperands();) {
          Instruction* opInst =
              get_def_use_mgr()->GetDef(annotation->GetSingleWordOperand(i));
          if (IsDead(opInst)) {
            // Don't increment |i|.
            annotation->RemoveOperand(i + 1);
            annotation->RemoveOperand(i);
            modified = true;
            removed_operand = true;
          } else {
            i += 2;
            dead = false;
          }
        }
        if (dead) {
          context()->KillInst(annotation);
          modified = true;
        } else if (removed_operand) {
          context()->UpdateDefUse(annotation);
        }
        break;
      }
      case SpvOpDecorationGroup:
        // By the time we hit decoration groups we've checked everything that
        // can target them. So if they have no uses they must be dead.
        if (get_def_use_mgr()->NumUsers(annotation) == 0) {
          context()->KillInst(annotation);
          modified = true;
        }
        break;
      default:
        assert(false);
        break;
    }
  }

  // Since ADCE is disabled for non-shaders, we don't check for export linkage
  // attributes here.
  for (auto& val : get_module()->types_values()) {
    if (IsDead(&val)) {
      to_kill_.push_back(&val);
    }
  }

  return modified;
}

AggressiveDCEPass::AggressiveDCEPass() = default;

Pass::Status AggressiveDCEPass::Process() {
  // Initialize extensions whitelist
  InitExtensions();
  return ProcessImpl();
}

void AggressiveDCEPass::InitExtensions() {
  extensions_whitelist_.clear();
  extensions_whitelist_.insert({
      "SPV_AMD_shader_explicit_vertex_parameter",
      "SPV_AMD_shader_trinary_minmax",
      "SPV_AMD_gcn_shader",
      "SPV_KHR_shader_ballot",
      "SPV_AMD_shader_ballot",
      "SPV_AMD_gpu_shader_half_float",
      "SPV_KHR_shader_draw_parameters",
      "SPV_KHR_subgroup_vote",
      "SPV_KHR_16bit_storage",
      "SPV_KHR_device_group",
      "SPV_KHR_multiview",
      "SPV_NVX_multiview_per_view_attributes",
      "SPV_NV_viewport_array2",
      "SPV_NV_stereo_view_rendering",
      "SPV_NV_sample_mask_override_coverage",
      "SPV_NV_geometry_shader_passthrough",
      "SPV_AMD_texture_gather_bias_lod",
      "SPV_KHR_storage_buffer_storage_class",
      // SPV_KHR_variable_pointers
      //   Currently do not support extended pointer expressions
      "SPV_AMD_gpu_shader_int16",
      "SPV_KHR_post_depth_coverage",
      "SPV_KHR_shader_atomic_counter_ops",
      "SPV_EXT_shader_stencil_export",
      "SPV_EXT_shader_viewport_index_layer",
      "SPV_AMD_shader_image_load_store_lod",
      "SPV_AMD_shader_fragment_mask",
      "SPV_EXT_fragment_fully_covered",
      "SPV_AMD_gpu_shader_half_float_fetch",
      "SPV_GOOGLE_decorate_string",
      "SPV_GOOGLE_hlsl_functionality1",
      "SPV_NV_shader_subgroup_partitioned",
      "SPV_EXT_descriptor_indexing",
      "SPV_NV_fragment_shader_barycentric",
      "SPV_NV_compute_shader_derivatives",
      "SPV_NV_shader_image_footprint",
      "SPV_NV_shading_rate",
      "SPV_NV_mesh_shader",
      "SPV_NV_ray_tracing",
      "SPV_EXT_fragment_invocation_density",
  });
}

}  // namespace opt
}  // namespace spvtools
