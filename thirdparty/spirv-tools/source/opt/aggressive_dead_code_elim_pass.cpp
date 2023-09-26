// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
// Copyright (c) 2018-2021 Google LLC
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
#include "source/opt/eliminate_dead_functions_util.h"
#include "source/opt/ir_builder.h"
#include "source/opt/iterator.h"
#include "source/opt/reflect.h"
#include "source/spirv_constant.h"
#include "source/util/string_utils.h"

namespace spvtools {
namespace opt {
namespace {

constexpr uint32_t kTypePointerStorageClassInIdx = 0;
constexpr uint32_t kEntryPointFunctionIdInIdx = 1;
constexpr uint32_t kSelectionMergeMergeBlockIdInIdx = 0;
constexpr uint32_t kLoopMergeContinueBlockIdInIdx = 1;
constexpr uint32_t kCopyMemoryTargetAddrInIdx = 0;
constexpr uint32_t kCopyMemorySourceAddrInIdx = 1;
constexpr uint32_t kLoadSourceAddrInIdx = 0;
constexpr uint32_t kDebugDeclareOperandVariableIndex = 5;
constexpr uint32_t kGlobalVariableVariableIndex = 12;

// Sorting functor to present annotation instructions in an easy-to-process
// order. The functor orders by opcode first and falls back on unique id
// ordering if both instructions have the same opcode.
//
// Desired priority:
// spv::Op::OpGroupDecorate
// spv::Op::OpGroupMemberDecorate
// spv::Op::OpDecorate
// spv::Op::OpMemberDecorate
// spv::Op::OpDecorateId
// spv::Op::OpDecorateStringGOOGLE
// spv::Op::OpDecorationGroup
struct DecorationLess {
  bool operator()(const Instruction* lhs, const Instruction* rhs) const {
    assert(lhs && rhs);
    spv::Op lhsOp = lhs->opcode();
    spv::Op rhsOp = rhs->opcode();
    if (lhsOp != rhsOp) {
#define PRIORITY_CASE(opcode)                          \
  if (lhsOp == opcode && rhsOp != opcode) return true; \
  if (rhsOp == opcode && lhsOp != opcode) return false;
      // OpGroupDecorate and OpGroupMember decorate are highest priority to
      // eliminate dead targets early and simplify subsequent checks.
      PRIORITY_CASE(spv::Op::OpGroupDecorate)
      PRIORITY_CASE(spv::Op::OpGroupMemberDecorate)
      PRIORITY_CASE(spv::Op::OpDecorate)
      PRIORITY_CASE(spv::Op::OpMemberDecorate)
      PRIORITY_CASE(spv::Op::OpDecorateId)
      PRIORITY_CASE(spv::Op::OpDecorateStringGOOGLE)
      // OpDecorationGroup is lowest priority to ensure use/def chains remain
      // usable for instructions that target this group.
      PRIORITY_CASE(spv::Op::OpDecorationGroup)
#undef PRIORITY_CASE
    }

    // Fall back to maintain total ordering (compare unique ids).
    return *lhs < *rhs;
  }
};

}  // namespace

bool AggressiveDCEPass::IsVarOfStorage(uint32_t varId,
                                       spv::StorageClass storageClass) {
  if (varId == 0) return false;
  const Instruction* varInst = get_def_use_mgr()->GetDef(varId);
  const spv::Op op = varInst->opcode();
  if (op != spv::Op::OpVariable) return false;
  const uint32_t varTypeId = varInst->type_id();
  const Instruction* varTypeInst = get_def_use_mgr()->GetDef(varTypeId);
  if (varTypeInst->opcode() != spv::Op::OpTypePointer) return false;
  return spv::StorageClass(varTypeInst->GetSingleWordInOperand(
             kTypePointerStorageClassInIdx)) == storageClass;
}

bool AggressiveDCEPass::IsLocalVar(uint32_t varId, Function* func) {
  if (IsVarOfStorage(varId, spv::StorageClass::Function)) {
    return true;
  }

  if (!IsVarOfStorage(varId, spv::StorageClass::Private) &&
      !IsVarOfStorage(varId, spv::StorageClass::Workgroup)) {
    return false;
  }

  // For a variable in the Private or WorkGroup storage class, the variable will
  // get a new instance for every call to an entry point.  If the entry point
  // does not have a call, then no other function can read or write to that
  // instance of the variable.
  return IsEntryPointWithNoCalls(func);
}

void AggressiveDCEPass::AddStores(Function* func, uint32_t ptrId) {
  get_def_use_mgr()->ForEachUser(ptrId, [this, ptrId, func](Instruction* user) {
    // If the user is not a part of |func|, skip it.
    BasicBlock* blk = context()->get_instr_block(user);
    if (blk && blk->GetParent() != func) return;

    switch (user->opcode()) {
      case spv::Op::OpAccessChain:
      case spv::Op::OpInBoundsAccessChain:
      case spv::Op::OpCopyObject:
        this->AddStores(func, user->result_id());
        break;
      case spv::Op::OpLoad:
        break;
      case spv::Op::OpCopyMemory:
      case spv::Op::OpCopyMemorySized:
        if (user->GetSingleWordInOperand(kCopyMemoryTargetAddrInIdx) == ptrId) {
          AddToWorklist(user);
        }
        break;
      // If default, assume it stores e.g. frexp, modf, function call
      case spv::Op::OpStore:
      default:
        AddToWorklist(user);
        break;
    }
  });
}

bool AggressiveDCEPass::AllExtensionsSupported() const {
  // If any extension not in allowlist, return false
  for (auto& ei : get_module()->extensions()) {
    const std::string extName = ei.GetInOperand(0).AsString();
    if (extensions_allowlist_.find(extName) == extensions_allowlist_.end())
      return false;
  }
  // Only allow NonSemantic.Shader.DebugInfo.100, we cannot safely optimise
  // around unknown extended instruction sets even if they are non-semantic
  for (auto& inst : context()->module()->ext_inst_imports()) {
    assert(inst.opcode() == spv::Op::OpExtInstImport &&
           "Expecting an import of an extension's instruction set.");
    const std::string extension_name = inst.GetInOperand(0).AsString();
    if (spvtools::utils::starts_with(extension_name, "NonSemantic.") &&
        extension_name != "NonSemantic.Shader.DebugInfo.100") {
      return false;
    }
  }
  return true;
}

bool AggressiveDCEPass::IsTargetDead(Instruction* inst) {
  const uint32_t tId = inst->GetSingleWordInOperand(0);
  Instruction* tInst = get_def_use_mgr()->GetDef(tId);
  if (IsAnnotationInst(tInst->opcode())) {
    // This must be a decoration group. We go through annotations in a specific
    // order. So if this is not used by any group or group member decorates, it
    // is dead.
    assert(tInst->opcode() == spv::Op::OpDecorationGroup);
    bool dead = true;
    get_def_use_mgr()->ForEachUser(tInst, [&dead](Instruction* user) {
      if (user->opcode() == spv::Op::OpGroupDecorate ||
          user->opcode() == spv::Op::OpGroupMemberDecorate)
        dead = false;
    });
    return dead;
  }
  return !IsLive(tInst);
}

void AggressiveDCEPass::ProcessLoad(Function* func, uint32_t varId) {
  // Only process locals
  if (!IsLocalVar(varId, func)) return;
  // Return if already processed
  if (live_local_vars_.find(varId) != live_local_vars_.end()) return;
  // Mark all stores to varId as live
  AddStores(func, varId);
  // Cache varId as processed
  live_local_vars_.insert(varId);
}

void AggressiveDCEPass::AddBranch(uint32_t labelId, BasicBlock* bp) {
  std::unique_ptr<Instruction> newBranch(
      new Instruction(context(), spv::Op::OpBranch, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {labelId}}}));
  context()->AnalyzeDefUse(&*newBranch);
  context()->set_instr_block(&*newBranch, bp);
  bp->AddInstruction(std::move(newBranch));
}

void AggressiveDCEPass::AddBreaksAndContinuesToWorklist(
    Instruction* mergeInst) {
  assert(mergeInst->opcode() == spv::Op::OpSelectionMerge ||
         mergeInst->opcode() == spv::Op::OpLoopMerge);

  BasicBlock* header = context()->get_instr_block(mergeInst);
  const uint32_t mergeId = mergeInst->GetSingleWordInOperand(0);
  get_def_use_mgr()->ForEachUser(mergeId, [header, this](Instruction* user) {
    if (!user->IsBranch()) return;
    BasicBlock* block = context()->get_instr_block(user);
    if (BlockIsInConstruct(header, block)) {
      // This is a break from the loop.
      AddToWorklist(user);
      // Add branch's merge if there is one.
      Instruction* userMerge = GetMergeInstruction(user);
      if (userMerge != nullptr) AddToWorklist(userMerge);
    }
  });

  if (mergeInst->opcode() != spv::Op::OpLoopMerge) {
    return;
  }

  // For loops we need to find the continues as well.
  const uint32_t contId =
      mergeInst->GetSingleWordInOperand(kLoopMergeContinueBlockIdInIdx);
  get_def_use_mgr()->ForEachUser(contId, [&contId, this](Instruction* user) {
    spv::Op op = user->opcode();
    if (op == spv::Op::OpBranchConditional || op == spv::Op::OpSwitch) {
      // A conditional branch or switch can only be a continue if it does not
      // have a merge instruction or its merge block is not the continue block.
      Instruction* hdrMerge = GetMergeInstruction(user);
      if (hdrMerge != nullptr &&
          hdrMerge->opcode() == spv::Op::OpSelectionMerge) {
        uint32_t hdrMergeId =
            hdrMerge->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx);
        if (hdrMergeId == contId) return;
        // Need to mark merge instruction too
        AddToWorklist(hdrMerge);
      }
    } else if (op == spv::Op::OpBranch) {
      // An unconditional branch can only be a continue if it is not
      // branching to its own merge block.
      BasicBlock* blk = context()->get_instr_block(user);
      Instruction* hdrBranch = GetHeaderBranch(blk);
      if (hdrBranch == nullptr) return;
      Instruction* hdrMerge = GetMergeInstruction(hdrBranch);
      if (hdrMerge->opcode() == spv::Op::OpLoopMerge) return;
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
  std::list<BasicBlock*> structured_order;
  cfg()->ComputeStructuredOrder(func, &*func->begin(), &structured_order);
  live_local_vars_.clear();
  InitializeWorkList(func, structured_order);
  ProcessWorkList(func);
  return KillDeadInstructions(func, structured_order);
}

bool AggressiveDCEPass::KillDeadInstructions(
    const Function* func, std::list<BasicBlock*>& structured_order) {
  bool modified = false;
  for (auto bi = structured_order.begin(); bi != structured_order.end();) {
    uint32_t merge_block_id = 0;
    (*bi)->ForEachInst([this, &modified, &merge_block_id](Instruction* inst) {
      if (IsLive(inst)) return;
      if (inst->opcode() == spv::Op::OpLabel) return;
      // If dead instruction is selection merge, remember merge block
      // for new branch at end of block
      if (inst->opcode() == spv::Op::OpSelectionMerge ||
          inst->opcode() == spv::Op::OpLoopMerge)
        merge_block_id = inst->GetSingleWordInOperand(0);
      to_kill_.push_back(inst);
      modified = true;
    });
    // If a structured if or loop was deleted, add a branch to its merge
    // block, and traverse to the merge block and continue processing there.
    // We know the block still exists because the label is not deleted.
    if (merge_block_id != 0) {
      AddBranch(merge_block_id, *bi);
      for (++bi; (*bi)->id() != merge_block_id; ++bi) {
      }

      auto merge_terminator = (*bi)->terminator();
      if (merge_terminator->opcode() == spv::Op::OpUnreachable) {
        // The merge was unreachable. This is undefined behaviour so just
        // return (or return an undef). Then mark the new return as live.
        auto func_ret_type_inst = get_def_use_mgr()->GetDef(func->type_id());
        if (func_ret_type_inst->opcode() == spv::Op::OpTypeVoid) {
          merge_terminator->SetOpcode(spv::Op::OpReturn);
        } else {
          // Find an undef for the return value and make sure it gets kept by
          // the pass.
          auto undef_id = Type2Undef(func->type_id());
          auto undef = get_def_use_mgr()->GetDef(undef_id);
          live_insts_.Set(undef->unique_id());
          merge_terminator->SetOpcode(spv::Op::OpReturnValue);
          merge_terminator->SetInOperands({{SPV_OPERAND_TYPE_ID, {undef_id}}});
          get_def_use_mgr()->AnalyzeInstUse(merge_terminator);
        }
        live_insts_.Set(merge_terminator->unique_id());
      }
    } else {
      Instruction* inst = (*bi)->terminator();
      if (!IsLive(inst)) {
        // If the terminator is not live, this block has no live instructions,
        // and it will be unreachable.
        AddUnreachable(*bi);
      }
      ++bi;
    }
  }
  return modified;
}

void AggressiveDCEPass::ProcessWorkList(Function* func) {
  while (!worklist_.empty()) {
    Instruction* live_inst = worklist_.front();
    worklist_.pop();
    AddOperandsToWorkList(live_inst);
    MarkBlockAsLive(live_inst);
    MarkLoadedVariablesAsLive(func, live_inst);
    AddDecorationsToWorkList(live_inst);
    AddDebugInstructionsToWorkList(live_inst);
  }
}

void AggressiveDCEPass::AddDebugScopeToWorkList(const Instruction* inst) {
  auto scope = inst->GetDebugScope();
  auto lex_scope_id = scope.GetLexicalScope();
  if (lex_scope_id != kNoDebugScope)
    AddToWorklist(get_def_use_mgr()->GetDef(lex_scope_id));
  auto inlined_at_id = scope.GetInlinedAt();
  if (inlined_at_id != kNoInlinedAt)
    AddToWorklist(get_def_use_mgr()->GetDef(inlined_at_id));
}

void AggressiveDCEPass::AddDebugInstructionsToWorkList(
    const Instruction* inst) {
  for (auto& line_inst : inst->dbg_line_insts()) {
    if (line_inst.IsDebugLineInst()) {
      AddOperandsToWorkList(&line_inst);
    }
    AddDebugScopeToWorkList(&line_inst);
  }
  AddDebugScopeToWorkList(inst);
}

void AggressiveDCEPass::AddDecorationsToWorkList(const Instruction* inst) {
  // Add OpDecorateId instructions that apply to this instruction to the work
  // list.  We use the decoration manager to look through the group
  // decorations to get to the OpDecorate* instructions themselves.
  auto decorations =
      get_decoration_mgr()->GetDecorationsFor(inst->result_id(), false);
  for (Instruction* dec : decorations) {
    // We only care about OpDecorateId instructions because the are the only
    // decorations that will reference an id that will have to be kept live
    // because of that use.
    if (dec->opcode() != spv::Op::OpDecorateId) {
      continue;
    }
    if (spv::Decoration(dec->GetSingleWordInOperand(1)) ==
        spv::Decoration::HlslCounterBufferGOOGLE) {
      // These decorations should not force the use id to be live.  It will be
      // removed if either the target or the in operand are dead.
      continue;
    }
    AddToWorklist(dec);
  }
}

void AggressiveDCEPass::MarkLoadedVariablesAsLive(Function* func,
                                                  Instruction* inst) {
  std::vector<uint32_t> live_variables = GetLoadedVariables(inst);
  for (uint32_t var_id : live_variables) {
    ProcessLoad(func, var_id);
  }
}

std::vector<uint32_t> AggressiveDCEPass::GetLoadedVariables(Instruction* inst) {
  if (inst->opcode() == spv::Op::OpFunctionCall) {
    return GetLoadedVariablesFromFunctionCall(inst);
  }
  uint32_t var_id = GetLoadedVariableFromNonFunctionCalls(inst);
  if (var_id == 0) {
    return {};
  }
  return {var_id};
}

uint32_t AggressiveDCEPass::GetLoadedVariableFromNonFunctionCalls(
    Instruction* inst) {
  std::vector<uint32_t> live_variables;
  if (inst->IsAtomicWithLoad()) {
    return GetVariableId(inst->GetSingleWordInOperand(kLoadSourceAddrInIdx));
  }

  switch (inst->opcode()) {
    case spv::Op::OpLoad:
    case spv::Op::OpImageTexelPointer:
      return GetVariableId(inst->GetSingleWordInOperand(kLoadSourceAddrInIdx));
    case spv::Op::OpCopyMemory:
    case spv::Op::OpCopyMemorySized:
      return GetVariableId(
          inst->GetSingleWordInOperand(kCopyMemorySourceAddrInIdx));
    default:
      break;
  }

  switch (inst->GetCommonDebugOpcode()) {
    case CommonDebugInfoDebugDeclare:
      return inst->GetSingleWordOperand(kDebugDeclareOperandVariableIndex);
    case CommonDebugInfoDebugValue: {
      analysis::DebugInfoManager* debug_info_mgr =
          context()->get_debug_info_mgr();
      return debug_info_mgr->GetVariableIdOfDebugValueUsedForDeclare(inst);
    }
    default:
      break;
  }
  return 0;
}

std::vector<uint32_t> AggressiveDCEPass::GetLoadedVariablesFromFunctionCall(
    const Instruction* inst) {
  assert(inst->opcode() == spv::Op::OpFunctionCall);
  std::vector<uint32_t> live_variables;
  inst->ForEachInId([this, &live_variables](const uint32_t* operand_id) {
    if (!IsPtr(*operand_id)) return;
    uint32_t var_id = GetVariableId(*operand_id);
    live_variables.push_back(var_id);
  });
  return live_variables;
}

uint32_t AggressiveDCEPass::GetVariableId(uint32_t ptr_id) {
  assert(IsPtr(ptr_id) &&
         "Cannot get the variable when input is not a pointer.");
  uint32_t varId = 0;
  (void)GetPtr(ptr_id, &varId);
  return varId;
}

void AggressiveDCEPass::MarkBlockAsLive(Instruction* inst) {
  BasicBlock* basic_block = context()->get_instr_block(inst);
  if (basic_block == nullptr) {
    return;
  }

  // If we intend to keep this instruction, we need the block label and
  // block terminator to have a valid block for the instruction.
  AddToWorklist(basic_block->GetLabelInst());

  // We need to mark the successors blocks that follow as live.  If this is
  // header of the merge construct, the construct may be folded, but we will
  // definitely need the merge label.  If it is not a construct, the terminator
  // must be live, and the successor blocks will be marked as live when
  // processing the terminator.
  uint32_t merge_id = basic_block->MergeBlockIdIfAny();
  if (merge_id == 0) {
    AddToWorklist(basic_block->terminator());
  } else {
    AddToWorklist(context()->get_def_use_mgr()->GetDef(merge_id));
  }

  // Mark the structured control flow constructs that contains this block as
  // live.  If |inst| is an instruction in the loop header, then it is part of
  // the loop, so the loop construct must be live.  We exclude the label because
  // it does not matter how many times it is executed.  This could be extended
  // to more instructions, but we will need it for now.
  if (inst->opcode() != spv::Op::OpLabel)
    MarkLoopConstructAsLiveIfLoopHeader(basic_block);

  Instruction* next_branch_inst = GetBranchForNextHeader(basic_block);
  if (next_branch_inst != nullptr) {
    AddToWorklist(next_branch_inst);
    Instruction* mergeInst = GetMergeInstruction(next_branch_inst);
    AddToWorklist(mergeInst);
  }

  if (inst->opcode() == spv::Op::OpLoopMerge ||
      inst->opcode() == spv::Op::OpSelectionMerge) {
    AddBreaksAndContinuesToWorklist(inst);
  }
}
void AggressiveDCEPass::MarkLoopConstructAsLiveIfLoopHeader(
    BasicBlock* basic_block) {
  // If this is the header for a loop, then loop structure needs to keep as well
  // because the loop header is also part of the loop.
  Instruction* merge_inst = basic_block->GetLoopMergeInst();
  if (merge_inst != nullptr) {
    AddToWorklist(basic_block->terminator());
    AddToWorklist(merge_inst);
  }
}

void AggressiveDCEPass::AddOperandsToWorkList(const Instruction* inst) {
  inst->ForEachInId([this](const uint32_t* iid) {
    Instruction* inInst = get_def_use_mgr()->GetDef(*iid);
    AddToWorklist(inInst);
  });
  if (inst->type_id() != 0) {
    AddToWorklist(get_def_use_mgr()->GetDef(inst->type_id()));
  }
}

void AggressiveDCEPass::InitializeWorkList(
    Function* func, std::list<BasicBlock*>& structured_order) {
  AddToWorklist(&func->DefInst());
  MarkFunctionParameterAsLive(func);
  MarkFirstBlockAsLive(func);

  // Add instructions with external side effects to the worklist. Also add
  // branches that are not attached to a structured construct.
  // TODO(s-perron): The handling of branch seems to be adhoc.  This needs to be
  // cleaned up.
  for (auto& bi : structured_order) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      spv::Op op = ii->opcode();
      if (ii->IsBranch()) {
        continue;
      }
      switch (op) {
        case spv::Op::OpStore: {
          uint32_t var_id = 0;
          (void)GetPtr(&*ii, &var_id);
          if (!IsLocalVar(var_id, func)) AddToWorklist(&*ii);
        } break;
        case spv::Op::OpCopyMemory:
        case spv::Op::OpCopyMemorySized: {
          uint32_t var_id = 0;
          uint32_t target_addr_id =
              ii->GetSingleWordInOperand(kCopyMemoryTargetAddrInIdx);
          (void)GetPtr(target_addr_id, &var_id);
          if (!IsLocalVar(var_id, func)) AddToWorklist(&*ii);
        } break;
        case spv::Op::OpLoopMerge:
        case spv::Op::OpSelectionMerge:
        case spv::Op::OpUnreachable:
          break;
        default: {
          // Function calls, atomics, function params, function returns, etc.
          if (!ii->IsOpcodeSafeToDelete()) {
            AddToWorklist(&*ii);
          }
        } break;
      }
    }
  }
}

void AggressiveDCEPass::InitializeModuleScopeLiveInstructions() {
  // Keep all execution modes.
  for (auto& exec : get_module()->execution_modes()) {
    AddToWorklist(&exec);
  }
  // Keep all entry points.
  for (auto& entry : get_module()->entry_points()) {
    if (!preserve_interface_) {
      live_insts_.Set(entry.unique_id());
      // The actual function is live always.
      AddToWorklist(
          get_def_use_mgr()->GetDef(entry.GetSingleWordInOperand(1u)));
      for (uint32_t i = 3; i < entry.NumInOperands(); ++i) {
        auto* var = get_def_use_mgr()->GetDef(entry.GetSingleWordInOperand(i));
        auto storage_class = var->GetSingleWordInOperand(0u);
        // Vulkan support outputs without an associated input, but not inputs
        // without an associated output. Don't remove outputs unless explicitly
        // allowed.
        if (!remove_outputs_ &&
            spv::StorageClass(storage_class) == spv::StorageClass::Output) {
          AddToWorklist(var);
        }
      }
    } else {
      AddToWorklist(&entry);
    }
  }
  for (auto& anno : get_module()->annotations()) {
    if (anno.opcode() == spv::Op::OpDecorate) {
      // Keep workgroup size.
      if (spv::Decoration(anno.GetSingleWordInOperand(1u)) ==
              spv::Decoration::BuiltIn &&
          spv::BuiltIn(anno.GetSingleWordInOperand(2u)) ==
              spv::BuiltIn::WorkgroupSize) {
        AddToWorklist(&anno);
      }

      if (context()->preserve_bindings()) {
        // Keep all bindings.
        if ((spv::Decoration(anno.GetSingleWordInOperand(1u)) ==
             spv::Decoration::DescriptorSet) ||
            (spv::Decoration(anno.GetSingleWordInOperand(1u)) ==
             spv::Decoration::Binding)) {
          AddToWorklist(&anno);
        }
      }

      if (context()->preserve_spec_constants()) {
        // Keep all specialization constant instructions
        if (spv::Decoration(anno.GetSingleWordInOperand(1u)) ==
            spv::Decoration::SpecId) {
          AddToWorklist(&anno);
        }
      }
    }
  }

  // For each DebugInfo GlobalVariable keep all operands except the Variable.
  // Later, if the variable is killed with KillInst(), we will set the operand
  // to DebugInfoNone. Create and save DebugInfoNone now for this possible
  // later use. This is slightly unoptimal, but it avoids generating it during
  // instruction killing when the module is not consistent.
  bool debug_global_seen = false;
  for (auto& dbg : get_module()->ext_inst_debuginfo()) {
    if (dbg.GetCommonDebugOpcode() != CommonDebugInfoDebugGlobalVariable)
      continue;
    debug_global_seen = true;
    dbg.ForEachInId([this](const uint32_t* iid) {
      Instruction* in_inst = get_def_use_mgr()->GetDef(*iid);
      if (in_inst->opcode() == spv::Op::OpVariable) return;
      AddToWorklist(in_inst);
    });
  }
  if (debug_global_seen) {
    auto dbg_none = context()->get_debug_info_mgr()->GetDebugInfoNone();
    AddToWorklist(dbg_none);
  }

  // Add top level DebugInfo to worklist
  for (auto& dbg : get_module()->ext_inst_debuginfo()) {
    auto op = dbg.GetShader100DebugOpcode();
    if (op == NonSemanticShaderDebugInfo100DebugCompilationUnit ||
        op == NonSemanticShaderDebugInfo100DebugEntryPoint ||
        op == NonSemanticShaderDebugInfo100DebugSourceContinued) {
      AddToWorklist(&dbg);
    }
  }
}

Pass::Status AggressiveDCEPass::ProcessImpl() {
  // Current functionality assumes shader capability
  // TODO(greg-lunarg): Handle additional capabilities
  if (!context()->get_feature_mgr()->HasCapability(spv::Capability::Shader))
    return Status::SuccessWithoutChange;

  // Current functionality assumes relaxed logical addressing (see
  // instruction.h)
  // TODO(greg-lunarg): Handle non-logical addressing
  if (context()->get_feature_mgr()->HasCapability(spv::Capability::Addresses))
    return Status::SuccessWithoutChange;

  // The variable pointer extension is no longer needed to use the capability,
  // so we have to look for the capability.
  if (context()->get_feature_mgr()->HasCapability(
          spv::Capability::VariablePointersStorageBuffer))
    return Status::SuccessWithoutChange;

  // If any extensions in the module are not explicitly supported,
  // return unmodified.
  if (!AllExtensionsSupported()) return Status::SuccessWithoutChange;

  // Eliminate Dead functions.
  bool modified = EliminateDeadFunctions();

  InitializeModuleScopeLiveInstructions();

  // Run |AggressiveDCE| on the remaining functions.  The order does not matter,
  // since |AggressiveDCE| is intra-procedural.  This can mean that function
  // will become dead if all function call to them are removed.  These dead
  // function will still be in the module after this pass.  We expect this to be
  // rare.
  for (Function& fp : *context()->module()) {
    modified |= AggressiveDCE(&fp);
  }

  // If the decoration manager is kept live then the context will try to keep it
  // up to date.  ADCE deals with group decorations by changing the operands in
  // |OpGroupDecorate| instruction directly without informing the decoration
  // manager.  This can put it in an invalid state which will cause an error
  // when the context tries to update it.  To avoid this problem invalidate
  // the decoration manager upfront.
  //
  // We kill it at now because it is used when processing the entry point
  // functions.
  context()->InvalidateAnalyses(IRContext::Analysis::kAnalysisDecorations);

  // Process module-level instructions. Now that all live instructions have
  // been marked, it is safe to remove dead global values.
  modified |= ProcessGlobalValues();

  assert((to_kill_.empty() || modified) &&
         "A dead instruction was identified, but no change recorded.");

  // Kill all dead instructions.
  for (auto inst : to_kill_) {
    context()->KillInst(inst);
  }

  // Cleanup all CFG including all unreachable blocks.
  for (Function& fp : *context()->module()) {
    modified |= CFGCleanup(&fp);
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool AggressiveDCEPass::EliminateDeadFunctions() {
  // Identify live functions first. Those that are not live
  // are dead.
  std::unordered_set<const Function*> live_function_set;
  ProcessFunction mark_live = [&live_function_set](Function* fp) {
    live_function_set.insert(fp);
    return false;
  };
  context()->ProcessReachableCallTree(mark_live);

  bool modified = false;
  for (auto funcIter = get_module()->begin();
       funcIter != get_module()->end();) {
    if (live_function_set.count(&*funcIter) == 0) {
      modified = true;
      funcIter =
          eliminatedeadfunctionsutil::EliminateFunction(context(), &funcIter);
    } else {
      ++funcIter;
    }
  }

  return modified;
}

bool AggressiveDCEPass::ProcessGlobalValues() {
  // Remove debug and annotation statements referencing dead instructions.
  // This must be done before killing the instructions, otherwise there are
  // dead objects in the def/use database.
  bool modified = false;
  Instruction* instruction = &*get_module()->debug2_begin();
  while (instruction) {
    if (instruction->opcode() != spv::Op::OpName) {
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
      case spv::Op::OpDecorate:
      case spv::Op::OpMemberDecorate:
      case spv::Op::OpDecorateStringGOOGLE:
      case spv::Op::OpMemberDecorateStringGOOGLE:
        if (IsTargetDead(annotation)) {
          context()->KillInst(annotation);
          modified = true;
        }
        break;
      case spv::Op::OpDecorateId:
        if (IsTargetDead(annotation)) {
          context()->KillInst(annotation);
          modified = true;
        } else {
          if (spv::Decoration(annotation->GetSingleWordInOperand(1)) ==
              spv::Decoration::HlslCounterBufferGOOGLE) {
            // HlslCounterBuffer will reference an id other than the target.
            // If that id is dead, then the decoration can be removed as well.
            uint32_t counter_buffer_id = annotation->GetSingleWordInOperand(2);
            Instruction* counter_buffer_inst =
                get_def_use_mgr()->GetDef(counter_buffer_id);
            if (!IsLive(counter_buffer_inst)) {
              context()->KillInst(annotation);
              modified = true;
            }
          }
        }
        break;
      case spv::Op::OpGroupDecorate: {
        // Go through the targets of this group decorate. Remove each dead
        // target. If all targets are dead, remove this decoration.
        bool dead = true;
        bool removed_operand = false;
        for (uint32_t i = 1; i < annotation->NumOperands();) {
          Instruction* opInst =
              get_def_use_mgr()->GetDef(annotation->GetSingleWordOperand(i));
          if (!IsLive(opInst)) {
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
      case spv::Op::OpGroupMemberDecorate: {
        // Go through the targets of this group member decorate. Remove each
        // dead target (and member index). If all targets are dead, remove this
        // decoration.
        bool dead = true;
        bool removed_operand = false;
        for (uint32_t i = 1; i < annotation->NumOperands();) {
          Instruction* opInst =
              get_def_use_mgr()->GetDef(annotation->GetSingleWordOperand(i));
          if (!IsLive(opInst)) {
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
      case spv::Op::OpDecorationGroup:
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

  for (auto& dbg : get_module()->ext_inst_debuginfo()) {
    if (IsLive(&dbg)) continue;
    // Save GlobalVariable if its variable is live, otherwise null out variable
    // index
    if (dbg.GetCommonDebugOpcode() == CommonDebugInfoDebugGlobalVariable) {
      auto var_id = dbg.GetSingleWordOperand(kGlobalVariableVariableIndex);
      Instruction* var_inst = get_def_use_mgr()->GetDef(var_id);
      if (IsLive(var_inst)) continue;
      context()->ForgetUses(&dbg);
      dbg.SetOperand(
          kGlobalVariableVariableIndex,
          {context()->get_debug_info_mgr()->GetDebugInfoNone()->result_id()});
      context()->AnalyzeUses(&dbg);
      continue;
    }
    to_kill_.push_back(&dbg);
    modified = true;
  }

  // Since ADCE is disabled for non-shaders, we don't check for export linkage
  // attributes here.
  for (auto& val : get_module()->types_values()) {
    if (!IsLive(&val)) {
      // Save forwarded pointer if pointer is live since closure does not mark
      // this live as it does not have a result id. This is a little too
      // conservative since it is not known if the structure type that needed
      // it is still live. TODO(greg-lunarg): Only save if needed.
      if (val.opcode() == spv::Op::OpTypeForwardPointer) {
        uint32_t ptr_ty_id = val.GetSingleWordInOperand(0);
        Instruction* ptr_ty_inst = get_def_use_mgr()->GetDef(ptr_ty_id);
        if (IsLive(ptr_ty_inst)) continue;
      }
      to_kill_.push_back(&val);
      modified = true;
    }
  }

  if (!preserve_interface_) {
    // Remove the dead interface variables from the entry point interface list.
    for (auto& entry : get_module()->entry_points()) {
      std::vector<Operand> new_operands;
      for (uint32_t i = 0; i < entry.NumInOperands(); ++i) {
        if (i < 3) {
          // Execution model, function id and name are always valid.
          new_operands.push_back(entry.GetInOperand(i));
        } else {
          auto* var =
              get_def_use_mgr()->GetDef(entry.GetSingleWordInOperand(i));
          if (IsLive(var)) {
            new_operands.push_back(entry.GetInOperand(i));
          }
        }
      }
      if (new_operands.size() != entry.NumInOperands()) {
        entry.SetInOperands(std::move(new_operands));
        get_def_use_mgr()->UpdateDefUse(&entry);
      }
    }
  }

  return modified;
}

Pass::Status AggressiveDCEPass::Process() {
  // Initialize extensions allowlist
  InitExtensions();
  return ProcessImpl();
}

void AggressiveDCEPass::InitExtensions() {
  extensions_allowlist_.clear();
  extensions_allowlist_.insert({
      "SPV_AMD_shader_explicit_vertex_parameter",
      "SPV_AMD_shader_trinary_minmax",
      "SPV_AMD_gcn_shader",
      "SPV_KHR_shader_ballot",
      "SPV_AMD_shader_ballot",
      "SPV_AMD_gpu_shader_half_float",
      "SPV_KHR_shader_draw_parameters",
      "SPV_KHR_subgroup_vote",
      "SPV_KHR_8bit_storage",
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
      "SPV_GOOGLE_user_type",
      "SPV_NV_shader_subgroup_partitioned",
      "SPV_EXT_demote_to_helper_invocation",
      "SPV_EXT_descriptor_indexing",
      "SPV_NV_fragment_shader_barycentric",
      "SPV_NV_compute_shader_derivatives",
      "SPV_NV_shader_image_footprint",
      "SPV_NV_shading_rate",
      "SPV_NV_mesh_shader",
      "SPV_NV_ray_tracing",
      "SPV_KHR_ray_tracing",
      "SPV_KHR_ray_query",
      "SPV_EXT_fragment_invocation_density",
      "SPV_EXT_physical_storage_buffer",
      "SPV_KHR_terminate_invocation",
      "SPV_KHR_shader_clock",
      "SPV_KHR_vulkan_memory_model",
      "SPV_KHR_subgroup_uniform_control_flow",
      "SPV_KHR_integer_dot_product",
      "SPV_EXT_shader_image_int64",
      "SPV_KHR_non_semantic_info",
      "SPV_KHR_uniform_group_instructions",
      "SPV_KHR_fragment_shader_barycentric",
  });
}

Instruction* AggressiveDCEPass::GetHeaderBranch(BasicBlock* blk) {
  if (blk == nullptr) {
    return nullptr;
  }
  BasicBlock* header_block = GetHeaderBlock(blk);
  if (header_block == nullptr) {
    return nullptr;
  }
  return header_block->terminator();
}

BasicBlock* AggressiveDCEPass::GetHeaderBlock(BasicBlock* blk) const {
  if (blk == nullptr) {
    return nullptr;
  }

  BasicBlock* header_block = nullptr;
  if (blk->IsLoopHeader()) {
    header_block = blk;
  } else {
    uint32_t header =
        context()->GetStructuredCFGAnalysis()->ContainingConstruct(blk->id());
    header_block = context()->get_instr_block(header);
  }
  return header_block;
}

Instruction* AggressiveDCEPass::GetMergeInstruction(Instruction* inst) {
  BasicBlock* bb = context()->get_instr_block(inst);
  if (bb == nullptr) {
    return nullptr;
  }
  return bb->GetMergeInst();
}

Instruction* AggressiveDCEPass::GetBranchForNextHeader(BasicBlock* blk) {
  if (blk == nullptr) {
    return nullptr;
  }

  if (blk->IsLoopHeader()) {
    uint32_t header =
        context()->GetStructuredCFGAnalysis()->ContainingConstruct(blk->id());
    blk = context()->get_instr_block(header);
  }
  return GetHeaderBranch(blk);
}

void AggressiveDCEPass::MarkFunctionParameterAsLive(const Function* func) {
  func->ForEachParam(
      [this](const Instruction* param) {
        AddToWorklist(const_cast<Instruction*>(param));
      },
      false);
}

bool AggressiveDCEPass::BlockIsInConstruct(BasicBlock* header_block,
                                           BasicBlock* bb) {
  if (bb == nullptr || header_block == nullptr) {
    return false;
  }

  uint32_t current_header = bb->id();
  while (current_header != 0) {
    if (current_header == header_block->id()) return true;
    current_header = context()->GetStructuredCFGAnalysis()->ContainingConstruct(
        current_header);
  }
  return false;
}

bool AggressiveDCEPass::IsEntryPointWithNoCalls(Function* func) {
  auto cached_result = entry_point_with_no_calls_cache_.find(func->result_id());
  if (cached_result != entry_point_with_no_calls_cache_.end()) {
    return cached_result->second;
  }
  bool result = IsEntryPoint(func) && !HasCall(func);
  entry_point_with_no_calls_cache_[func->result_id()] = result;
  return result;
}

bool AggressiveDCEPass::IsEntryPoint(Function* func) {
  for (const Instruction& entry_point : get_module()->entry_points()) {
    uint32_t entry_point_id =
        entry_point.GetSingleWordInOperand(kEntryPointFunctionIdInIdx);
    if (entry_point_id == func->result_id()) {
      return true;
    }
  }
  return false;
}

bool AggressiveDCEPass::HasCall(Function* func) {
  return !func->WhileEachInst([](Instruction* inst) {
    return inst->opcode() != spv::Op::OpFunctionCall;
  });
}

void AggressiveDCEPass::MarkFirstBlockAsLive(Function* func) {
  BasicBlock* first_block = &*func->begin();
  MarkBlockAsLive(first_block->GetLabelInst());
}

void AggressiveDCEPass::AddUnreachable(BasicBlock*& block) {
  InstructionBuilder builder(
      context(), block,
      IRContext::kAnalysisInstrToBlockMapping | IRContext::kAnalysisDefUse);
  builder.AddUnreachable();
}

}  // namespace opt
}  // namespace spvtools
