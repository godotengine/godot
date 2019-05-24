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

#include "source/opt/inline_pass.h"

#include <unordered_set>
#include <utility>

#include "source/cfa.h"
#include "source/util/make_unique.h"

// Indices of operands in SPIR-V instructions

static const int kSpvFunctionCallFunctionId = 2;
static const int kSpvFunctionCallArgumentId = 3;
static const int kSpvReturnValueId = 0;
static const int kSpvLoopMergeContinueTargetIdInIdx = 1;

namespace spvtools {
namespace opt {

uint32_t InlinePass::AddPointerToType(uint32_t type_id,
                                      SpvStorageClass storage_class) {
  uint32_t resultId = context()->TakeNextId();
  if (resultId == 0) {
    return resultId;
  }

  std::unique_ptr<Instruction> type_inst(
      new Instruction(context(), SpvOpTypePointer, 0, resultId,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
                        {uint32_t(storage_class)}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {type_id}}}));
  context()->AddType(std::move(type_inst));
  analysis::Type* pointeeTy;
  std::unique_ptr<analysis::Pointer> pointerTy;
  std::tie(pointeeTy, pointerTy) =
      context()->get_type_mgr()->GetTypeAndPointerType(type_id,
                                                       SpvStorageClassFunction);
  context()->get_type_mgr()->RegisterType(resultId, *pointerTy);
  return resultId;
}

void InlinePass::AddBranch(uint32_t label_id,
                           std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newBranch(
      new Instruction(context(), SpvOpBranch, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {label_id}}}));
  (*block_ptr)->AddInstruction(std::move(newBranch));
}

void InlinePass::AddBranchCond(uint32_t cond_id, uint32_t true_id,
                               uint32_t false_id,
                               std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newBranch(
      new Instruction(context(), SpvOpBranchConditional, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {cond_id}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {true_id}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {false_id}}}));
  (*block_ptr)->AddInstruction(std::move(newBranch));
}

void InlinePass::AddLoopMerge(uint32_t merge_id, uint32_t continue_id,
                              std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newLoopMerge(new Instruction(
      context(), SpvOpLoopMerge, 0, 0,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {merge_id}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {continue_id}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LOOP_CONTROL, {0}}}));
  (*block_ptr)->AddInstruction(std::move(newLoopMerge));
}

void InlinePass::AddStore(uint32_t ptr_id, uint32_t val_id,
                          std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newStore(
      new Instruction(context(), SpvOpStore, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {ptr_id}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {val_id}}}));
  (*block_ptr)->AddInstruction(std::move(newStore));
}

void InlinePass::AddLoad(uint32_t type_id, uint32_t resultId, uint32_t ptr_id,
                         std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newLoad(
      new Instruction(context(), SpvOpLoad, type_id, resultId,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {ptr_id}}}));
  (*block_ptr)->AddInstruction(std::move(newLoad));
}

std::unique_ptr<Instruction> InlinePass::NewLabel(uint32_t label_id) {
  std::unique_ptr<Instruction> newLabel(
      new Instruction(context(), SpvOpLabel, 0, label_id, {}));
  return newLabel;
}

uint32_t InlinePass::GetFalseId() {
  if (false_id_ != 0) return false_id_;
  false_id_ = get_module()->GetGlobalValue(SpvOpConstantFalse);
  if (false_id_ != 0) return false_id_;
  uint32_t boolId = get_module()->GetGlobalValue(SpvOpTypeBool);
  if (boolId == 0) {
    boolId = context()->TakeNextId();
    if (boolId == 0) {
      return 0;
    }
    get_module()->AddGlobalValue(SpvOpTypeBool, boolId, 0);
  }
  false_id_ = context()->TakeNextId();
  if (false_id_ == 0) {
    return 0;
  }
  get_module()->AddGlobalValue(SpvOpConstantFalse, false_id_, boolId);
  return false_id_;
}

void InlinePass::MapParams(
    Function* calleeFn, BasicBlock::iterator call_inst_itr,
    std::unordered_map<uint32_t, uint32_t>* callee2caller) {
  int param_idx = 0;
  calleeFn->ForEachParam(
      [&call_inst_itr, &param_idx, &callee2caller](const Instruction* cpi) {
        const uint32_t pid = cpi->result_id();
        (*callee2caller)[pid] = call_inst_itr->GetSingleWordOperand(
            kSpvFunctionCallArgumentId + param_idx);
        ++param_idx;
      });
}

bool InlinePass::CloneAndMapLocals(
    Function* calleeFn, std::vector<std::unique_ptr<Instruction>>* new_vars,
    std::unordered_map<uint32_t, uint32_t>* callee2caller) {
  auto callee_block_itr = calleeFn->begin();
  auto callee_var_itr = callee_block_itr->begin();
  while (callee_var_itr->opcode() == SpvOp::SpvOpVariable) {
    std::unique_ptr<Instruction> var_inst(callee_var_itr->Clone(context()));
    uint32_t newId = context()->TakeNextId();
    if (newId == 0) {
      return false;
    }
    get_decoration_mgr()->CloneDecorations(callee_var_itr->result_id(), newId);
    var_inst->SetResultId(newId);
    (*callee2caller)[callee_var_itr->result_id()] = newId;
    new_vars->push_back(std::move(var_inst));
    ++callee_var_itr;
  }
  return true;
}

uint32_t InlinePass::CreateReturnVar(
    Function* calleeFn, std::vector<std::unique_ptr<Instruction>>* new_vars) {
  uint32_t returnVarId = 0;
  const uint32_t calleeTypeId = calleeFn->type_id();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  assert(type_mgr->GetType(calleeTypeId)->AsVoid() == nullptr &&
         "Cannot create a return variable of type void.");
  // Find or create ptr to callee return type.
  uint32_t returnVarTypeId =
      type_mgr->FindPointerToType(calleeTypeId, SpvStorageClassFunction);

  if (returnVarTypeId == 0) {
    returnVarTypeId = AddPointerToType(calleeTypeId, SpvStorageClassFunction);
    if (returnVarTypeId == 0) {
      return 0;
    }
  }

  // Add return var to new function scope variables.
  returnVarId = context()->TakeNextId();
  if (returnVarId == 0) {
    return 0;
  }

  std::unique_ptr<Instruction> var_inst(
      new Instruction(context(), SpvOpVariable, returnVarTypeId, returnVarId,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
                        {SpvStorageClassFunction}}}));
  new_vars->push_back(std::move(var_inst));
  get_decoration_mgr()->CloneDecorations(calleeFn->result_id(), returnVarId);
  return returnVarId;
}

bool InlinePass::IsSameBlockOp(const Instruction* inst) const {
  return inst->opcode() == SpvOpSampledImage || inst->opcode() == SpvOpImage;
}

bool InlinePass::CloneSameBlockOps(
    std::unique_ptr<Instruction>* inst,
    std::unordered_map<uint32_t, uint32_t>* postCallSB,
    std::unordered_map<uint32_t, Instruction*>* preCallSB,
    std::unique_ptr<BasicBlock>* block_ptr) {
  return (*inst)->WhileEachInId([&postCallSB, &preCallSB, &block_ptr,
                                 this](uint32_t* iid) {
    const auto mapItr = (*postCallSB).find(*iid);
    if (mapItr == (*postCallSB).end()) {
      const auto mapItr2 = (*preCallSB).find(*iid);
      if (mapItr2 != (*preCallSB).end()) {
        // Clone pre-call same-block ops, map result id.
        const Instruction* inInst = mapItr2->second;
        std::unique_ptr<Instruction> sb_inst(inInst->Clone(context()));
        if (!CloneSameBlockOps(&sb_inst, postCallSB, preCallSB, block_ptr)) {
          return false;
        }

        const uint32_t rid = sb_inst->result_id();
        const uint32_t nid = context()->TakeNextId();
        if (nid == 0) {
          return false;
        }
        get_decoration_mgr()->CloneDecorations(rid, nid);
        sb_inst->SetResultId(nid);
        (*postCallSB)[rid] = nid;
        *iid = nid;
        (*block_ptr)->AddInstruction(std::move(sb_inst));
      }
    } else {
      // Reset same-block op operand.
      *iid = mapItr->second;
    }
    return true;
  });
}

bool InlinePass::GenInlineCode(
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::vector<std::unique_ptr<Instruction>>* new_vars,
    BasicBlock::iterator call_inst_itr,
    UptrVectorIterator<BasicBlock> call_block_itr) {
  // Map from all ids in the callee to their equivalent id in the caller
  // as callee instructions are copied into caller.
  std::unordered_map<uint32_t, uint32_t> callee2caller;
  // Pre-call same-block insts
  std::unordered_map<uint32_t, Instruction*> preCallSB;
  // Post-call same-block op ids
  std::unordered_map<uint32_t, uint32_t> postCallSB;

  // Invalidate the def-use chains.  They are not kept up to date while
  // inlining.  However, certain calls try to keep them up-to-date if they are
  // valid.  These operations can fail.
  context()->InvalidateAnalyses(IRContext::kAnalysisDefUse);

  Function* calleeFn = id2function_[call_inst_itr->GetSingleWordOperand(
      kSpvFunctionCallFunctionId)];

  // Check for multiple returns in the callee.
  auto fi = early_return_funcs_.find(calleeFn->result_id());
  const bool earlyReturn = fi != early_return_funcs_.end();

  // Map parameters to actual arguments.
  MapParams(calleeFn, call_inst_itr, &callee2caller);

  // Define caller local variables for all callee variables and create map to
  // them.
  if (!CloneAndMapLocals(calleeFn, new_vars, &callee2caller)) {
    return false;
  }

  // Create return var if needed.
  const uint32_t calleeTypeId = calleeFn->type_id();
  uint32_t returnVarId = 0;
  analysis::Type* calleeType = context()->get_type_mgr()->GetType(calleeTypeId);
  if (calleeType->AsVoid() == nullptr) {
    returnVarId = CreateReturnVar(calleeFn, new_vars);
    if (returnVarId == 0) {
      return false;
    }
  }

  // Create set of callee result ids. Used to detect forward references
  std::unordered_set<uint32_t> callee_result_ids;
  calleeFn->ForEachInst([&callee_result_ids](const Instruction* cpi) {
    const uint32_t rid = cpi->result_id();
    if (rid != 0) callee_result_ids.insert(rid);
  });

  // If the caller is in a single-block loop, and the callee has multiple
  // blocks, then the normal inlining logic will place the OpLoopMerge in
  // the last of several blocks in the loop.  Instead, it should be placed
  // at the end of the first block.  First determine if the caller is in a
  // single block loop.  We'll wait to move the OpLoopMerge until the end
  // of the regular inlining logic, and only if necessary.
  bool caller_is_single_block_loop = false;
  bool caller_is_loop_header = false;
  if (auto* loop_merge = call_block_itr->GetLoopMergeInst()) {
    caller_is_loop_header = true;
    caller_is_single_block_loop =
        call_block_itr->id() ==
        loop_merge->GetSingleWordInOperand(kSpvLoopMergeContinueTargetIdInIdx);
  }

  bool callee_begins_with_structured_header =
      (*(calleeFn->begin())).GetMergeInst() != nullptr;

  // Clone and map callee code. Copy caller block code to beginning of
  // first block and end of last block.
  bool prevInstWasReturn = false;
  uint32_t singleTripLoopHeaderId = 0;
  uint32_t singleTripLoopContinueId = 0;
  uint32_t returnLabelId = 0;
  bool multiBlocks = false;
  // new_blk_ptr is a new basic block in the caller.  New instructions are
  // written to it.  It is created when we encounter the OpLabel
  // of the first callee block.  It is appended to new_blocks only when
  // it is complete.
  std::unique_ptr<BasicBlock> new_blk_ptr;
  bool successful = calleeFn->WhileEachInst(
      [&new_blocks, &callee2caller, &call_block_itr, &call_inst_itr,
       &new_blk_ptr, &prevInstWasReturn, &returnLabelId, &returnVarId,
       caller_is_loop_header, callee_begins_with_structured_header,
       &calleeTypeId, &multiBlocks, &postCallSB, &preCallSB, earlyReturn,
       &singleTripLoopHeaderId, &singleTripLoopContinueId, &callee_result_ids,
       this](const Instruction* cpi) {
        switch (cpi->opcode()) {
          case SpvOpFunction:
          case SpvOpFunctionParameter:
            // Already processed
            break;
          case SpvOpVariable:
            if (cpi->NumInOperands() == 2) {
              assert(callee2caller.count(cpi->result_id()) &&
                     "Expected the variable to have already been mapped.");
              uint32_t new_var_id = callee2caller.at(cpi->result_id());

              // The initializer must be a constant or global value.  No mapped
              // should be used.
              uint32_t val_id = cpi->GetSingleWordInOperand(1);
              AddStore(new_var_id, val_id, &new_blk_ptr);
            }
            break;
          case SpvOpUnreachable:
          case SpvOpKill: {
            // Generate a return label so that we split the block with the
            // function call. Copy the terminator into the new block.
            if (returnLabelId == 0) {
              returnLabelId = context()->TakeNextId();
              if (returnLabelId == 0) {
                return false;
              }
            }
            std::unique_ptr<Instruction> terminator(
                new Instruction(context(), cpi->opcode(), 0, 0, {}));
            new_blk_ptr->AddInstruction(std::move(terminator));
            break;
          }
          case SpvOpLabel: {
            // If previous instruction was early return, insert branch
            // instruction to return block.
            if (prevInstWasReturn) {
              if (returnLabelId == 0) {
                returnLabelId = context()->TakeNextId();
                if (returnLabelId == 0) {
                  return false;
                }
              }
              AddBranch(returnLabelId, &new_blk_ptr);
              prevInstWasReturn = false;
            }
            // Finish current block (if it exists) and get label for next block.
            uint32_t labelId;
            bool firstBlock = false;
            if (new_blk_ptr != nullptr) {
              new_blocks->push_back(std::move(new_blk_ptr));
              // If result id is already mapped, use it, otherwise get a new
              // one.
              const uint32_t rid = cpi->result_id();
              const auto mapItr = callee2caller.find(rid);
              labelId = (mapItr != callee2caller.end())
                            ? mapItr->second
                            : context()->TakeNextId();
              if (labelId == 0) {
                return false;
              }
            } else {
              // First block needs to use label of original block
              // but map callee label in case of phi reference.
              labelId = call_block_itr->id();
              callee2caller[cpi->result_id()] = labelId;
              firstBlock = true;
            }
            // Create first/next block.
            new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(labelId));
            if (firstBlock) {
              // Copy contents of original caller block up to call instruction.
              for (auto cii = call_block_itr->begin(); cii != call_inst_itr;
                   cii = call_block_itr->begin()) {
                Instruction* inst = &*cii;
                inst->RemoveFromList();
                std::unique_ptr<Instruction> cp_inst(inst);
                // Remember same-block ops for possible regeneration.
                if (IsSameBlockOp(&*cp_inst)) {
                  auto* sb_inst_ptr = cp_inst.get();
                  preCallSB[cp_inst->result_id()] = sb_inst_ptr;
                }
                new_blk_ptr->AddInstruction(std::move(cp_inst));
              }
              if (caller_is_loop_header &&
                  callee_begins_with_structured_header) {
                // We can't place both the caller's merge instruction and
                // another merge instruction in the same block.  So split the
                // calling block. Insert an unconditional branch to a new guard
                // block.  Later, once we know the ID of the last block,  we
                // will move the caller's OpLoopMerge from the last generated
                // block into the first block. We also wait to avoid
                // invalidating various iterators.
                const auto guard_block_id = context()->TakeNextId();
                if (guard_block_id == 0) {
                  return false;
                }
                AddBranch(guard_block_id, &new_blk_ptr);
                new_blocks->push_back(std::move(new_blk_ptr));
                // Start the next block.
                new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(guard_block_id));
                // Reset the mapping of the callee's entry block to point to
                // the guard block.  Do this so we can fix up phis later on to
                // satisfy dominance.
                callee2caller[cpi->result_id()] = guard_block_id;
              }
              // If callee has early return, insert a header block for
              // single-trip loop that will encompass callee code.  Start
              // postheader block.
              //
              // Note: Consider the following combination:
              //  - the caller is a single block loop
              //  - the callee does not begin with a structure header
              //  - the callee has multiple returns.
              // We still need to split the caller block and insert a guard
              // block. But we only need to do it once. We haven't done it yet,
              // but the single-trip loop header will serve the same purpose.
              if (earlyReturn) {
                singleTripLoopHeaderId = context()->TakeNextId();
                if (singleTripLoopHeaderId == 0) {
                  return false;
                }
                AddBranch(singleTripLoopHeaderId, &new_blk_ptr);
                new_blocks->push_back(std::move(new_blk_ptr));
                new_blk_ptr =
                    MakeUnique<BasicBlock>(NewLabel(singleTripLoopHeaderId));
                returnLabelId = context()->TakeNextId();
                singleTripLoopContinueId = context()->TakeNextId();
                if (returnLabelId == 0 || singleTripLoopContinueId == 0) {
                  return false;
                }
                AddLoopMerge(returnLabelId, singleTripLoopContinueId,
                             &new_blk_ptr);
                uint32_t postHeaderId = context()->TakeNextId();
                if (postHeaderId == 0) {
                  return false;
                }
                AddBranch(postHeaderId, &new_blk_ptr);
                new_blocks->push_back(std::move(new_blk_ptr));
                new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(postHeaderId));
                multiBlocks = true;
                // Reset the mapping of the callee's entry block to point to
                // the post-header block.  Do this so we can fix up phis later
                // on to satisfy dominance.
                callee2caller[cpi->result_id()] = postHeaderId;
              }
            } else {
              multiBlocks = true;
            }
          } break;
          case SpvOpReturnValue: {
            // Store return value to return variable.
            assert(returnVarId != 0);
            uint32_t valId = cpi->GetInOperand(kSpvReturnValueId).words[0];
            const auto mapItr = callee2caller.find(valId);
            if (mapItr != callee2caller.end()) {
              valId = mapItr->second;
            }
            AddStore(returnVarId, valId, &new_blk_ptr);

            // Remember we saw a return; if followed by a label, will need to
            // insert branch.
            prevInstWasReturn = true;
          } break;
          case SpvOpReturn: {
            // Remember we saw a return; if followed by a label, will need to
            // insert branch.
            prevInstWasReturn = true;
          } break;
          case SpvOpFunctionEnd: {
            // If there was an early return, we generated a return label id
            // for it.  Now we have to generate the return block with that Id.
            if (returnLabelId != 0) {
              // If previous instruction was return, insert branch instruction
              // to return block.
              if (prevInstWasReturn) AddBranch(returnLabelId, &new_blk_ptr);
              if (earlyReturn) {
                // If we generated a loop header for the single-trip loop
                // to accommodate early returns, insert the continue
                // target block now, with a false branch back to the loop
                // header.
                new_blocks->push_back(std::move(new_blk_ptr));
                new_blk_ptr =
                    MakeUnique<BasicBlock>(NewLabel(singleTripLoopContinueId));
                uint32_t false_id = GetFalseId();
                if (false_id == 0) {
                  return false;
                }
                AddBranchCond(false_id, singleTripLoopHeaderId, returnLabelId,
                              &new_blk_ptr);
              }
              // Generate the return block.
              new_blocks->push_back(std::move(new_blk_ptr));
              new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(returnLabelId));
              multiBlocks = true;
            }
            // Load return value into result id of call, if it exists.
            if (returnVarId != 0) {
              const uint32_t resId = call_inst_itr->result_id();
              assert(resId != 0);
              AddLoad(calleeTypeId, resId, returnVarId, &new_blk_ptr);
            }
            // Copy remaining instructions from caller block.
            for (Instruction* inst = call_inst_itr->NextNode(); inst;
                 inst = call_inst_itr->NextNode()) {
              inst->RemoveFromList();
              std::unique_ptr<Instruction> cp_inst(inst);
              // If multiple blocks generated, regenerate any same-block
              // instruction that has not been seen in this last block.
              if (multiBlocks) {
                if (!CloneSameBlockOps(&cp_inst, &postCallSB, &preCallSB,
                                       &new_blk_ptr)) {
                  return false;
                }

                // Remember same-block ops in this block.
                if (IsSameBlockOp(&*cp_inst)) {
                  const uint32_t rid = cp_inst->result_id();
                  postCallSB[rid] = rid;
                }
              }
              new_blk_ptr->AddInstruction(std::move(cp_inst));
            }
            // Finalize inline code.
            new_blocks->push_back(std::move(new_blk_ptr));
          } break;
          default: {
            // Copy callee instruction and remap all input Ids.
            std::unique_ptr<Instruction> cp_inst(cpi->Clone(context()));
            bool succeeded = cp_inst->WhileEachInId(
                [&callee2caller, &callee_result_ids, this](uint32_t* iid) {
                  const auto mapItr = callee2caller.find(*iid);
                  if (mapItr != callee2caller.end()) {
                    *iid = mapItr->second;
                  } else if (callee_result_ids.find(*iid) !=
                             callee_result_ids.end()) {
                    // Forward reference. Allocate a new id, map it,
                    // use it and check for it when remapping result ids
                    const uint32_t nid = context()->TakeNextId();
                    if (nid == 0) {
                      return false;
                    }
                    callee2caller[*iid] = nid;
                    *iid = nid;
                  }
                  return true;
                });
            if (!succeeded) {
              return false;
            }
            // If result id is non-zero, remap it. If already mapped, use mapped
            // value, else use next id.
            const uint32_t rid = cp_inst->result_id();
            if (rid != 0) {
              const auto mapItr = callee2caller.find(rid);
              uint32_t nid;
              if (mapItr != callee2caller.end()) {
                nid = mapItr->second;
              } else {
                nid = context()->TakeNextId();
                if (nid == 0) {
                  return false;
                }
                callee2caller[rid] = nid;
              }
              cp_inst->SetResultId(nid);
              get_decoration_mgr()->CloneDecorations(rid, nid);
            }
            new_blk_ptr->AddInstruction(std::move(cp_inst));
          } break;
        }
        return true;
      });

  if (!successful) {
    return false;
  }

  if (caller_is_loop_header && (new_blocks->size() > 1)) {
    // Move the OpLoopMerge from the last block back to the first, where
    // it belongs.
    auto& first = new_blocks->front();
    auto& last = new_blocks->back();
    assert(first != last);

    // Insert a modified copy of the loop merge into the first block.
    auto loop_merge_itr = last->tail();
    --loop_merge_itr;
    assert(loop_merge_itr->opcode() == SpvOpLoopMerge);
    std::unique_ptr<Instruction> cp_inst(loop_merge_itr->Clone(context()));
    if (caller_is_single_block_loop) {
      // Also, update its continue target to point to the last block.
      cp_inst->SetInOperand(kSpvLoopMergeContinueTargetIdInIdx, {last->id()});
    }
    first->tail().InsertBefore(std::move(cp_inst));

    // Remove the loop merge from the last block.
    loop_merge_itr->RemoveFromList();
    delete &*loop_merge_itr;
  }

  // Update block map given replacement blocks.
  for (auto& blk : *new_blocks) {
    id2block_[blk->id()] = &*blk;
  }
  return true;
}

bool InlinePass::IsInlinableFunctionCall(const Instruction* inst) {
  if (inst->opcode() != SpvOp::SpvOpFunctionCall) return false;
  const uint32_t calleeFnId =
      inst->GetSingleWordOperand(kSpvFunctionCallFunctionId);
  const auto ci = inlinable_.find(calleeFnId);
  return ci != inlinable_.cend();
}

void InlinePass::UpdateSucceedingPhis(
    std::vector<std::unique_ptr<BasicBlock>>& new_blocks) {
  const auto firstBlk = new_blocks.begin();
  const auto lastBlk = new_blocks.end() - 1;
  const uint32_t firstId = (*firstBlk)->id();
  const uint32_t lastId = (*lastBlk)->id();
  const BasicBlock& const_last_block = *lastBlk->get();
  const_last_block.ForEachSuccessorLabel(
      [&firstId, &lastId, this](const uint32_t succ) {
        BasicBlock* sbp = this->id2block_[succ];
        sbp->ForEachPhiInst([&firstId, &lastId](Instruction* phi) {
          phi->ForEachInId([&firstId, &lastId](uint32_t* id) {
            if (*id == firstId) *id = lastId;
          });
        });
      });
}

bool InlinePass::HasNoReturnInStructuredConstruct(Function* func) {
  // If control not structured, do not do loop/return analysis
  // TODO: Analyze returns in non-structured control flow
  if (!context()->get_feature_mgr()->HasCapability(SpvCapabilityShader))
    return false;
  const auto structured_analysis = context()->GetStructuredCFGAnalysis();
  // Search for returns in structured construct.
  bool return_in_construct = false;
  for (auto& blk : *func) {
    auto terminal_ii = blk.cend();
    --terminal_ii;
    if (spvOpcodeIsReturn(terminal_ii->opcode()) &&
        structured_analysis->ContainingConstruct(blk.id()) != 0) {
      return_in_construct = true;
      break;
    }
  }
  return !return_in_construct;
}

bool InlinePass::HasNoReturnInLoop(Function* func) {
  // If control not structured, do not do loop/return analysis
  // TODO: Analyze returns in non-structured control flow
  if (!context()->get_feature_mgr()->HasCapability(SpvCapabilityShader))
    return false;
  const auto structured_analysis = context()->GetStructuredCFGAnalysis();
  // Search for returns in structured construct.
  bool return_in_loop = false;
  for (auto& blk : *func) {
    auto terminal_ii = blk.cend();
    --terminal_ii;
    if (spvOpcodeIsReturn(terminal_ii->opcode()) &&
        structured_analysis->ContainingLoop(blk.id()) != 0) {
      return_in_loop = true;
      break;
    }
  }
  return !return_in_loop;
}

void InlinePass::AnalyzeReturns(Function* func) {
  if (HasNoReturnInLoop(func)) {
    no_return_in_loop_.insert(func->result_id());
    if (!HasNoReturnInStructuredConstruct(func))
      early_return_funcs_.insert(func->result_id());
  }
}

bool InlinePass::IsInlinableFunction(Function* func) {
  // We can only inline a function if it has blocks.
  if (func->cbegin() == func->cend()) return false;
  // Do not inline functions with returns in loops. Currently early return
  // functions are inlined by wrapping them in a one trip loop and implementing
  // the returns as a branch to the loop's merge block. However, this can only
  // done validly if the return was not in a loop in the original function.
  // Also remember functions with multiple (early) returns.
  AnalyzeReturns(func);
  if (no_return_in_loop_.find(func->result_id()) == no_return_in_loop_.cend()) {
    return false;
  }

  if (func->IsRecursive()) {
    return false;
  }

  return true;
}

void InlinePass::InitializeInline() {
  false_id_ = 0;

  // clear collections
  id2function_.clear();
  id2block_.clear();
  inlinable_.clear();
  no_return_in_loop_.clear();
  early_return_funcs_.clear();

  for (auto& fn : *get_module()) {
    // Initialize function and block maps.
    id2function_[fn.result_id()] = &fn;
    for (auto& blk : fn) {
      id2block_[blk.id()] = &blk;
    }
    // Compute inlinability
    if (IsInlinableFunction(&fn)) inlinable_.insert(fn.result_id());
  }
}

InlinePass::InlinePass() {}

}  // namespace opt
}  // namespace spvtools
