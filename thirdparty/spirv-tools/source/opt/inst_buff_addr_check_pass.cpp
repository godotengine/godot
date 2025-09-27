// Copyright (c) 2019 The Khronos Group Inc.
// Copyright (c) 2019 Valve Corporation
// Copyright (c) 2019 LunarG Inc.
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

#include "inst_buff_addr_check_pass.h"

namespace spvtools {
namespace opt {

uint32_t InstBuffAddrCheckPass::CloneOriginalReference(
    Instruction* ref_inst, InstructionBuilder* builder) {
  // Clone original ref with new result id (if load)
  assert((ref_inst->opcode() == spv::Op::OpLoad ||
          ref_inst->opcode() == spv::Op::OpStore) &&
         "unexpected ref");
  std::unique_ptr<Instruction> new_ref_inst(ref_inst->Clone(context()));
  uint32_t ref_result_id = ref_inst->result_id();
  uint32_t new_ref_id = 0;
  if (ref_result_id != 0) {
    new_ref_id = TakeNextId();
    new_ref_inst->SetResultId(new_ref_id);
  }
  // Register new reference and add to new block
  Instruction* added_inst = builder->AddInstruction(std::move(new_ref_inst));
  uid2offset_[added_inst->unique_id()] = uid2offset_[ref_inst->unique_id()];
  if (new_ref_id != 0)
    get_decoration_mgr()->CloneDecorations(ref_result_id, new_ref_id);
  return new_ref_id;
}

bool InstBuffAddrCheckPass::IsPhysicalBuffAddrReference(Instruction* ref_inst) {
  if (ref_inst->opcode() != spv::Op::OpLoad &&
      ref_inst->opcode() != spv::Op::OpStore)
    return false;
  uint32_t ptr_id = ref_inst->GetSingleWordInOperand(0);
  analysis::DefUseManager* du_mgr = get_def_use_mgr();
  Instruction* ptr_inst = du_mgr->GetDef(ptr_id);
  if (ptr_inst->opcode() != spv::Op::OpAccessChain) return false;
  uint32_t ptr_ty_id = ptr_inst->type_id();
  Instruction* ptr_ty_inst = du_mgr->GetDef(ptr_ty_id);
  if (spv::StorageClass(ptr_ty_inst->GetSingleWordInOperand(0)) !=
      spv::StorageClass::PhysicalStorageBufferEXT)
    return false;
  return true;
}

// TODO(greg-lunarg): Refactor with InstBindlessCheckPass::GenCheckCode() ??
void InstBuffAddrCheckPass::GenCheckCode(
    uint32_t check_id, uint32_t error_id, uint32_t ref_uptr_id,
    uint32_t stage_idx, Instruction* ref_inst,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  BasicBlock* back_blk_ptr = &*new_blocks->back();
  InstructionBuilder builder(
      context(), back_blk_ptr,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  // Gen conditional branch on check_id. Valid branch generates original
  // reference. Invalid generates debug output and zero result (if needed).
  uint32_t merge_blk_id = TakeNextId();
  uint32_t valid_blk_id = TakeNextId();
  uint32_t invalid_blk_id = TakeNextId();
  std::unique_ptr<Instruction> merge_label(NewLabel(merge_blk_id));
  std::unique_ptr<Instruction> valid_label(NewLabel(valid_blk_id));
  std::unique_ptr<Instruction> invalid_label(NewLabel(invalid_blk_id));
  (void)builder.AddConditionalBranch(
      check_id, valid_blk_id, invalid_blk_id, merge_blk_id,
      uint32_t(spv::SelectionControlMask::MaskNone));
  // Gen valid branch
  std::unique_ptr<BasicBlock> new_blk_ptr(
      new BasicBlock(std::move(valid_label)));
  builder.SetInsertPoint(&*new_blk_ptr);
  uint32_t new_ref_id = CloneOriginalReference(ref_inst, &builder);
  (void)builder.AddBranch(merge_blk_id);
  new_blocks->push_back(std::move(new_blk_ptr));
  // Gen invalid block
  new_blk_ptr.reset(new BasicBlock(std::move(invalid_label)));
  builder.SetInsertPoint(&*new_blk_ptr);
  // Convert uptr from uint64 to 2 uint32
  Instruction* lo_uptr_inst =
      builder.AddUnaryOp(GetUintId(), spv::Op::OpUConvert, ref_uptr_id);
  Instruction* rshift_uptr_inst =
      builder.AddBinaryOp(GetUint64Id(), spv::Op::OpShiftRightLogical,
                          ref_uptr_id, builder.GetUintConstantId(32));
  Instruction* hi_uptr_inst = builder.AddUnaryOp(
      GetUintId(), spv::Op::OpUConvert, rshift_uptr_inst->result_id());
  GenDebugStreamWrite(
      uid2offset_[ref_inst->unique_id()], stage_idx,
      {error_id, lo_uptr_inst->result_id(), hi_uptr_inst->result_id()},
      &builder);
  // Gen zero for invalid load. If pointer type, need to convert uint64
  // zero to pointer; cannot create ConstantNull of pointer type.
  uint32_t null_id = 0;
  if (new_ref_id != 0) {
    uint32_t ref_type_id = ref_inst->type_id();
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Type* ref_type = type_mgr->GetType(ref_type_id);
    if (ref_type->AsPointer() != nullptr) {
      uint32_t null_u64_id = GetNullId(GetUint64Id());
      Instruction* null_ptr_inst = builder.AddUnaryOp(
          ref_type_id, spv::Op::OpConvertUToPtr, null_u64_id);
      null_id = null_ptr_inst->result_id();
    } else {
      null_id = GetNullId(ref_type_id);
    }
  }
  (void)builder.AddBranch(merge_blk_id);
  new_blocks->push_back(std::move(new_blk_ptr));
  // Gen merge block
  new_blk_ptr.reset(new BasicBlock(std::move(merge_label)));
  builder.SetInsertPoint(&*new_blk_ptr);
  // Gen phi of new reference and zero, if necessary, and replace the
  // result id of the original reference with that of the Phi. Kill original
  // reference.
  if (new_ref_id != 0) {
    Instruction* phi_inst =
        builder.AddPhi(ref_inst->type_id(),
                       {new_ref_id, valid_blk_id, null_id, invalid_blk_id});
    context()->ReplaceAllUsesWith(ref_inst->result_id(), phi_inst->result_id());
  }
  new_blocks->push_back(std::move(new_blk_ptr));
  context()->KillInst(ref_inst);
}

uint32_t InstBuffAddrCheckPass::GetTypeAlignment(uint32_t type_id) {
  Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
  switch (type_inst->opcode()) {
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeVector:
      return GetTypeLength(type_id);
    case spv::Op::OpTypeMatrix:
      return GetTypeAlignment(type_inst->GetSingleWordInOperand(0));
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
      return GetTypeAlignment(type_inst->GetSingleWordInOperand(0));
    case spv::Op::OpTypeStruct: {
      uint32_t max = 0;
      type_inst->ForEachInId([&max, this](const uint32_t* iid) {
        uint32_t alignment = GetTypeAlignment(*iid);
        max = (alignment > max) ? alignment : max;
      });
      return max;
    }
    case spv::Op::OpTypePointer:
      assert(spv::StorageClass(type_inst->GetSingleWordInOperand(0)) ==
                 spv::StorageClass::PhysicalStorageBufferEXT &&
             "unexpected pointer type");
      return 8u;
    default:
      assert(false && "unexpected type");
      return 0;
  }
}

uint32_t InstBuffAddrCheckPass::GetTypeLength(uint32_t type_id) {
  Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
  switch (type_inst->opcode()) {
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeInt:
      return type_inst->GetSingleWordInOperand(0) / 8u;
    case spv::Op::OpTypeVector: {
      uint32_t raw_cnt = type_inst->GetSingleWordInOperand(1);
      uint32_t adj_cnt = (raw_cnt == 3u) ? 4u : raw_cnt;
      return adj_cnt * GetTypeLength(type_inst->GetSingleWordInOperand(0));
    }
    case spv::Op::OpTypeMatrix:
      return type_inst->GetSingleWordInOperand(1) *
             GetTypeLength(type_inst->GetSingleWordInOperand(0));
    case spv::Op::OpTypePointer:
      assert(spv::StorageClass(type_inst->GetSingleWordInOperand(0)) ==
                 spv::StorageClass::PhysicalStorageBufferEXT &&
             "unexpected pointer type");
      return 8u;
    case spv::Op::OpTypeArray: {
      uint32_t const_id = type_inst->GetSingleWordInOperand(1);
      Instruction* const_inst = get_def_use_mgr()->GetDef(const_id);
      uint32_t cnt = const_inst->GetSingleWordInOperand(0);
      return cnt * GetTypeLength(type_inst->GetSingleWordInOperand(0));
    }
    case spv::Op::OpTypeStruct: {
      uint32_t len = 0;
      type_inst->ForEachInId([&len, this](const uint32_t* iid) {
        // Align struct length
        uint32_t alignment = GetTypeAlignment(*iid);
        uint32_t mod = len % alignment;
        uint32_t diff = (mod != 0) ? alignment - mod : 0;
        len += diff;
        // Increment struct length by component length
        uint32_t comp_len = GetTypeLength(*iid);
        len += comp_len;
      });
      return len;
    }
    case spv::Op::OpTypeRuntimeArray:
    default:
      assert(false && "unexpected type");
      return 0;
  }
}

void InstBuffAddrCheckPass::AddParam(uint32_t type_id,
                                     std::vector<uint32_t>* param_vec,
                                     std::unique_ptr<Function>* input_func) {
  uint32_t pid = TakeNextId();
  param_vec->push_back(pid);
  std::unique_ptr<Instruction> param_inst(new Instruction(
      get_module()->context(), spv::Op::OpFunctionParameter, type_id, pid, {}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*param_inst);
  (*input_func)->AddParameter(std::move(param_inst));
}

uint32_t InstBuffAddrCheckPass::GetSearchAndTestFuncId() {
  if (search_test_func_id_ == 0) {
    // Generate function "bool search_and_test(uint64_t ref_ptr, uint32_t len)"
    // which searches input buffer for buffer which most likely contains the
    // pointer value |ref_ptr| and verifies that the entire reference of
    // length |len| bytes is contained in the buffer.
    search_test_func_id_ = TakeNextId();
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    std::vector<const analysis::Type*> param_types = {
        type_mgr->GetType(GetUint64Id()), type_mgr->GetType(GetUintId())};
    analysis::Function func_ty(type_mgr->GetType(GetBoolId()), param_types);
    analysis::Type* reg_func_ty = type_mgr->GetRegisteredType(&func_ty);
    std::unique_ptr<Instruction> func_inst(
        new Instruction(get_module()->context(), spv::Op::OpFunction,
                        GetBoolId(), search_test_func_id_,
                        {{spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                          {uint32_t(spv::FunctionControlMask::MaskNone)}},
                         {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                          {type_mgr->GetTypeInstruction(reg_func_ty)}}}));
    get_def_use_mgr()->AnalyzeInstDefUse(&*func_inst);
    std::unique_ptr<Function> input_func =
        MakeUnique<Function>(std::move(func_inst));
    std::vector<uint32_t> param_vec;
    // Add ref_ptr and length parameters
    AddParam(GetUint64Id(), &param_vec, &input_func);
    AddParam(GetUintId(), &param_vec, &input_func);
    // Empty first block.
    uint32_t first_blk_id = TakeNextId();
    std::unique_ptr<Instruction> first_blk_label(NewLabel(first_blk_id));
    std::unique_ptr<BasicBlock> first_blk_ptr =
        MakeUnique<BasicBlock>(std::move(first_blk_label));
    InstructionBuilder builder(
        context(), &*first_blk_ptr,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    uint32_t hdr_blk_id = TakeNextId();
    // Branch to search loop header
    std::unique_ptr<Instruction> hdr_blk_label(NewLabel(hdr_blk_id));
    (void)builder.AddBranch(hdr_blk_id);
    input_func->AddBasicBlock(std::move(first_blk_ptr));
    // Linear search loop header block
    // TODO(greg-lunarg): Implement binary search
    std::unique_ptr<BasicBlock> hdr_blk_ptr =
        MakeUnique<BasicBlock>(std::move(hdr_blk_label));
    builder.SetInsertPoint(&*hdr_blk_ptr);
    // Phi for search index. Starts with 1.
    uint32_t cont_blk_id = TakeNextId();
    std::unique_ptr<Instruction> cont_blk_label(NewLabel(cont_blk_id));
    // Deal with def-use cycle caused by search loop index computation.
    // Create Add and Phi instructions first, then do Def analysis on Add.
    // Add Phi and Add instructions and do Use analysis later.
    uint32_t idx_phi_id = TakeNextId();
    uint32_t idx_inc_id = TakeNextId();
    std::unique_ptr<Instruction> idx_inc_inst(new Instruction(
        context(), spv::Op::OpIAdd, GetUintId(), idx_inc_id,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {idx_phi_id}},
         {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          {builder.GetUintConstantId(1u)}}}));
    std::unique_ptr<Instruction> idx_phi_inst(new Instruction(
        context(), spv::Op::OpPhi, GetUintId(), idx_phi_id,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          {builder.GetUintConstantId(1u)}},
         {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {first_blk_id}},
         {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {idx_inc_id}},
         {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {cont_blk_id}}}));
    get_def_use_mgr()->AnalyzeInstDef(&*idx_inc_inst);
    // Add (previously created) search index phi
    (void)builder.AddInstruction(std::move(idx_phi_inst));
    // LoopMerge
    uint32_t bound_test_blk_id = TakeNextId();
    std::unique_ptr<Instruction> bound_test_blk_label(
        NewLabel(bound_test_blk_id));
    (void)builder.AddLoopMerge(bound_test_blk_id, cont_blk_id,
                               uint32_t(spv::LoopControlMask::MaskNone));
    // Branch to continue/work block
    (void)builder.AddBranch(cont_blk_id);
    input_func->AddBasicBlock(std::move(hdr_blk_ptr));
    // Continue/Work Block. Read next buffer pointer and break if greater
    // than ref_ptr arg.
    std::unique_ptr<BasicBlock> cont_blk_ptr =
        MakeUnique<BasicBlock>(std::move(cont_blk_label));
    builder.SetInsertPoint(&*cont_blk_ptr);
    // Add (previously created) search index increment now.
    (void)builder.AddInstruction(std::move(idx_inc_inst));
    // Load next buffer address from debug input buffer
    uint32_t ibuf_id = GetInputBufferId();
    uint32_t ibuf_ptr_id = GetInputBufferPtrId();
    Instruction* uptr_ac_inst = builder.AddTernaryOp(
        ibuf_ptr_id, spv::Op::OpAccessChain, ibuf_id,
        builder.GetUintConstantId(kDebugInputDataOffset), idx_inc_id);
    uint32_t ibuf_type_id = GetInputBufferTypeId();
    Instruction* uptr_load_inst = builder.AddUnaryOp(
        ibuf_type_id, spv::Op::OpLoad, uptr_ac_inst->result_id());
    // If loaded address greater than ref_ptr arg, break, else branch back to
    // loop header
    Instruction* uptr_test_inst =
        builder.AddBinaryOp(GetBoolId(), spv::Op::OpUGreaterThan,
                            uptr_load_inst->result_id(), param_vec[0]);
    (void)builder.AddConditionalBranch(
        uptr_test_inst->result_id(), bound_test_blk_id, hdr_blk_id, kInvalidId,
        uint32_t(spv::SelectionControlMask::MaskNone));
    input_func->AddBasicBlock(std::move(cont_blk_ptr));
    // Bounds test block. Read length of selected buffer and test that
    // all len arg bytes are in buffer.
    std::unique_ptr<BasicBlock> bound_test_blk_ptr =
        MakeUnique<BasicBlock>(std::move(bound_test_blk_label));
    builder.SetInsertPoint(&*bound_test_blk_ptr);
    // Decrement index to point to previous/candidate buffer address
    Instruction* cand_idx_inst =
        builder.AddBinaryOp(GetUintId(), spv::Op::OpISub, idx_inc_id,
                            builder.GetUintConstantId(1u));
    // Load candidate buffer address
    Instruction* cand_ac_inst =
        builder.AddTernaryOp(ibuf_ptr_id, spv::Op::OpAccessChain, ibuf_id,
                             builder.GetUintConstantId(kDebugInputDataOffset),
                             cand_idx_inst->result_id());
    Instruction* cand_load_inst = builder.AddUnaryOp(
        ibuf_type_id, spv::Op::OpLoad, cand_ac_inst->result_id());
    // Compute offset of ref_ptr from candidate buffer address
    Instruction* offset_inst =
        builder.AddBinaryOp(ibuf_type_id, spv::Op::OpISub, param_vec[0],
                            cand_load_inst->result_id());
    // Convert ref length to uint64
    Instruction* ref_len_64_inst =
        builder.AddUnaryOp(ibuf_type_id, spv::Op::OpUConvert, param_vec[1]);
    // Add ref length to ref offset to compute end of reference
    Instruction* ref_end_inst = builder.AddBinaryOp(
        ibuf_type_id, spv::Op::OpIAdd, offset_inst->result_id(),
        ref_len_64_inst->result_id());
    // Load starting index of lengths in input buffer and convert to uint32
    Instruction* len_start_ac_inst =
        builder.AddTernaryOp(ibuf_ptr_id, spv::Op::OpAccessChain, ibuf_id,
                             builder.GetUintConstantId(kDebugInputDataOffset),
                             builder.GetUintConstantId(0u));
    Instruction* len_start_load_inst = builder.AddUnaryOp(
        ibuf_type_id, spv::Op::OpLoad, len_start_ac_inst->result_id());
    Instruction* len_start_32_inst = builder.AddUnaryOp(
        GetUintId(), spv::Op::OpUConvert, len_start_load_inst->result_id());
    // Decrement search index to get candidate buffer length index
    Instruction* cand_len_idx_inst = builder.AddBinaryOp(
        GetUintId(), spv::Op::OpISub, cand_idx_inst->result_id(),
        builder.GetUintConstantId(1u));
    // Add candidate length index to start index
    Instruction* len_idx_inst = builder.AddBinaryOp(
        GetUintId(), spv::Op::OpIAdd, cand_len_idx_inst->result_id(),
        len_start_32_inst->result_id());
    // Load candidate buffer length
    Instruction* len_ac_inst =
        builder.AddTernaryOp(ibuf_ptr_id, spv::Op::OpAccessChain, ibuf_id,
                             builder.GetUintConstantId(kDebugInputDataOffset),
                             len_idx_inst->result_id());
    Instruction* len_load_inst = builder.AddUnaryOp(
        ibuf_type_id, spv::Op::OpLoad, len_ac_inst->result_id());
    // Test if reference end within candidate buffer length
    Instruction* len_test_inst = builder.AddBinaryOp(
        GetBoolId(), spv::Op::OpULessThanEqual, ref_end_inst->result_id(),
        len_load_inst->result_id());
    // Return test result
    (void)builder.AddUnaryOp(0, spv::Op::OpReturnValue,
                             len_test_inst->result_id());
    // Close block
    input_func->AddBasicBlock(std::move(bound_test_blk_ptr));
    // Close function and add function to module
    std::unique_ptr<Instruction> func_end_inst(new Instruction(
        get_module()->context(), spv::Op::OpFunctionEnd, 0, 0, {}));
    get_def_use_mgr()->AnalyzeInstDefUse(&*func_end_inst);
    input_func->SetFunctionEnd(std::move(func_end_inst));
    context()->AddFunction(std::move(input_func));
    context()->AddDebug2Inst(
        NewGlobalName(search_test_func_id_, "search_and_test"));
  }
  return search_test_func_id_;
}

uint32_t InstBuffAddrCheckPass::GenSearchAndTest(Instruction* ref_inst,
                                                 InstructionBuilder* builder,
                                                 uint32_t* ref_uptr_id) {
  // Enable Int64 if necessary
  context()->AddCapability(spv::Capability::Int64);
  // Convert reference pointer to uint64
  uint32_t ref_ptr_id = ref_inst->GetSingleWordInOperand(0);
  Instruction* ref_uptr_inst =
      builder->AddUnaryOp(GetUint64Id(), spv::Op::OpConvertPtrToU, ref_ptr_id);
  *ref_uptr_id = ref_uptr_inst->result_id();
  // Compute reference length in bytes
  analysis::DefUseManager* du_mgr = get_def_use_mgr();
  Instruction* ref_ptr_inst = du_mgr->GetDef(ref_ptr_id);
  uint32_t ref_ptr_ty_id = ref_ptr_inst->type_id();
  Instruction* ref_ptr_ty_inst = du_mgr->GetDef(ref_ptr_ty_id);
  uint32_t ref_len = GetTypeLength(ref_ptr_ty_inst->GetSingleWordInOperand(1));
  uint32_t ref_len_id = builder->GetUintConstantId(ref_len);
  // Gen call to search and test function
  Instruction* call_inst = builder->AddFunctionCall(
      GetBoolId(), GetSearchAndTestFuncId(), {*ref_uptr_id, ref_len_id});
  uint32_t retval = call_inst->result_id();
  return retval;
}

void InstBuffAddrCheckPass::GenBuffAddrCheckCode(
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  // Look for reference through indexed descriptor. If found, analyze and
  // save components. If not, return.
  Instruction* ref_inst = &*ref_inst_itr;
  if (!IsPhysicalBuffAddrReference(ref_inst)) return;
  // Move original block's preceding instructions into first new block
  std::unique_ptr<BasicBlock> new_blk_ptr;
  MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
  InstructionBuilder builder(
      context(), &*new_blk_ptr,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  new_blocks->push_back(std::move(new_blk_ptr));
  uint32_t error_id = builder.GetUintConstantId(kInstErrorBuffAddrUnallocRef);
  // Generate code to do search and test if all bytes of reference
  // are within a listed buffer. Return reference pointer converted to uint64.
  uint32_t ref_uptr_id;
  uint32_t valid_id = GenSearchAndTest(ref_inst, &builder, &ref_uptr_id);
  // Generate test of search results with true branch
  // being full reference and false branch being debug output and zero
  // for the referenced value.
  GenCheckCode(valid_id, error_id, ref_uptr_id, stage_idx, ref_inst,
               new_blocks);
  // Move original block's remaining code into remainder/merge block and add
  // to new blocks
  BasicBlock* back_blk_ptr = &*new_blocks->back();
  MovePostludeCode(ref_block_itr, back_blk_ptr);
}

void InstBuffAddrCheckPass::InitInstBuffAddrCheck() {
  // Initialize base class
  InitializeInstrument();
  // Initialize class
  search_test_func_id_ = 0;
}

Pass::Status InstBuffAddrCheckPass::ProcessImpl() {
  // Perform bindless bounds check on each entry point function in module
  InstProcessFunction pfn =
      [this](BasicBlock::iterator ref_inst_itr,
             UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
             std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
        return GenBuffAddrCheckCode(ref_inst_itr, ref_block_itr, stage_idx,
                                    new_blocks);
      };
  bool modified = InstProcessEntryPointCallTree(pfn);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Pass::Status InstBuffAddrCheckPass::Process() {
  if (!get_feature_mgr()->HasCapability(
          spv::Capability::PhysicalStorageBufferAddressesEXT))
    return Status::SuccessWithoutChange;
  InitInstBuffAddrCheck();
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
