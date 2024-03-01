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
    uint32_t check_id, Instruction* ref_inst,
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

uint32_t InstBuffAddrCheckPass::GetTypeLength(uint32_t type_id) {
  Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
  switch (type_inst->opcode()) {
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeInt:
      return type_inst->GetSingleWordInOperand(0) / 8u;
    case spv::Op::OpTypeVector:
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
      // Figure out the location of the last byte of the last member of the
      // structure.
      uint32_t last_offset = 0, last_len = 0;

      get_decoration_mgr()->ForEachDecoration(
          type_id, uint32_t(spv::Decoration::Offset),
          [&last_offset](const Instruction& deco_inst) {
            last_offset = deco_inst.GetSingleWordInOperand(3);
          });
      type_inst->ForEachInId([&last_len, this](const uint32_t* iid) {
        last_len = GetTypeLength(*iid);
      });
      return last_offset + last_len;
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

// This is a stub function for use with Import linkage
// clang-format off
// GLSL:
//bool inst_bindless_search_and_test(const uint shader_id, const uint inst_num, const uvec4 stage_info,
//				     const uint64 ref_ptr, const uint length) {
//}
// clang-format on
uint32_t InstBuffAddrCheckPass::GetSearchAndTestFuncId() {
  enum {
    kShaderId = 0,
    kInstructionIndex = 1,
    kStageInfo = 2,
    kRefPtr = 3,
    kLength = 4,
    kNumArgs
  };
  if (search_test_func_id_ != 0) {
    return search_test_func_id_;
  }
  // Generate function "bool search_and_test(uint64_t ref_ptr, uint32_t len)"
  // which searches input buffer for buffer which most likely contains the
  // pointer value |ref_ptr| and verifies that the entire reference of
  // length |len| bytes is contained in the buffer.
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  const analysis::Integer* uint_type = GetInteger(32, false);
  const analysis::Vector v4uint(uint_type, 4);
  const analysis::Type* v4uint_type = type_mgr->GetRegisteredType(&v4uint);

  std::vector<const analysis::Type*> param_types = {
      uint_type, uint_type, v4uint_type, type_mgr->GetType(GetUint64Id()),
      uint_type};

  const std::string func_name{"inst_buff_addr_search_and_test"};
  const uint32_t func_id = TakeNextId();
  std::unique_ptr<Function> func =
      StartFunction(func_id, type_mgr->GetBoolType(), param_types);
  func->SetFunctionEnd(EndFunction());
  context()->AddFunctionDeclaration(std::move(func));
  context()->AddDebug2Inst(NewName(func_id, func_name));

  std::vector<Operand> operands{
      {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {func_id}},
      {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
       {uint32_t(spv::Decoration::LinkageAttributes)}},
      {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_STRING,
       utils::MakeVector(func_name.c_str())},
      {spv_operand_type_t::SPV_OPERAND_TYPE_LINKAGE_TYPE,
       {uint32_t(spv::LinkageType::Import)}},
  };
  get_decoration_mgr()->AddDecoration(spv::Op::OpDecorate, operands);

  search_test_func_id_ = func_id;
  return search_test_func_id_;
}

uint32_t InstBuffAddrCheckPass::GenSearchAndTest(Instruction* ref_inst,
                                                 InstructionBuilder* builder,
                                                 uint32_t* ref_uptr_id,
                                                 uint32_t stage_idx) {
  // Enable Int64 if necessary
  // Convert reference pointer to uint64
  const uint32_t ref_ptr_id = ref_inst->GetSingleWordInOperand(0);
  Instruction* ref_uptr_inst =
      builder->AddUnaryOp(GetUint64Id(), spv::Op::OpConvertPtrToU, ref_ptr_id);
  *ref_uptr_id = ref_uptr_inst->result_id();
  // Compute reference length in bytes
  analysis::DefUseManager* du_mgr = get_def_use_mgr();
  Instruction* ref_ptr_inst = du_mgr->GetDef(ref_ptr_id);
  const uint32_t ref_ptr_ty_id = ref_ptr_inst->type_id();
  Instruction* ref_ptr_ty_inst = du_mgr->GetDef(ref_ptr_ty_id);
  const uint32_t ref_len =
      GetTypeLength(ref_ptr_ty_inst->GetSingleWordInOperand(1));
  // Gen call to search and test function
  const uint32_t func_id = GetSearchAndTestFuncId();
  const std::vector<uint32_t> args = {
      builder->GetUintConstantId(shader_id_),
      builder->GetUintConstantId(ref_inst->unique_id()),
      GenStageInfo(stage_idx, builder), *ref_uptr_id,
      builder->GetUintConstantId(ref_len)};
  return GenReadFunctionCall(GetBoolId(), func_id, args, builder);
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
  // Generate code to do search and test if all bytes of reference
  // are within a listed buffer. Return reference pointer converted to uint64.
  uint32_t ref_uptr_id;
  uint32_t valid_id =
      GenSearchAndTest(ref_inst, &builder, &ref_uptr_id, stage_idx);
  // Generate test of search results with true branch
  // being full reference and false branch being debug output and zero
  // for the referenced value.
  GenCheckCode(valid_id, ref_inst, new_blocks);

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
  // The memory model and linkage must always be updated for spirv-link to work
  // correctly.
  AddStorageBufferExt();
  if (!get_feature_mgr()->HasExtension(kSPV_KHR_physical_storage_buffer)) {
    context()->AddExtension("SPV_KHR_physical_storage_buffer");
  }

  context()->AddCapability(spv::Capability::PhysicalStorageBufferAddresses);
  Instruction* memory_model = get_module()->GetMemoryModel();
  memory_model->SetInOperand(
      0u, {uint32_t(spv::AddressingModel::PhysicalStorageBuffer64)});

  context()->AddCapability(spv::Capability::Int64);
  context()->AddCapability(spv::Capability::Linkage);
  // Perform bindless bounds check on each entry point function in module
  InstProcessFunction pfn =
      [this](BasicBlock::iterator ref_inst_itr,
             UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
             std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
        return GenBuffAddrCheckCode(ref_inst_itr, ref_block_itr, stage_idx,
                                    new_blocks);
      };
  InstProcessEntryPointCallTree(pfn);
  // This pass always changes the memory model, so that linking will work
  // properly.
  return Status::SuccessWithChange;
}

Pass::Status InstBuffAddrCheckPass::Process() {
  InitInstBuffAddrCheck();
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
