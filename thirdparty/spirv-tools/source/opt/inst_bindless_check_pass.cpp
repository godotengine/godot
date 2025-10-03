// Copyright (c) 2018 The Khronos Group Inc.
// Copyright (c) 2018 Valve Corporation
// Copyright (c) 2018 LunarG Inc.
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

#include "inst_bindless_check_pass.h"

namespace spvtools {
namespace opt {
namespace {
// Input Operand Indices
constexpr int kSpvImageSampleImageIdInIdx = 0;
constexpr int kSpvSampledImageImageIdInIdx = 0;
constexpr int kSpvSampledImageSamplerIdInIdx = 1;
constexpr int kSpvImageSampledImageIdInIdx = 0;
constexpr int kSpvCopyObjectOperandIdInIdx = 0;
constexpr int kSpvLoadPtrIdInIdx = 0;
constexpr int kSpvAccessChainBaseIdInIdx = 0;
constexpr int kSpvAccessChainIndex0IdInIdx = 1;
constexpr int kSpvTypeArrayTypeIdInIdx = 0;
constexpr int kSpvTypeArrayLengthIdInIdx = 1;
constexpr int kSpvConstantValueInIdx = 0;
constexpr int kSpvVariableStorageClassInIdx = 0;
constexpr int kSpvTypePtrTypeIdInIdx = 1;
constexpr int kSpvTypeImageDim = 1;
constexpr int kSpvTypeImageDepth = 2;
constexpr int kSpvTypeImageArrayed = 3;
constexpr int kSpvTypeImageMS = 4;
constexpr int kSpvTypeImageSampled = 5;
}  // namespace

uint32_t InstBindlessCheckPass::GenDebugReadLength(
    uint32_t var_id, InstructionBuilder* builder) {
  uint32_t desc_set_idx =
      var2desc_set_[var_id] + kDebugInputBindlessOffsetLengths;
  uint32_t desc_set_idx_id = builder->GetUintConstantId(desc_set_idx);
  uint32_t binding_idx_id = builder->GetUintConstantId(var2binding_[var_id]);
  return GenDebugDirectRead({desc_set_idx_id, binding_idx_id}, builder);
}

uint32_t InstBindlessCheckPass::GenDebugReadInit(uint32_t var_id,
                                                 uint32_t desc_idx_id,
                                                 InstructionBuilder* builder) {
  uint32_t binding_idx_id = builder->GetUintConstantId(var2binding_[var_id]);
  uint32_t u_desc_idx_id = GenUintCastCode(desc_idx_id, builder);
  // If desc index checking is not enabled, we know the offset of initialization
  // entries is 1, so we can avoid loading this value and just add 1 to the
  // descriptor set.
  if (!desc_idx_enabled_) {
    uint32_t desc_set_idx_id =
        builder->GetUintConstantId(var2desc_set_[var_id] + 1);
    return GenDebugDirectRead({desc_set_idx_id, binding_idx_id, u_desc_idx_id},
                              builder);
  } else {
    uint32_t desc_set_base_id =
        builder->GetUintConstantId(kDebugInputBindlessInitOffset);
    uint32_t desc_set_idx_id =
        builder->GetUintConstantId(var2desc_set_[var_id]);
    return GenDebugDirectRead(
        {desc_set_base_id, desc_set_idx_id, binding_idx_id, u_desc_idx_id},
        builder);
  }
}

uint32_t InstBindlessCheckPass::CloneOriginalImage(
    uint32_t old_image_id, InstructionBuilder* builder) {
  Instruction* new_image_inst;
  Instruction* old_image_inst = get_def_use_mgr()->GetDef(old_image_id);
  if (old_image_inst->opcode() == spv::Op::OpLoad) {
    new_image_inst = builder->AddLoad(
        old_image_inst->type_id(),
        old_image_inst->GetSingleWordInOperand(kSpvLoadPtrIdInIdx));
  } else if (old_image_inst->opcode() == spv::Op::OpSampledImage) {
    uint32_t clone_id = CloneOriginalImage(
        old_image_inst->GetSingleWordInOperand(kSpvSampledImageImageIdInIdx),
        builder);
    new_image_inst = builder->AddBinaryOp(
        old_image_inst->type_id(), spv::Op::OpSampledImage, clone_id,
        old_image_inst->GetSingleWordInOperand(kSpvSampledImageSamplerIdInIdx));
  } else if (old_image_inst->opcode() == spv::Op::OpImage) {
    uint32_t clone_id = CloneOriginalImage(
        old_image_inst->GetSingleWordInOperand(kSpvImageSampledImageIdInIdx),
        builder);
    new_image_inst = builder->AddUnaryOp(old_image_inst->type_id(),
                                         spv::Op::OpImage, clone_id);
  } else {
    assert(old_image_inst->opcode() == spv::Op::OpCopyObject &&
           "expecting OpCopyObject");
    uint32_t clone_id = CloneOriginalImage(
        old_image_inst->GetSingleWordInOperand(kSpvCopyObjectOperandIdInIdx),
        builder);
    // Since we are cloning, no need to create new copy
    new_image_inst = get_def_use_mgr()->GetDef(clone_id);
  }
  uid2offset_[new_image_inst->unique_id()] =
      uid2offset_[old_image_inst->unique_id()];
  uint32_t new_image_id = new_image_inst->result_id();
  get_decoration_mgr()->CloneDecorations(old_image_id, new_image_id);
  return new_image_id;
}

uint32_t InstBindlessCheckPass::CloneOriginalReference(
    RefAnalysis* ref, InstructionBuilder* builder) {
  // If original is image based, start by cloning descriptor load
  uint32_t new_image_id = 0;
  if (ref->desc_load_id != 0) {
    uint32_t old_image_id =
        ref->ref_inst->GetSingleWordInOperand(kSpvImageSampleImageIdInIdx);
    new_image_id = CloneOriginalImage(old_image_id, builder);
  }
  // Clone original reference
  std::unique_ptr<Instruction> new_ref_inst(ref->ref_inst->Clone(context()));
  uint32_t ref_result_id = ref->ref_inst->result_id();
  uint32_t new_ref_id = 0;
  if (ref_result_id != 0) {
    new_ref_id = TakeNextId();
    new_ref_inst->SetResultId(new_ref_id);
  }
  // Update new ref with new image if created
  if (new_image_id != 0)
    new_ref_inst->SetInOperand(kSpvImageSampleImageIdInIdx, {new_image_id});
  // Register new reference and add to new block
  Instruction* added_inst = builder->AddInstruction(std::move(new_ref_inst));
  uid2offset_[added_inst->unique_id()] =
      uid2offset_[ref->ref_inst->unique_id()];
  if (new_ref_id != 0)
    get_decoration_mgr()->CloneDecorations(ref_result_id, new_ref_id);
  return new_ref_id;
}

uint32_t InstBindlessCheckPass::GetImageId(Instruction* inst) {
  switch (inst->opcode()) {
    case spv::Op::OpImageSampleImplicitLod:
    case spv::Op::OpImageSampleExplicitLod:
    case spv::Op::OpImageSampleDrefImplicitLod:
    case spv::Op::OpImageSampleDrefExplicitLod:
    case spv::Op::OpImageSampleProjImplicitLod:
    case spv::Op::OpImageSampleProjExplicitLod:
    case spv::Op::OpImageSampleProjDrefImplicitLod:
    case spv::Op::OpImageSampleProjDrefExplicitLod:
    case spv::Op::OpImageGather:
    case spv::Op::OpImageDrefGather:
    case spv::Op::OpImageQueryLod:
    case spv::Op::OpImageSparseSampleImplicitLod:
    case spv::Op::OpImageSparseSampleExplicitLod:
    case spv::Op::OpImageSparseSampleDrefImplicitLod:
    case spv::Op::OpImageSparseSampleDrefExplicitLod:
    case spv::Op::OpImageSparseSampleProjImplicitLod:
    case spv::Op::OpImageSparseSampleProjExplicitLod:
    case spv::Op::OpImageSparseSampleProjDrefImplicitLod:
    case spv::Op::OpImageSparseSampleProjDrefExplicitLod:
    case spv::Op::OpImageSparseGather:
    case spv::Op::OpImageSparseDrefGather:
    case spv::Op::OpImageFetch:
    case spv::Op::OpImageRead:
    case spv::Op::OpImageQueryFormat:
    case spv::Op::OpImageQueryOrder:
    case spv::Op::OpImageQuerySizeLod:
    case spv::Op::OpImageQuerySize:
    case spv::Op::OpImageQueryLevels:
    case spv::Op::OpImageQuerySamples:
    case spv::Op::OpImageSparseFetch:
    case spv::Op::OpImageSparseRead:
    case spv::Op::OpImageWrite:
      return inst->GetSingleWordInOperand(kSpvImageSampleImageIdInIdx);
    default:
      break;
  }
  return 0;
}

Instruction* InstBindlessCheckPass::GetPointeeTypeInst(Instruction* ptr_inst) {
  uint32_t pte_ty_id = GetPointeeTypeId(ptr_inst);
  return get_def_use_mgr()->GetDef(pte_ty_id);
}

bool InstBindlessCheckPass::AnalyzeDescriptorReference(Instruction* ref_inst,
                                                       RefAnalysis* ref) {
  ref->ref_inst = ref_inst;
  if (ref_inst->opcode() == spv::Op::OpLoad ||
      ref_inst->opcode() == spv::Op::OpStore) {
    ref->desc_load_id = 0;
    ref->ptr_id = ref_inst->GetSingleWordInOperand(kSpvLoadPtrIdInIdx);
    Instruction* ptr_inst = get_def_use_mgr()->GetDef(ref->ptr_id);
    if (ptr_inst->opcode() != spv::Op::OpAccessChain) return false;
    ref->var_id = ptr_inst->GetSingleWordInOperand(kSpvAccessChainBaseIdInIdx);
    Instruction* var_inst = get_def_use_mgr()->GetDef(ref->var_id);
    if (var_inst->opcode() != spv::Op::OpVariable) return false;
    spv::StorageClass storage_class = spv::StorageClass(
        var_inst->GetSingleWordInOperand(kSpvVariableStorageClassInIdx));
    switch (storage_class) {
      case spv::StorageClass::Uniform:
      case spv::StorageClass::StorageBuffer:
        break;
      default:
        return false;
        break;
    }
    // Check for deprecated storage block form
    if (storage_class == spv::StorageClass::Uniform) {
      uint32_t var_ty_id = var_inst->type_id();
      Instruction* var_ty_inst = get_def_use_mgr()->GetDef(var_ty_id);
      uint32_t ptr_ty_id =
          var_ty_inst->GetSingleWordInOperand(kSpvTypePtrTypeIdInIdx);
      Instruction* ptr_ty_inst = get_def_use_mgr()->GetDef(ptr_ty_id);
      spv::Op ptr_ty_op = ptr_ty_inst->opcode();
      uint32_t block_ty_id =
          (ptr_ty_op == spv::Op::OpTypeArray ||
           ptr_ty_op == spv::Op::OpTypeRuntimeArray)
              ? ptr_ty_inst->GetSingleWordInOperand(kSpvTypeArrayTypeIdInIdx)
              : ptr_ty_id;
      assert(get_def_use_mgr()->GetDef(block_ty_id)->opcode() ==
                 spv::Op::OpTypeStruct &&
             "unexpected block type");
      bool block_found = get_decoration_mgr()->FindDecoration(
          block_ty_id, uint32_t(spv::Decoration::Block),
          [](const Instruction&) { return true; });
      if (!block_found) {
        // If block decoration not found, verify deprecated form of SSBO
        bool buffer_block_found = get_decoration_mgr()->FindDecoration(
            block_ty_id, uint32_t(spv::Decoration::BufferBlock),
            [](const Instruction&) { return true; });
        USE_ASSERT(buffer_block_found && "block decoration not found");
        storage_class = spv::StorageClass::StorageBuffer;
      }
    }
    ref->strg_class = uint32_t(storage_class);
    Instruction* desc_type_inst = GetPointeeTypeInst(var_inst);
    switch (desc_type_inst->opcode()) {
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeRuntimeArray:
        // A load through a descriptor array will have at least 3 operands. We
        // do not want to instrument loads of descriptors here which are part of
        // an image-based reference.
        if (ptr_inst->NumInOperands() < 3) return false;
        ref->desc_idx_id =
            ptr_inst->GetSingleWordInOperand(kSpvAccessChainIndex0IdInIdx);
        break;
      default:
        ref->desc_idx_id = 0;
        break;
    }
    return true;
  }
  // Reference is not load or store. If not an image-based reference, return.
  ref->image_id = GetImageId(ref_inst);
  if (ref->image_id == 0) return false;
  // Search for descriptor load
  uint32_t desc_load_id = ref->image_id;
  Instruction* desc_load_inst;
  for (;;) {
    desc_load_inst = get_def_use_mgr()->GetDef(desc_load_id);
    if (desc_load_inst->opcode() == spv::Op::OpSampledImage)
      desc_load_id =
          desc_load_inst->GetSingleWordInOperand(kSpvSampledImageImageIdInIdx);
    else if (desc_load_inst->opcode() == spv::Op::OpImage)
      desc_load_id =
          desc_load_inst->GetSingleWordInOperand(kSpvImageSampledImageIdInIdx);
    else if (desc_load_inst->opcode() == spv::Op::OpCopyObject)
      desc_load_id =
          desc_load_inst->GetSingleWordInOperand(kSpvCopyObjectOperandIdInIdx);
    else
      break;
  }
  if (desc_load_inst->opcode() != spv::Op::OpLoad) {
    // TODO(greg-lunarg): Handle additional possibilities?
    return false;
  }
  ref->desc_load_id = desc_load_id;
  ref->ptr_id = desc_load_inst->GetSingleWordInOperand(kSpvLoadPtrIdInIdx);
  Instruction* ptr_inst = get_def_use_mgr()->GetDef(ref->ptr_id);
  if (ptr_inst->opcode() == spv::Op::OpVariable) {
    ref->desc_idx_id = 0;
    ref->var_id = ref->ptr_id;
  } else if (ptr_inst->opcode() == spv::Op::OpAccessChain) {
    if (ptr_inst->NumInOperands() != 2) {
      assert(false && "unexpected bindless index number");
      return false;
    }
    ref->desc_idx_id =
        ptr_inst->GetSingleWordInOperand(kSpvAccessChainIndex0IdInIdx);
    ref->var_id = ptr_inst->GetSingleWordInOperand(kSpvAccessChainBaseIdInIdx);
    Instruction* var_inst = get_def_use_mgr()->GetDef(ref->var_id);
    if (var_inst->opcode() != spv::Op::OpVariable) {
      assert(false && "unexpected bindless base");
      return false;
    }
  } else {
    // TODO(greg-lunarg): Handle additional possibilities?
    return false;
  }
  return true;
}

uint32_t InstBindlessCheckPass::FindStride(uint32_t ty_id,
                                           uint32_t stride_deco) {
  uint32_t stride = 0xdeadbeef;
  bool found = get_decoration_mgr()->FindDecoration(
      ty_id, stride_deco, [&stride](const Instruction& deco_inst) {
        stride = deco_inst.GetSingleWordInOperand(2u);
        return true;
      });
  USE_ASSERT(found && "stride not found");
  return stride;
}

uint32_t InstBindlessCheckPass::ByteSize(uint32_t ty_id, uint32_t matrix_stride,
                                         bool col_major, bool in_matrix) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  const analysis::Type* sz_ty = type_mgr->GetType(ty_id);
  if (sz_ty->kind() == analysis::Type::kPointer) {
    // Assuming PhysicalStorageBuffer pointer
    return 8;
  }
  if (sz_ty->kind() == analysis::Type::kMatrix) {
    assert(matrix_stride != 0 && "missing matrix stride");
    const analysis::Matrix* m_ty = sz_ty->AsMatrix();
    if (col_major) {
      return m_ty->element_count() * matrix_stride;
    } else {
      const analysis::Vector* v_ty = m_ty->element_type()->AsVector();
      return v_ty->element_count() * matrix_stride;
    }
  }
  uint32_t size = 1;
  if (sz_ty->kind() == analysis::Type::kVector) {
    const analysis::Vector* v_ty = sz_ty->AsVector();
    size = v_ty->element_count();
    const analysis::Type* comp_ty = v_ty->element_type();
    // if vector in row major matrix, the vector is strided so return the
    // number of bytes spanned by the vector
    if (in_matrix && !col_major && matrix_stride > 0) {
      uint32_t comp_ty_id = type_mgr->GetId(comp_ty);
      return (size - 1) * matrix_stride + ByteSize(comp_ty_id, 0, false, false);
    }
    sz_ty = comp_ty;
  }
  switch (sz_ty->kind()) {
    case analysis::Type::kFloat: {
      const analysis::Float* f_ty = sz_ty->AsFloat();
      size *= f_ty->width();
    } break;
    case analysis::Type::kInteger: {
      const analysis::Integer* i_ty = sz_ty->AsInteger();
      size *= i_ty->width();
    } break;
    default: { assert(false && "unexpected type"); } break;
  }
  size /= 8;
  return size;
}

uint32_t InstBindlessCheckPass::GenLastByteIdx(RefAnalysis* ref,
                                               InstructionBuilder* builder) {
  // Find outermost buffer type and its access chain index
  Instruction* var_inst = get_def_use_mgr()->GetDef(ref->var_id);
  Instruction* desc_ty_inst = GetPointeeTypeInst(var_inst);
  uint32_t buff_ty_id;
  uint32_t ac_in_idx = 1;
  switch (desc_ty_inst->opcode()) {
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
      buff_ty_id = desc_ty_inst->GetSingleWordInOperand(0);
      ++ac_in_idx;
      break;
    default:
      assert(desc_ty_inst->opcode() == spv::Op::OpTypeStruct &&
             "unexpected descriptor type");
      buff_ty_id = desc_ty_inst->result_id();
      break;
  }
  // Process remaining access chain indices
  Instruction* ac_inst = get_def_use_mgr()->GetDef(ref->ptr_id);
  uint32_t curr_ty_id = buff_ty_id;
  uint32_t sum_id = 0u;
  uint32_t matrix_stride = 0u;
  bool col_major = false;
  uint32_t matrix_stride_id = 0u;
  bool in_matrix = false;
  while (ac_in_idx < ac_inst->NumInOperands()) {
    uint32_t curr_idx_id = ac_inst->GetSingleWordInOperand(ac_in_idx);
    Instruction* curr_ty_inst = get_def_use_mgr()->GetDef(curr_ty_id);
    uint32_t curr_offset_id = 0;
    switch (curr_ty_inst->opcode()) {
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeRuntimeArray: {
        // Get array stride and multiply by current index
        uint32_t arr_stride =
            FindStride(curr_ty_id, uint32_t(spv::Decoration::ArrayStride));
        uint32_t arr_stride_id = builder->GetUintConstantId(arr_stride);
        uint32_t curr_idx_32b_id = Gen32BitCvtCode(curr_idx_id, builder);
        Instruction* curr_offset_inst = builder->AddBinaryOp(
            GetUintId(), spv::Op::OpIMul, arr_stride_id, curr_idx_32b_id);
        curr_offset_id = curr_offset_inst->result_id();
        // Get element type for next step
        curr_ty_id = curr_ty_inst->GetSingleWordInOperand(0);
      } break;
      case spv::Op::OpTypeMatrix: {
        assert(matrix_stride != 0 && "missing matrix stride");
        matrix_stride_id = builder->GetUintConstantId(matrix_stride);
        uint32_t vec_ty_id = curr_ty_inst->GetSingleWordInOperand(0);
        // If column major, multiply column index by matrix stride, otherwise
        // by vector component size and save matrix stride for vector (row)
        // index
        uint32_t col_stride_id;
        if (col_major) {
          col_stride_id = matrix_stride_id;
        } else {
          Instruction* vec_ty_inst = get_def_use_mgr()->GetDef(vec_ty_id);
          uint32_t comp_ty_id = vec_ty_inst->GetSingleWordInOperand(0u);
          uint32_t col_stride = ByteSize(comp_ty_id, 0u, false, false);
          col_stride_id = builder->GetUintConstantId(col_stride);
        }
        uint32_t curr_idx_32b_id = Gen32BitCvtCode(curr_idx_id, builder);
        Instruction* curr_offset_inst = builder->AddBinaryOp(
            GetUintId(), spv::Op::OpIMul, col_stride_id, curr_idx_32b_id);
        curr_offset_id = curr_offset_inst->result_id();
        // Get element type for next step
        curr_ty_id = vec_ty_id;
        in_matrix = true;
      } break;
      case spv::Op::OpTypeVector: {
        // If inside a row major matrix type, multiply index by matrix stride,
        // else multiply by component size
        uint32_t comp_ty_id = curr_ty_inst->GetSingleWordInOperand(0u);
        uint32_t curr_idx_32b_id = Gen32BitCvtCode(curr_idx_id, builder);
        if (in_matrix && !col_major) {
          Instruction* curr_offset_inst = builder->AddBinaryOp(
              GetUintId(), spv::Op::OpIMul, matrix_stride_id, curr_idx_32b_id);
          curr_offset_id = curr_offset_inst->result_id();
        } else {
          uint32_t comp_ty_sz = ByteSize(comp_ty_id, 0u, false, false);
          uint32_t comp_ty_sz_id = builder->GetUintConstantId(comp_ty_sz);
          Instruction* curr_offset_inst = builder->AddBinaryOp(
              GetUintId(), spv::Op::OpIMul, comp_ty_sz_id, curr_idx_32b_id);
          curr_offset_id = curr_offset_inst->result_id();
        }
        // Get element type for next step
        curr_ty_id = comp_ty_id;
      } break;
      case spv::Op::OpTypeStruct: {
        // Get buffer byte offset for the referenced member
        Instruction* curr_idx_inst = get_def_use_mgr()->GetDef(curr_idx_id);
        assert(curr_idx_inst->opcode() == spv::Op::OpConstant &&
               "unexpected struct index");
        uint32_t member_idx = curr_idx_inst->GetSingleWordInOperand(0);
        uint32_t member_offset = 0xdeadbeef;
        bool found = get_decoration_mgr()->FindDecoration(
            curr_ty_id, uint32_t(spv::Decoration::Offset),
            [&member_idx, &member_offset](const Instruction& deco_inst) {
              if (deco_inst.GetSingleWordInOperand(1u) != member_idx)
                return false;
              member_offset = deco_inst.GetSingleWordInOperand(3u);
              return true;
            });
        USE_ASSERT(found && "member offset not found");
        curr_offset_id = builder->GetUintConstantId(member_offset);
        // Look for matrix stride for this member if there is one. The matrix
        // stride is not on the matrix type, but in a OpMemberDecorate on the
        // enclosing struct type at the member index. If none found, reset
        // stride to 0.
        found = get_decoration_mgr()->FindDecoration(
            curr_ty_id, uint32_t(spv::Decoration::MatrixStride),
            [&member_idx, &matrix_stride](const Instruction& deco_inst) {
              if (deco_inst.GetSingleWordInOperand(1u) != member_idx)
                return false;
              matrix_stride = deco_inst.GetSingleWordInOperand(3u);
              return true;
            });
        if (!found) matrix_stride = 0;
        // Look for column major decoration
        found = get_decoration_mgr()->FindDecoration(
            curr_ty_id, uint32_t(spv::Decoration::ColMajor),
            [&member_idx, &col_major](const Instruction& deco_inst) {
              if (deco_inst.GetSingleWordInOperand(1u) != member_idx)
                return false;
              col_major = true;
              return true;
            });
        if (!found) col_major = false;
        // Get element type for next step
        curr_ty_id = curr_ty_inst->GetSingleWordInOperand(member_idx);
      } break;
      default: { assert(false && "unexpected non-composite type"); } break;
    }
    if (sum_id == 0)
      sum_id = curr_offset_id;
    else {
      Instruction* sum_inst =
          builder->AddIAdd(GetUintId(), sum_id, curr_offset_id);
      sum_id = sum_inst->result_id();
    }
    ++ac_in_idx;
  }
  // Add in offset of last byte of referenced object
  uint32_t bsize = ByteSize(curr_ty_id, matrix_stride, col_major, in_matrix);
  uint32_t last = bsize - 1;
  uint32_t last_id = builder->GetUintConstantId(last);
  Instruction* sum_inst = builder->AddIAdd(GetUintId(), sum_id, last_id);
  return sum_inst->result_id();
}

void InstBindlessCheckPass::GenCheckCode(
    uint32_t check_id, uint32_t error_id, uint32_t offset_id,
    uint32_t length_id, uint32_t stage_idx, RefAnalysis* ref,
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
  // Gen valid bounds branch
  std::unique_ptr<BasicBlock> new_blk_ptr(
      new BasicBlock(std::move(valid_label)));
  builder.SetInsertPoint(&*new_blk_ptr);
  uint32_t new_ref_id = CloneOriginalReference(ref, &builder);
  uint32_t null_id = 0;
  uint32_t ref_type_id = ref->ref_inst->type_id();
  (void)builder.AddBranch(merge_blk_id);
  new_blocks->push_back(std::move(new_blk_ptr));
  // Gen invalid block
  new_blk_ptr.reset(new BasicBlock(std::move(invalid_label)));
  builder.SetInsertPoint(&*new_blk_ptr);
  uint32_t u_index_id = GenUintCastCode(ref->desc_idx_id, &builder);
  if (offset_id != 0) {
    // Buffer OOB
    uint32_t u_offset_id = GenUintCastCode(offset_id, &builder);
    uint32_t u_length_id = GenUintCastCode(length_id, &builder);
    GenDebugStreamWrite(uid2offset_[ref->ref_inst->unique_id()], stage_idx,
                        {error_id, u_index_id, u_offset_id, u_length_id},
                        &builder);
  } else if (buffer_bounds_enabled_ || texel_buffer_enabled_) {
    // Uninitialized Descriptor - Return additional unused zero so all error
    // modes will use same debug stream write function
    uint32_t u_length_id = GenUintCastCode(length_id, &builder);
    GenDebugStreamWrite(
        uid2offset_[ref->ref_inst->unique_id()], stage_idx,
        {error_id, u_index_id, u_length_id, builder.GetUintConstantId(0)},
        &builder);
  } else {
    // Uninitialized Descriptor - Normal error return
    uint32_t u_length_id = GenUintCastCode(length_id, &builder);
    GenDebugStreamWrite(uid2offset_[ref->ref_inst->unique_id()], stage_idx,
                        {error_id, u_index_id, u_length_id}, &builder);
  }
  // Generate a ConstantNull, converting to uint64 if the type cannot be a null.
  if (new_ref_id != 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Type* ref_type = type_mgr->GetType(ref_type_id);
    if (ref_type->AsPointer() != nullptr) {
      context()->AddCapability(spv::Capability::Int64);
      uint32_t null_u64_id = GetNullId(GetUint64Id());
      Instruction* null_ptr_inst = builder.AddUnaryOp(
          ref_type_id, spv::Op::OpConvertUToPtr, null_u64_id);
      null_id = null_ptr_inst->result_id();
    } else {
      null_id = GetNullId(ref_type_id);
    }
  }
  // Remember last invalid block id
  uint32_t last_invalid_blk_id = new_blk_ptr->GetLabelInst()->result_id();
  // Gen zero for invalid  reference
  (void)builder.AddBranch(merge_blk_id);
  new_blocks->push_back(std::move(new_blk_ptr));
  // Gen merge block
  new_blk_ptr.reset(new BasicBlock(std::move(merge_label)));
  builder.SetInsertPoint(&*new_blk_ptr);
  // Gen phi of new reference and zero, if necessary, and replace the
  // result id of the original reference with that of the Phi. Kill original
  // reference.
  if (new_ref_id != 0) {
    Instruction* phi_inst = builder.AddPhi(
        ref_type_id, {new_ref_id, valid_blk_id, null_id, last_invalid_blk_id});
    context()->ReplaceAllUsesWith(ref->ref_inst->result_id(),
                                  phi_inst->result_id());
  }
  new_blocks->push_back(std::move(new_blk_ptr));
  context()->KillInst(ref->ref_inst);
}

void InstBindlessCheckPass::GenDescIdxCheckCode(
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  // Look for reference through indexed descriptor. If found, analyze and
  // save components. If not, return.
  RefAnalysis ref;
  if (!AnalyzeDescriptorReference(&*ref_inst_itr, &ref)) return;
  Instruction* ptr_inst = get_def_use_mgr()->GetDef(ref.ptr_id);
  if (ptr_inst->opcode() != spv::Op::OpAccessChain) return;
  // If index and bound both compile-time constants and index < bound,
  // return without changing
  Instruction* var_inst = get_def_use_mgr()->GetDef(ref.var_id);
  Instruction* desc_type_inst = GetPointeeTypeInst(var_inst);
  uint32_t length_id = 0;
  if (desc_type_inst->opcode() == spv::Op::OpTypeArray) {
    length_id =
        desc_type_inst->GetSingleWordInOperand(kSpvTypeArrayLengthIdInIdx);
    Instruction* index_inst = get_def_use_mgr()->GetDef(ref.desc_idx_id);
    Instruction* length_inst = get_def_use_mgr()->GetDef(length_id);
    if (index_inst->opcode() == spv::Op::OpConstant &&
        length_inst->opcode() == spv::Op::OpConstant &&
        index_inst->GetSingleWordInOperand(kSpvConstantValueInIdx) <
            length_inst->GetSingleWordInOperand(kSpvConstantValueInIdx))
      return;
  } else if (!desc_idx_enabled_ ||
             desc_type_inst->opcode() != spv::Op::OpTypeRuntimeArray) {
    return;
  }
  // Move original block's preceding instructions into first new block
  std::unique_ptr<BasicBlock> new_blk_ptr;
  MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
  InstructionBuilder builder(
      context(), &*new_blk_ptr,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  new_blocks->push_back(std::move(new_blk_ptr));
  uint32_t error_id = builder.GetUintConstantId(kInstErrorBindlessBounds);
  // If length id not yet set, descriptor array is runtime size so
  // generate load of length from stage's debug input buffer.
  if (length_id == 0) {
    assert(desc_type_inst->opcode() == spv::Op::OpTypeRuntimeArray &&
           "unexpected bindless type");
    length_id = GenDebugReadLength(ref.var_id, &builder);
  }
  // Generate full runtime bounds test code with true branch
  // being full reference and false branch being debug output and zero
  // for the referenced value.
  uint32_t desc_idx_32b_id = Gen32BitCvtCode(ref.desc_idx_id, &builder);
  uint32_t length_32b_id = Gen32BitCvtCode(length_id, &builder);
  Instruction* ult_inst = builder.AddBinaryOp(GetBoolId(), spv::Op::OpULessThan,
                                              desc_idx_32b_id, length_32b_id);
  ref.desc_idx_id = desc_idx_32b_id;
  GenCheckCode(ult_inst->result_id(), error_id, 0u, length_id, stage_idx, &ref,
               new_blocks);
  // Move original block's remaining code into remainder/merge block and add
  // to new blocks
  BasicBlock* back_blk_ptr = &*new_blocks->back();
  MovePostludeCode(ref_block_itr, back_blk_ptr);
}

void InstBindlessCheckPass::GenDescInitCheckCode(
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  // Look for reference through descriptor. If not, return.
  RefAnalysis ref;
  if (!AnalyzeDescriptorReference(&*ref_inst_itr, &ref)) return;
  // Determine if we can only do initialization check
  bool init_check = false;
  if (ref.desc_load_id != 0 || !buffer_bounds_enabled_) {
    init_check = true;
  } else {
    // For now, only do bounds check for non-aggregate types. Otherwise
    // just do descriptor initialization check.
    // TODO(greg-lunarg): Do bounds check for aggregate loads and stores
    Instruction* ref_ptr_inst = get_def_use_mgr()->GetDef(ref.ptr_id);
    Instruction* pte_type_inst = GetPointeeTypeInst(ref_ptr_inst);
    spv::Op pte_type_op = pte_type_inst->opcode();
    if (pte_type_op == spv::Op::OpTypeArray ||
        pte_type_op == spv::Op::OpTypeRuntimeArray ||
        pte_type_op == spv::Op::OpTypeStruct)
      init_check = true;
  }
  // If initialization check and not enabled, return
  if (init_check && !desc_init_enabled_) return;
  // Move original block's preceding instructions into first new block
  std::unique_ptr<BasicBlock> new_blk_ptr;
  MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
  InstructionBuilder builder(
      context(), &*new_blk_ptr,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  new_blocks->push_back(std::move(new_blk_ptr));
  // If initialization check, use reference value of zero.
  // Else use the index of the last byte referenced.
  uint32_t ref_id = init_check ? builder.GetUintConstantId(0u)
                               : GenLastByteIdx(&ref, &builder);
  // Read initialization/bounds from debug input buffer. If index id not yet
  // set, binding is single descriptor, so set index to constant 0.
  if (ref.desc_idx_id == 0) ref.desc_idx_id = builder.GetUintConstantId(0u);
  uint32_t init_id = GenDebugReadInit(ref.var_id, ref.desc_idx_id, &builder);
  // Generate runtime initialization/bounds test code with true branch
  // being full reference and false branch being debug output and zero
  // for the referenced value.
  Instruction* ult_inst =
      builder.AddBinaryOp(GetBoolId(), spv::Op::OpULessThan, ref_id, init_id);
  uint32_t error =
      init_check
          ? kInstErrorBindlessUninit
          : (spv::StorageClass(ref.strg_class) == spv::StorageClass::Uniform
                 ? kInstErrorBuffOOBUniform
                 : kInstErrorBuffOOBStorage);
  uint32_t error_id = builder.GetUintConstantId(error);
  GenCheckCode(ult_inst->result_id(), error_id, init_check ? 0 : ref_id,
               init_check ? builder.GetUintConstantId(0u) : init_id, stage_idx,
               &ref, new_blocks);
  // Move original block's remaining code into remainder/merge block and add
  // to new blocks
  BasicBlock* back_blk_ptr = &*new_blocks->back();
  MovePostludeCode(ref_block_itr, back_blk_ptr);
}

void InstBindlessCheckPass::GenTexBuffCheckCode(
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  // Only process OpImageRead and OpImageWrite with no optional operands
  Instruction* ref_inst = &*ref_inst_itr;
  spv::Op op = ref_inst->opcode();
  uint32_t num_in_oprnds = ref_inst->NumInOperands();
  if (!((op == spv::Op::OpImageRead && num_in_oprnds == 2) ||
        (op == spv::Op::OpImageFetch && num_in_oprnds == 2) ||
        (op == spv::Op::OpImageWrite && num_in_oprnds == 3)))
    return;
  // Pull components from descriptor reference
  RefAnalysis ref;
  if (!AnalyzeDescriptorReference(ref_inst, &ref)) return;
  // Only process if image is texel buffer
  Instruction* image_inst = get_def_use_mgr()->GetDef(ref.image_id);
  uint32_t image_ty_id = image_inst->type_id();
  Instruction* image_ty_inst = get_def_use_mgr()->GetDef(image_ty_id);
  if (spv::Dim(image_ty_inst->GetSingleWordInOperand(kSpvTypeImageDim)) !=
      spv::Dim::Buffer) {
    return;
  }
  if (image_ty_inst->GetSingleWordInOperand(kSpvTypeImageDepth) != 0) return;
  if (image_ty_inst->GetSingleWordInOperand(kSpvTypeImageArrayed) != 0) return;
  if (image_ty_inst->GetSingleWordInOperand(kSpvTypeImageMS) != 0) return;
  // Enable ImageQuery Capability if not yet enabled
  context()->AddCapability(spv::Capability::ImageQuery);
  // Move original block's preceding instructions into first new block
  std::unique_ptr<BasicBlock> new_blk_ptr;
  MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
  InstructionBuilder builder(
      context(), &*new_blk_ptr,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  new_blocks->push_back(std::move(new_blk_ptr));
  // Get texel coordinate
  uint32_t coord_id =
      GenUintCastCode(ref_inst->GetSingleWordInOperand(1), &builder);
  // If index id not yet set, binding is single descriptor, so set index to
  // constant 0.
  if (ref.desc_idx_id == 0) ref.desc_idx_id = builder.GetUintConstantId(0u);
  // Get texel buffer size.
  Instruction* size_inst =
      builder.AddUnaryOp(GetUintId(), spv::Op::OpImageQuerySize, ref.image_id);
  uint32_t size_id = size_inst->result_id();
  // Generate runtime initialization/bounds test code with true branch
  // being full reference and false branch being debug output and zero
  // for the referenced value.
  Instruction* ult_inst =
      builder.AddBinaryOp(GetBoolId(), spv::Op::OpULessThan, coord_id, size_id);
  uint32_t error =
      (image_ty_inst->GetSingleWordInOperand(kSpvTypeImageSampled) == 2)
          ? kInstErrorBuffOOBStorageTexel
          : kInstErrorBuffOOBUniformTexel;
  uint32_t error_id = builder.GetUintConstantId(error);
  GenCheckCode(ult_inst->result_id(), error_id, coord_id, size_id, stage_idx,
               &ref, new_blocks);
  // Move original block's remaining code into remainder/merge block and add
  // to new blocks
  BasicBlock* back_blk_ptr = &*new_blocks->back();
  MovePostludeCode(ref_block_itr, back_blk_ptr);
}

void InstBindlessCheckPass::InitializeInstBindlessCheck() {
  // Initialize base class
  InitializeInstrument();
  // If runtime array length support or buffer bounds checking are enabled,
  // create variable mappings. Length support is always enabled if descriptor
  // init check is enabled.
  if (desc_idx_enabled_ || buffer_bounds_enabled_ || texel_buffer_enabled_)
    for (auto& anno : get_module()->annotations())
      if (anno.opcode() == spv::Op::OpDecorate) {
        if (spv::Decoration(anno.GetSingleWordInOperand(1u)) ==
            spv::Decoration::DescriptorSet) {
          var2desc_set_[anno.GetSingleWordInOperand(0u)] =
              anno.GetSingleWordInOperand(2u);
        } else if (spv::Decoration(anno.GetSingleWordInOperand(1u)) ==
                   spv::Decoration::Binding) {
          var2binding_[anno.GetSingleWordInOperand(0u)] =
              anno.GetSingleWordInOperand(2u);
        }
      }
}

Pass::Status InstBindlessCheckPass::ProcessImpl() {
  // Perform bindless bounds check on each entry point function in module
  InstProcessFunction pfn =
      [this](BasicBlock::iterator ref_inst_itr,
             UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
             std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
        return GenDescIdxCheckCode(ref_inst_itr, ref_block_itr, stage_idx,
                                   new_blocks);
      };
  bool modified = InstProcessEntryPointCallTree(pfn);
  if (desc_init_enabled_ || buffer_bounds_enabled_) {
    // Perform descriptor initialization and/or buffer bounds check on each
    // entry point function in module
    pfn = [this](BasicBlock::iterator ref_inst_itr,
                 UptrVectorIterator<BasicBlock> ref_block_itr,
                 uint32_t stage_idx,
                 std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
      return GenDescInitCheckCode(ref_inst_itr, ref_block_itr, stage_idx,
                                  new_blocks);
    };
    modified |= InstProcessEntryPointCallTree(pfn);
  }
  if (texel_buffer_enabled_) {
    // Perform texel buffer bounds check on each entry point function in
    // module. Generate after descriptor bounds and initialization checks.
    pfn = [this](BasicBlock::iterator ref_inst_itr,
                 UptrVectorIterator<BasicBlock> ref_block_itr,
                 uint32_t stage_idx,
                 std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
      return GenTexBuffCheckCode(ref_inst_itr, ref_block_itr, stage_idx,
                                 new_blocks);
    };
    modified |= InstProcessEntryPointCallTree(pfn);
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Pass::Status InstBindlessCheckPass::Process() {
  InitializeInstBindlessCheck();
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
