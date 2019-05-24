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

#include "instrument_pass.h"

#include "source/cfa.h"

namespace {

// Common Parameter Positions
static const int kInstCommonParamInstIdx = 0;
static const int kInstCommonParamCnt = 1;

// Indices of operands in SPIR-V instructions
static const int kEntryPointExecutionModelInIdx = 0;
static const int kEntryPointFunctionIdInIdx = 1;

}  // anonymous namespace

namespace spvtools {
namespace opt {

void InstrumentPass::MovePreludeCode(
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  same_block_pre_.clear();
  same_block_post_.clear();
  // Initialize new block. Reuse label from original block.
  new_blk_ptr->reset(new BasicBlock(std::move(ref_block_itr->GetLabel())));
  // Move contents of original ref block up to ref instruction.
  for (auto cii = ref_block_itr->begin(); cii != ref_inst_itr;
       cii = ref_block_itr->begin()) {
    Instruction* inst = &*cii;
    inst->RemoveFromList();
    std::unique_ptr<Instruction> mv_ptr(inst);
    // Remember same-block ops for possible regeneration.
    if (IsSameBlockOp(&*mv_ptr)) {
      auto* sb_inst_ptr = mv_ptr.get();
      same_block_pre_[mv_ptr->result_id()] = sb_inst_ptr;
    }
    (*new_blk_ptr)->AddInstruction(std::move(mv_ptr));
  }
}

void InstrumentPass::MovePostludeCode(
    UptrVectorIterator<BasicBlock> ref_block_itr, BasicBlock* new_blk_ptr) {
  // new_blk_ptr->reset(new BasicBlock(NewLabel(ref_block_itr->id())));
  // Move contents of original ref block.
  for (auto cii = ref_block_itr->begin(); cii != ref_block_itr->end();
       cii = ref_block_itr->begin()) {
    Instruction* inst = &*cii;
    inst->RemoveFromList();
    std::unique_ptr<Instruction> mv_inst(inst);
    // Regenerate any same-block instruction that has not been seen in the
    // current block.
    if (same_block_pre_.size() > 0) {
      CloneSameBlockOps(&mv_inst, &same_block_post_, &same_block_pre_,
                        new_blk_ptr);
      // Remember same-block ops in this block.
      if (IsSameBlockOp(&*mv_inst)) {
        const uint32_t rid = mv_inst->result_id();
        same_block_post_[rid] = rid;
      }
    }
    new_blk_ptr->AddInstruction(std::move(mv_inst));
  }
}

std::unique_ptr<Instruction> InstrumentPass::NewLabel(uint32_t label_id) {
  std::unique_ptr<Instruction> newLabel(
      new Instruction(context(), SpvOpLabel, 0, label_id, {}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newLabel);
  return newLabel;
}

uint32_t InstrumentPass::GenUintCastCode(uint32_t val_id,
                                         InstructionBuilder* builder) {
  // Cast value to 32-bit unsigned if necessary
  if (get_def_use_mgr()->GetDef(val_id)->type_id() == GetUintId())
    return val_id;
  return builder->AddUnaryOp(GetUintId(), SpvOpBitcast, val_id)->result_id();
}

void InstrumentPass::GenDebugOutputFieldCode(uint32_t base_offset_id,
                                             uint32_t field_offset,
                                             uint32_t field_value_id,
                                             InstructionBuilder* builder) {
  // Cast value to 32-bit unsigned if necessary
  uint32_t val_id = GenUintCastCode(field_value_id, builder);
  // Store value
  Instruction* data_idx_inst =
      builder->AddBinaryOp(GetUintId(), SpvOpIAdd, base_offset_id,
                           builder->GetUintConstantId(field_offset));
  uint32_t buf_id = GetOutputBufferId();
  uint32_t buf_uint_ptr_id = GetBufferUintPtrId();
  Instruction* achain_inst =
      builder->AddTernaryOp(buf_uint_ptr_id, SpvOpAccessChain, buf_id,
                            builder->GetUintConstantId(kDebugOutputDataOffset),
                            data_idx_inst->result_id());
  (void)builder->AddBinaryOp(0, SpvOpStore, achain_inst->result_id(), val_id);
}

void InstrumentPass::GenCommonStreamWriteCode(uint32_t record_sz,
                                              uint32_t inst_id,
                                              uint32_t stage_idx,
                                              uint32_t base_offset_id,
                                              InstructionBuilder* builder) {
  // Store record size
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutSize,
                          builder->GetUintConstantId(record_sz), builder);
  // Store Shader Id
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutShaderId,
                          builder->GetUintConstantId(shader_id_), builder);
  // Store Instruction Idx
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutInstructionIdx, inst_id,
                          builder);
  // Store Stage Idx
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutStageIdx,
                          builder->GetUintConstantId(stage_idx), builder);
}

void InstrumentPass::GenFragCoordEltDebugOutputCode(
    uint32_t base_offset_id, uint32_t uint_frag_coord_id, uint32_t element,
    InstructionBuilder* builder) {
  Instruction* element_val_inst = builder->AddIdLiteralOp(
      GetUintId(), SpvOpCompositeExtract, uint_frag_coord_id, element);
  GenDebugOutputFieldCode(base_offset_id, kInstFragOutFragCoordX + element,
                          element_val_inst->result_id(), builder);
}

void InstrumentPass::GenBuiltinOutputCode(uint32_t builtin_id,
                                          uint32_t builtin_off,
                                          uint32_t base_offset_id,
                                          InstructionBuilder* builder) {
  // Load and store builtin
  Instruction* load_inst =
      builder->AddUnaryOp(GetUintId(), SpvOpLoad, builtin_id);
  GenDebugOutputFieldCode(base_offset_id, builtin_off, load_inst->result_id(),
                          builder);
}

void InstrumentPass::GenUintNullOutputCode(uint32_t field_off,
                                           uint32_t base_offset_id,
                                           InstructionBuilder* builder) {
  GenDebugOutputFieldCode(base_offset_id, field_off,
                          builder->GetNullId(GetUintId()), builder);
}

void InstrumentPass::GenStageStreamWriteCode(uint32_t stage_idx,
                                             uint32_t base_offset_id,
                                             InstructionBuilder* builder) {
  // TODO(greg-lunarg): Add support for all stages
  switch (stage_idx) {
    case SpvExecutionModelVertex: {
      // Load and store VertexId and InstanceId
      GenBuiltinOutputCode(context()->GetBuiltinVarId(SpvBuiltInVertexIndex),
                           kInstVertOutVertexIndex, base_offset_id, builder);
      GenBuiltinOutputCode(context()->GetBuiltinVarId(SpvBuiltInInstanceIndex),
                           kInstVertOutInstanceIndex, base_offset_id, builder);
    } break;
    case SpvExecutionModelGLCompute: {
      // Load and store GlobalInvocationId. Second word is unused; store zero.
      GenBuiltinOutputCode(
          context()->GetBuiltinVarId(SpvBuiltInGlobalInvocationId),
          kInstCompOutGlobalInvocationId, base_offset_id, builder);
      GenUintNullOutputCode(kInstCompOutUnused, base_offset_id, builder);
    } break;
    case SpvExecutionModelGeometry: {
      // Load and store PrimitiveId and InvocationId.
      GenBuiltinOutputCode(context()->GetBuiltinVarId(SpvBuiltInPrimitiveId),
                           kInstGeomOutPrimitiveId, base_offset_id, builder);
      GenBuiltinOutputCode(context()->GetBuiltinVarId(SpvBuiltInInvocationId),
                           kInstGeomOutInvocationId, base_offset_id, builder);
    } break;
    case SpvExecutionModelTessellationControl:
    case SpvExecutionModelTessellationEvaluation: {
      // Load and store InvocationId. Second word is unused; store zero.
      GenBuiltinOutputCode(context()->GetBuiltinVarId(SpvBuiltInInvocationId),
                           kInstTessOutInvocationId, base_offset_id, builder);
      GenUintNullOutputCode(kInstTessOutUnused, base_offset_id, builder);
    } break;
    case SpvExecutionModelFragment: {
      // Load FragCoord and convert to Uint
      Instruction* frag_coord_inst =
          builder->AddUnaryOp(GetVec4FloatId(), SpvOpLoad,
                              context()->GetBuiltinVarId(SpvBuiltInFragCoord));
      Instruction* uint_frag_coord_inst = builder->AddUnaryOp(
          GetVec4UintId(), SpvOpBitcast, frag_coord_inst->result_id());
      for (uint32_t u = 0; u < 2u; ++u)
        GenFragCoordEltDebugOutputCode(
            base_offset_id, uint_frag_coord_inst->result_id(), u, builder);
    } break;
    default: { assert(false && "unsupported stage"); } break;
  }
}

void InstrumentPass::GenDebugStreamWrite(
    uint32_t instruction_idx, uint32_t stage_idx,
    const std::vector<uint32_t>& validation_ids, InstructionBuilder* builder) {
  // Call debug output function. Pass func_idx, instruction_idx and
  // validation ids as args.
  uint32_t val_id_cnt = static_cast<uint32_t>(validation_ids.size());
  uint32_t output_func_id = GetStreamWriteFunctionId(stage_idx, val_id_cnt);
  std::vector<uint32_t> args = {output_func_id,
                                builder->GetUintConstantId(instruction_idx)};
  (void)args.insert(args.end(), validation_ids.begin(), validation_ids.end());
  (void)builder->AddNaryOp(GetVoidId(), SpvOpFunctionCall, args);
}

uint32_t InstrumentPass::GenDebugDirectRead(
    const std::vector<uint32_t>& offset_ids, InstructionBuilder* builder) {
  // Call debug input function. Pass func_idx and offset ids as args.
  uint32_t off_id_cnt = static_cast<uint32_t>(offset_ids.size());
  uint32_t input_func_id = GetDirectReadFunctionId(off_id_cnt);
  std::vector<uint32_t> args = {input_func_id};
  (void)args.insert(args.end(), offset_ids.begin(), offset_ids.end());
  return builder->AddNaryOp(GetUintId(), SpvOpFunctionCall, args)->result_id();
}

bool InstrumentPass::IsSameBlockOp(const Instruction* inst) const {
  return inst->opcode() == SpvOpSampledImage || inst->opcode() == SpvOpImage;
}

void InstrumentPass::CloneSameBlockOps(
    std::unique_ptr<Instruction>* inst,
    std::unordered_map<uint32_t, uint32_t>* same_blk_post,
    std::unordered_map<uint32_t, Instruction*>* same_blk_pre,
    BasicBlock* block_ptr) {
  (*inst)->ForEachInId(
      [&same_blk_post, &same_blk_pre, &block_ptr, this](uint32_t* iid) {
        const auto map_itr = (*same_blk_post).find(*iid);
        if (map_itr == (*same_blk_post).end()) {
          const auto map_itr2 = (*same_blk_pre).find(*iid);
          if (map_itr2 != (*same_blk_pre).end()) {
            // Clone pre-call same-block ops, map result id.
            const Instruction* in_inst = map_itr2->second;
            std::unique_ptr<Instruction> sb_inst(in_inst->Clone(context()));
            CloneSameBlockOps(&sb_inst, same_blk_post, same_blk_pre, block_ptr);
            const uint32_t rid = sb_inst->result_id();
            const uint32_t nid = this->TakeNextId();
            get_decoration_mgr()->CloneDecorations(rid, nid);
            sb_inst->SetResultId(nid);
            (*same_blk_post)[rid] = nid;
            *iid = nid;
            block_ptr->AddInstruction(std::move(sb_inst));
          }
        } else {
          // Reset same-block op operand.
          *iid = map_itr->second;
        }
      });
}

void InstrumentPass::UpdateSucceedingPhis(
    std::vector<std::unique_ptr<BasicBlock>>& new_blocks) {
  const auto first_blk = new_blocks.begin();
  const auto last_blk = new_blocks.end() - 1;
  const uint32_t first_id = (*first_blk)->id();
  const uint32_t last_id = (*last_blk)->id();
  const BasicBlock& const_last_block = *last_blk->get();
  const_last_block.ForEachSuccessorLabel(
      [&first_id, &last_id, this](const uint32_t succ) {
        BasicBlock* sbp = this->id2block_[succ];
        sbp->ForEachPhiInst([&first_id, &last_id, this](Instruction* phi) {
          bool changed = false;
          phi->ForEachInId([&first_id, &last_id, &changed](uint32_t* id) {
            if (*id == first_id) {
              *id = last_id;
              changed = true;
            }
          });
          if (changed) get_def_use_mgr()->AnalyzeInstUse(phi);
        });
      });
}

// Return id for output buffer uint ptr type
uint32_t InstrumentPass::GetBufferUintPtrId() {
  if (buffer_uint_ptr_id_ == 0) {
    buffer_uint_ptr_id_ = context()->get_type_mgr()->FindPointerToType(
        GetUintId(), SpvStorageClassStorageBuffer);
  }
  return buffer_uint_ptr_id_;
}

uint32_t InstrumentPass::GetOutputBufferBinding() {
  switch (validation_id_) {
    case kInstValidationIdBindless:
      return kDebugOutputBindingStream;
    default:
      assert(false && "unexpected validation id");
  }
  return 0;
}

uint32_t InstrumentPass::GetInputBufferBinding() {
  switch (validation_id_) {
    case kInstValidationIdBindless:
      return kDebugInputBindingBindless;
    default:
      assert(false && "unexpected validation id");
  }
  return 0;
}

analysis::Type* InstrumentPass::GetUintRuntimeArrayType(
    analysis::DecorationManager* deco_mgr, analysis::TypeManager* type_mgr) {
  if (uint_rarr_ty_ == nullptr) {
    analysis::Integer uint_ty(32, false);
    analysis::Type* reg_uint_ty = type_mgr->GetRegisteredType(&uint_ty);
    analysis::RuntimeArray uint_rarr_ty_tmp(reg_uint_ty);
    uint_rarr_ty_ = type_mgr->GetRegisteredType(&uint_rarr_ty_tmp);
    uint32_t uint_arr_ty_id = type_mgr->GetTypeInstruction(uint_rarr_ty_);
    // By the Vulkan spec, a pre-existing RuntimeArray of uint must be part of
    // a block, and will therefore be decorated with an ArrayStride. Therefore
    // the undecorated type returned here will not be pre-existing and can
    // safely be decorated. Since this type is now decorated, it is out of
    // sync with the TypeManager and therefore the TypeManager must be
    // invalidated after this pass.
    assert(context()->get_def_use_mgr()->NumUses(uint_arr_ty_id) == 0 &&
           "used RuntimeArray type returned");
    deco_mgr->AddDecorationVal(uint_arr_ty_id, SpvDecorationArrayStride, 4u);
  }
  return uint_rarr_ty_;
}

void InstrumentPass::AddStorageBufferExt() {
  if (storage_buffer_ext_defined_) return;
  if (!get_feature_mgr()->HasExtension(kSPV_KHR_storage_buffer_storage_class)) {
    const std::string ext_name("SPV_KHR_storage_buffer_storage_class");
    const auto num_chars = ext_name.size();
    // Compute num words, accommodate the terminating null character.
    const auto num_words = (num_chars + 1 + 3) / 4;
    std::vector<uint32_t> ext_words(num_words, 0u);
    std::memcpy(ext_words.data(), ext_name.data(), num_chars);
    context()->AddExtension(std::unique_ptr<Instruction>(
        new Instruction(context(), SpvOpExtension, 0u, 0u,
                        {{SPV_OPERAND_TYPE_LITERAL_STRING, ext_words}})));
  }
  storage_buffer_ext_defined_ = true;
}

// Return id for output buffer
uint32_t InstrumentPass::GetOutputBufferId() {
  if (output_buffer_id_ == 0) {
    // If not created yet, create one
    analysis::DecorationManager* deco_mgr = get_decoration_mgr();
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Type* reg_uint_rarr_ty =
        GetUintRuntimeArrayType(deco_mgr, type_mgr);
    analysis::Integer uint_ty(32, false);
    analysis::Type* reg_uint_ty = type_mgr->GetRegisteredType(&uint_ty);
    analysis::Struct buf_ty({reg_uint_ty, reg_uint_rarr_ty});
    analysis::Type* reg_buf_ty = type_mgr->GetRegisteredType(&buf_ty);
    uint32_t obufTyId = type_mgr->GetTypeInstruction(reg_buf_ty);
    // By the Vulkan spec, a pre-existing struct containing a RuntimeArray
    // must be a block, and will therefore be decorated with Block. Therefore
    // the undecorated type returned here will not be pre-existing and can
    // safely be decorated. Since this type is now decorated, it is out of
    // sync with the TypeManager and therefore the TypeManager must be
    // invalidated after this pass.
    assert(context()->get_def_use_mgr()->NumUses(obufTyId) == 0 &&
           "used struct type returned");
    deco_mgr->AddDecoration(obufTyId, SpvDecorationBlock);
    deco_mgr->AddMemberDecoration(obufTyId, kDebugOutputSizeOffset,
                                  SpvDecorationOffset, 0);
    deco_mgr->AddMemberDecoration(obufTyId, kDebugOutputDataOffset,
                                  SpvDecorationOffset, 4);
    uint32_t obufTyPtrId_ =
        type_mgr->FindPointerToType(obufTyId, SpvStorageClassStorageBuffer);
    output_buffer_id_ = TakeNextId();
    std::unique_ptr<Instruction> newVarOp(new Instruction(
        context(), SpvOpVariable, obufTyPtrId_, output_buffer_id_,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          {SpvStorageClassStorageBuffer}}}));
    context()->AddGlobalValue(std::move(newVarOp));
    deco_mgr->AddDecorationVal(output_buffer_id_, SpvDecorationDescriptorSet,
                               desc_set_);
    deco_mgr->AddDecorationVal(output_buffer_id_, SpvDecorationBinding,
                               GetOutputBufferBinding());
    AddStorageBufferExt();
  }
  return output_buffer_id_;
}

uint32_t InstrumentPass::GetInputBufferId() {
  if (input_buffer_id_ == 0) {
    // If not created yet, create one
    analysis::DecorationManager* deco_mgr = get_decoration_mgr();
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Type* reg_uint_rarr_ty =
        GetUintRuntimeArrayType(deco_mgr, type_mgr);
    analysis::Struct buf_ty({reg_uint_rarr_ty});
    analysis::Type* reg_buf_ty = type_mgr->GetRegisteredType(&buf_ty);
    uint32_t ibufTyId = type_mgr->GetTypeInstruction(reg_buf_ty);
    // By the Vulkan spec, a pre-existing struct containing a RuntimeArray
    // must be a block, and will therefore be decorated with Block. Therefore
    // the undecorated type returned here will not be pre-existing and can
    // safely be decorated. Since this type is now decorated, it is out of
    // sync with the TypeManager and therefore the TypeManager must be
    // invalidated after this pass.
    assert(context()->get_def_use_mgr()->NumUses(ibufTyId) == 0 &&
           "used struct type returned");
    deco_mgr->AddDecoration(ibufTyId, SpvDecorationBlock);
    deco_mgr->AddMemberDecoration(ibufTyId, 0, SpvDecorationOffset, 0);
    uint32_t ibufTyPtrId_ =
        type_mgr->FindPointerToType(ibufTyId, SpvStorageClassStorageBuffer);
    input_buffer_id_ = TakeNextId();
    std::unique_ptr<Instruction> newVarOp(new Instruction(
        context(), SpvOpVariable, ibufTyPtrId_, input_buffer_id_,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          {SpvStorageClassStorageBuffer}}}));
    context()->AddGlobalValue(std::move(newVarOp));
    deco_mgr->AddDecorationVal(input_buffer_id_, SpvDecorationDescriptorSet,
                               desc_set_);
    deco_mgr->AddDecorationVal(input_buffer_id_, SpvDecorationBinding,
                               GetInputBufferBinding());
    AddStorageBufferExt();
  }
  return input_buffer_id_;
}

uint32_t InstrumentPass::GetVec4FloatId() {
  if (v4float_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Float float_ty(32);
    analysis::Type* reg_float_ty = type_mgr->GetRegisteredType(&float_ty);
    analysis::Vector v4float_ty(reg_float_ty, 4);
    analysis::Type* reg_v4float_ty = type_mgr->GetRegisteredType(&v4float_ty);
    v4float_id_ = type_mgr->GetTypeInstruction(reg_v4float_ty);
  }
  return v4float_id_;
}

uint32_t InstrumentPass::GetUintId() {
  if (uint_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Integer uint_ty(32, false);
    analysis::Type* reg_uint_ty = type_mgr->GetRegisteredType(&uint_ty);
    uint_id_ = type_mgr->GetTypeInstruction(reg_uint_ty);
  }
  return uint_id_;
}

uint32_t InstrumentPass::GetVec4UintId() {
  if (v4uint_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Integer uint_ty(32, false);
    analysis::Type* reg_uint_ty = type_mgr->GetRegisteredType(&uint_ty);
    analysis::Vector v4uint_ty(reg_uint_ty, 4);
    analysis::Type* reg_v4uint_ty = type_mgr->GetRegisteredType(&v4uint_ty);
    v4uint_id_ = type_mgr->GetTypeInstruction(reg_v4uint_ty);
  }
  return v4uint_id_;
}

uint32_t InstrumentPass::GetBoolId() {
  if (bool_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Bool bool_ty;
    analysis::Type* reg_bool_ty = type_mgr->GetRegisteredType(&bool_ty);
    bool_id_ = type_mgr->GetTypeInstruction(reg_bool_ty);
  }
  return bool_id_;
}

uint32_t InstrumentPass::GetVoidId() {
  if (void_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Void void_ty;
    analysis::Type* reg_void_ty = type_mgr->GetRegisteredType(&void_ty);
    void_id_ = type_mgr->GetTypeInstruction(reg_void_ty);
  }
  return void_id_;
}

uint32_t InstrumentPass::GetStreamWriteFunctionId(uint32_t stage_idx,
                                                  uint32_t val_spec_param_cnt) {
  // Total param count is common params plus validation-specific
  // params
  uint32_t param_cnt = kInstCommonParamCnt + val_spec_param_cnt;
  if (output_func_id_ == 0) {
    // Create function
    output_func_id_ = TakeNextId();
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    std::vector<const analysis::Type*> param_types;
    for (uint32_t c = 0; c < param_cnt; ++c)
      param_types.push_back(type_mgr->GetType(GetUintId()));
    analysis::Function func_ty(type_mgr->GetType(GetVoidId()), param_types);
    analysis::Type* reg_func_ty = type_mgr->GetRegisteredType(&func_ty);
    std::unique_ptr<Instruction> func_inst(new Instruction(
        get_module()->context(), SpvOpFunction, GetVoidId(), output_func_id_,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          {SpvFunctionControlMaskNone}},
         {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          {type_mgr->GetTypeInstruction(reg_func_ty)}}}));
    get_def_use_mgr()->AnalyzeInstDefUse(&*func_inst);
    std::unique_ptr<Function> output_func =
        MakeUnique<Function>(std::move(func_inst));
    // Add parameters
    std::vector<uint32_t> param_vec;
    for (uint32_t c = 0; c < param_cnt; ++c) {
      uint32_t pid = TakeNextId();
      param_vec.push_back(pid);
      std::unique_ptr<Instruction> param_inst(
          new Instruction(get_module()->context(), SpvOpFunctionParameter,
                          GetUintId(), pid, {}));
      get_def_use_mgr()->AnalyzeInstDefUse(&*param_inst);
      output_func->AddParameter(std::move(param_inst));
    }
    // Create first block
    uint32_t test_blk_id = TakeNextId();
    std::unique_ptr<Instruction> test_label(NewLabel(test_blk_id));
    std::unique_ptr<BasicBlock> new_blk_ptr =
        MakeUnique<BasicBlock>(std::move(test_label));
    InstructionBuilder builder(
        context(), &*new_blk_ptr,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    // Gen test if debug output buffer size will not be exceeded.
    uint32_t obuf_record_sz = kInstStageOutCnt + val_spec_param_cnt;
    uint32_t buf_id = GetOutputBufferId();
    uint32_t buf_uint_ptr_id = GetBufferUintPtrId();
    Instruction* obuf_curr_sz_ac_inst =
        builder.AddBinaryOp(buf_uint_ptr_id, SpvOpAccessChain, buf_id,
                            builder.GetUintConstantId(kDebugOutputSizeOffset));
    // Fetch the current debug buffer written size atomically, adding the
    // size of the record to be written.
    uint32_t obuf_record_sz_id = builder.GetUintConstantId(obuf_record_sz);
    uint32_t mask_none_id = builder.GetUintConstantId(SpvMemoryAccessMaskNone);
    uint32_t scope_invok_id = builder.GetUintConstantId(SpvScopeInvocation);
    Instruction* obuf_curr_sz_inst = builder.AddQuadOp(
        GetUintId(), SpvOpAtomicIAdd, obuf_curr_sz_ac_inst->result_id(),
        scope_invok_id, mask_none_id, obuf_record_sz_id);
    uint32_t obuf_curr_sz_id = obuf_curr_sz_inst->result_id();
    // Compute new written size
    Instruction* obuf_new_sz_inst =
        builder.AddBinaryOp(GetUintId(), SpvOpIAdd, obuf_curr_sz_id,
                            builder.GetUintConstantId(obuf_record_sz));
    // Fetch the data bound
    Instruction* obuf_bnd_inst =
        builder.AddIdLiteralOp(GetUintId(), SpvOpArrayLength,
                               GetOutputBufferId(), kDebugOutputDataOffset);
    // Test that new written size is less than or equal to debug output
    // data bound
    Instruction* obuf_safe_inst = builder.AddBinaryOp(
        GetBoolId(), SpvOpULessThanEqual, obuf_new_sz_inst->result_id(),
        obuf_bnd_inst->result_id());
    uint32_t merge_blk_id = TakeNextId();
    uint32_t write_blk_id = TakeNextId();
    std::unique_ptr<Instruction> merge_label(NewLabel(merge_blk_id));
    std::unique_ptr<Instruction> write_label(NewLabel(write_blk_id));
    (void)builder.AddConditionalBranch(obuf_safe_inst->result_id(),
                                       write_blk_id, merge_blk_id, merge_blk_id,
                                       SpvSelectionControlMaskNone);
    // Close safety test block and gen write block
    new_blk_ptr->SetParent(&*output_func);
    output_func->AddBasicBlock(std::move(new_blk_ptr));
    new_blk_ptr = MakeUnique<BasicBlock>(std::move(write_label));
    builder.SetInsertPoint(&*new_blk_ptr);
    // Generate common and stage-specific debug record members
    GenCommonStreamWriteCode(obuf_record_sz, param_vec[kInstCommonParamInstIdx],
                             stage_idx, obuf_curr_sz_id, &builder);
    GenStageStreamWriteCode(stage_idx, obuf_curr_sz_id, &builder);
    // Gen writes of validation specific data
    for (uint32_t i = 0; i < val_spec_param_cnt; ++i) {
      GenDebugOutputFieldCode(obuf_curr_sz_id, kInstStageOutCnt + i,
                              param_vec[kInstCommonParamCnt + i], &builder);
    }
    // Close write block and gen merge block
    (void)builder.AddBranch(merge_blk_id);
    new_blk_ptr->SetParent(&*output_func);
    output_func->AddBasicBlock(std::move(new_blk_ptr));
    new_blk_ptr = MakeUnique<BasicBlock>(std::move(merge_label));
    builder.SetInsertPoint(&*new_blk_ptr);
    // Close merge block and function and add function to module
    (void)builder.AddNullaryOp(0, SpvOpReturn);
    new_blk_ptr->SetParent(&*output_func);
    output_func->AddBasicBlock(std::move(new_blk_ptr));
    std::unique_ptr<Instruction> func_end_inst(
        new Instruction(get_module()->context(), SpvOpFunctionEnd, 0, 0, {}));
    get_def_use_mgr()->AnalyzeInstDefUse(&*func_end_inst);
    output_func->SetFunctionEnd(std::move(func_end_inst));
    context()->AddFunction(std::move(output_func));
    output_func_param_cnt_ = param_cnt;
  }
  assert(param_cnt == output_func_param_cnt_ && "bad arg count");
  return output_func_id_;
}

uint32_t InstrumentPass::GetDirectReadFunctionId(uint32_t param_cnt) {
  uint32_t func_id = param2input_func_id_[param_cnt];
  if (func_id != 0) return func_id;
  // Create input function for param_cnt
  func_id = TakeNextId();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  std::vector<const analysis::Type*> param_types;
  for (uint32_t c = 0; c < param_cnt; ++c)
    param_types.push_back(type_mgr->GetType(GetUintId()));
  analysis::Function func_ty(type_mgr->GetType(GetUintId()), param_types);
  analysis::Type* reg_func_ty = type_mgr->GetRegisteredType(&func_ty);
  std::unique_ptr<Instruction> func_inst(new Instruction(
      get_module()->context(), SpvOpFunction, GetUintId(), func_id,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
        {SpvFunctionControlMaskNone}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
        {type_mgr->GetTypeInstruction(reg_func_ty)}}}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*func_inst);
  std::unique_ptr<Function> input_func =
      MakeUnique<Function>(std::move(func_inst));
  // Add parameters
  std::vector<uint32_t> param_vec;
  for (uint32_t c = 0; c < param_cnt; ++c) {
    uint32_t pid = TakeNextId();
    param_vec.push_back(pid);
    std::unique_ptr<Instruction> param_inst(new Instruction(
        get_module()->context(), SpvOpFunctionParameter, GetUintId(), pid, {}));
    get_def_use_mgr()->AnalyzeInstDefUse(&*param_inst);
    input_func->AddParameter(std::move(param_inst));
  }
  // Create block
  uint32_t blk_id = TakeNextId();
  std::unique_ptr<Instruction> blk_label(NewLabel(blk_id));
  std::unique_ptr<BasicBlock> new_blk_ptr =
      MakeUnique<BasicBlock>(std::move(blk_label));
  InstructionBuilder builder(
      context(), &*new_blk_ptr,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  // For each offset parameter, generate new offset with parameter, adding last
  // loaded value if it exists, and load value from input buffer at new offset.
  // Return last loaded value.
  uint32_t buf_id = GetInputBufferId();
  uint32_t buf_uint_ptr_id = GetBufferUintPtrId();
  uint32_t last_value_id = 0;
  for (uint32_t p = 0; p < param_cnt; ++p) {
    uint32_t offset_id;
    if (p == 0) {
      offset_id = param_vec[0];
    } else {
      Instruction* offset_inst = builder.AddBinaryOp(
          GetUintId(), SpvOpIAdd, last_value_id, param_vec[p]);
      offset_id = offset_inst->result_id();
    }
    Instruction* ac_inst = builder.AddTernaryOp(
        buf_uint_ptr_id, SpvOpAccessChain, buf_id,
        builder.GetUintConstantId(kDebugInputDataOffset), offset_id);
    Instruction* load_inst =
        builder.AddUnaryOp(GetUintId(), SpvOpLoad, ac_inst->result_id());
    last_value_id = load_inst->result_id();
  }
  (void)builder.AddInstruction(MakeUnique<Instruction>(
      context(), SpvOpReturnValue, 0, 0,
      std::initializer_list<Operand>{{SPV_OPERAND_TYPE_ID, {last_value_id}}}));
  // Close block and function and add function to module
  new_blk_ptr->SetParent(&*input_func);
  input_func->AddBasicBlock(std::move(new_blk_ptr));
  std::unique_ptr<Instruction> func_end_inst(
      new Instruction(get_module()->context(), SpvOpFunctionEnd, 0, 0, {}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*func_end_inst);
  input_func->SetFunctionEnd(std::move(func_end_inst));
  context()->AddFunction(std::move(input_func));
  param2input_func_id_[param_cnt] = func_id;
  return func_id;
}

bool InstrumentPass::InstrumentFunction(Function* func, uint32_t stage_idx,
                                        InstProcessFunction& pfn) {
  bool modified = false;
  // Compute function index
  uint32_t function_idx = 0;
  for (auto fii = get_module()->begin(); fii != get_module()->end(); ++fii) {
    if (&*fii == func) break;
    ++function_idx;
  }
  std::vector<std::unique_ptr<BasicBlock>> new_blks;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      // Generate instrumentation if warranted
      pfn(ii, bi, stage_idx, &new_blks);
      if (new_blks.size() == 0) {
        ++ii;
        continue;
      }
      // Add new blocks to label id map
      for (auto& blk : new_blks) id2block_[blk->id()] = &*blk;
      // If there are new blocks we know there will always be two or
      // more, so update succeeding phis with label of new last block.
      size_t newBlocksSize = new_blks.size();
      assert(newBlocksSize > 1);
      UpdateSucceedingPhis(new_blks);
      // Replace original block with new block(s)
      bi = bi.Erase();
      for (auto& bb : new_blks) {
        bb->SetParent(func);
      }
      bi = bi.InsertBefore(&new_blks);
      // Reset block iterator to last new block
      for (size_t i = 0; i < newBlocksSize - 1; i++) ++bi;
      modified = true;
      // Restart instrumenting at beginning of last new block,
      // but skip over any new phi or copy instruction.
      ii = bi->begin();
      if (ii->opcode() == SpvOpPhi || ii->opcode() == SpvOpCopyObject) ++ii;
      new_blks.clear();
    }
  }
  return modified;
}

bool InstrumentPass::InstProcessCallTreeFromRoots(InstProcessFunction& pfn,
                                                  std::queue<uint32_t>* roots,
                                                  uint32_t stage_idx) {
  bool modified = false;
  std::unordered_set<uint32_t> done;
  // Don't process input and output functions
  for (auto& ifn : param2input_func_id_) done.insert(ifn.second);
  if (output_func_id_ != 0) done.insert(output_func_id_);
  // Process all functions from roots
  while (!roots->empty()) {
    const uint32_t fi = roots->front();
    roots->pop();
    if (done.insert(fi).second) {
      Function* fn = id2function_.at(fi);
      // Add calls first so we don't add new output function
      context()->AddCalls(fn, roots);
      modified = InstrumentFunction(fn, stage_idx, pfn) || modified;
    }
  }
  return modified;
}

bool InstrumentPass::InstProcessEntryPointCallTree(InstProcessFunction& pfn) {
  // Make sure all entry points have the same execution model. Do not
  // instrument if they do not.
  // TODO(greg-lunarg): Handle mixed stages. Technically, a shader module
  // can contain entry points with different execution models, although
  // such modules will likely be rare as GLSL and HLSL are geared toward
  // one model per module. In such cases we will need
  // to clone any functions which are in the call trees of entrypoints
  // with differing execution models.
  uint32_t ecnt = 0;
  uint32_t stage = SpvExecutionModelMax;
  for (auto& e : get_module()->entry_points()) {
    if (ecnt == 0)
      stage = e.GetSingleWordInOperand(kEntryPointExecutionModelInIdx);
    else if (e.GetSingleWordInOperand(kEntryPointExecutionModelInIdx) != stage)
      return false;
    ++ecnt;
  }
  // Only supporting vertex, fragment and compute shaders at the moment.
  // TODO(greg-lunarg): Handle all stages.
  if (stage != SpvExecutionModelVertex && stage != SpvExecutionModelFragment &&
      stage != SpvExecutionModelGeometry &&
      stage != SpvExecutionModelGLCompute &&
      stage != SpvExecutionModelTessellationControl &&
      stage != SpvExecutionModelTessellationEvaluation)
    return false;
  // Add together the roots of all entry points
  std::queue<uint32_t> roots;
  for (auto& e : get_module()->entry_points()) {
    roots.push(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));
  }
  bool modified = InstProcessCallTreeFromRoots(pfn, &roots, stage);
  return modified;
}

void InstrumentPass::InitializeInstrument() {
  output_buffer_id_ = 0;
  buffer_uint_ptr_id_ = 0;
  output_func_id_ = 0;
  output_func_param_cnt_ = 0;
  input_buffer_id_ = 0;
  v4float_id_ = 0;
  uint_id_ = 0;
  v4uint_id_ = 0;
  bool_id_ = 0;
  void_id_ = 0;
  storage_buffer_ext_defined_ = false;
  uint_rarr_ty_ = nullptr;

  // clear collections
  id2function_.clear();
  id2block_.clear();

  // Initialize function and block maps.
  for (auto& fn : *get_module()) {
    id2function_[fn.result_id()] = &fn;
    for (auto& blk : fn) {
      id2block_[blk.id()] = &blk;
    }
  }

  // Remember original instruction offsets
  uint32_t module_offset = 0;
  Module* module = get_module();
  for (auto& i : context()->capabilities()) {
    (void)i;
    ++module_offset;
  }
  for (auto& i : module->extensions()) {
    (void)i;
    ++module_offset;
  }
  for (auto& i : module->ext_inst_imports()) {
    (void)i;
    ++module_offset;
  }
  ++module_offset;  // memory_model
  for (auto& i : module->entry_points()) {
    (void)i;
    ++module_offset;
  }
  for (auto& i : module->execution_modes()) {
    (void)i;
    ++module_offset;
  }
  for (auto& i : module->debugs1()) {
    (void)i;
    ++module_offset;
  }
  for (auto& i : module->debugs2()) {
    (void)i;
    ++module_offset;
  }
  for (auto& i : module->debugs3()) {
    (void)i;
    ++module_offset;
  }
  for (auto& i : module->annotations()) {
    (void)i;
    ++module_offset;
  }
  for (auto& i : module->types_values()) {
    module_offset += 1;
    module_offset += static_cast<uint32_t>(i.dbg_line_insts().size());
  }

  auto curr_fn = get_module()->begin();
  for (; curr_fn != get_module()->end(); ++curr_fn) {
    // Count function instruction
    module_offset += 1;
    curr_fn->ForEachParam(
        [&module_offset](const Instruction*) { module_offset += 1; }, true);
    for (auto& blk : *curr_fn) {
      // Count label
      module_offset += 1;
      for (auto& inst : blk) {
        module_offset += static_cast<uint32_t>(inst.dbg_line_insts().size());
        uid2offset_[inst.unique_id()] = module_offset;
        module_offset += 1;
      }
    }
    // Count function end instruction
    module_offset += 1;
  }
}

}  // namespace opt
}  // namespace spvtools
