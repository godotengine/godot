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
#include "source/spirv_constant.h"

namespace spvtools {
namespace opt {
namespace {
// Common Parameter Positions
constexpr int kInstCommonParamInstIdx = 0;
constexpr int kInstCommonParamCnt = 1;
// Indices of operands in SPIR-V instructions
constexpr int kEntryPointFunctionIdInIdx = 1;
}  // namespace

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
  auto new_label =
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{});
  get_def_use_mgr()->AnalyzeInstDefUse(&*new_label);
  return new_label;
}

std::unique_ptr<Function> InstrumentPass::StartFunction(
    uint32_t func_id, const analysis::Type* return_type,
    const std::vector<const analysis::Type*>& param_types) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Function* func_type = GetFunction(return_type, param_types);

  const std::vector<Operand> operands{
      {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
       {uint32_t(spv::FunctionControlMask::MaskNone)}},
      {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {type_mgr->GetId(func_type)}},
  };
  auto func_inst =
      MakeUnique<Instruction>(context(), spv::Op::OpFunction,
                              type_mgr->GetId(return_type), func_id, operands);
  get_def_use_mgr()->AnalyzeInstDefUse(&*func_inst);
  return MakeUnique<Function>(std::move(func_inst));
}

std::unique_ptr<Instruction> InstrumentPass::EndFunction() {
  auto end = MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                                     std::initializer_list<Operand>{});
  get_def_use_mgr()->AnalyzeInstDefUse(end.get());
  return end;
}

std::vector<uint32_t> InstrumentPass::AddParameters(
    Function& func, const std::vector<const analysis::Type*>& param_types) {
  std::vector<uint32_t> param_ids;
  param_ids.reserve(param_types.size());
  for (const analysis::Type* param : param_types) {
    uint32_t pid = TakeNextId();
    param_ids.push_back(pid);
    auto param_inst =
        MakeUnique<Instruction>(context(), spv::Op::OpFunctionParameter,
                                context()->get_type_mgr()->GetId(param), pid,
                                std::initializer_list<Operand>{});
    get_def_use_mgr()->AnalyzeInstDefUse(param_inst.get());
    func.AddParameter(std::move(param_inst));
  }
  return param_ids;
}

std::unique_ptr<Instruction> InstrumentPass::NewName(
    uint32_t id, const std::string& name_str) {
  return MakeUnique<Instruction>(
      context(), spv::Op::OpName, 0, 0,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_ID, {id}},
          {SPV_OPERAND_TYPE_LITERAL_STRING, utils::MakeVector(name_str)}});
}

std::unique_ptr<Instruction> InstrumentPass::NewGlobalName(
    uint32_t id, const std::string& name_str) {
  std::string prefixed_name;
  switch (validation_id_) {
    case kInstValidationIdBindless:
      prefixed_name = "inst_bindless_";
      break;
    case kInstValidationIdBuffAddr:
      prefixed_name = "inst_buff_addr_";
      break;
    case kInstValidationIdDebugPrintf:
      prefixed_name = "inst_printf_";
      break;
    default:
      assert(false);  // add new instrumentation pass here
      prefixed_name = "inst_pass_";
      break;
  }
  prefixed_name += name_str;
  return NewName(id, prefixed_name);
}

std::unique_ptr<Instruction> InstrumentPass::NewMemberName(
    uint32_t id, uint32_t member_index, const std::string& name_str) {
  return MakeUnique<Instruction>(
      context(), spv::Op::OpMemberName, 0, 0,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_ID, {id}},
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {member_index}},
          {SPV_OPERAND_TYPE_LITERAL_STRING, utils::MakeVector(name_str)}});
}

uint32_t InstrumentPass::Gen32BitCvtCode(uint32_t val_id,
                                         InstructionBuilder* builder) {
  // Convert integer value to 32-bit if necessary
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  uint32_t val_ty_id = get_def_use_mgr()->GetDef(val_id)->type_id();
  analysis::Integer* val_ty = type_mgr->GetType(val_ty_id)->AsInteger();
  if (val_ty->width() == 32) return val_id;
  bool is_signed = val_ty->IsSigned();
  analysis::Integer val_32b_ty(32, is_signed);
  analysis::Type* val_32b_reg_ty = type_mgr->GetRegisteredType(&val_32b_ty);
  uint32_t val_32b_reg_ty_id = type_mgr->GetId(val_32b_reg_ty);
  if (is_signed)
    return builder->AddUnaryOp(val_32b_reg_ty_id, spv::Op::OpSConvert, val_id)
        ->result_id();
  else
    return builder->AddUnaryOp(val_32b_reg_ty_id, spv::Op::OpUConvert, val_id)
        ->result_id();
}

uint32_t InstrumentPass::GenUintCastCode(uint32_t val_id,
                                         InstructionBuilder* builder) {
  // Convert value to 32-bit if necessary
  uint32_t val_32b_id = Gen32BitCvtCode(val_id, builder);
  // Cast value to unsigned if necessary
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  uint32_t val_ty_id = get_def_use_mgr()->GetDef(val_32b_id)->type_id();
  analysis::Integer* val_ty = type_mgr->GetType(val_ty_id)->AsInteger();
  if (!val_ty->IsSigned()) return val_32b_id;
  return builder->AddUnaryOp(GetUintId(), spv::Op::OpBitcast, val_32b_id)
      ->result_id();
}

void InstrumentPass::GenDebugOutputFieldCode(uint32_t base_offset_id,
                                             uint32_t field_offset,
                                             uint32_t field_value_id,
                                             InstructionBuilder* builder) {
  // Cast value to 32-bit unsigned if necessary
  uint32_t val_id = GenUintCastCode(field_value_id, builder);
  // Store value
  Instruction* data_idx_inst = builder->AddIAdd(
      GetUintId(), base_offset_id, builder->GetUintConstantId(field_offset));
  uint32_t buf_id = GetOutputBufferId();
  uint32_t buf_uint_ptr_id = GetOutputBufferPtrId();
  Instruction* achain_inst = builder->AddAccessChain(
      buf_uint_ptr_id, buf_id,
      {builder->GetUintConstantId(kDebugOutputDataOffset),
       data_idx_inst->result_id()});
  (void)builder->AddStore(achain_inst->result_id(), val_id);
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
  Instruction* element_val_inst =
      builder->AddCompositeExtract(GetUintId(), uint_frag_coord_id, {element});
  GenDebugOutputFieldCode(base_offset_id, kInstFragOutFragCoordX + element,
                          element_val_inst->result_id(), builder);
}

uint32_t InstrumentPass::GenVarLoad(uint32_t var_id,
                                    InstructionBuilder* builder) {
  Instruction* var_inst = get_def_use_mgr()->GetDef(var_id);
  uint32_t type_id = GetPointeeTypeId(var_inst);
  Instruction* load_inst = builder->AddLoad(type_id, var_id);
  return load_inst->result_id();
}

void InstrumentPass::GenBuiltinOutputCode(uint32_t builtin_id,
                                          uint32_t builtin_off,
                                          uint32_t base_offset_id,
                                          InstructionBuilder* builder) {
  // Load and store builtin
  uint32_t load_id = GenVarLoad(builtin_id, builder);
  GenDebugOutputFieldCode(base_offset_id, builtin_off, load_id, builder);
}

void InstrumentPass::GenStageStreamWriteCode(uint32_t stage_idx,
                                             uint32_t base_offset_id,
                                             InstructionBuilder* builder) {
  // TODO(greg-lunarg): Add support for all stages
  switch (spv::ExecutionModel(stage_idx)) {
    case spv::ExecutionModel::Vertex: {
      // Load and store VertexId and InstanceId
      GenBuiltinOutputCode(
          context()->GetBuiltinInputVarId(uint32_t(spv::BuiltIn::VertexIndex)),
          kInstVertOutVertexIndex, base_offset_id, builder);
      GenBuiltinOutputCode(context()->GetBuiltinInputVarId(
                               uint32_t(spv::BuiltIn::InstanceIndex)),
                           kInstVertOutInstanceIndex, base_offset_id, builder);
    } break;
    case spv::ExecutionModel::GLCompute:
    case spv::ExecutionModel::TaskNV:
    case spv::ExecutionModel::MeshNV:
    case spv::ExecutionModel::TaskEXT:
    case spv::ExecutionModel::MeshEXT: {
      // Load and store GlobalInvocationId.
      uint32_t load_id = GenVarLoad(context()->GetBuiltinInputVarId(uint32_t(
                                        spv::BuiltIn::GlobalInvocationId)),
                                    builder);
      Instruction* x_inst =
          builder->AddCompositeExtract(GetUintId(), load_id, {0});
      Instruction* y_inst =
          builder->AddCompositeExtract(GetUintId(), load_id, {1});
      Instruction* z_inst =
          builder->AddCompositeExtract(GetUintId(), load_id, {2});
      GenDebugOutputFieldCode(base_offset_id, kInstCompOutGlobalInvocationIdX,
                              x_inst->result_id(), builder);
      GenDebugOutputFieldCode(base_offset_id, kInstCompOutGlobalInvocationIdY,
                              y_inst->result_id(), builder);
      GenDebugOutputFieldCode(base_offset_id, kInstCompOutGlobalInvocationIdZ,
                              z_inst->result_id(), builder);
    } break;
    case spv::ExecutionModel::Geometry: {
      // Load and store PrimitiveId and InvocationId.
      GenBuiltinOutputCode(
          context()->GetBuiltinInputVarId(uint32_t(spv::BuiltIn::PrimitiveId)),
          kInstGeomOutPrimitiveId, base_offset_id, builder);
      GenBuiltinOutputCode(
          context()->GetBuiltinInputVarId(uint32_t(spv::BuiltIn::InvocationId)),
          kInstGeomOutInvocationId, base_offset_id, builder);
    } break;
    case spv::ExecutionModel::TessellationControl: {
      // Load and store InvocationId and PrimitiveId
      GenBuiltinOutputCode(
          context()->GetBuiltinInputVarId(uint32_t(spv::BuiltIn::InvocationId)),
          kInstTessCtlOutInvocationId, base_offset_id, builder);
      GenBuiltinOutputCode(
          context()->GetBuiltinInputVarId(uint32_t(spv::BuiltIn::PrimitiveId)),
          kInstTessCtlOutPrimitiveId, base_offset_id, builder);
    } break;
    case spv::ExecutionModel::TessellationEvaluation: {
      // Load and store PrimitiveId and TessCoord.uv
      GenBuiltinOutputCode(
          context()->GetBuiltinInputVarId(uint32_t(spv::BuiltIn::PrimitiveId)),
          kInstTessEvalOutPrimitiveId, base_offset_id, builder);
      uint32_t load_id = GenVarLoad(
          context()->GetBuiltinInputVarId(uint32_t(spv::BuiltIn::TessCoord)),
          builder);
      Instruction* uvec3_cast_inst =
          builder->AddUnaryOp(GetVec3UintId(), spv::Op::OpBitcast, load_id);
      uint32_t uvec3_cast_id = uvec3_cast_inst->result_id();
      Instruction* u_inst =
          builder->AddCompositeExtract(GetUintId(), uvec3_cast_id, {0});
      Instruction* v_inst =
          builder->AddCompositeExtract(GetUintId(), uvec3_cast_id, {1});
      GenDebugOutputFieldCode(base_offset_id, kInstTessEvalOutTessCoordU,
                              u_inst->result_id(), builder);
      GenDebugOutputFieldCode(base_offset_id, kInstTessEvalOutTessCoordV,
                              v_inst->result_id(), builder);
    } break;
    case spv::ExecutionModel::Fragment: {
      // Load FragCoord and convert to Uint
      Instruction* frag_coord_inst = builder->AddLoad(
          GetVec4FloatId(),
          context()->GetBuiltinInputVarId(uint32_t(spv::BuiltIn::FragCoord)));
      Instruction* uint_frag_coord_inst = builder->AddUnaryOp(
          GetVec4UintId(), spv::Op::OpBitcast, frag_coord_inst->result_id());
      for (uint32_t u = 0; u < 2u; ++u)
        GenFragCoordEltDebugOutputCode(
            base_offset_id, uint_frag_coord_inst->result_id(), u, builder);
    } break;
    case spv::ExecutionModel::RayGenerationNV:
    case spv::ExecutionModel::IntersectionNV:
    case spv::ExecutionModel::AnyHitNV:
    case spv::ExecutionModel::ClosestHitNV:
    case spv::ExecutionModel::MissNV:
    case spv::ExecutionModel::CallableNV: {
      // Load and store LaunchIdNV.
      uint32_t launch_id = GenVarLoad(
          context()->GetBuiltinInputVarId(uint32_t(spv::BuiltIn::LaunchIdNV)),
          builder);
      Instruction* x_launch_inst =
          builder->AddCompositeExtract(GetUintId(), launch_id, {0});
      Instruction* y_launch_inst =
          builder->AddCompositeExtract(GetUintId(), launch_id, {1});
      Instruction* z_launch_inst =
          builder->AddCompositeExtract(GetUintId(), launch_id, {2});
      GenDebugOutputFieldCode(base_offset_id, kInstRayTracingOutLaunchIdX,
                              x_launch_inst->result_id(), builder);
      GenDebugOutputFieldCode(base_offset_id, kInstRayTracingOutLaunchIdY,
                              y_launch_inst->result_id(), builder);
      GenDebugOutputFieldCode(base_offset_id, kInstRayTracingOutLaunchIdZ,
                              z_launch_inst->result_id(), builder);
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
  std::vector<uint32_t> args = {builder->GetUintConstantId(instruction_idx)};
  (void)args.insert(args.end(), validation_ids.begin(), validation_ids.end());
  (void)builder->AddFunctionCall(
      GetVoidId(), GetStreamWriteFunctionId(stage_idx, val_id_cnt), args);
}

bool InstrumentPass::AllConstant(const std::vector<uint32_t>& ids) {
  for (auto& id : ids) {
    Instruction* id_inst = context()->get_def_use_mgr()->GetDef(id);
    if (!spvOpcodeIsConstant(id_inst->opcode())) return false;
  }
  return true;
}

uint32_t InstrumentPass::GenDebugDirectRead(
    const std::vector<uint32_t>& offset_ids, InstructionBuilder* builder) {
  // Call debug input function. Pass func_idx and offset ids as args.
  const uint32_t off_id_cnt = static_cast<uint32_t>(offset_ids.size());
  const uint32_t input_func_id = GetDirectReadFunctionId(off_id_cnt);
  return GenReadFunctionCall(input_func_id, offset_ids, builder);
}

uint32_t InstrumentPass::GenReadFunctionCall(
    uint32_t func_id, const std::vector<uint32_t>& func_call_args,
    InstructionBuilder* ref_builder) {
  // If optimizing direct reads and the call has already been generated,
  // use its result
  if (opt_direct_reads_) {
    uint32_t res_id = call2id_[func_call_args];
    if (res_id != 0) return res_id;
  }
  // If the function arguments are all constants, the call can be moved to the
  // first block of the function where its result can be reused. One example
  // where this is profitable is for uniform buffer references, of which there
  // are often many.
  InstructionBuilder builder(ref_builder->GetContext(),
                             &*ref_builder->GetInsertPoint(),
                             ref_builder->GetPreservedAnalysis());
  bool insert_in_first_block = opt_direct_reads_ && AllConstant(func_call_args);
  if (insert_in_first_block) {
    Instruction* insert_before = &*curr_func_->begin()->tail();
    builder.SetInsertPoint(insert_before);
  }
  uint32_t res_id =
      builder.AddFunctionCall(GetUintId(), func_id, func_call_args)
          ->result_id();
  if (insert_in_first_block) call2id_[func_call_args] = res_id;
  return res_id;
}

bool InstrumentPass::IsSameBlockOp(const Instruction* inst) const {
  return inst->opcode() == spv::Op::OpSampledImage ||
         inst->opcode() == spv::Op::OpImage;
}

void InstrumentPass::CloneSameBlockOps(
    std::unique_ptr<Instruction>* inst,
    std::unordered_map<uint32_t, uint32_t>* same_blk_post,
    std::unordered_map<uint32_t, Instruction*>* same_blk_pre,
    BasicBlock* block_ptr) {
  bool changed = false;
  (*inst)->ForEachInId([&same_blk_post, &same_blk_pre, &block_ptr, &changed,
                        this](uint32_t* iid) {
    const auto map_itr = (*same_blk_post).find(*iid);
    if (map_itr == (*same_blk_post).end()) {
      const auto map_itr2 = (*same_blk_pre).find(*iid);
      if (map_itr2 != (*same_blk_pre).end()) {
        // Clone pre-call same-block ops, map result id.
        const Instruction* in_inst = map_itr2->second;
        std::unique_ptr<Instruction> sb_inst(in_inst->Clone(context()));
        const uint32_t rid = sb_inst->result_id();
        const uint32_t nid = this->TakeNextId();
        get_decoration_mgr()->CloneDecorations(rid, nid);
        sb_inst->SetResultId(nid);
        get_def_use_mgr()->AnalyzeInstDefUse(&*sb_inst);
        (*same_blk_post)[rid] = nid;
        *iid = nid;
        changed = true;
        CloneSameBlockOps(&sb_inst, same_blk_post, same_blk_pre, block_ptr);
        block_ptr->AddInstruction(std::move(sb_inst));
      }
    } else {
      // Reset same-block op operand if necessary
      if (*iid != map_itr->second) {
        *iid = map_itr->second;
        changed = true;
      }
    }
  });
  if (changed) get_def_use_mgr()->AnalyzeInstUse(&**inst);
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

uint32_t InstrumentPass::GetOutputBufferPtrId() {
  if (output_buffer_ptr_id_ == 0) {
    output_buffer_ptr_id_ = context()->get_type_mgr()->FindPointerToType(
        GetUintId(), spv::StorageClass::StorageBuffer);
  }
  return output_buffer_ptr_id_;
}

uint32_t InstrumentPass::GetInputBufferTypeId() {
  return (validation_id_ == kInstValidationIdBuffAddr) ? GetUint64Id()
                                                       : GetUintId();
}

uint32_t InstrumentPass::GetInputBufferPtrId() {
  if (input_buffer_ptr_id_ == 0) {
    input_buffer_ptr_id_ = context()->get_type_mgr()->FindPointerToType(
        GetInputBufferTypeId(), spv::StorageClass::StorageBuffer);
  }
  return input_buffer_ptr_id_;
}

uint32_t InstrumentPass::GetOutputBufferBinding() {
  switch (validation_id_) {
    case kInstValidationIdBindless:
      return kDebugOutputBindingStream;
    case kInstValidationIdBuffAddr:
      return kDebugOutputBindingStream;
    case kInstValidationIdDebugPrintf:
      return kDebugOutputPrintfStream;
    default:
      assert(false && "unexpected validation id");
  }
  return 0;
}

uint32_t InstrumentPass::GetInputBufferBinding() {
  switch (validation_id_) {
    case kInstValidationIdBindless:
      return kDebugInputBindingBindless;
    case kInstValidationIdBuffAddr:
      return kDebugInputBindingBuffAddr;
    default:
      assert(false && "unexpected validation id");
  }
  return 0;
}

analysis::Integer* InstrumentPass::GetInteger(uint32_t width, bool is_signed) {
  analysis::Integer i(width, is_signed);
  analysis::Type* type = context()->get_type_mgr()->GetRegisteredType(&i);
  assert(type && type->AsInteger());
  return type->AsInteger();
}

analysis::Struct* InstrumentPass::GetStruct(
    const std::vector<const analysis::Type*>& fields) {
  analysis::Struct s(fields);
  analysis::Type* type = context()->get_type_mgr()->GetRegisteredType(&s);
  assert(type && type->AsStruct());
  return type->AsStruct();
}

analysis::RuntimeArray* InstrumentPass::GetRuntimeArray(
    const analysis::Type* element) {
  analysis::RuntimeArray r(element);
  analysis::Type* type = context()->get_type_mgr()->GetRegisteredType(&r);
  assert(type && type->AsRuntimeArray());
  return type->AsRuntimeArray();
}

analysis::Function* InstrumentPass::GetFunction(
    const analysis::Type* return_val,
    const std::vector<const analysis::Type*>& args) {
  analysis::Function func(return_val, args);
  analysis::Type* type = context()->get_type_mgr()->GetRegisteredType(&func);
  assert(type && type->AsFunction());
  return type->AsFunction();
}

analysis::RuntimeArray* InstrumentPass::GetUintXRuntimeArrayType(
    uint32_t width, analysis::RuntimeArray** rarr_ty) {
  if (*rarr_ty == nullptr) {
    *rarr_ty = GetRuntimeArray(GetInteger(width, false));
    uint32_t uint_arr_ty_id =
        context()->get_type_mgr()->GetTypeInstruction(*rarr_ty);
    // By the Vulkan spec, a pre-existing RuntimeArray of uint must be part of
    // a block, and will therefore be decorated with an ArrayStride. Therefore
    // the undecorated type returned here will not be pre-existing and can
    // safely be decorated. Since this type is now decorated, it is out of
    // sync with the TypeManager and therefore the TypeManager must be
    // invalidated after this pass.
    assert(get_def_use_mgr()->NumUses(uint_arr_ty_id) == 0 &&
           "used RuntimeArray type returned");
    get_decoration_mgr()->AddDecorationVal(
        uint_arr_ty_id, uint32_t(spv::Decoration::ArrayStride), width / 8u);
  }
  return *rarr_ty;
}

analysis::RuntimeArray* InstrumentPass::GetUintRuntimeArrayType(
    uint32_t width) {
  analysis::RuntimeArray** rarr_ty =
      (width == 64) ? &uint64_rarr_ty_ : &uint32_rarr_ty_;
  return GetUintXRuntimeArrayType(width, rarr_ty);
}

void InstrumentPass::AddStorageBufferExt() {
  if (storage_buffer_ext_defined_) return;
  if (!get_feature_mgr()->HasExtension(kSPV_KHR_storage_buffer_storage_class)) {
    context()->AddExtension("SPV_KHR_storage_buffer_storage_class");
  }
  storage_buffer_ext_defined_ = true;
}

// Return id for output buffer
uint32_t InstrumentPass::GetOutputBufferId() {
  if (output_buffer_id_ == 0) {
    // If not created yet, create one
    analysis::DecorationManager* deco_mgr = get_decoration_mgr();
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::RuntimeArray* reg_uint_rarr_ty = GetUintRuntimeArrayType(32);
    analysis::Integer* reg_uint_ty = GetInteger(32, false);
    analysis::Type* reg_buf_ty =
        GetStruct({reg_uint_ty, reg_uint_ty, reg_uint_rarr_ty});
    uint32_t obufTyId = type_mgr->GetTypeInstruction(reg_buf_ty);
    // By the Vulkan spec, a pre-existing struct containing a RuntimeArray
    // must be a block, and will therefore be decorated with Block. Therefore
    // the undecorated type returned here will not be pre-existing and can
    // safely be decorated. Since this type is now decorated, it is out of
    // sync with the TypeManager and therefore the TypeManager must be
    // invalidated after this pass.
    assert(context()->get_def_use_mgr()->NumUses(obufTyId) == 0 &&
           "used struct type returned");
    deco_mgr->AddDecoration(obufTyId, uint32_t(spv::Decoration::Block));
    deco_mgr->AddMemberDecoration(obufTyId, kDebugOutputFlagsOffset,
                                  uint32_t(spv::Decoration::Offset), 0);
    deco_mgr->AddMemberDecoration(obufTyId, kDebugOutputSizeOffset,
                                  uint32_t(spv::Decoration::Offset), 4);
    deco_mgr->AddMemberDecoration(obufTyId, kDebugOutputDataOffset,
                                  uint32_t(spv::Decoration::Offset), 8);
    uint32_t obufTyPtrId_ =
        type_mgr->FindPointerToType(obufTyId, spv::StorageClass::StorageBuffer);
    output_buffer_id_ = TakeNextId();
    std::unique_ptr<Instruction> newVarOp(new Instruction(
        context(), spv::Op::OpVariable, obufTyPtrId_, output_buffer_id_,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          {uint32_t(spv::StorageClass::StorageBuffer)}}}));
    context()->AddGlobalValue(std::move(newVarOp));
    context()->AddDebug2Inst(NewGlobalName(obufTyId, "OutputBuffer"));
    context()->AddDebug2Inst(NewMemberName(obufTyId, 0, "flags"));
    context()->AddDebug2Inst(NewMemberName(obufTyId, 1, "written_count"));
    context()->AddDebug2Inst(NewMemberName(obufTyId, 2, "data"));
    context()->AddDebug2Inst(NewGlobalName(output_buffer_id_, "output_buffer"));
    deco_mgr->AddDecorationVal(
        output_buffer_id_, uint32_t(spv::Decoration::DescriptorSet), desc_set_);
    deco_mgr->AddDecorationVal(output_buffer_id_,
                               uint32_t(spv::Decoration::Binding),
                               GetOutputBufferBinding());
    AddStorageBufferExt();
    if (get_module()->version() >= SPV_SPIRV_VERSION_WORD(1, 4)) {
      // Add the new buffer to all entry points.
      for (auto& entry : get_module()->entry_points()) {
        entry.AddOperand({SPV_OPERAND_TYPE_ID, {output_buffer_id_}});
        context()->AnalyzeUses(&entry);
      }
    }
  }
  return output_buffer_id_;
}

uint32_t InstrumentPass::GetInputBufferId() {
  if (input_buffer_id_ == 0) {
    // If not created yet, create one
    analysis::DecorationManager* deco_mgr = get_decoration_mgr();
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    uint32_t width = (validation_id_ == kInstValidationIdBuffAddr) ? 64u : 32u;
    analysis::Type* reg_uint_rarr_ty = GetUintRuntimeArrayType(width);
    analysis::Struct* reg_buf_ty = GetStruct({reg_uint_rarr_ty});
    uint32_t ibufTyId = type_mgr->GetTypeInstruction(reg_buf_ty);
    // By the Vulkan spec, a pre-existing struct containing a RuntimeArray
    // must be a block, and will therefore be decorated with Block. Therefore
    // the undecorated type returned here will not be pre-existing and can
    // safely be decorated. Since this type is now decorated, it is out of
    // sync with the TypeManager and therefore the TypeManager must be
    // invalidated after this pass.
    assert(context()->get_def_use_mgr()->NumUses(ibufTyId) == 0 &&
           "used struct type returned");
    deco_mgr->AddDecoration(ibufTyId, uint32_t(spv::Decoration::Block));
    deco_mgr->AddMemberDecoration(ibufTyId, 0,
                                  uint32_t(spv::Decoration::Offset), 0);
    uint32_t ibufTyPtrId_ =
        type_mgr->FindPointerToType(ibufTyId, spv::StorageClass::StorageBuffer);
    input_buffer_id_ = TakeNextId();
    std::unique_ptr<Instruction> newVarOp(new Instruction(
        context(), spv::Op::OpVariable, ibufTyPtrId_, input_buffer_id_,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          {uint32_t(spv::StorageClass::StorageBuffer)}}}));
    context()->AddGlobalValue(std::move(newVarOp));
    context()->AddDebug2Inst(NewGlobalName(ibufTyId, "InputBuffer"));
    context()->AddDebug2Inst(NewMemberName(ibufTyId, 0, "data"));
    context()->AddDebug2Inst(NewGlobalName(input_buffer_id_, "input_buffer"));
    deco_mgr->AddDecorationVal(
        input_buffer_id_, uint32_t(spv::Decoration::DescriptorSet), desc_set_);
    deco_mgr->AddDecorationVal(input_buffer_id_,
                               uint32_t(spv::Decoration::Binding),
                               GetInputBufferBinding());
    AddStorageBufferExt();
    if (get_module()->version() >= SPV_SPIRV_VERSION_WORD(1, 4)) {
      // Add the new buffer to all entry points.
      for (auto& entry : get_module()->entry_points()) {
        entry.AddOperand({SPV_OPERAND_TYPE_ID, {input_buffer_id_}});
        context()->AnalyzeUses(&entry);
      }
    }
  }
  return input_buffer_id_;
}

uint32_t InstrumentPass::GetFloatId() {
  if (float_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Float float_ty(32);
    analysis::Type* reg_float_ty = type_mgr->GetRegisteredType(&float_ty);
    float_id_ = type_mgr->GetTypeInstruction(reg_float_ty);
  }
  return float_id_;
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

uint32_t InstrumentPass::GetUint64Id() {
  if (uint64_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Integer uint64_ty(64, false);
    analysis::Type* reg_uint64_ty = type_mgr->GetRegisteredType(&uint64_ty);
    uint64_id_ = type_mgr->GetTypeInstruction(reg_uint64_ty);
  }
  return uint64_id_;
}

uint32_t InstrumentPass::GetUint8Id() {
  if (uint8_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Integer uint8_ty(8, false);
    analysis::Type* reg_uint8_ty = type_mgr->GetRegisteredType(&uint8_ty);
    uint8_id_ = type_mgr->GetTypeInstruction(reg_uint8_ty);
  }
  return uint8_id_;
}

uint32_t InstrumentPass::GetVecUintId(uint32_t len) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Integer uint_ty(32, false);
  analysis::Type* reg_uint_ty = type_mgr->GetRegisteredType(&uint_ty);
  analysis::Vector v_uint_ty(reg_uint_ty, len);
  analysis::Type* reg_v_uint_ty = type_mgr->GetRegisteredType(&v_uint_ty);
  uint32_t v_uint_id = type_mgr->GetTypeInstruction(reg_v_uint_ty);
  return v_uint_id;
}

uint32_t InstrumentPass::GetVec4UintId() {
  if (v4uint_id_ == 0) v4uint_id_ = GetVecUintId(4u);
  return v4uint_id_;
}

uint32_t InstrumentPass::GetVec3UintId() {
  if (v3uint_id_ == 0) v3uint_id_ = GetVecUintId(3u);
  return v3uint_id_;
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
  if (param2output_func_id_[param_cnt] == 0) {
    // Create function
    param2output_func_id_[param_cnt] = TakeNextId();
    analysis::TypeManager* type_mgr = context()->get_type_mgr();

    const std::vector<const analysis::Type*> param_types(param_cnt,
                                                         GetInteger(32, false));
    std::unique_ptr<Function> output_func = StartFunction(
        param2output_func_id_[param_cnt], type_mgr->GetVoidType(), param_types);

    std::vector<uint32_t> param_ids = AddParameters(*output_func, param_types);

    // Create first block
    auto new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(TakeNextId()));

    InstructionBuilder builder(
        context(), &*new_blk_ptr,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    // Gen test if debug output buffer size will not be exceeded.
    uint32_t val_spec_offset = kInstStageOutCnt;
    uint32_t obuf_record_sz = val_spec_offset + val_spec_param_cnt;
    uint32_t buf_id = GetOutputBufferId();
    uint32_t buf_uint_ptr_id = GetOutputBufferPtrId();
    Instruction* obuf_curr_sz_ac_inst = builder.AddAccessChain(
        buf_uint_ptr_id, buf_id,
        {builder.GetUintConstantId(kDebugOutputSizeOffset)});
    // Fetch the current debug buffer written size atomically, adding the
    // size of the record to be written.
    uint32_t obuf_record_sz_id = builder.GetUintConstantId(obuf_record_sz);
    uint32_t mask_none_id =
        builder.GetUintConstantId(uint32_t(spv::MemoryAccessMask::MaskNone));
    uint32_t scope_invok_id =
        builder.GetUintConstantId(uint32_t(spv::Scope::Invocation));
    Instruction* obuf_curr_sz_inst = builder.AddQuadOp(
        GetUintId(), spv::Op::OpAtomicIAdd, obuf_curr_sz_ac_inst->result_id(),
        scope_invok_id, mask_none_id, obuf_record_sz_id);
    uint32_t obuf_curr_sz_id = obuf_curr_sz_inst->result_id();
    // Compute new written size
    Instruction* obuf_new_sz_inst =
        builder.AddIAdd(GetUintId(), obuf_curr_sz_id,
                        builder.GetUintConstantId(obuf_record_sz));
    // Fetch the data bound
    Instruction* obuf_bnd_inst =
        builder.AddIdLiteralOp(GetUintId(), spv::Op::OpArrayLength,
                               GetOutputBufferId(), kDebugOutputDataOffset);
    // Test that new written size is less than or equal to debug output
    // data bound
    Instruction* obuf_safe_inst = builder.AddBinaryOp(
        GetBoolId(), spv::Op::OpULessThanEqual, obuf_new_sz_inst->result_id(),
        obuf_bnd_inst->result_id());
    uint32_t merge_blk_id = TakeNextId();
    uint32_t write_blk_id = TakeNextId();
    std::unique_ptr<Instruction> merge_label(NewLabel(merge_blk_id));
    std::unique_ptr<Instruction> write_label(NewLabel(write_blk_id));
    (void)builder.AddConditionalBranch(
        obuf_safe_inst->result_id(), write_blk_id, merge_blk_id, merge_blk_id,
        uint32_t(spv::SelectionControlMask::MaskNone));
    // Close safety test block and gen write block
    output_func->AddBasicBlock(std::move(new_blk_ptr));
    new_blk_ptr = MakeUnique<BasicBlock>(std::move(write_label));
    builder.SetInsertPoint(&*new_blk_ptr);
    // Generate common and stage-specific debug record members
    GenCommonStreamWriteCode(obuf_record_sz, param_ids[kInstCommonParamInstIdx],
                             stage_idx, obuf_curr_sz_id, &builder);
    GenStageStreamWriteCode(stage_idx, obuf_curr_sz_id, &builder);
    // Gen writes of validation specific data
    for (uint32_t i = 0; i < val_spec_param_cnt; ++i) {
      GenDebugOutputFieldCode(obuf_curr_sz_id, val_spec_offset + i,
                              param_ids[kInstCommonParamCnt + i], &builder);
    }
    // Close write block and gen merge block
    (void)builder.AddBranch(merge_blk_id);
    output_func->AddBasicBlock(std::move(new_blk_ptr));
    new_blk_ptr = MakeUnique<BasicBlock>(std::move(merge_label));
    builder.SetInsertPoint(&*new_blk_ptr);
    // Close merge block and function and add function to module
    (void)builder.AddNullaryOp(0, spv::Op::OpReturn);

    output_func->AddBasicBlock(std::move(new_blk_ptr));
    output_func->SetFunctionEnd(EndFunction());
    context()->AddFunction(std::move(output_func));

    std::string name("stream_write_");
    name += std::to_string(param_cnt);

    context()->AddDebug2Inst(
        NewGlobalName(param2output_func_id_[param_cnt], name));
  }
  return param2output_func_id_[param_cnt];
}

uint32_t InstrumentPass::GetDirectReadFunctionId(uint32_t param_cnt) {
  uint32_t func_id = param2input_func_id_[param_cnt];
  if (func_id != 0) return func_id;
  // Create input function for param_cnt.
  func_id = TakeNextId();
  analysis::Integer* uint_type = GetInteger(32, false);
  std::vector<const analysis::Type*> param_types(param_cnt, uint_type);

  std::unique_ptr<Function> input_func =
      StartFunction(func_id, uint_type, param_types);
  std::vector<uint32_t> param_ids = AddParameters(*input_func, param_types);

  // Create block
  auto new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(TakeNextId()));
  InstructionBuilder builder(
      context(), &*new_blk_ptr,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  // For each offset parameter, generate new offset with parameter, adding last
  // loaded value if it exists, and load value from input buffer at new offset.
  // Return last loaded value.
  uint32_t ibuf_type_id = GetInputBufferTypeId();
  uint32_t buf_id = GetInputBufferId();
  uint32_t buf_ptr_id = GetInputBufferPtrId();
  uint32_t last_value_id = 0;
  for (uint32_t p = 0; p < param_cnt; ++p) {
    uint32_t offset_id;
    if (p == 0) {
      offset_id = param_ids[0];
    } else {
      if (ibuf_type_id != GetUintId()) {
        last_value_id =
            builder.AddUnaryOp(GetUintId(), spv::Op::OpUConvert, last_value_id)
                ->result_id();
      }
      offset_id = builder.AddIAdd(GetUintId(), last_value_id, param_ids[p])
                      ->result_id();
    }
    Instruction* ac_inst = builder.AddAccessChain(
        buf_ptr_id, buf_id,
        {builder.GetUintConstantId(kDebugInputDataOffset), offset_id});
    last_value_id =
        builder.AddLoad(ibuf_type_id, ac_inst->result_id())->result_id();
  }
  (void)builder.AddUnaryOp(0, spv::Op::OpReturnValue, last_value_id);
  // Close block and function and add function to module
  input_func->AddBasicBlock(std::move(new_blk_ptr));
  input_func->SetFunctionEnd(EndFunction());
  context()->AddFunction(std::move(input_func));

  std::string name("direct_read_");
  name += std::to_string(param_cnt);
  context()->AddDebug2Inst(NewGlobalName(func_id, name));

  param2input_func_id_[param_cnt] = func_id;
  return func_id;
}

void InstrumentPass::SplitBlock(
    BasicBlock::iterator inst_itr, UptrVectorIterator<BasicBlock> block_itr,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  // Make sure def/use analysis is done before we start moving instructions
  // out of function
  (void)get_def_use_mgr();
  // Move original block's preceding instructions into first new block
  std::unique_ptr<BasicBlock> first_blk_ptr;
  MovePreludeCode(inst_itr, block_itr, &first_blk_ptr);
  InstructionBuilder builder(
      context(), &*first_blk_ptr,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  uint32_t split_blk_id = TakeNextId();
  std::unique_ptr<Instruction> split_label(NewLabel(split_blk_id));
  (void)builder.AddBranch(split_blk_id);
  new_blocks->push_back(std::move(first_blk_ptr));
  // Move remaining instructions into split block and add to new blocks
  std::unique_ptr<BasicBlock> split_blk_ptr(
      new BasicBlock(std::move(split_label)));
  MovePostludeCode(block_itr, &*split_blk_ptr);
  new_blocks->push_back(std::move(split_blk_ptr));
}

bool InstrumentPass::InstrumentFunction(Function* func, uint32_t stage_idx,
                                        InstProcessFunction& pfn) {
  curr_func_ = func;
  call2id_.clear();
  bool first_block_split = false;
  bool modified = false;
  // Apply instrumentation function to each instruction.
  // Using block iterators here because of block erasures and insertions.
  std::vector<std::unique_ptr<BasicBlock>> new_blks;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      // Split all executable instructions out of first block into a following
      // block. This will allow function calls to be inserted into the first
      // block without interfering with the instrumentation algorithm.
      if (opt_direct_reads_ && !first_block_split) {
        if (ii->opcode() != spv::Op::OpVariable) {
          SplitBlock(ii, bi, &new_blks);
          first_block_split = true;
        }
      } else {
        pfn(ii, bi, stage_idx, &new_blks);
      }
      // If no new code, continue
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
      if (ii->opcode() == spv::Op::OpPhi ||
          ii->opcode() == spv::Op::OpCopyObject)
        ++ii;
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
  for (auto& ofn : param2output_func_id_) done.insert(ofn.second);
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
  spv::ExecutionModel stage = context()->GetStage();
  // Check for supported stages
  if (stage != spv::ExecutionModel::Vertex &&
      stage != spv::ExecutionModel::Fragment &&
      stage != spv::ExecutionModel::Geometry &&
      stage != spv::ExecutionModel::GLCompute &&
      stage != spv::ExecutionModel::TessellationControl &&
      stage != spv::ExecutionModel::TessellationEvaluation &&
      stage != spv::ExecutionModel::TaskNV &&
      stage != spv::ExecutionModel::MeshNV &&
      stage != spv::ExecutionModel::RayGenerationNV &&
      stage != spv::ExecutionModel::IntersectionNV &&
      stage != spv::ExecutionModel::AnyHitNV &&
      stage != spv::ExecutionModel::ClosestHitNV &&
      stage != spv::ExecutionModel::MissNV &&
      stage != spv::ExecutionModel::CallableNV &&
      stage != spv::ExecutionModel::TaskEXT &&
      stage != spv::ExecutionModel::MeshEXT) {
    if (consumer()) {
      std::string message = "Stage not supported by instrumentation";
      consumer()(SPV_MSG_ERROR, 0, {0, 0, 0}, message.c_str());
    }
    return false;
  }
  // Add together the roots of all entry points
  std::queue<uint32_t> roots;
  for (auto& e : get_module()->entry_points()) {
    roots.push(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));
  }
  bool modified = InstProcessCallTreeFromRoots(pfn, &roots, uint32_t(stage));
  return modified;
}

void InstrumentPass::InitializeInstrument() {
  output_buffer_id_ = 0;
  output_buffer_ptr_id_ = 0;
  input_buffer_ptr_id_ = 0;
  input_buffer_id_ = 0;
  float_id_ = 0;
  v4float_id_ = 0;
  uint_id_ = 0;
  uint64_id_ = 0;
  uint8_id_ = 0;
  v4uint_id_ = 0;
  v3uint_id_ = 0;
  bool_id_ = 0;
  void_id_ = 0;
  storage_buffer_ext_defined_ = false;
  uint32_rarr_ty_ = nullptr;
  uint64_rarr_ty_ = nullptr;

  // clear collections
  id2function_.clear();
  id2block_.clear();

  // clear maps
  param2input_func_id_.clear();
  param2output_func_id_.clear();

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
  for (auto& i : module->ext_inst_debuginfo()) {
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
