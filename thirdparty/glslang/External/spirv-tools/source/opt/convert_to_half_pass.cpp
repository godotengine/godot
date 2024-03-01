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

#include "convert_to_half_pass.h"

#include "source/opt/ir_builder.h"

namespace spvtools {
namespace opt {
namespace {
// Indices of operands in SPIR-V instructions
constexpr int kImageSampleDrefIdInIdx = 2;
}  // namespace

bool ConvertToHalfPass::IsArithmetic(Instruction* inst) {
  return target_ops_core_.count(inst->opcode()) != 0 ||
         (inst->opcode() == spv::Op::OpExtInst &&
          inst->GetSingleWordInOperand(0) ==
              context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450() &&
          target_ops_450_.count(inst->GetSingleWordInOperand(1)) != 0);
}

bool ConvertToHalfPass::IsFloat(Instruction* inst, uint32_t width) {
  uint32_t ty_id = inst->type_id();
  if (ty_id == 0) return false;
  return Pass::IsFloat(ty_id, width);
}

bool ConvertToHalfPass::IsStruct(Instruction* inst) {
  uint32_t ty_id = inst->type_id();
  if (ty_id == 0) return false;
  Instruction* ty_inst = Pass::GetBaseType(ty_id);
  return (ty_inst->opcode() == spv::Op::OpTypeStruct);
}

bool ConvertToHalfPass::IsDecoratedRelaxed(Instruction* inst) {
  uint32_t r_id = inst->result_id();
  for (auto r_inst : get_decoration_mgr()->GetDecorationsFor(r_id, false))
    if (r_inst->opcode() == spv::Op::OpDecorate &&
        spv::Decoration(r_inst->GetSingleWordInOperand(1)) ==
            spv::Decoration::RelaxedPrecision) {
      return true;
    }
  return false;
}

bool ConvertToHalfPass::IsRelaxed(uint32_t id) {
  return relaxed_ids_set_.count(id) > 0;
}

void ConvertToHalfPass::AddRelaxed(uint32_t id) { relaxed_ids_set_.insert(id); }

bool ConvertToHalfPass::CanRelaxOpOperands(Instruction* inst) {
  return image_ops_.count(inst->opcode()) == 0;
}

analysis::Type* ConvertToHalfPass::FloatScalarType(uint32_t width) {
  analysis::Float float_ty(width);
  return context()->get_type_mgr()->GetRegisteredType(&float_ty);
}

analysis::Type* ConvertToHalfPass::FloatVectorType(uint32_t v_len,
                                                   uint32_t width) {
  analysis::Type* reg_float_ty = FloatScalarType(width);
  analysis::Vector vec_ty(reg_float_ty, v_len);
  return context()->get_type_mgr()->GetRegisteredType(&vec_ty);
}

analysis::Type* ConvertToHalfPass::FloatMatrixType(uint32_t v_cnt,
                                                   uint32_t vty_id,
                                                   uint32_t width) {
  Instruction* vty_inst = get_def_use_mgr()->GetDef(vty_id);
  uint32_t v_len = vty_inst->GetSingleWordInOperand(1);
  analysis::Type* reg_vec_ty = FloatVectorType(v_len, width);
  analysis::Matrix mat_ty(reg_vec_ty, v_cnt);
  return context()->get_type_mgr()->GetRegisteredType(&mat_ty);
}

uint32_t ConvertToHalfPass::EquivFloatTypeId(uint32_t ty_id, uint32_t width) {
  analysis::Type* reg_equiv_ty;
  Instruction* ty_inst = get_def_use_mgr()->GetDef(ty_id);
  if (ty_inst->opcode() == spv::Op::OpTypeMatrix)
    reg_equiv_ty = FloatMatrixType(ty_inst->GetSingleWordInOperand(1),
                                   ty_inst->GetSingleWordInOperand(0), width);
  else if (ty_inst->opcode() == spv::Op::OpTypeVector)
    reg_equiv_ty = FloatVectorType(ty_inst->GetSingleWordInOperand(1), width);
  else  // spv::Op::OpTypeFloat
    reg_equiv_ty = FloatScalarType(width);
  return context()->get_type_mgr()->GetTypeInstruction(reg_equiv_ty);
}

void ConvertToHalfPass::GenConvert(uint32_t* val_idp, uint32_t width,
                                   Instruction* inst) {
  Instruction* val_inst = get_def_use_mgr()->GetDef(*val_idp);
  uint32_t ty_id = val_inst->type_id();
  uint32_t nty_id = EquivFloatTypeId(ty_id, width);
  if (nty_id == ty_id) return;
  Instruction* cvt_inst;
  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (val_inst->opcode() == spv::Op::OpUndef)
    cvt_inst = builder.AddNullaryOp(nty_id, spv::Op::OpUndef);
  else
    cvt_inst = builder.AddUnaryOp(nty_id, spv::Op::OpFConvert, *val_idp);
  *val_idp = cvt_inst->result_id();
}

bool ConvertToHalfPass::MatConvertCleanup(Instruction* inst) {
  if (inst->opcode() != spv::Op::OpFConvert) return false;
  uint32_t mty_id = inst->type_id();
  Instruction* mty_inst = get_def_use_mgr()->GetDef(mty_id);
  if (mty_inst->opcode() != spv::Op::OpTypeMatrix) return false;
  uint32_t vty_id = mty_inst->GetSingleWordInOperand(0);
  uint32_t v_cnt = mty_inst->GetSingleWordInOperand(1);
  Instruction* vty_inst = get_def_use_mgr()->GetDef(vty_id);
  uint32_t cty_id = vty_inst->GetSingleWordInOperand(0);
  Instruction* cty_inst = get_def_use_mgr()->GetDef(cty_id);
  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  // Convert each component vector, combine them with OpCompositeConstruct
  // and replace original instruction.
  uint32_t orig_width = (cty_inst->GetSingleWordInOperand(0) == 16) ? 32 : 16;
  uint32_t orig_mat_id = inst->GetSingleWordInOperand(0);
  uint32_t orig_vty_id = EquivFloatTypeId(vty_id, orig_width);
  std::vector<Operand> opnds = {};
  for (uint32_t vidx = 0; vidx < v_cnt; ++vidx) {
    Instruction* ext_inst = builder.AddIdLiteralOp(
        orig_vty_id, spv::Op::OpCompositeExtract, orig_mat_id, vidx);
    Instruction* cvt_inst =
        builder.AddUnaryOp(vty_id, spv::Op::OpFConvert, ext_inst->result_id());
    opnds.push_back({SPV_OPERAND_TYPE_ID, {cvt_inst->result_id()}});
  }
  uint32_t mat_id = TakeNextId();
  std::unique_ptr<Instruction> mat_inst(new Instruction(
      context(), spv::Op::OpCompositeConstruct, mty_id, mat_id, opnds));
  (void)builder.AddInstruction(std::move(mat_inst));
  context()->ReplaceAllUsesWith(inst->result_id(), mat_id);
  // Turn original instruction into copy so it is valid.
  inst->SetOpcode(spv::Op::OpCopyObject);
  inst->SetResultType(EquivFloatTypeId(mty_id, orig_width));
  get_def_use_mgr()->AnalyzeInstUse(inst);
  return true;
}

bool ConvertToHalfPass::RemoveRelaxedDecoration(uint32_t id) {
  return context()->get_decoration_mgr()->RemoveDecorationsFrom(
      id, [](const Instruction& dec) {
        if (dec.opcode() == spv::Op::OpDecorate &&
            spv::Decoration(dec.GetSingleWordInOperand(1u)) ==
                spv::Decoration::RelaxedPrecision) {
          return true;
        } else
          return false;
      });
}

bool ConvertToHalfPass::GenHalfArith(Instruction* inst) {
  bool modified = false;
  // If this is a OpCompositeExtract instruction and has a struct operand, we
  // should not relax this instruction. Doing so could cause a mismatch between
  // the result type and the struct member type.
  bool hasStructOperand = false;
  if (inst->opcode() == spv::Op::OpCompositeExtract) {
    inst->ForEachInId([&hasStructOperand, this](uint32_t* idp) {
      Instruction* op_inst = get_def_use_mgr()->GetDef(*idp);
      if (IsStruct(op_inst)) hasStructOperand = true;
    });
    if (hasStructOperand) {
      return false;
    }
  }
  // Convert all float32 based operands to float16 equivalent and change
  // instruction type to float16 equivalent.
  inst->ForEachInId([&inst, &modified, this](uint32_t* idp) {
    Instruction* op_inst = get_def_use_mgr()->GetDef(*idp);
    if (!IsFloat(op_inst, 32)) return;
    GenConvert(idp, 16, inst);
    modified = true;
  });
  if (IsFloat(inst, 32)) {
    inst->SetResultType(EquivFloatTypeId(inst->type_id(), 16));
    converted_ids_.insert(inst->result_id());
    modified = true;
  }
  if (modified) get_def_use_mgr()->AnalyzeInstUse(inst);
  return modified;
}

bool ConvertToHalfPass::ProcessPhi(Instruction* inst, uint32_t from_width,
                                   uint32_t to_width) {
  // Add converts of any float operands to to_width if they are of from_width.
  // If converting to 16, change type of phi to float16 equivalent and remember
  // result id. Converts need to be added to preceding blocks.
  uint32_t ocnt = 0;
  uint32_t* prev_idp;
  bool modified = false;
  inst->ForEachInId([&ocnt, &prev_idp, &from_width, &to_width, &modified,
                     this](uint32_t* idp) {
    if (ocnt % 2 == 0) {
      prev_idp = idp;
    } else {
      Instruction* val_inst = get_def_use_mgr()->GetDef(*prev_idp);
      if (IsFloat(val_inst, from_width)) {
        BasicBlock* bp = context()->get_instr_block(*idp);
        auto insert_before = bp->tail();
        if (insert_before != bp->begin()) {
          --insert_before;
          if (insert_before->opcode() != spv::Op::OpSelectionMerge &&
              insert_before->opcode() != spv::Op::OpLoopMerge)
            ++insert_before;
        }
        GenConvert(prev_idp, to_width, &*insert_before);
        modified = true;
      }
    }
    ++ocnt;
  });
  if (to_width == 16u) {
    inst->SetResultType(EquivFloatTypeId(inst->type_id(), 16u));
    converted_ids_.insert(inst->result_id());
    modified = true;
  }
  if (modified) get_def_use_mgr()->AnalyzeInstUse(inst);
  return modified;
}

bool ConvertToHalfPass::ProcessConvert(Instruction* inst) {
  // If float32 and relaxed, change to float16 convert
  if (IsFloat(inst, 32) && IsRelaxed(inst->result_id())) {
    inst->SetResultType(EquivFloatTypeId(inst->type_id(), 16));
    get_def_use_mgr()->AnalyzeInstUse(inst);
    converted_ids_.insert(inst->result_id());
  }
  // If operand and result types are the same, change FConvert to CopyObject to
  // keep validator happy; simplification and DCE will clean it up
  // One way this can happen is if an FConvert generated during this pass
  // (likely by ProcessPhi) is later encountered here and its operand has been
  // changed to half.
  uint32_t val_id = inst->GetSingleWordInOperand(0);
  Instruction* val_inst = get_def_use_mgr()->GetDef(val_id);
  if (inst->type_id() == val_inst->type_id())
    inst->SetOpcode(spv::Op::OpCopyObject);
  return true;  // modified
}

bool ConvertToHalfPass::ProcessImageRef(Instruction* inst) {
  bool modified = false;
  // If image reference, only need to convert dref args back to float32
  if (dref_image_ops_.count(inst->opcode()) != 0) {
    uint32_t dref_id = inst->GetSingleWordInOperand(kImageSampleDrefIdInIdx);
    if (converted_ids_.count(dref_id) > 0) {
      GenConvert(&dref_id, 32, inst);
      inst->SetInOperand(kImageSampleDrefIdInIdx, {dref_id});
      get_def_use_mgr()->AnalyzeInstUse(inst);
      modified = true;
    }
  }
  return modified;
}

bool ConvertToHalfPass::ProcessDefault(Instruction* inst) {
  // If non-relaxed instruction has changed operands, need to convert
  // them back to float32
  if (inst->opcode() == spv::Op::OpPhi) return ProcessPhi(inst, 16u, 32u);
  bool modified = false;
  inst->ForEachInId([&inst, &modified, this](uint32_t* idp) {
    if (converted_ids_.count(*idp) == 0) return;
    uint32_t old_id = *idp;
    GenConvert(idp, 32, inst);
    if (*idp != old_id) modified = true;
  });
  if (modified) get_def_use_mgr()->AnalyzeInstUse(inst);
  return modified;
}

bool ConvertToHalfPass::GenHalfInst(Instruction* inst) {
  bool modified = false;
  // Remember id for later deletion of RelaxedPrecision decoration
  bool inst_relaxed = IsRelaxed(inst->result_id());
  if (IsArithmetic(inst) && inst_relaxed)
    modified = GenHalfArith(inst);
  else if (inst->opcode() == spv::Op::OpPhi && inst_relaxed)
    modified = ProcessPhi(inst, 32u, 16u);
  else if (inst->opcode() == spv::Op::OpFConvert)
    modified = ProcessConvert(inst);
  else if (image_ops_.count(inst->opcode()) != 0)
    modified = ProcessImageRef(inst);
  else
    modified = ProcessDefault(inst);
  return modified;
}

bool ConvertToHalfPass::CloseRelaxInst(Instruction* inst) {
  if (inst->result_id() == 0) return false;
  if (IsRelaxed(inst->result_id())) return false;
  if (!IsFloat(inst, 32)) return false;
  if (IsDecoratedRelaxed(inst)) {
    AddRelaxed(inst->result_id());
    return true;
  }
  if (closure_ops_.count(inst->opcode()) == 0) return false;
  // Can relax if all float operands are relaxed
  bool relax = true;
  bool hasStructOperand = false;
  inst->ForEachInId([&relax, &hasStructOperand, this](uint32_t* idp) {
    Instruction* op_inst = get_def_use_mgr()->GetDef(*idp);
    if (IsStruct(op_inst)) hasStructOperand = true;
    if (!IsFloat(op_inst, 32)) return;
    if (!IsRelaxed(*idp)) relax = false;
  });
  // If the instruction has a struct operand, we should not relax it, even if
  // all its uses are relaxed. Doing so could cause a mismatch between the
  // result type and the struct member type.
  if (hasStructOperand) {
    return false;
  }
  if (relax) {
    AddRelaxed(inst->result_id());
    return true;
  }
  // Can relax if all uses are relaxed
  relax = true;
  get_def_use_mgr()->ForEachUser(inst, [&relax, this](Instruction* uinst) {
    if (uinst->result_id() == 0 || !IsFloat(uinst, 32) ||
        (!IsDecoratedRelaxed(uinst) && !IsRelaxed(uinst->result_id())) ||
        !CanRelaxOpOperands(uinst)) {
      relax = false;
      return;
    }
  });
  if (relax) {
    AddRelaxed(inst->result_id());
    return true;
  }
  return false;
}

bool ConvertToHalfPass::ProcessFunction(Function* func) {
  // Do a closure of Relaxed on composite and phi instructions
  bool changed = true;
  while (changed) {
    changed = false;
    cfg()->ForEachBlockInReversePostOrder(
        func->entry().get(), [&changed, this](BasicBlock* bb) {
          for (auto ii = bb->begin(); ii != bb->end(); ++ii)
            changed |= CloseRelaxInst(&*ii);
        });
  }
  // Do convert of relaxed instructions to half precision
  bool modified = false;
  cfg()->ForEachBlockInReversePostOrder(
      func->entry().get(), [&modified, this](BasicBlock* bb) {
        for (auto ii = bb->begin(); ii != bb->end(); ++ii)
          modified |= GenHalfInst(&*ii);
      });
  // Replace invalid converts of matrix into equivalent vector extracts,
  // converts and finally a composite construct
  cfg()->ForEachBlockInReversePostOrder(
      func->entry().get(), [&modified, this](BasicBlock* bb) {
        for (auto ii = bb->begin(); ii != bb->end(); ++ii)
          modified |= MatConvertCleanup(&*ii);
      });
  return modified;
}

Pass::Status ConvertToHalfPass::ProcessImpl() {
  Pass::ProcessFunction pfn = [this](Function* fp) {
    return ProcessFunction(fp);
  };
  bool modified = context()->ProcessReachableCallTree(pfn);
  // If modified, make sure module has Float16 capability
  if (modified) context()->AddCapability(spv::Capability::Float16);
  // Remove all RelaxedPrecision decorations from instructions and globals
  for (auto c_id : relaxed_ids_set_) {
    modified |= RemoveRelaxedDecoration(c_id);
  }
  for (auto& val : get_module()->types_values()) {
    uint32_t v_id = val.result_id();
    if (v_id != 0) {
      modified |= RemoveRelaxedDecoration(v_id);
    }
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Pass::Status ConvertToHalfPass::Process() {
  Initialize();
  return ProcessImpl();
}

void ConvertToHalfPass::Initialize() {
  target_ops_core_ = {
      spv::Op::OpVectorExtractDynamic,
      spv::Op::OpVectorInsertDynamic,
      spv::Op::OpVectorShuffle,
      spv::Op::OpCompositeConstruct,
      spv::Op::OpCompositeInsert,
      spv::Op::OpCompositeExtract,
      spv::Op::OpCopyObject,
      spv::Op::OpTranspose,
      spv::Op::OpConvertSToF,
      spv::Op::OpConvertUToF,
      // spv::Op::OpFConvert,
      // spv::Op::OpQuantizeToF16,
      spv::Op::OpFNegate,
      spv::Op::OpFAdd,
      spv::Op::OpFSub,
      spv::Op::OpFMul,
      spv::Op::OpFDiv,
      spv::Op::OpFMod,
      spv::Op::OpVectorTimesScalar,
      spv::Op::OpMatrixTimesScalar,
      spv::Op::OpVectorTimesMatrix,
      spv::Op::OpMatrixTimesVector,
      spv::Op::OpMatrixTimesMatrix,
      spv::Op::OpOuterProduct,
      spv::Op::OpDot,
      spv::Op::OpSelect,
      spv::Op::OpFOrdEqual,
      spv::Op::OpFUnordEqual,
      spv::Op::OpFOrdNotEqual,
      spv::Op::OpFUnordNotEqual,
      spv::Op::OpFOrdLessThan,
      spv::Op::OpFUnordLessThan,
      spv::Op::OpFOrdGreaterThan,
      spv::Op::OpFUnordGreaterThan,
      spv::Op::OpFOrdLessThanEqual,
      spv::Op::OpFUnordLessThanEqual,
      spv::Op::OpFOrdGreaterThanEqual,
      spv::Op::OpFUnordGreaterThanEqual,
  };
  target_ops_450_ = {
      GLSLstd450Round, GLSLstd450RoundEven, GLSLstd450Trunc, GLSLstd450FAbs,
      GLSLstd450FSign, GLSLstd450Floor, GLSLstd450Ceil, GLSLstd450Fract,
      GLSLstd450Radians, GLSLstd450Degrees, GLSLstd450Sin, GLSLstd450Cos,
      GLSLstd450Tan, GLSLstd450Asin, GLSLstd450Acos, GLSLstd450Atan,
      GLSLstd450Sinh, GLSLstd450Cosh, GLSLstd450Tanh, GLSLstd450Asinh,
      GLSLstd450Acosh, GLSLstd450Atanh, GLSLstd450Atan2, GLSLstd450Pow,
      GLSLstd450Exp, GLSLstd450Log, GLSLstd450Exp2, GLSLstd450Log2,
      GLSLstd450Sqrt, GLSLstd450InverseSqrt, GLSLstd450Determinant,
      GLSLstd450MatrixInverse,
      // TODO(greg-lunarg): GLSLstd450ModfStruct,
      GLSLstd450FMin, GLSLstd450FMax, GLSLstd450FClamp, GLSLstd450FMix,
      GLSLstd450Step, GLSLstd450SmoothStep, GLSLstd450Fma,
      // TODO(greg-lunarg): GLSLstd450FrexpStruct,
      GLSLstd450Ldexp, GLSLstd450Length, GLSLstd450Distance, GLSLstd450Cross,
      GLSLstd450Normalize, GLSLstd450FaceForward, GLSLstd450Reflect,
      GLSLstd450Refract, GLSLstd450NMin, GLSLstd450NMax, GLSLstd450NClamp};
  image_ops_ = {spv::Op::OpImageSampleImplicitLod,
                spv::Op::OpImageSampleExplicitLod,
                spv::Op::OpImageSampleDrefImplicitLod,
                spv::Op::OpImageSampleDrefExplicitLod,
                spv::Op::OpImageSampleProjImplicitLod,
                spv::Op::OpImageSampleProjExplicitLod,
                spv::Op::OpImageSampleProjDrefImplicitLod,
                spv::Op::OpImageSampleProjDrefExplicitLod,
                spv::Op::OpImageFetch,
                spv::Op::OpImageGather,
                spv::Op::OpImageDrefGather,
                spv::Op::OpImageRead,
                spv::Op::OpImageSparseSampleImplicitLod,
                spv::Op::OpImageSparseSampleExplicitLod,
                spv::Op::OpImageSparseSampleDrefImplicitLod,
                spv::Op::OpImageSparseSampleDrefExplicitLod,
                spv::Op::OpImageSparseSampleProjImplicitLod,
                spv::Op::OpImageSparseSampleProjExplicitLod,
                spv::Op::OpImageSparseSampleProjDrefImplicitLod,
                spv::Op::OpImageSparseSampleProjDrefExplicitLod,
                spv::Op::OpImageSparseFetch,
                spv::Op::OpImageSparseGather,
                spv::Op::OpImageSparseDrefGather,
                spv::Op::OpImageSparseTexelsResident,
                spv::Op::OpImageSparseRead};
  dref_image_ops_ = {
      spv::Op::OpImageSampleDrefImplicitLod,
      spv::Op::OpImageSampleDrefExplicitLod,
      spv::Op::OpImageSampleProjDrefImplicitLod,
      spv::Op::OpImageSampleProjDrefExplicitLod,
      spv::Op::OpImageDrefGather,
      spv::Op::OpImageSparseSampleDrefImplicitLod,
      spv::Op::OpImageSparseSampleDrefExplicitLod,
      spv::Op::OpImageSparseSampleProjDrefImplicitLod,
      spv::Op::OpImageSparseSampleProjDrefExplicitLod,
      spv::Op::OpImageSparseDrefGather,
  };
  closure_ops_ = {
      spv::Op::OpVectorExtractDynamic,
      spv::Op::OpVectorInsertDynamic,
      spv::Op::OpVectorShuffle,
      spv::Op::OpCompositeConstruct,
      spv::Op::OpCompositeInsert,
      spv::Op::OpCompositeExtract,
      spv::Op::OpCopyObject,
      spv::Op::OpTranspose,
      spv::Op::OpPhi,
  };
  relaxed_ids_set_.clear();
  converted_ids_.clear();
}

}  // namespace opt
}  // namespace spvtools
