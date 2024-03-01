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

#include "relax_float_ops_pass.h"

#include "source/opt/ir_builder.h"

namespace spvtools {
namespace opt {

bool RelaxFloatOpsPass::IsRelaxable(Instruction* inst) {
  return target_ops_core_f_rslt_.count(inst->opcode()) != 0 ||
         target_ops_core_f_opnd_.count(inst->opcode()) != 0 ||
         sample_ops_.count(inst->opcode()) != 0 ||
         (inst->opcode() == spv::Op::OpExtInst &&
          inst->GetSingleWordInOperand(0) ==
              context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450() &&
          target_ops_450_.count(inst->GetSingleWordInOperand(1)) != 0);
}

bool RelaxFloatOpsPass::IsFloat32(Instruction* inst) {
  uint32_t ty_id;
  if (target_ops_core_f_opnd_.count(inst->opcode()) != 0) {
    uint32_t opnd_id = inst->GetSingleWordInOperand(0);
    Instruction* opnd_inst = get_def_use_mgr()->GetDef(opnd_id);
    ty_id = opnd_inst->type_id();
  } else {
    ty_id = inst->type_id();
    if (ty_id == 0) return false;
  }
  return IsFloat(ty_id, 32);
}

bool RelaxFloatOpsPass::IsRelaxed(uint32_t r_id) {
  for (auto r_inst : get_decoration_mgr()->GetDecorationsFor(r_id, false))
    if (r_inst->opcode() == spv::Op::OpDecorate &&
        spv::Decoration(r_inst->GetSingleWordInOperand(1)) ==
            spv::Decoration::RelaxedPrecision)
      return true;
  return false;
}

bool RelaxFloatOpsPass::ProcessInst(Instruction* r_inst) {
  uint32_t r_id = r_inst->result_id();
  if (r_id == 0) return false;
  if (!IsFloat32(r_inst)) return false;
  if (IsRelaxed(r_id)) return false;
  if (!IsRelaxable(r_inst)) return false;
  get_decoration_mgr()->AddDecoration(
      r_id, uint32_t(spv::Decoration::RelaxedPrecision));
  return true;
}

bool RelaxFloatOpsPass::ProcessFunction(Function* func) {
  bool modified = false;
  cfg()->ForEachBlockInReversePostOrder(
      func->entry().get(), [&modified, this](BasicBlock* bb) {
        for (auto ii = bb->begin(); ii != bb->end(); ++ii)
          modified |= ProcessInst(&*ii);
      });
  return modified;
}

Pass::Status RelaxFloatOpsPass::ProcessImpl() {
  Pass::ProcessFunction pfn = [this](Function* fp) {
    return ProcessFunction(fp);
  };
  bool modified = context()->ProcessReachableCallTree(pfn);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Pass::Status RelaxFloatOpsPass::Process() {
  Initialize();
  return ProcessImpl();
}

void RelaxFloatOpsPass::Initialize() {
  target_ops_core_f_rslt_ = {
      spv::Op::OpLoad,
      spv::Op::OpPhi,
      spv::Op::OpVectorExtractDynamic,
      spv::Op::OpVectorInsertDynamic,
      spv::Op::OpVectorShuffle,
      spv::Op::OpCompositeExtract,
      spv::Op::OpCompositeConstruct,
      spv::Op::OpCompositeInsert,
      spv::Op::OpCopyObject,
      spv::Op::OpTranspose,
      spv::Op::OpConvertSToF,
      spv::Op::OpConvertUToF,
      spv::Op::OpFConvert,
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
  };
  target_ops_core_f_opnd_ = {
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
  sample_ops_ = {spv::Op::OpImageSampleImplicitLod,
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
}

}  // namespace opt
}  // namespace spvtools
