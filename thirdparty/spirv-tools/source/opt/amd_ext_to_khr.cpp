// Copyright (c) 2019 Google LLC.
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

#include "source/opt/amd_ext_to_khr.h"

#include <set>
#include <string>

#include "ir_builder.h"
#include "source/opt/ir_context.h"
#include "spv-amd-shader-ballot.insts.inc"
#include "type_manager.h"

namespace spvtools {
namespace opt {
namespace {

enum AmdShaderBallotExtOpcodes {
  AmdShaderBallotSwizzleInvocationsAMD = 1,
  AmdShaderBallotSwizzleInvocationsMaskedAMD = 2,
  AmdShaderBallotWriteInvocationAMD = 3,
  AmdShaderBallotMbcntAMD = 4
};

enum AmdShaderTrinaryMinMaxExtOpCodes {
  FMin3AMD = 1,
  UMin3AMD = 2,
  SMin3AMD = 3,
  FMax3AMD = 4,
  UMax3AMD = 5,
  SMax3AMD = 6,
  FMid3AMD = 7,
  UMid3AMD = 8,
  SMid3AMD = 9
};

enum AmdGcnShader { CubeFaceCoordAMD = 2, CubeFaceIndexAMD = 1, TimeAMD = 3 };

analysis::Type* GetUIntType(IRContext* ctx) {
  analysis::Integer int_type(32, false);
  return ctx->get_type_mgr()->GetRegisteredType(&int_type);
}

// Returns a folding rule that replaces |op(a,b,c)| by |op(op(a,b),c)|, where
// |op| is either min or max. |opcode| is the binary opcode in the GLSLstd450
// extended instruction set that corresponds to the trinary instruction being
// replaced.
template <GLSLstd450 opcode>
bool ReplaceTrinaryMinMax(IRContext* ctx, Instruction* inst,
                          const std::vector<const analysis::Constant*>&) {
  uint32_t glsl405_ext_inst_id =
      ctx->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  if (glsl405_ext_inst_id == 0) {
    ctx->AddExtInstImport("GLSL.std.450");
    glsl405_ext_inst_id =
        ctx->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  }

  InstructionBuilder ir_builder(
      ctx, inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  uint32_t op1 = inst->GetSingleWordInOperand(2);
  uint32_t op2 = inst->GetSingleWordInOperand(3);
  uint32_t op3 = inst->GetSingleWordInOperand(4);

  Instruction* temp = ir_builder.AddNaryExtendedInstruction(
      inst->type_id(), glsl405_ext_inst_id, opcode, {op1, op2});

  Instruction::OperandList new_operands;
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {glsl405_ext_inst_id}});
  new_operands.push_back({SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
                          {static_cast<uint32_t>(opcode)}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {temp->result_id()}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {op3}});

  inst->SetInOperands(std::move(new_operands));
  ctx->UpdateDefUse(inst);
  return true;
}

// Returns a folding rule that replaces |mid(a,b,c)| by |clamp(a, min(b,c),
// max(b,c)|. The three parameters are the opcode that correspond to the min,
// max, and clamp operations for the type of the instruction being replaced.
template <GLSLstd450 min_opcode, GLSLstd450 max_opcode, GLSLstd450 clamp_opcode>
bool ReplaceTrinaryMid(IRContext* ctx, Instruction* inst,
                       const std::vector<const analysis::Constant*>&) {
  uint32_t glsl405_ext_inst_id =
      ctx->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  if (glsl405_ext_inst_id == 0) {
    ctx->AddExtInstImport("GLSL.std.450");
    glsl405_ext_inst_id =
        ctx->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  }

  InstructionBuilder ir_builder(
      ctx, inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  uint32_t op1 = inst->GetSingleWordInOperand(2);
  uint32_t op2 = inst->GetSingleWordInOperand(3);
  uint32_t op3 = inst->GetSingleWordInOperand(4);

  Instruction* min = ir_builder.AddNaryExtendedInstruction(
      inst->type_id(), glsl405_ext_inst_id, static_cast<uint32_t>(min_opcode),
      {op2, op3});
  Instruction* max = ir_builder.AddNaryExtendedInstruction(
      inst->type_id(), glsl405_ext_inst_id, static_cast<uint32_t>(max_opcode),
      {op2, op3});

  Instruction::OperandList new_operands;
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {glsl405_ext_inst_id}});
  new_operands.push_back({SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
                          {static_cast<uint32_t>(clamp_opcode)}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {op1}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {min->result_id()}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {max->result_id()}});

  inst->SetInOperands(std::move(new_operands));
  ctx->UpdateDefUse(inst);
  return true;
}

// Returns a folding rule that will replace the opcode with |opcode| and add
// the capabilities required.  The folding rule assumes it is folding an
// OpGroup*NonUniformAMD instruction from the SPV_AMD_shader_ballot extension.
template <spv::Op new_opcode>
bool ReplaceGroupNonuniformOperationOpCode(
    IRContext* ctx, Instruction* inst,
    const std::vector<const analysis::Constant*>&) {
  switch (new_opcode) {
    case spv::Op::OpGroupNonUniformIAdd:
    case spv::Op::OpGroupNonUniformFAdd:
    case spv::Op::OpGroupNonUniformUMin:
    case spv::Op::OpGroupNonUniformSMin:
    case spv::Op::OpGroupNonUniformFMin:
    case spv::Op::OpGroupNonUniformUMax:
    case spv::Op::OpGroupNonUniformSMax:
    case spv::Op::OpGroupNonUniformFMax:
      break;
    default:
      assert(
          false &&
          "Should be replacing with a group non uniform arithmetic operation.");
  }

  switch (inst->opcode()) {
    case spv::Op::OpGroupIAddNonUniformAMD:
    case spv::Op::OpGroupFAddNonUniformAMD:
    case spv::Op::OpGroupUMinNonUniformAMD:
    case spv::Op::OpGroupSMinNonUniformAMD:
    case spv::Op::OpGroupFMinNonUniformAMD:
    case spv::Op::OpGroupUMaxNonUniformAMD:
    case spv::Op::OpGroupSMaxNonUniformAMD:
    case spv::Op::OpGroupFMaxNonUniformAMD:
      break;
    default:
      assert(false &&
             "Should be replacing a group non uniform arithmetic operation.");
  }

  ctx->AddCapability(spv::Capability::GroupNonUniformArithmetic);
  inst->SetOpcode(new_opcode);
  return true;
}

// Returns a folding rule that will replace the SwizzleInvocationsAMD extended
// instruction in the SPV_AMD_shader_ballot extension.
//
// The instruction
//
//  %offset = OpConstantComposite %v3uint %x %y %z %w
//  %result = OpExtInst %type %1 SwizzleInvocationsAMD %data %offset
//
// is replaced with
//
// potentially new constants and types
//
// clang-format off
//         %uint_max = OpConstant %uint 0xFFFFFFFF
//           %v4uint = OpTypeVector %uint 4
//     %ballot_value = OpConstantComposite %v4uint %uint_max %uint_max %uint_max %uint_max
//             %null = OpConstantNull %type
// clang-format on
//
// and the following code in the function body
//
// clang-format off
//         %id = OpLoad %uint %SubgroupLocalInvocationId
//   %quad_idx = OpBitwiseAnd %uint %id %uint_3
//   %quad_ldr = OpBitwiseXor %uint %id %quad_idx
//  %my_offset = OpVectorExtractDynamic %uint %offset %quad_idx
// %target_inv = OpIAdd %uint %quad_ldr %my_offset
//  %is_active = OpGroupNonUniformBallotBitExtract %bool %uint_3 %ballot_value %target_inv
//    %shuffle = OpGroupNonUniformShuffle %type %uint_3 %data %target_inv
//     %result = OpSelect %type %is_active %shuffle %null
// clang-format on
//
// Also adding the capabilities and builtins that are needed.
bool ReplaceSwizzleInvocations(IRContext* ctx, Instruction* inst,
                               const std::vector<const analysis::Constant*>&) {
  analysis::TypeManager* type_mgr = ctx->get_type_mgr();
  analysis::ConstantManager* const_mgr = ctx->get_constant_mgr();

  ctx->AddExtension("SPV_KHR_shader_ballot");
  ctx->AddCapability(spv::Capability::GroupNonUniformBallot);
  ctx->AddCapability(spv::Capability::GroupNonUniformShuffle);

  InstructionBuilder ir_builder(
      ctx, inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  uint32_t data_id = inst->GetSingleWordInOperand(2);
  uint32_t offset_id = inst->GetSingleWordInOperand(3);

  // Get the subgroup invocation id.
  uint32_t var_id = ctx->GetBuiltinInputVarId(
      uint32_t(spv::BuiltIn::SubgroupLocalInvocationId));
  assert(var_id != 0 && "Could not get SubgroupLocalInvocationId variable.");
  Instruction* var_inst = ctx->get_def_use_mgr()->GetDef(var_id);
  Instruction* var_ptr_type =
      ctx->get_def_use_mgr()->GetDef(var_inst->type_id());
  uint32_t uint_type_id = var_ptr_type->GetSingleWordInOperand(1);

  Instruction* id = ir_builder.AddLoad(uint_type_id, var_id);

  uint32_t quad_mask = ir_builder.GetUintConstantId(3);

  // This gives the offset in the group of 4 of this invocation.
  Instruction* quad_idx = ir_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpBitwiseAnd, id->result_id(), quad_mask);

  // Get the invocation id of the first invocation in the group of 4.
  Instruction* quad_ldr =
      ir_builder.AddBinaryOp(uint_type_id, spv::Op::OpBitwiseXor,
                             id->result_id(), quad_idx->result_id());

  // Get the offset of the target invocation from the offset vector.
  Instruction* my_offset =
      ir_builder.AddBinaryOp(uint_type_id, spv::Op::OpVectorExtractDynamic,
                             offset_id, quad_idx->result_id());

  // Determine the index of the invocation to read from.
  Instruction* target_inv =
      ir_builder.AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                             quad_ldr->result_id(), my_offset->result_id());

  // Do the group operations
  uint32_t uint_max_id = ir_builder.GetUintConstantId(0xFFFFFFFF);
  uint32_t subgroup_scope =
      ir_builder.GetUintConstantId(uint32_t(spv::Scope::Subgroup));
  const auto* ballot_value_const = const_mgr->GetConstant(
      type_mgr->GetUIntVectorType(4),
      {uint_max_id, uint_max_id, uint_max_id, uint_max_id});
  Instruction* ballot_value =
      const_mgr->GetDefiningInstruction(ballot_value_const);
  Instruction* is_active = ir_builder.AddNaryOp(
      type_mgr->GetBoolTypeId(), spv::Op::OpGroupNonUniformBallotBitExtract,
      {subgroup_scope, ballot_value->result_id(), target_inv->result_id()});
  Instruction* shuffle =
      ir_builder.AddNaryOp(inst->type_id(), spv::Op::OpGroupNonUniformShuffle,
                           {subgroup_scope, data_id, target_inv->result_id()});

  // Create the null constant to use in the select.
  const auto* null = const_mgr->GetConstant(type_mgr->GetType(inst->type_id()),
                                            std::vector<uint32_t>());
  Instruction* null_inst = const_mgr->GetDefiningInstruction(null);

  // Build the select.
  inst->SetOpcode(spv::Op::OpSelect);
  Instruction::OperandList new_operands;
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {is_active->result_id()}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {shuffle->result_id()}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {null_inst->result_id()}});

  inst->SetInOperands(std::move(new_operands));
  ctx->UpdateDefUse(inst);
  return true;
}

// Returns a folding rule that will replace the SwizzleInvocationsMaskedAMD
// extended instruction in the SPV_AMD_shader_ballot extension.
//
// The instruction
//
//    %mask = OpConstantComposite %v3uint %uint_x %uint_y %uint_z
//  %result = OpExtInst %uint %1 SwizzleInvocationsMaskedAMD %data %mask
//
// is replaced with
//
// potentially new constants and types
//
// clang-format off
// %uint_mask_extend = OpConstant %uint 0xFFFFFFE0
//         %uint_max = OpConstant %uint 0xFFFFFFFF
//           %v4uint = OpTypeVector %uint 4
//     %ballot_value = OpConstantComposite %v4uint %uint_max %uint_max %uint_max %uint_max
// clang-format on
//
// and the following code in the function body
//
// clang-format off
//         %id = OpLoad %uint %SubgroupLocalInvocationId
//   %and_mask = OpBitwiseOr %uint %uint_x %uint_mask_extend
//        %and = OpBitwiseAnd %uint %id %and_mask
//         %or = OpBitwiseOr %uint %and %uint_y
// %target_inv = OpBitwiseXor %uint %or %uint_z
//  %is_active = OpGroupNonUniformBallotBitExtract %bool %uint_3 %ballot_value %target_inv
//    %shuffle = OpGroupNonUniformShuffle %type %uint_3 %data %target_inv
//     %result = OpSelect %type %is_active %shuffle %uint_0
// clang-format on
//
// Also adding the capabilities and builtins that are needed.
bool ReplaceSwizzleInvocationsMasked(
    IRContext* ctx, Instruction* inst,
    const std::vector<const analysis::Constant*>&) {
  analysis::TypeManager* type_mgr = ctx->get_type_mgr();
  analysis::DefUseManager* def_use_mgr = ctx->get_def_use_mgr();
  analysis::ConstantManager* const_mgr = ctx->get_constant_mgr();

  ctx->AddCapability(spv::Capability::GroupNonUniformBallot);
  ctx->AddCapability(spv::Capability::GroupNonUniformShuffle);

  InstructionBuilder ir_builder(
      ctx, inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  // Get the operands to inst, and the components of the mask
  uint32_t data_id = inst->GetSingleWordInOperand(2);

  Instruction* mask_inst = def_use_mgr->GetDef(inst->GetSingleWordInOperand(3));
  assert(mask_inst->opcode() == spv::Op::OpConstantComposite &&
         "The mask is suppose to be a vector constant.");
  assert(mask_inst->NumInOperands() == 3 &&
         "The mask is suppose to have 3 components.");

  uint32_t uint_x = mask_inst->GetSingleWordInOperand(0);
  uint32_t uint_y = mask_inst->GetSingleWordInOperand(1);
  uint32_t uint_z = mask_inst->GetSingleWordInOperand(2);

  // Get the subgroup invocation id.
  uint32_t var_id = ctx->GetBuiltinInputVarId(
      uint32_t(spv::BuiltIn::SubgroupLocalInvocationId));
  ctx->AddExtension("SPV_KHR_shader_ballot");
  assert(var_id != 0 && "Could not get SubgroupLocalInvocationId variable.");
  Instruction* var_inst = ctx->get_def_use_mgr()->GetDef(var_id);
  Instruction* var_ptr_type =
      ctx->get_def_use_mgr()->GetDef(var_inst->type_id());
  uint32_t uint_type_id = var_ptr_type->GetSingleWordInOperand(1);

  Instruction* id = ir_builder.AddLoad(uint_type_id, var_id);

  // Do the bitwise operations.
  uint32_t mask_extended = ir_builder.GetUintConstantId(0xFFFFFFE0);
  Instruction* and_mask = ir_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpBitwiseOr, uint_x, mask_extended);
  Instruction* and_result =
      ir_builder.AddBinaryOp(uint_type_id, spv::Op::OpBitwiseAnd,
                             id->result_id(), and_mask->result_id());
  Instruction* or_result = ir_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpBitwiseOr, and_result->result_id(), uint_y);
  Instruction* target_inv = ir_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpBitwiseXor, or_result->result_id(), uint_z);

  // Do the group operations
  uint32_t uint_max_id = ir_builder.GetUintConstantId(0xFFFFFFFF);
  uint32_t subgroup_scope =
      ir_builder.GetUintConstantId(uint32_t(spv::Scope::Subgroup));
  const auto* ballot_value_const = const_mgr->GetConstant(
      type_mgr->GetUIntVectorType(4),
      {uint_max_id, uint_max_id, uint_max_id, uint_max_id});
  Instruction* ballot_value =
      const_mgr->GetDefiningInstruction(ballot_value_const);
  Instruction* is_active = ir_builder.AddNaryOp(
      type_mgr->GetBoolTypeId(), spv::Op::OpGroupNonUniformBallotBitExtract,
      {subgroup_scope, ballot_value->result_id(), target_inv->result_id()});
  Instruction* shuffle =
      ir_builder.AddNaryOp(inst->type_id(), spv::Op::OpGroupNonUniformShuffle,
                           {subgroup_scope, data_id, target_inv->result_id()});

  // Create the null constant to use in the select.
  const auto* null = const_mgr->GetConstant(type_mgr->GetType(inst->type_id()),
                                            std::vector<uint32_t>());
  Instruction* null_inst = const_mgr->GetDefiningInstruction(null);

  // Build the select.
  inst->SetOpcode(spv::Op::OpSelect);
  Instruction::OperandList new_operands;
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {is_active->result_id()}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {shuffle->result_id()}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {null_inst->result_id()}});

  inst->SetInOperands(std::move(new_operands));
  ctx->UpdateDefUse(inst);
  return true;
}

// Returns a folding rule that will replace the WriteInvocationAMD extended
// instruction in the SPV_AMD_shader_ballot extension.
//
// The instruction
//
// clang-format off
//    %result = OpExtInst %type %1 WriteInvocationAMD %input_value %write_value %invocation_index
// clang-format on
//
// with
//
//     %id = OpLoad %uint %SubgroupLocalInvocationId
//    %cmp = OpIEqual %bool %id %invocation_index
// %result = OpSelect %type %cmp %write_value %input_value
//
// Also adding the capabilities and builtins that are needed.
bool ReplaceWriteInvocation(IRContext* ctx, Instruction* inst,
                            const std::vector<const analysis::Constant*>&) {
  uint32_t var_id = ctx->GetBuiltinInputVarId(
      uint32_t(spv::BuiltIn::SubgroupLocalInvocationId));
  ctx->AddCapability(spv::Capability::SubgroupBallotKHR);
  ctx->AddExtension("SPV_KHR_shader_ballot");
  assert(var_id != 0 && "Could not get SubgroupLocalInvocationId variable.");
  Instruction* var_inst = ctx->get_def_use_mgr()->GetDef(var_id);
  Instruction* var_ptr_type =
      ctx->get_def_use_mgr()->GetDef(var_inst->type_id());

  InstructionBuilder ir_builder(
      ctx, inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* t =
      ir_builder.AddLoad(var_ptr_type->GetSingleWordInOperand(1), var_id);
  analysis::Bool bool_type;
  uint32_t bool_type_id = ctx->get_type_mgr()->GetTypeInstruction(&bool_type);
  Instruction* cmp =
      ir_builder.AddBinaryOp(bool_type_id, spv::Op::OpIEqual, t->result_id(),
                             inst->GetSingleWordInOperand(4));

  // Build a select.
  inst->SetOpcode(spv::Op::OpSelect);
  Instruction::OperandList new_operands;
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {cmp->result_id()}});
  new_operands.push_back(inst->GetInOperand(3));
  new_operands.push_back(inst->GetInOperand(2));

  inst->SetInOperands(std::move(new_operands));
  ctx->UpdateDefUse(inst);
  return true;
}

// Returns a folding rule that will replace the MbcntAMD extended instruction in
// the SPV_AMD_shader_ballot extension.
//
// The instruction
//
//  %result = OpExtInst %uint %1 MbcntAMD %mask
//
// with
//
// Get SubgroupLtMask and convert the first 64-bits into a uint64_t because
// AMD's shader compiler expects a 64-bit integer mask.
//
//     %var = OpLoad %v4uint %SubgroupLtMaskKHR
// %shuffle = OpVectorShuffle %v2uint %var %var 0 1
//    %cast = OpBitcast %ulong %shuffle
//
// Perform the mask and count the bits.
//
//     %and = OpBitwiseAnd %ulong %cast %mask
//  %result = OpBitCount %uint %and
//
// Also adding the capabilities and builtins that are needed.
bool ReplaceMbcnt(IRContext* context, Instruction* inst,
                  const std::vector<const analysis::Constant*>&) {
  analysis::TypeManager* type_mgr = context->get_type_mgr();
  analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();

  uint32_t var_id =
      context->GetBuiltinInputVarId(uint32_t(spv::BuiltIn::SubgroupLtMask));
  assert(var_id != 0 && "Could not get SubgroupLtMask variable.");
  context->AddCapability(spv::Capability::GroupNonUniformBallot);
  Instruction* var_inst = def_use_mgr->GetDef(var_id);
  Instruction* var_ptr_type = def_use_mgr->GetDef(var_inst->type_id());
  Instruction* var_type =
      def_use_mgr->GetDef(var_ptr_type->GetSingleWordInOperand(1));
  assert(var_type->opcode() == spv::Op::OpTypeVector &&
         "Variable is suppose to be a vector of 4 ints");

  // Get the type for the shuffle.
  analysis::Vector temp_type(GetUIntType(context), 2);
  const analysis::Type* shuffle_type =
      context->get_type_mgr()->GetRegisteredType(&temp_type);
  uint32_t shuffle_type_id = type_mgr->GetTypeInstruction(shuffle_type);

  uint32_t mask_id = inst->GetSingleWordInOperand(2);
  Instruction* mask_inst = def_use_mgr->GetDef(mask_id);

  // Testing with amd's shader compiler shows that a 64-bit mask is expected.
  assert(type_mgr->GetType(mask_inst->type_id())->AsInteger() != nullptr);
  assert(type_mgr->GetType(mask_inst->type_id())->AsInteger()->width() == 64);

  InstructionBuilder ir_builder(
      context, inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* load = ir_builder.AddLoad(var_type->result_id(), var_id);
  Instruction* shuffle = ir_builder.AddVectorShuffle(
      shuffle_type_id, load->result_id(), load->result_id(), {0, 1});
  Instruction* bitcast = ir_builder.AddUnaryOp(
      mask_inst->type_id(), spv::Op::OpBitcast, shuffle->result_id());
  Instruction* t =
      ir_builder.AddBinaryOp(mask_inst->type_id(), spv::Op::OpBitwiseAnd,
                             bitcast->result_id(), mask_id);

  inst->SetOpcode(spv::Op::OpBitCount);
  inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {t->result_id()}}});
  context->UpdateDefUse(inst);
  return true;
}

// A folding rule that will replace the CubeFaceCoordAMD extended
// instruction in the SPV_AMD_gcn_shader_ballot.  Returns true if the folding is
// successful.
//
// The instruction
//
//  %result = OpExtInst %v2float %1 CubeFaceCoordAMD %input
//
// with
//
//             %x = OpCompositeExtract %float %input 0
//             %y = OpCompositeExtract %float %input 1
//             %z = OpCompositeExtract %float %input 2
//            %nx = OpFNegate %float %x
//            %ny = OpFNegate %float %y
//            %nz = OpFNegate %float %z
//            %ax = OpExtInst %float %n_1 FAbs %x
//            %ay = OpExtInst %float %n_1 FAbs %y
//            %az = OpExtInst %float %n_1 FAbs %z
//      %amax_x_y = OpExtInst %float %n_1 FMax %ay %ax
//          %amax = OpExtInst %float %n_1 FMax %az %amax_x_y
//        %cubema = OpFMul %float %float_2 %amax
//      %is_z_max = OpFOrdGreaterThanEqual %bool %az %amax_x_y
//  %not_is_z_max = OpLogicalNot %bool %is_z_max
//        %y_gt_x = OpFOrdGreaterThanEqual %bool %ay %ax
//      %is_y_max = OpLogicalAnd %bool %not_is_z_max %y_gt_x
//      %is_z_neg = OpFOrdLessThan %bool %z %float_0
// %cubesc_case_1 = OpSelect %float %is_z_neg %nx %x
//      %is_x_neg = OpFOrdLessThan %bool %x %float_0
// %cubesc_case_2 = OpSelect %float %is_x_neg %z %nz
//           %sel = OpSelect %float %is_y_max %x %cubesc_case_2
//        %cubesc = OpSelect %float %is_z_max %cubesc_case_1 %sel
//      %is_y_neg = OpFOrdLessThan %bool %y %float_0
// %cubetc_case_1 = OpSelect %float %is_y_neg %nz %z
//        %cubetc = OpSelect %float %is_y_max %cubetc_case_1 %ny
//          %cube = OpCompositeConstruct %v2float %cubesc %cubetc
//         %denom = OpCompositeConstruct %v2float %cubema %cubema
//           %div = OpFDiv %v2float %cube %denom
//        %result = OpFAdd %v2float %div %const
//
// Also adding the capabilities and builtins that are needed.
bool ReplaceCubeFaceCoord(IRContext* ctx, Instruction* inst,
                          const std::vector<const analysis::Constant*>&) {
  analysis::TypeManager* type_mgr = ctx->get_type_mgr();
  analysis::ConstantManager* const_mgr = ctx->get_constant_mgr();

  uint32_t float_type_id = type_mgr->GetFloatTypeId();
  const analysis::Type* v2_float_type = type_mgr->GetFloatVectorType(2);
  uint32_t v2_float_type_id = type_mgr->GetId(v2_float_type);
  uint32_t bool_id = type_mgr->GetBoolTypeId();

  InstructionBuilder ir_builder(
      ctx, inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  uint32_t input_id = inst->GetSingleWordInOperand(2);
  uint32_t glsl405_ext_inst_id =
      ctx->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  if (glsl405_ext_inst_id == 0) {
    ctx->AddExtInstImport("GLSL.std.450");
    glsl405_ext_inst_id =
        ctx->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  }

  // Get the constants that will be used.
  uint32_t f0_const_id = const_mgr->GetFloatConstId(0.0);
  uint32_t f2_const_id = const_mgr->GetFloatConstId(2.0);
  uint32_t f0_5_const_id = const_mgr->GetFloatConstId(0.5);
  const analysis::Constant* vec_const =
      const_mgr->GetConstant(v2_float_type, {f0_5_const_id, f0_5_const_id});
  uint32_t vec_const_id =
      const_mgr->GetDefiningInstruction(vec_const)->result_id();

  // Extract the input values.
  Instruction* x = ir_builder.AddCompositeExtract(float_type_id, input_id, {0});
  Instruction* y = ir_builder.AddCompositeExtract(float_type_id, input_id, {1});
  Instruction* z = ir_builder.AddCompositeExtract(float_type_id, input_id, {2});

  // Negate the input values.
  Instruction* nx =
      ir_builder.AddUnaryOp(float_type_id, spv::Op::OpFNegate, x->result_id());
  Instruction* ny =
      ir_builder.AddUnaryOp(float_type_id, spv::Op::OpFNegate, y->result_id());
  Instruction* nz =
      ir_builder.AddUnaryOp(float_type_id, spv::Op::OpFNegate, z->result_id());

  // Get the abolsute values of the inputs.
  Instruction* ax = ir_builder.AddNaryExtendedInstruction(
      float_type_id, glsl405_ext_inst_id, GLSLstd450FAbs, {x->result_id()});
  Instruction* ay = ir_builder.AddNaryExtendedInstruction(
      float_type_id, glsl405_ext_inst_id, GLSLstd450FAbs, {y->result_id()});
  Instruction* az = ir_builder.AddNaryExtendedInstruction(
      float_type_id, glsl405_ext_inst_id, GLSLstd450FAbs, {z->result_id()});

  // Find which values are negative.  Used in later computations.
  Instruction* is_z_neg = ir_builder.AddBinaryOp(
      bool_id, spv::Op::OpFOrdLessThan, z->result_id(), f0_const_id);
  Instruction* is_y_neg = ir_builder.AddBinaryOp(
      bool_id, spv::Op::OpFOrdLessThan, y->result_id(), f0_const_id);
  Instruction* is_x_neg = ir_builder.AddBinaryOp(
      bool_id, spv::Op::OpFOrdLessThan, x->result_id(), f0_const_id);

  // Compute cubema
  Instruction* amax_x_y = ir_builder.AddNaryExtendedInstruction(
      float_type_id, glsl405_ext_inst_id, GLSLstd450FMax,
      {ax->result_id(), ay->result_id()});
  Instruction* amax = ir_builder.AddNaryExtendedInstruction(
      float_type_id, glsl405_ext_inst_id, GLSLstd450FMax,
      {az->result_id(), amax_x_y->result_id()});
  Instruction* cubema = ir_builder.AddBinaryOp(float_type_id, spv::Op::OpFMul,
                                               f2_const_id, amax->result_id());

  // Do the comparisons needed for computing cubesc and cubetc.
  Instruction* is_z_max =
      ir_builder.AddBinaryOp(bool_id, spv::Op::OpFOrdGreaterThanEqual,
                             az->result_id(), amax_x_y->result_id());
  Instruction* not_is_z_max = ir_builder.AddUnaryOp(
      bool_id, spv::Op::OpLogicalNot, is_z_max->result_id());
  Instruction* y_gr_x =
      ir_builder.AddBinaryOp(bool_id, spv::Op::OpFOrdGreaterThanEqual,
                             ay->result_id(), ax->result_id());
  Instruction* is_y_max =
      ir_builder.AddBinaryOp(bool_id, spv::Op::OpLogicalAnd,
                             not_is_z_max->result_id(), y_gr_x->result_id());

  // Select the correct value for cubesc.
  Instruction* cubesc_case_1 = ir_builder.AddSelect(
      float_type_id, is_z_neg->result_id(), nx->result_id(), x->result_id());
  Instruction* cubesc_case_2 = ir_builder.AddSelect(
      float_type_id, is_x_neg->result_id(), z->result_id(), nz->result_id());
  Instruction* sel =
      ir_builder.AddSelect(float_type_id, is_y_max->result_id(), x->result_id(),
                           cubesc_case_2->result_id());
  Instruction* cubesc =
      ir_builder.AddSelect(float_type_id, is_z_max->result_id(),
                           cubesc_case_1->result_id(), sel->result_id());

  // Select the correct value for cubetc.
  Instruction* cubetc_case_1 = ir_builder.AddSelect(
      float_type_id, is_y_neg->result_id(), nz->result_id(), z->result_id());
  Instruction* cubetc =
      ir_builder.AddSelect(float_type_id, is_y_max->result_id(),
                           cubetc_case_1->result_id(), ny->result_id());

  // Do the division
  Instruction* cube = ir_builder.AddCompositeConstruct(
      v2_float_type_id, {cubesc->result_id(), cubetc->result_id()});
  Instruction* denom = ir_builder.AddCompositeConstruct(
      v2_float_type_id, {cubema->result_id(), cubema->result_id()});
  Instruction* div = ir_builder.AddBinaryOp(
      v2_float_type_id, spv::Op::OpFDiv, cube->result_id(), denom->result_id());

  // Get the final result by adding 0.5 to |div|.
  inst->SetOpcode(spv::Op::OpFAdd);
  Instruction::OperandList new_operands;
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {div->result_id()}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {vec_const_id}});

  inst->SetInOperands(std::move(new_operands));
  ctx->UpdateDefUse(inst);
  return true;
}

// A folding rule that will replace the CubeFaceIndexAMD extended
// instruction in the SPV_AMD_gcn_shader_ballot.  Returns true if the folding
// is successful.
//
// The instruction
//
//  %result = OpExtInst %float %1 CubeFaceIndexAMD %input
//
// with
//
//             %x = OpCompositeExtract %float %input 0
//             %y = OpCompositeExtract %float %input 1
//             %z = OpCompositeExtract %float %input 2
//            %ax = OpExtInst %float %n_1 FAbs %x
//            %ay = OpExtInst %float %n_1 FAbs %y
//            %az = OpExtInst %float %n_1 FAbs %z
//      %is_z_neg = OpFOrdLessThan %bool %z %float_0
//      %is_y_neg = OpFOrdLessThan %bool %y %float_0
//      %is_x_neg = OpFOrdLessThan %bool %x %float_0
//      %amax_x_y = OpExtInst %float %n_1 FMax %ax %ay
//      %is_z_max = OpFOrdGreaterThanEqual %bool %az %amax_x_y
//        %y_gt_x = OpFOrdGreaterThanEqual %bool %ay %ax
//        %case_z = OpSelect %float %is_z_neg %float_5 %float4
//        %case_y = OpSelect %float %is_y_neg %float_3 %float2
//        %case_x = OpSelect %float %is_x_neg %float_1 %float0
//           %sel = OpSelect %float %y_gt_x %case_y %case_x
//        %result = OpSelect %float %is_z_max %case_z %sel
//
// Also adding the capabilities and builtins that are needed.
bool ReplaceCubeFaceIndex(IRContext* ctx, Instruction* inst,
                          const std::vector<const analysis::Constant*>&) {
  analysis::TypeManager* type_mgr = ctx->get_type_mgr();
  analysis::ConstantManager* const_mgr = ctx->get_constant_mgr();

  uint32_t float_type_id = type_mgr->GetFloatTypeId();
  uint32_t bool_id = type_mgr->GetBoolTypeId();

  InstructionBuilder ir_builder(
      ctx, inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  uint32_t input_id = inst->GetSingleWordInOperand(2);
  uint32_t glsl405_ext_inst_id =
      ctx->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  if (glsl405_ext_inst_id == 0) {
    ctx->AddExtInstImport("GLSL.std.450");
    glsl405_ext_inst_id =
        ctx->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  }

  // Get the constants that will be used.
  uint32_t f0_const_id = const_mgr->GetFloatConstId(0.0);
  uint32_t f1_const_id = const_mgr->GetFloatConstId(1.0);
  uint32_t f2_const_id = const_mgr->GetFloatConstId(2.0);
  uint32_t f3_const_id = const_mgr->GetFloatConstId(3.0);
  uint32_t f4_const_id = const_mgr->GetFloatConstId(4.0);
  uint32_t f5_const_id = const_mgr->GetFloatConstId(5.0);

  // Extract the input values.
  Instruction* x = ir_builder.AddCompositeExtract(float_type_id, input_id, {0});
  Instruction* y = ir_builder.AddCompositeExtract(float_type_id, input_id, {1});
  Instruction* z = ir_builder.AddCompositeExtract(float_type_id, input_id, {2});

  // Get the absolute values of the inputs.
  Instruction* ax = ir_builder.AddNaryExtendedInstruction(
      float_type_id, glsl405_ext_inst_id, GLSLstd450FAbs, {x->result_id()});
  Instruction* ay = ir_builder.AddNaryExtendedInstruction(
      float_type_id, glsl405_ext_inst_id, GLSLstd450FAbs, {y->result_id()});
  Instruction* az = ir_builder.AddNaryExtendedInstruction(
      float_type_id, glsl405_ext_inst_id, GLSLstd450FAbs, {z->result_id()});

  // Find which values are negative.  Used in later computations.
  Instruction* is_z_neg = ir_builder.AddBinaryOp(
      bool_id, spv::Op::OpFOrdLessThan, z->result_id(), f0_const_id);
  Instruction* is_y_neg = ir_builder.AddBinaryOp(
      bool_id, spv::Op::OpFOrdLessThan, y->result_id(), f0_const_id);
  Instruction* is_x_neg = ir_builder.AddBinaryOp(
      bool_id, spv::Op::OpFOrdLessThan, x->result_id(), f0_const_id);

  // Find the max value.
  Instruction* amax_x_y = ir_builder.AddNaryExtendedInstruction(
      float_type_id, glsl405_ext_inst_id, GLSLstd450FMax,
      {ax->result_id(), ay->result_id()});
  Instruction* is_z_max =
      ir_builder.AddBinaryOp(bool_id, spv::Op::OpFOrdGreaterThanEqual,
                             az->result_id(), amax_x_y->result_id());
  Instruction* y_gr_x =
      ir_builder.AddBinaryOp(bool_id, spv::Op::OpFOrdGreaterThanEqual,
                             ay->result_id(), ax->result_id());

  // Get the value for each case.
  Instruction* case_z = ir_builder.AddSelect(
      float_type_id, is_z_neg->result_id(), f5_const_id, f4_const_id);
  Instruction* case_y = ir_builder.AddSelect(
      float_type_id, is_y_neg->result_id(), f3_const_id, f2_const_id);
  Instruction* case_x = ir_builder.AddSelect(
      float_type_id, is_x_neg->result_id(), f1_const_id, f0_const_id);

  // Select the correct case.
  Instruction* sel =
      ir_builder.AddSelect(float_type_id, y_gr_x->result_id(),
                           case_y->result_id(), case_x->result_id());

  // Get the final result by adding 0.5 to |div|.
  inst->SetOpcode(spv::Op::OpSelect);
  Instruction::OperandList new_operands;
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {is_z_max->result_id()}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {case_z->result_id()}});
  new_operands.push_back({SPV_OPERAND_TYPE_ID, {sel->result_id()}});

  inst->SetInOperands(std::move(new_operands));
  ctx->UpdateDefUse(inst);
  return true;
}

// A folding rule that will replace the TimeAMD extended instruction in the
// SPV_AMD_gcn_shader_ballot.  It returns true if the folding is successful.
// It returns False, otherwise.
//
// The instruction
//
//  %result = OpExtInst %uint64 %1 TimeAMD
//
// with
//
//  %result = OpReadClockKHR %uint64 %uint_3
//
// NOTE: TimeAMD uses subgroup scope (it is not a real time clock).
bool ReplaceTimeAMD(IRContext* ctx, Instruction* inst,
                    const std::vector<const analysis::Constant*>&) {
  InstructionBuilder ir_builder(
      ctx, inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  ctx->AddExtension("SPV_KHR_shader_clock");
  ctx->AddCapability(spv::Capability::ShaderClockKHR);

  inst->SetOpcode(spv::Op::OpReadClockKHR);
  Instruction::OperandList args;
  uint32_t subgroup_scope_id =
      ir_builder.GetUintConstantId(uint32_t(spv::Scope::Subgroup));
  args.push_back({SPV_OPERAND_TYPE_ID, {subgroup_scope_id}});
  inst->SetInOperands(std::move(args));
  ctx->UpdateDefUse(inst);

  return true;
}

class AmdExtFoldingRules : public FoldingRules {
 public:
  explicit AmdExtFoldingRules(IRContext* ctx) : FoldingRules(ctx) {}

 protected:
  virtual void AddFoldingRules() override {
    rules_[spv::Op::OpGroupIAddNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode<spv::Op::OpGroupNonUniformIAdd>);
    rules_[spv::Op::OpGroupFAddNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode<spv::Op::OpGroupNonUniformFAdd>);
    rules_[spv::Op::OpGroupUMinNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode<spv::Op::OpGroupNonUniformUMin>);
    rules_[spv::Op::OpGroupSMinNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode<spv::Op::OpGroupNonUniformSMin>);
    rules_[spv::Op::OpGroupFMinNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode<spv::Op::OpGroupNonUniformFMin>);
    rules_[spv::Op::OpGroupUMaxNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode<spv::Op::OpGroupNonUniformUMax>);
    rules_[spv::Op::OpGroupSMaxNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode<spv::Op::OpGroupNonUniformSMax>);
    rules_[spv::Op::OpGroupFMaxNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode<spv::Op::OpGroupNonUniformFMax>);

    uint32_t extension_id =
        context()->module()->GetExtInstImportId("SPV_AMD_shader_ballot");

    if (extension_id != 0) {
      ext_rules_[{extension_id, AmdShaderBallotSwizzleInvocationsAMD}]
          .push_back(ReplaceSwizzleInvocations);
      ext_rules_[{extension_id, AmdShaderBallotSwizzleInvocationsMaskedAMD}]
          .push_back(ReplaceSwizzleInvocationsMasked);
      ext_rules_[{extension_id, AmdShaderBallotWriteInvocationAMD}].push_back(
          ReplaceWriteInvocation);
      ext_rules_[{extension_id, AmdShaderBallotMbcntAMD}].push_back(
          ReplaceMbcnt);
    }

    extension_id = context()->module()->GetExtInstImportId(
        "SPV_AMD_shader_trinary_minmax");

    if (extension_id != 0) {
      ext_rules_[{extension_id, FMin3AMD}].push_back(
          ReplaceTrinaryMinMax<GLSLstd450FMin>);
      ext_rules_[{extension_id, UMin3AMD}].push_back(
          ReplaceTrinaryMinMax<GLSLstd450UMin>);
      ext_rules_[{extension_id, SMin3AMD}].push_back(
          ReplaceTrinaryMinMax<GLSLstd450SMin>);
      ext_rules_[{extension_id, FMax3AMD}].push_back(
          ReplaceTrinaryMinMax<GLSLstd450FMax>);
      ext_rules_[{extension_id, UMax3AMD}].push_back(
          ReplaceTrinaryMinMax<GLSLstd450UMax>);
      ext_rules_[{extension_id, SMax3AMD}].push_back(
          ReplaceTrinaryMinMax<GLSLstd450SMax>);
      ext_rules_[{extension_id, FMid3AMD}].push_back(
          ReplaceTrinaryMid<GLSLstd450FMin, GLSLstd450FMax, GLSLstd450FClamp>);
      ext_rules_[{extension_id, UMid3AMD}].push_back(
          ReplaceTrinaryMid<GLSLstd450UMin, GLSLstd450UMax, GLSLstd450UClamp>);
      ext_rules_[{extension_id, SMid3AMD}].push_back(
          ReplaceTrinaryMid<GLSLstd450SMin, GLSLstd450SMax, GLSLstd450SClamp>);
    }

    extension_id =
        context()->module()->GetExtInstImportId("SPV_AMD_gcn_shader");

    if (extension_id != 0) {
      ext_rules_[{extension_id, CubeFaceCoordAMD}].push_back(
          ReplaceCubeFaceCoord);
      ext_rules_[{extension_id, CubeFaceIndexAMD}].push_back(
          ReplaceCubeFaceIndex);
      ext_rules_[{extension_id, TimeAMD}].push_back(ReplaceTimeAMD);
    }
  }
};

class AmdExtConstFoldingRules : public ConstantFoldingRules {
 public:
  AmdExtConstFoldingRules(IRContext* ctx) : ConstantFoldingRules(ctx) {}

 protected:
  virtual void AddFoldingRules() override {}
};

}  // namespace

Pass::Status AmdExtensionToKhrPass::Process() {
  bool changed = false;

  // Traverse the body of the functions to replace instructions that require
  // the extensions.
  InstructionFolder folder(
      context(),
      std::unique_ptr<AmdExtFoldingRules>(new AmdExtFoldingRules(context())),
      MakeUnique<AmdExtConstFoldingRules>(context()));
  for (Function& func : *get_module()) {
    func.ForEachInst([&changed, &folder](Instruction* inst) {
      if (folder.FoldInstruction(inst)) {
        changed = true;
      }
    });
  }

  // Now that instruction that require the extensions have been removed, we can
  // remove the extension instructions.
  std::set<std::string> ext_to_remove = {"SPV_AMD_shader_ballot",
                                         "SPV_AMD_shader_trinary_minmax",
                                         "SPV_AMD_gcn_shader"};

  std::vector<Instruction*> to_be_killed;
  for (Instruction& inst : context()->module()->extensions()) {
    if (inst.opcode() == spv::Op::OpExtension) {
      if (ext_to_remove.count(inst.GetInOperand(0).AsString()) != 0) {
        to_be_killed.push_back(&inst);
      }
    }
  }

  for (Instruction& inst : context()->ext_inst_imports()) {
    if (inst.opcode() == spv::Op::OpExtInstImport) {
      if (ext_to_remove.count(inst.GetInOperand(0).AsString()) != 0) {
        to_be_killed.push_back(&inst);
      }
    }
  }

  for (Instruction* inst : to_be_killed) {
    context()->KillInst(inst);
    changed = true;
  }

  // The replacements that take place use instructions that are missing before
  // SPIR-V 1.3. If we changed something, we will have to make sure the version
  // is at least SPIR-V 1.3 to make sure those instruction can be used.
  if (changed) {
    uint32_t version = get_module()->version();
    if (version < 0x00010300 /*1.3*/) {
      get_module()->set_version(0x00010300);
    }
  }
  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
