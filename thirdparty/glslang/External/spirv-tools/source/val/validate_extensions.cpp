// Copyright (c) 2018 Google Inc.
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

// Validates correctness of extension SPIR-V instructions.
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "NonSemanticShaderDebugInfo100.h"
#include "OpenCLDebugInfo100.h"
#include "source/common_debug_info.h"
#include "source/enum_string_mapping.h"
#include "source/extensions.h"
#include "source/latest_version_glsl_std_450_header.h"
#include "source/latest_version_opencl_std_header.h"
#include "source/spirv_constant.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"
#include "spirv/unified1/NonSemanticClspvReflection.h"

namespace spvtools {
namespace val {
namespace {

std::string ReflectionInstructionName(ValidationState_t& _,
                                      const Instruction* inst) {
  spv_ext_inst_desc desc = nullptr;
  if (_.grammar().lookupExtInst(SPV_EXT_INST_TYPE_NONSEMANTIC_CLSPVREFLECTION,
                                inst->word(4), &desc) != SPV_SUCCESS ||
      !desc) {
    return std::string("Unknown ExtInst");
  }
  std::ostringstream ss;
  ss << desc->name;

  return ss.str();
}

uint32_t GetSizeTBitWidth(const ValidationState_t& _) {
  if (_.addressing_model() == spv::AddressingModel::Physical32) return 32;

  if (_.addressing_model() == spv::AddressingModel::Physical64) return 64;

  return 0;
}

bool IsIntScalar(ValidationState_t& _, uint32_t id, bool must_len32,
                 bool must_unsigned) {
  auto type = _.FindDef(id);
  if (!type || type->opcode() != spv::Op::OpTypeInt) {
    return false;
  }

  if (must_len32 && type->GetOperandAs<uint32_t>(1) != 32) {
    return false;
  }

  return !must_unsigned || type->GetOperandAs<uint32_t>(2) == 0;
}

bool IsUint32Constant(ValidationState_t& _, uint32_t id) {
  auto inst = _.FindDef(id);
  if (!inst || inst->opcode() != spv::Op::OpConstant) {
    return false;
  }

  return IsIntScalar(_, inst->type_id(), true, true);
}

uint32_t GetUint32Constant(ValidationState_t& _, uint32_t id) {
  auto inst = _.FindDef(id);
  return inst->word(3);
}

// Check that the operand of a debug info instruction |inst| at |word_index|
// is a result id of an instruction with |expected_opcode|.
spv_result_t ValidateOperandForDebugInfo(
    ValidationState_t& _, const std::string& operand_name,
    spv::Op expected_opcode, const Instruction* inst, uint32_t word_index,
    const std::function<std::string()>& ext_inst_name) {
  auto* operand = _.FindDef(inst->word(word_index));
  if (operand->opcode() != expected_opcode) {
    spv_opcode_desc desc = nullptr;
    if (_.grammar().lookupOpcode(expected_opcode, &desc) != SPV_SUCCESS ||
        !desc) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << ext_inst_name() << ": "
             << "expected operand " << operand_name << " is invalid";
    }
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << ext_inst_name() << ": "
           << "expected operand " << operand_name << " must be a result id of "
           << "Op" << desc->name;
  }
  return SPV_SUCCESS;
}

// For NonSemantic.Shader.DebugInfo.100 check that the operand of a debug info
// instruction |inst| at |word_index| is a result id of a 32-bit integer
// OpConstant instruction. For OpenCL.DebugInfo.100 the parameter is a literal
// word so cannot be validated.
spv_result_t ValidateUint32ConstantOperandForDebugInfo(
    ValidationState_t& _, const std::string& operand_name,
    const Instruction* inst, uint32_t word_index,
    const std::function<std::string()>& ext_inst_name) {
  if (!IsUint32Constant(_, inst->word(word_index))) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << ext_inst_name() << ": expected operand " << operand_name
           << " must be a result id of 32-bit unsigned OpConstant";
  }
  return SPV_SUCCESS;
}

#define CHECK_OPERAND(NAME, opcode, index)                                  \
  do {                                                                      \
    auto result = ValidateOperandForDebugInfo(_, NAME, opcode, inst, index, \
                                              ext_inst_name);               \
    if (result != SPV_SUCCESS) return result;                               \
  } while (0)

#define CHECK_CONST_UINT_OPERAND(NAME, index)                \
  if (vulkanDebugInfo) {                                     \
    auto result = ValidateUint32ConstantOperandForDebugInfo( \
        _, NAME, inst, index, ext_inst_name);                \
    if (result != SPV_SUCCESS) return result;                \
  }

// True if the operand of a debug info instruction |inst| at |word_index|
// satisfies |expectation| that is given as a function. Otherwise,
// returns false.
bool DoesDebugInfoOperandMatchExpectation(
    const ValidationState_t& _,
    const std::function<bool(CommonDebugInfoInstructions)>& expectation,
    const Instruction* inst, uint32_t word_index) {
  if (inst->words().size() <= word_index) return false;
  auto* debug_inst = _.FindDef(inst->word(word_index));
  if (debug_inst->opcode() != spv::Op::OpExtInst ||
      (debug_inst->ext_inst_type() != SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100 &&
       debug_inst->ext_inst_type() !=
           SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100) ||
      !expectation(CommonDebugInfoInstructions(debug_inst->word(4)))) {
    return false;
  }
  return true;
}

// Overload for NonSemanticShaderDebugInfo100Instructions.
bool DoesDebugInfoOperandMatchExpectation(
    const ValidationState_t& _,
    const std::function<bool(NonSemanticShaderDebugInfo100Instructions)>&
        expectation,
    const Instruction* inst, uint32_t word_index) {
  if (inst->words().size() <= word_index) return false;
  auto* debug_inst = _.FindDef(inst->word(word_index));
  if (debug_inst->opcode() != spv::Op::OpExtInst ||
      (debug_inst->ext_inst_type() !=
       SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100) ||
      !expectation(
          NonSemanticShaderDebugInfo100Instructions(debug_inst->word(4)))) {
    return false;
  }
  return true;
}

// Check that the operand of a debug info instruction |inst| at |word_index|
// is a result id of an debug info instruction whose debug instruction type
// is |expected_debug_inst|.
spv_result_t ValidateDebugInfoOperand(
    ValidationState_t& _, const std::string& debug_inst_name,
    CommonDebugInfoInstructions expected_debug_inst, const Instruction* inst,
    uint32_t word_index, const std::function<std::string()>& ext_inst_name) {
  std::function<bool(CommonDebugInfoInstructions)> expectation =
      [expected_debug_inst](CommonDebugInfoInstructions dbg_inst) {
        return dbg_inst == expected_debug_inst;
      };
  if (DoesDebugInfoOperandMatchExpectation(_, expectation, inst, word_index))
    return SPV_SUCCESS;

  spv_ext_inst_desc desc = nullptr;
  if (_.grammar().lookupExtInst(inst->ext_inst_type(), expected_debug_inst,
                                &desc) != SPV_SUCCESS ||
      !desc) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << ext_inst_name() << ": "
           << "expected operand " << debug_inst_name << " is invalid";
  }
  return _.diag(SPV_ERROR_INVALID_DATA, inst)
         << ext_inst_name() << ": "
         << "expected operand " << debug_inst_name << " must be a result id of "
         << desc->name;
}

#define CHECK_DEBUG_OPERAND(NAME, debug_opcode, index)                         \
  do {                                                                         \
    auto result = ValidateDebugInfoOperand(_, NAME, debug_opcode, inst, index, \
                                           ext_inst_name);                     \
    if (result != SPV_SUCCESS) return result;                                  \
  } while (0)

// Check that the operand of a debug info instruction |inst| at |word_index|
// is a result id of an debug info instruction with DebugTypeBasic.
spv_result_t ValidateOperandBaseType(
    ValidationState_t& _, const Instruction* inst, uint32_t word_index,
    const std::function<std::string()>& ext_inst_name) {
  return ValidateDebugInfoOperand(_, "Base Type", CommonDebugInfoDebugTypeBasic,
                                  inst, word_index, ext_inst_name);
}

// Check that the operand of a debug info instruction |inst| at |word_index|
// is a result id of a debug lexical scope instruction which is one of
// DebugCompilationUnit, DebugFunction, DebugLexicalBlock, or
// DebugTypeComposite.
spv_result_t ValidateOperandLexicalScope(
    ValidationState_t& _, const std::string& debug_inst_name,
    const Instruction* inst, uint32_t word_index,
    const std::function<std::string()>& ext_inst_name) {
  std::function<bool(CommonDebugInfoInstructions)> expectation =
      [](CommonDebugInfoInstructions dbg_inst) {
        return dbg_inst == CommonDebugInfoDebugCompilationUnit ||
               dbg_inst == CommonDebugInfoDebugFunction ||
               dbg_inst == CommonDebugInfoDebugLexicalBlock ||
               dbg_inst == CommonDebugInfoDebugTypeComposite;
      };
  if (DoesDebugInfoOperandMatchExpectation(_, expectation, inst, word_index))
    return SPV_SUCCESS;

  return _.diag(SPV_ERROR_INVALID_DATA, inst)
         << ext_inst_name() << ": "
         << "expected operand " << debug_inst_name
         << " must be a result id of a lexical scope";
}

// Check that the operand of a debug info instruction |inst| at |word_index|
// is a result id of a debug type instruction (See DebugTypeXXX in
// "4.3. Type instructions" section of OpenCL.DebugInfo.100 spec.
spv_result_t ValidateOperandDebugType(
    ValidationState_t& _, const std::string& debug_inst_name,
    const Instruction* inst, uint32_t word_index,
    const std::function<std::string()>& ext_inst_name,
    bool allow_template_param) {
  // Check for NonSemanticShaderDebugInfo100 specific types.
  if (inst->ext_inst_type() ==
      SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100) {
    std::function<bool(NonSemanticShaderDebugInfo100Instructions)> expectation =
        [](NonSemanticShaderDebugInfo100Instructions dbg_inst) {
          return dbg_inst == NonSemanticShaderDebugInfo100DebugTypeMatrix;
        };
    if (DoesDebugInfoOperandMatchExpectation(_, expectation, inst, word_index))
      return SPV_SUCCESS;
  }

  // Check for common types.
  std::function<bool(CommonDebugInfoInstructions)> expectation =
      [&allow_template_param](CommonDebugInfoInstructions dbg_inst) {
        if (allow_template_param &&
            (dbg_inst == CommonDebugInfoDebugTypeTemplateParameter ||
             dbg_inst == CommonDebugInfoDebugTypeTemplateTemplateParameter)) {
          return true;
        }
        return CommonDebugInfoDebugTypeBasic <= dbg_inst &&
               dbg_inst <= CommonDebugInfoDebugTypeTemplate;
      };
  if (DoesDebugInfoOperandMatchExpectation(_, expectation, inst, word_index))
    return SPV_SUCCESS;

  return _.diag(SPV_ERROR_INVALID_DATA, inst)
         << ext_inst_name() << ": "
         << "expected operand " << debug_inst_name
         << " is not a valid debug type";
}

spv_result_t ValidateClspvReflectionKernel(ValidationState_t& _,
                                           const Instruction* inst,
                                           uint32_t version) {
  const auto inst_name = ReflectionInstructionName(_, inst);
  const auto kernel_id = inst->GetOperandAs<uint32_t>(4);
  const auto kernel = _.FindDef(kernel_id);
  if (kernel->opcode() != spv::Op::OpFunction) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << inst_name << " does not reference a function";
  }

  bool found_kernel = false;
  for (auto entry_point : _.entry_points()) {
    if (entry_point == kernel_id) {
      found_kernel = true;
      break;
    }
  }
  if (!found_kernel) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << inst_name << " does not reference an entry-point";
  }

  const auto* exec_models = _.GetExecutionModels(kernel_id);
  if (!exec_models || exec_models->empty()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << inst_name << " does not reference an entry-point";
  }
  for (auto exec_model : *exec_models) {
    if (exec_model != spv::ExecutionModel::GLCompute) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << inst_name << " must refer only to GLCompute entry-points";
    }
  }

  auto name = _.FindDef(inst->GetOperandAs<uint32_t>(5));
  if (!name || name->opcode() != spv::Op::OpString) {
    return _.diag(SPV_ERROR_INVALID_ID, inst) << "Name must be an OpString";
  }

  const std::string name_str = name->GetOperandAs<std::string>(1);
  bool found = false;
  for (auto& desc : _.entry_point_descriptions(kernel_id)) {
    if (name_str == desc.name) {
      found = true;
      break;
    }
  }
  if (!found) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Name must match an entry-point for Kernel";
  }

  const auto num_operands = inst->operands().size();
  if (version < 5 && num_operands > 6) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Version " << version << " of the " << inst_name
           << " instruction can only have 2 additional operands";
  }

  if (num_operands > 6) {
    const auto num_args_id = inst->GetOperandAs<uint32_t>(6);
    if (!IsUint32Constant(_, num_args_id)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "NumArguments must be a 32-bit unsigned integer OpConstant";
    }
  }

  if (num_operands > 7) {
    const auto flags_id = inst->GetOperandAs<uint32_t>(7);
    if (!IsUint32Constant(_, flags_id)) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Flags must be a 32-bit unsigned integer OpConstant";
    }
  }

  if (num_operands > 8) {
    const auto atts_id = inst->GetOperandAs<uint32_t>(8);
    if (_.GetIdOpcode(atts_id) != spv::Op::OpString) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Attributes must be an OpString";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionArgumentInfo(ValidationState_t& _,
                                                 const Instruction* inst) {
  const auto num_operands = inst->operands().size();
  if (_.GetIdOpcode(inst->GetOperandAs<uint32_t>(4)) != spv::Op::OpString) {
    return _.diag(SPV_ERROR_INVALID_ID, inst) << "Name must be an OpString";
  }
  if (num_operands > 5) {
    if (_.GetIdOpcode(inst->GetOperandAs<uint32_t>(5)) != spv::Op::OpString) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "TypeName must be an OpString";
    }
  }
  if (num_operands > 6) {
    if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "AddressQualifier must be a 32-bit unsigned integer "
                "OpConstant";
    }
  }
  if (num_operands > 7) {
    if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(7))) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "AccessQualifier must be a 32-bit unsigned integer "
                "OpConstant";
    }
  }
  if (num_operands > 8) {
    if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(8))) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "TypeQualifier must be a 32-bit unsigned integer "
                "OpConstant";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateKernelDecl(ValidationState_t& _, const Instruction* inst) {
  const auto decl_id = inst->GetOperandAs<uint32_t>(4);
  const auto decl = _.FindDef(decl_id);
  if (!decl || decl->opcode() != spv::Op::OpExtInst) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Kernel must be a Kernel extended instruction";
  }

  if (decl->GetOperandAs<uint32_t>(2) != inst->GetOperandAs<uint32_t>(2)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Kernel must be from the same extended instruction import";
  }

  const auto ext_inst =
      decl->GetOperandAs<NonSemanticClspvReflectionInstructions>(3);
  if (ext_inst != NonSemanticClspvReflectionKernel) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Kernel must be a Kernel extended instruction";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateArgInfo(ValidationState_t& _, const Instruction* inst,
                             uint32_t info_index) {
  auto info = _.FindDef(inst->GetOperandAs<uint32_t>(info_index));
  if (!info || info->opcode() != spv::Op::OpExtInst) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "ArgInfo must be an ArgumentInfo extended instruction";
  }

  if (info->GetOperandAs<uint32_t>(2) != inst->GetOperandAs<uint32_t>(2)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "ArgInfo must be from the same extended instruction import";
  }

  auto ext_inst = info->GetOperandAs<NonSemanticClspvReflectionInstructions>(3);
  if (ext_inst != NonSemanticClspvReflectionArgumentInfo) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "ArgInfo must be an ArgumentInfo extended instruction";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionArgumentBuffer(ValidationState_t& _,
                                                   const Instruction* inst) {
  const auto num_operands = inst->operands().size();
  if (auto error = ValidateKernelDecl(_, inst)) {
    return error;
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Ordinal must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "DescriptorSet must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(7))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Binding must be a 32-bit unsigned integer OpConstant";
  }

  if (num_operands == 9) {
    if (auto error = ValidateArgInfo(_, inst, 8)) {
      return error;
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionArgumentOffsetBuffer(
    ValidationState_t& _, const Instruction* inst) {
  const auto num_operands = inst->operands().size();
  if (auto error = ValidateKernelDecl(_, inst)) {
    return error;
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Ordinal must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "DescriptorSet must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(7))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Binding must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(8))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Offset must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(9))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Size must be a 32-bit unsigned integer OpConstant";
  }

  if (num_operands == 11) {
    if (auto error = ValidateArgInfo(_, inst, 10)) {
      return error;
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionArgumentPushConstant(
    ValidationState_t& _, const Instruction* inst) {
  const auto num_operands = inst->operands().size();
  if (auto error = ValidateKernelDecl(_, inst)) {
    return error;
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Ordinal must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Offset must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(7))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Size must be a 32-bit unsigned integer OpConstant";
  }

  if (num_operands == 9) {
    if (auto error = ValidateArgInfo(_, inst, 8)) {
      return error;
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionArgumentWorkgroup(ValidationState_t& _,
                                                      const Instruction* inst) {
  const auto num_operands = inst->operands().size();
  if (auto error = ValidateKernelDecl(_, inst)) {
    return error;
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Ordinal must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "SpecId must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(7))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "ElemSize must be a 32-bit unsigned integer OpConstant";
  }

  if (num_operands == 9) {
    if (auto error = ValidateArgInfo(_, inst, 8)) {
      return error;
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionSpecConstantTriple(
    ValidationState_t& _, const Instruction* inst) {
  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(4))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "X must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Y must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Z must be a 32-bit unsigned integer OpConstant";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionSpecConstantWorkDim(
    ValidationState_t& _, const Instruction* inst) {
  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(4))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Dim must be a 32-bit unsigned integer OpConstant";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionPushConstant(ValidationState_t& _,
                                                 const Instruction* inst) {
  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(4))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Offset must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Size must be a 32-bit unsigned integer OpConstant";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionInitializedData(ValidationState_t& _,
                                                    const Instruction* inst) {
  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(4))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "DescriptorSet must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Binding must be a 32-bit unsigned integer OpConstant";
  }

  if (_.GetIdOpcode(inst->GetOperandAs<uint32_t>(6)) != spv::Op::OpString) {
    return _.diag(SPV_ERROR_INVALID_ID, inst) << "Data must be an OpString";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionSampler(ValidationState_t& _,
                                            const Instruction* inst) {
  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(4))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "DescriptorSet must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Binding must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Mask must be a 32-bit unsigned integer OpConstant";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionPropertyRequiredWorkgroupSize(
    ValidationState_t& _, const Instruction* inst) {
  if (auto error = ValidateKernelDecl(_, inst)) {
    return error;
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "X must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Y must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(7))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Z must be a 32-bit unsigned integer OpConstant";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionSubgroupMaxSize(ValidationState_t& _,
                                                    const Instruction* inst) {
  const auto size_id = inst->GetOperandAs<uint32_t>(4);
  if (!IsUint32Constant(_, size_id)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Size must be a 32-bit unsigned integer OpConstant";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionPointerRelocation(ValidationState_t& _,
                                                      const Instruction* inst) {
  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(4))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "ObjectOffset must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "PointerOffset must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "PointerSize must be a 32-bit unsigned integer OpConstant";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionImageMetadataPushConstant(
    ValidationState_t& _, const Instruction* inst) {
  if (auto error = ValidateKernelDecl(_, inst)) {
    return error;
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Ordinal must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Offset must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(7))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Size must be a 32-bit unsigned integer OpConstant";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionImageMetadataUniform(
    ValidationState_t& _, const Instruction* inst) {
  if (auto error = ValidateKernelDecl(_, inst)) {
    return error;
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Ordinal must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "DescriptorSet must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(7))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Binding must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(8))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Offset must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(9))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Size must be a 32-bit unsigned integer OpConstant";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionPushConstantData(ValidationState_t& _,
                                                     const Instruction* inst) {
  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(4))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Offset must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Size must be a 32-bit unsigned integer OpConstant";
  }

  if (_.GetIdOpcode(inst->GetOperandAs<uint32_t>(6)) != spv::Op::OpString) {
    return _.diag(SPV_ERROR_INVALID_ID, inst) << "Data must be an OpString";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionPrintfInfo(ValidationState_t& _,
                                               const Instruction* inst) {
  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(4))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "PrintfID must be a 32-bit unsigned integer OpConstant";
  }

  if (_.GetIdOpcode(inst->GetOperandAs<uint32_t>(5)) != spv::Op::OpString) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "FormatString must be an OpString";
  }

  for (size_t i = 6; i < inst->operands().size(); ++i) {
    if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(i))) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "ArgumentSizes must be a 32-bit unsigned integer OpConstant";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionPrintfStorageBuffer(
    ValidationState_t& _, const Instruction* inst) {
  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(4))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "DescriptorSet must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Binding must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Size must be a 32-bit unsigned integer OpConstant";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionPrintfPushConstant(
    ValidationState_t& _, const Instruction* inst) {
  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(4))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Offset must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(5))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Size must be a 32-bit unsigned integer OpConstant";
  }

  if (!IsUint32Constant(_, inst->GetOperandAs<uint32_t>(6))) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "BufferSize must be a 32-bit unsigned integer OpConstant";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateClspvReflectionInstruction(ValidationState_t& _,
                                                const Instruction* inst,
                                                uint32_t version) {
  if (!_.IsVoidType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Return Type must be OpTypeVoid";
  }

  uint32_t required_version = 0;
  const auto ext_inst =
      inst->GetOperandAs<NonSemanticClspvReflectionInstructions>(3);
  switch (ext_inst) {
    case NonSemanticClspvReflectionKernel:
    case NonSemanticClspvReflectionArgumentInfo:
    case NonSemanticClspvReflectionArgumentStorageBuffer:
    case NonSemanticClspvReflectionArgumentUniform:
    case NonSemanticClspvReflectionArgumentPodStorageBuffer:
    case NonSemanticClspvReflectionArgumentPodUniform:
    case NonSemanticClspvReflectionArgumentPodPushConstant:
    case NonSemanticClspvReflectionArgumentSampledImage:
    case NonSemanticClspvReflectionArgumentStorageImage:
    case NonSemanticClspvReflectionArgumentSampler:
    case NonSemanticClspvReflectionArgumentWorkgroup:
    case NonSemanticClspvReflectionSpecConstantWorkgroupSize:
    case NonSemanticClspvReflectionSpecConstantGlobalOffset:
    case NonSemanticClspvReflectionSpecConstantWorkDim:
    case NonSemanticClspvReflectionPushConstantGlobalOffset:
    case NonSemanticClspvReflectionPushConstantEnqueuedLocalSize:
    case NonSemanticClspvReflectionPushConstantGlobalSize:
    case NonSemanticClspvReflectionPushConstantRegionOffset:
    case NonSemanticClspvReflectionPushConstantNumWorkgroups:
    case NonSemanticClspvReflectionPushConstantRegionGroupOffset:
    case NonSemanticClspvReflectionConstantDataStorageBuffer:
    case NonSemanticClspvReflectionConstantDataUniform:
    case NonSemanticClspvReflectionLiteralSampler:
    case NonSemanticClspvReflectionPropertyRequiredWorkgroupSize:
      required_version = 1;
      break;
    case NonSemanticClspvReflectionSpecConstantSubgroupMaxSize:
      required_version = 2;
      break;
    case NonSemanticClspvReflectionArgumentPointerPushConstant:
    case NonSemanticClspvReflectionArgumentPointerUniform:
    case NonSemanticClspvReflectionProgramScopeVariablesStorageBuffer:
    case NonSemanticClspvReflectionProgramScopeVariablePointerRelocation:
    case NonSemanticClspvReflectionImageArgumentInfoChannelOrderPushConstant:
    case NonSemanticClspvReflectionImageArgumentInfoChannelDataTypePushConstant:
    case NonSemanticClspvReflectionImageArgumentInfoChannelOrderUniform:
    case NonSemanticClspvReflectionImageArgumentInfoChannelDataTypeUniform:
      required_version = 3;
      break;
    case NonSemanticClspvReflectionArgumentStorageTexelBuffer:
    case NonSemanticClspvReflectionArgumentUniformTexelBuffer:
      required_version = 4;
      break;
    case NonSemanticClspvReflectionConstantDataPointerPushConstant:
    case NonSemanticClspvReflectionProgramScopeVariablePointerPushConstant:
    case NonSemanticClspvReflectionPrintfInfo:
    case NonSemanticClspvReflectionPrintfBufferStorageBuffer:
    case NonSemanticClspvReflectionPrintfBufferPointerPushConstant:
      required_version = 5;
      break;
    default:
      break;
  }
  if (version < required_version) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << ReflectionInstructionName(_, inst) << " requires version "
           << required_version << ", but parsed version is " << version;
  }

  switch (ext_inst) {
    case NonSemanticClspvReflectionKernel:
      return ValidateClspvReflectionKernel(_, inst, version);
    case NonSemanticClspvReflectionArgumentInfo:
      return ValidateClspvReflectionArgumentInfo(_, inst);
    case NonSemanticClspvReflectionArgumentStorageBuffer:
    case NonSemanticClspvReflectionArgumentUniform:
    case NonSemanticClspvReflectionArgumentSampledImage:
    case NonSemanticClspvReflectionArgumentStorageImage:
    case NonSemanticClspvReflectionArgumentSampler:
    case NonSemanticClspvReflectionArgumentStorageTexelBuffer:
    case NonSemanticClspvReflectionArgumentUniformTexelBuffer:
      return ValidateClspvReflectionArgumentBuffer(_, inst);
    case NonSemanticClspvReflectionArgumentPodStorageBuffer:
    case NonSemanticClspvReflectionArgumentPodUniform:
    case NonSemanticClspvReflectionArgumentPointerUniform:
      return ValidateClspvReflectionArgumentOffsetBuffer(_, inst);
    case NonSemanticClspvReflectionArgumentPodPushConstant:
    case NonSemanticClspvReflectionArgumentPointerPushConstant:
      return ValidateClspvReflectionArgumentPushConstant(_, inst);
    case NonSemanticClspvReflectionArgumentWorkgroup:
      return ValidateClspvReflectionArgumentWorkgroup(_, inst);
    case NonSemanticClspvReflectionSpecConstantWorkgroupSize:
    case NonSemanticClspvReflectionSpecConstantGlobalOffset:
      return ValidateClspvReflectionSpecConstantTriple(_, inst);
    case NonSemanticClspvReflectionSpecConstantWorkDim:
      return ValidateClspvReflectionSpecConstantWorkDim(_, inst);
    case NonSemanticClspvReflectionPushConstantGlobalOffset:
    case NonSemanticClspvReflectionPushConstantEnqueuedLocalSize:
    case NonSemanticClspvReflectionPushConstantGlobalSize:
    case NonSemanticClspvReflectionPushConstantRegionOffset:
    case NonSemanticClspvReflectionPushConstantNumWorkgroups:
    case NonSemanticClspvReflectionPushConstantRegionGroupOffset:
      return ValidateClspvReflectionPushConstant(_, inst);
    case NonSemanticClspvReflectionConstantDataStorageBuffer:
    case NonSemanticClspvReflectionConstantDataUniform:
    case NonSemanticClspvReflectionProgramScopeVariablesStorageBuffer:
      return ValidateClspvReflectionInitializedData(_, inst);
    case NonSemanticClspvReflectionLiteralSampler:
      return ValidateClspvReflectionSampler(_, inst);
    case NonSemanticClspvReflectionPropertyRequiredWorkgroupSize:
      return ValidateClspvReflectionPropertyRequiredWorkgroupSize(_, inst);
    case NonSemanticClspvReflectionSpecConstantSubgroupMaxSize:
      return ValidateClspvReflectionSubgroupMaxSize(_, inst);
    case NonSemanticClspvReflectionProgramScopeVariablePointerRelocation:
      return ValidateClspvReflectionPointerRelocation(_, inst);
    case NonSemanticClspvReflectionImageArgumentInfoChannelOrderPushConstant:
    case NonSemanticClspvReflectionImageArgumentInfoChannelDataTypePushConstant:
      return ValidateClspvReflectionImageMetadataPushConstant(_, inst);
    case NonSemanticClspvReflectionImageArgumentInfoChannelOrderUniform:
    case NonSemanticClspvReflectionImageArgumentInfoChannelDataTypeUniform:
      return ValidateClspvReflectionImageMetadataUniform(_, inst);
    case NonSemanticClspvReflectionConstantDataPointerPushConstant:
    case NonSemanticClspvReflectionProgramScopeVariablePointerPushConstant:
      return ValidateClspvReflectionPushConstantData(_, inst);
    case NonSemanticClspvReflectionPrintfInfo:
      return ValidateClspvReflectionPrintfInfo(_, inst);
    case NonSemanticClspvReflectionPrintfBufferStorageBuffer:
      return ValidateClspvReflectionPrintfStorageBuffer(_, inst);
    case NonSemanticClspvReflectionPrintfBufferPointerPushConstant:
      return ValidateClspvReflectionPrintfPushConstant(_, inst);
    default:
      break;
  }

  return SPV_SUCCESS;
}

bool IsConstIntScalarTypeWith32Or64Bits(ValidationState_t& _,
                                        Instruction* instr) {
  if (instr->opcode() != spv::Op::OpConstant) return false;
  if (!_.IsIntScalarType(instr->type_id())) return false;
  uint32_t size_in_bits = _.GetBitWidth(instr->type_id());
  return size_in_bits == 32 || size_in_bits == 64;
}

bool IsConstWithIntScalarType(ValidationState_t& _, const Instruction* inst,
                              uint32_t word_index) {
  auto* int_scalar_const = _.FindDef(inst->word(word_index));
  if (int_scalar_const->opcode() == spv::Op::OpConstant &&
      _.IsIntScalarType(int_scalar_const->type_id())) {
    return true;
  }
  return false;
}

bool IsDebugVariableWithIntScalarType(ValidationState_t& _,
                                      const Instruction* inst,
                                      uint32_t word_index) {
  auto* dbg_int_scalar_var = _.FindDef(inst->word(word_index));
  if (CommonDebugInfoInstructions(dbg_int_scalar_var->word(4)) ==
          CommonDebugInfoDebugLocalVariable ||
      CommonDebugInfoInstructions(dbg_int_scalar_var->word(4)) ==
          CommonDebugInfoDebugGlobalVariable) {
    auto* dbg_type = _.FindDef(dbg_int_scalar_var->word(6));
    if (CommonDebugInfoInstructions(dbg_type->word(4)) ==
        CommonDebugInfoDebugTypeBasic) {
      const spv_ext_inst_type_t ext_inst_type =
          spv_ext_inst_type_t(inst->ext_inst_type());
      const bool vulkanDebugInfo =
          ext_inst_type == SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100;
      uint32_t encoding = dbg_type->word(7);
      if (!vulkanDebugInfo || IsUint32Constant(_, encoding)) {
        auto ocl_encoding = OpenCLDebugInfo100DebugBaseTypeAttributeEncoding(
            vulkanDebugInfo ? GetUint32Constant(_, encoding) : encoding);
        if (ocl_encoding == OpenCLDebugInfo100Signed ||
            ocl_encoding == OpenCLDebugInfo100Unsigned) {
          return true;
        }
      }
    }
  }
  return false;
}

}  // anonymous namespace

spv_result_t ValidateExtension(ValidationState_t& _, const Instruction* inst) {
  if (_.version() < SPV_SPIRV_VERSION_WORD(1, 4)) {
    std::string extension = GetExtensionString(&(inst->c_inst()));
    if (extension ==
            ExtensionToString(kSPV_KHR_workgroup_memory_explicit_layout) ||
        extension == ExtensionToString(kSPV_EXT_mesh_shader) ||
        extension == ExtensionToString(kSPV_NV_shader_invocation_reorder)) {
      return _.diag(SPV_ERROR_WRONG_VERSION, inst)
             << extension << " extension requires SPIR-V version 1.4 or later.";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateExtInstImport(ValidationState_t& _,
                                   const Instruction* inst) {
  const auto name_id = 1;
  if (_.version() <= SPV_SPIRV_VERSION_WORD(1, 5) &&
      !_.HasExtension(kSPV_KHR_non_semantic_info)) {
    const std::string name = inst->GetOperandAs<std::string>(name_id);
    if (name.find("NonSemantic.") == 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "NonSemantic extended instruction sets cannot be declared "
                "without SPV_KHR_non_semantic_info.";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateExtInst(ValidationState_t& _, const Instruction* inst) {
  const uint32_t result_type = inst->type_id();
  const uint32_t num_operands = static_cast<uint32_t>(inst->operands().size());

  const uint32_t ext_inst_set = inst->word(3);
  const uint32_t ext_inst_index = inst->word(4);
  const spv_ext_inst_type_t ext_inst_type =
      spv_ext_inst_type_t(inst->ext_inst_type());

  auto ext_inst_name = [&_, ext_inst_set, ext_inst_type, ext_inst_index]() {
    spv_ext_inst_desc desc = nullptr;
    if (_.grammar().lookupExtInst(ext_inst_type, ext_inst_index, &desc) !=
            SPV_SUCCESS ||
        !desc) {
      return std::string("Unknown ExtInst");
    }

    auto* import_inst = _.FindDef(ext_inst_set);
    assert(import_inst);

    std::ostringstream ss;
    ss << import_inst->GetOperandAs<std::string>(1);
    ss << " ";
    ss << desc->name;

    return ss.str();
  };

  if (ext_inst_type == SPV_EXT_INST_TYPE_GLSL_STD_450) {
    const GLSLstd450 ext_inst_key = GLSLstd450(ext_inst_index);
    switch (ext_inst_key) {
      case GLSLstd450Round:
      case GLSLstd450RoundEven:
      case GLSLstd450FAbs:
      case GLSLstd450Trunc:
      case GLSLstd450FSign:
      case GLSLstd450Floor:
      case GLSLstd450Ceil:
      case GLSLstd450Fract:
      case GLSLstd450Sqrt:
      case GLSLstd450InverseSqrt:
      case GLSLstd450FMin:
      case GLSLstd450FMax:
      case GLSLstd450FClamp:
      case GLSLstd450FMix:
      case GLSLstd450Step:
      case GLSLstd450SmoothStep:
      case GLSLstd450Fma:
      case GLSLstd450Normalize:
      case GLSLstd450FaceForward:
      case GLSLstd450Reflect:
      case GLSLstd450NMin:
      case GLSLstd450NMax:
      case GLSLstd450NClamp: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case GLSLstd450SAbs:
      case GLSLstd450SSign:
      case GLSLstd450UMin:
      case GLSLstd450SMin:
      case GLSLstd450UMax:
      case GLSLstd450SMax:
      case GLSLstd450UClamp:
      case GLSLstd450SClamp:
      case GLSLstd450FindILsb:
      case GLSLstd450FindUMsb:
      case GLSLstd450FindSMsb: {
        if (!_.IsIntScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int scalar or vector type";
        }

        const uint32_t result_type_bit_width = _.GetBitWidth(result_type);
        const uint32_t result_type_dimension = _.GetDimension(result_type);

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (!operand_type || !_.IsIntScalarOrVectorType(operand_type)) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected all operands to be int scalars or vectors";
          }

          if (result_type_dimension != _.GetDimension(operand_type)) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected all operands to have the same dimension as "
                   << "Result Type";
          }

          if (result_type_bit_width != _.GetBitWidth(operand_type)) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected all operands to have the same bit width as "
                   << "Result Type";
          }

          if (ext_inst_key == GLSLstd450FindUMsb ||
              ext_inst_key == GLSLstd450FindSMsb) {
            if (result_type_bit_width != 32) {
              return _.diag(SPV_ERROR_INVALID_DATA, inst)
                     << ext_inst_name() << ": "
                     << "this instruction is currently limited to 32-bit width "
                     << "components";
            }
          }
        }
        break;
      }

      case GLSLstd450Radians:
      case GLSLstd450Degrees:
      case GLSLstd450Sin:
      case GLSLstd450Cos:
      case GLSLstd450Tan:
      case GLSLstd450Asin:
      case GLSLstd450Acos:
      case GLSLstd450Atan:
      case GLSLstd450Sinh:
      case GLSLstd450Cosh:
      case GLSLstd450Tanh:
      case GLSLstd450Asinh:
      case GLSLstd450Acosh:
      case GLSLstd450Atanh:
      case GLSLstd450Exp:
      case GLSLstd450Exp2:
      case GLSLstd450Log:
      case GLSLstd450Log2:
      case GLSLstd450Atan2:
      case GLSLstd450Pow: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 16 or 32-bit scalar or "
                    "vector float type";
        }

        const uint32_t result_type_bit_width = _.GetBitWidth(result_type);
        if (result_type_bit_width != 16 && result_type_bit_width != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 16 or 32-bit scalar or "
                    "vector float type";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case GLSLstd450Determinant: {
        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        uint32_t num_rows = 0;
        uint32_t num_cols = 0;
        uint32_t col_type = 0;
        uint32_t component_type = 0;
        if (!_.GetMatrixTypeInfo(x_type, &num_rows, &num_cols, &col_type,
                                 &component_type) ||
            num_rows != num_cols) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X to be a square matrix";
        }

        if (result_type != component_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X component type to be equal to "
                 << "Result Type";
        }
        break;
      }

      case GLSLstd450MatrixInverse: {
        uint32_t num_rows = 0;
        uint32_t num_cols = 0;
        uint32_t col_type = 0;
        uint32_t component_type = 0;
        if (!_.GetMatrixTypeInfo(result_type, &num_rows, &num_cols, &col_type,
                                 &component_type) ||
            num_rows != num_cols) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a square matrix";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (result_type != x_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }
        break;
      }

      case GLSLstd450Modf: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or vector float type";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t i_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        spv::StorageClass i_storage_class;
        uint32_t i_data_type = 0;
        if (!_.GetPointerTypeInfo(i_type, &i_data_type, &i_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand I to be a pointer";
        }

        if (i_data_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand I data type to be equal to Result Type";
        }

        break;
      }

      case GLSLstd450ModfStruct: {
        std::vector<uint32_t> result_types;
        if (!_.GetStructMemberTypes(result_type, &result_types) ||
            result_types.size() != 2 ||
            !_.IsFloatScalarOrVectorType(result_types[0]) ||
            result_types[1] != result_types[0]) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a struct with two identical "
                 << "scalar or vector float type members";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (x_type != result_types[0]) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to members of "
                 << "Result Type struct";
        }
        break;
      }

      case GLSLstd450Frexp: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or vector float type";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t exp_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        spv::StorageClass exp_storage_class;
        uint32_t exp_data_type = 0;
        if (!_.GetPointerTypeInfo(exp_type, &exp_data_type,
                                  &exp_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Exp to be a pointer";
        }

        if (!_.IsIntScalarOrVectorType(exp_data_type) ||
            (!_.HasExtension(kSPV_AMD_gpu_shader_int16) &&
             _.GetBitWidth(exp_data_type) != 32) ||
            (_.HasExtension(kSPV_AMD_gpu_shader_int16) &&
             _.GetBitWidth(exp_data_type) != 16 &&
             _.GetBitWidth(exp_data_type) != 32)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Exp data type to be a "
                 << (_.HasExtension(kSPV_AMD_gpu_shader_int16)
                         ? "16-bit or 32-bit "
                         : "32-bit ")
                 << "int scalar or vector type";
        }

        if (_.GetDimension(result_type) != _.GetDimension(exp_data_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Exp data type to have the same component "
                 << "number as Result Type";
        }

        break;
      }

      case GLSLstd450Ldexp: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or vector float type";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t exp_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        if (!_.IsIntScalarOrVectorType(exp_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Exp to be a 32-bit int scalar "
                 << "or vector type";
        }

        if (_.GetDimension(result_type) != _.GetDimension(exp_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Exp to have the same component "
                 << "number as Result Type";
        }

        break;
      }

      case GLSLstd450FrexpStruct: {
        std::vector<uint32_t> result_types;
        if (!_.GetStructMemberTypes(result_type, &result_types) ||
            result_types.size() != 2 ||
            !_.IsFloatScalarOrVectorType(result_types[0]) ||
            !_.IsIntScalarOrVectorType(result_types[1]) ||
            (!_.HasExtension(kSPV_AMD_gpu_shader_int16) &&
             _.GetBitWidth(result_types[1]) != 32) ||
            (_.HasExtension(kSPV_AMD_gpu_shader_int16) &&
             _.GetBitWidth(result_types[1]) != 16 &&
             _.GetBitWidth(result_types[1]) != 32) ||
            _.GetDimension(result_types[0]) !=
                _.GetDimension(result_types[1])) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a struct with two members, "
                 << "first member a float scalar or vector, second member a "
                 << (_.HasExtension(kSPV_AMD_gpu_shader_int16)
                         ? "16-bit or 32-bit "
                         : "32-bit ")
                 << "int scalar or vector with the same number of "
                 << "components as the first member";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (x_type != result_types[0]) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to the first member "
                 << "of Result Type struct";
        }
        break;
      }

      case GLSLstd450PackSnorm4x8:
      case GLSLstd450PackUnorm4x8: {
        if (!_.IsIntScalarType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be 32-bit int scalar type";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatVectorType(v_type) || _.GetDimension(v_type) != 4 ||
            _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand V to be a 32-bit float vector of size 4";
        }
        break;
      }

      case GLSLstd450PackSnorm2x16:
      case GLSLstd450PackUnorm2x16:
      case GLSLstd450PackHalf2x16: {
        if (!_.IsIntScalarType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be 32-bit int scalar type";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatVectorType(v_type) || _.GetDimension(v_type) != 2 ||
            _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand V to be a 32-bit float vector of size 2";
        }
        break;
      }

      case GLSLstd450PackDouble2x32: {
        if (!_.IsFloatScalarType(result_type) ||
            _.GetBitWidth(result_type) != 64) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be 64-bit float scalar type";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsIntVectorType(v_type) || _.GetDimension(v_type) != 2 ||
            _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand V to be a 32-bit int vector of size 2";
        }
        break;
      }

      case GLSLstd450UnpackSnorm4x8:
      case GLSLstd450UnpackUnorm4x8: {
        if (!_.IsFloatVectorType(result_type) ||
            _.GetDimension(result_type) != 4 ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit float vector of size "
                    "4";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsIntScalarType(v_type) || _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a 32-bit int scalar";
        }
        break;
      }

      case GLSLstd450UnpackSnorm2x16:
      case GLSLstd450UnpackUnorm2x16:
      case GLSLstd450UnpackHalf2x16: {
        if (!_.IsFloatVectorType(result_type) ||
            _.GetDimension(result_type) != 2 ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit float vector of size "
                    "2";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsIntScalarType(v_type) || _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a 32-bit int scalar";
        }
        break;
      }

      case GLSLstd450UnpackDouble2x32: {
        if (!_.IsIntVectorType(result_type) ||
            _.GetDimension(result_type) != 2 ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit int vector of size "
                    "2";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarType(v_type) || _.GetBitWidth(v_type) != 64) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand V to be a 64-bit float scalar";
        }
        break;
      }

      case GLSLstd450Length: {
        if (!_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar type";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarOrVectorType(x_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X to be of float scalar or vector type";
        }

        if (result_type != _.GetComponentType(x_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X component type to be equal to Result "
                    "Type";
        }
        break;
      }

      case GLSLstd450Distance: {
        if (!_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar type";
        }

        const uint32_t p0_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarOrVectorType(p0_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P0 to be of float scalar or vector type";
        }

        if (result_type != _.GetComponentType(p0_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P0 component type to be equal to "
                 << "Result Type";
        }

        const uint32_t p1_type = _.GetOperandTypeId(inst, 5);
        if (!_.IsFloatScalarOrVectorType(p1_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P1 to be of float scalar or vector type";
        }

        if (result_type != _.GetComponentType(p1_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P1 component type to be equal to "
                 << "Result Type";
        }

        if (_.GetDimension(p0_type) != _.GetDimension(p1_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operands P0 and P1 to have the same number of "
                 << "components";
        }
        break;
      }

      case GLSLstd450Cross: {
        if (!_.IsFloatVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float vector type";
        }

        if (_.GetDimension(result_type) != 3) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have 3 components";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t y_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        if (y_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Y type to be equal to Result Type";
        }
        break;
      }

      case GLSLstd450Refract: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t i_type = _.GetOperandTypeId(inst, 4);
        const uint32_t n_type = _.GetOperandTypeId(inst, 5);
        const uint32_t eta_type = _.GetOperandTypeId(inst, 6);

        if (result_type != i_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand I to be of type equal to Result Type";
        }

        if (result_type != n_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand N to be of type equal to Result Type";
        }

        if (!_.IsFloatScalarType(eta_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Eta to be a float scalar";
        }
        break;
      }

      case GLSLstd450InterpolateAtCentroid:
      case GLSLstd450InterpolateAtSample:
      case GLSLstd450InterpolateAtOffset: {
        if (!_.HasCapability(spv::Capability::InterpolationFunction)) {
          return _.diag(SPV_ERROR_INVALID_CAPABILITY, inst)
                 << ext_inst_name()
                 << " requires capability InterpolationFunction";
        }

        if (!_.IsFloatScalarOrVectorType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit float scalar "
                 << "or vector type";
        }

        // If HLSL legalization and first operand is an OpLoad, use load
        // pointer as the interpolant lvalue. Else use interpolate first
        // operand.
        uint32_t interp_id = inst->GetOperandAs<uint32_t>(4);
        auto* interp_inst = _.FindDef(interp_id);
        uint32_t interpolant_type = (_.options()->before_hlsl_legalization &&
                                     interp_inst->opcode() == spv::Op::OpLoad)
                                        ? _.GetOperandTypeId(interp_inst, 2)
                                        : _.GetOperandTypeId(inst, 4);

        spv::StorageClass interpolant_storage_class;
        uint32_t interpolant_data_type = 0;
        if (!_.GetPointerTypeInfo(interpolant_type, &interpolant_data_type,
                                  &interpolant_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Interpolant to be a pointer";
        }

        if (result_type != interpolant_data_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Interpolant data type to be equal to Result Type";
        }

        if (interpolant_storage_class != spv::StorageClass::Input) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Interpolant storage class to be Input";
        }

        if (ext_inst_key == GLSLstd450InterpolateAtSample) {
          const uint32_t sample_type = _.GetOperandTypeId(inst, 5);
          if (!_.IsIntScalarType(sample_type) ||
              _.GetBitWidth(sample_type) != 32) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected Sample to be 32-bit integer";
          }
        }

        if (ext_inst_key == GLSLstd450InterpolateAtOffset) {
          const uint32_t offset_type = _.GetOperandTypeId(inst, 5);
          if (!_.IsFloatVectorType(offset_type) ||
              _.GetDimension(offset_type) != 2 ||
              _.GetBitWidth(offset_type) != 32) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected Offset to be a vector of 2 32-bit floats";
          }
        }

        _.function(inst->function()->id())
            ->RegisterExecutionModelLimitation(
                spv::ExecutionModel::Fragment,
                ext_inst_name() +
                    std::string(" requires Fragment execution model"));
        break;
      }

      case GLSLstd450IMix: {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Extended instruction GLSLstd450IMix is not supported";
      }

      case GLSLstd450Bad: {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Encountered extended instruction GLSLstd450Bad";
      }

      case GLSLstd450Count: {
        assert(0);
        break;
      }
    }
  } else if (ext_inst_type == SPV_EXT_INST_TYPE_OPENCL_STD) {
    const OpenCLLIB::Entrypoints ext_inst_key =
        OpenCLLIB::Entrypoints(ext_inst_index);
    switch (ext_inst_key) {
      case OpenCLLIB::Acos:
      case OpenCLLIB::Acosh:
      case OpenCLLIB::Acospi:
      case OpenCLLIB::Asin:
      case OpenCLLIB::Asinh:
      case OpenCLLIB::Asinpi:
      case OpenCLLIB::Atan:
      case OpenCLLIB::Atan2:
      case OpenCLLIB::Atanh:
      case OpenCLLIB::Atanpi:
      case OpenCLLIB::Atan2pi:
      case OpenCLLIB::Cbrt:
      case OpenCLLIB::Ceil:
      case OpenCLLIB::Copysign:
      case OpenCLLIB::Cos:
      case OpenCLLIB::Cosh:
      case OpenCLLIB::Cospi:
      case OpenCLLIB::Erfc:
      case OpenCLLIB::Erf:
      case OpenCLLIB::Exp:
      case OpenCLLIB::Exp2:
      case OpenCLLIB::Exp10:
      case OpenCLLIB::Expm1:
      case OpenCLLIB::Fabs:
      case OpenCLLIB::Fdim:
      case OpenCLLIB::Floor:
      case OpenCLLIB::Fma:
      case OpenCLLIB::Fmax:
      case OpenCLLIB::Fmin:
      case OpenCLLIB::Fmod:
      case OpenCLLIB::Hypot:
      case OpenCLLIB::Lgamma:
      case OpenCLLIB::Log:
      case OpenCLLIB::Log2:
      case OpenCLLIB::Log10:
      case OpenCLLIB::Log1p:
      case OpenCLLIB::Logb:
      case OpenCLLIB::Mad:
      case OpenCLLIB::Maxmag:
      case OpenCLLIB::Minmag:
      case OpenCLLIB::Nextafter:
      case OpenCLLIB::Pow:
      case OpenCLLIB::Powr:
      case OpenCLLIB::Remainder:
      case OpenCLLIB::Rint:
      case OpenCLLIB::Round:
      case OpenCLLIB::Rsqrt:
      case OpenCLLIB::Sin:
      case OpenCLLIB::Sinh:
      case OpenCLLIB::Sinpi:
      case OpenCLLIB::Sqrt:
      case OpenCLLIB::Tan:
      case OpenCLLIB::Tanh:
      case OpenCLLIB::Tanpi:
      case OpenCLLIB::Tgamma:
      case OpenCLLIB::Trunc:
      case OpenCLLIB::Half_cos:
      case OpenCLLIB::Half_divide:
      case OpenCLLIB::Half_exp:
      case OpenCLLIB::Half_exp2:
      case OpenCLLIB::Half_exp10:
      case OpenCLLIB::Half_log:
      case OpenCLLIB::Half_log2:
      case OpenCLLIB::Half_log10:
      case OpenCLLIB::Half_powr:
      case OpenCLLIB::Half_recip:
      case OpenCLLIB::Half_rsqrt:
      case OpenCLLIB::Half_sin:
      case OpenCLLIB::Half_sqrt:
      case OpenCLLIB::Half_tan:
      case OpenCLLIB::Native_cos:
      case OpenCLLIB::Native_divide:
      case OpenCLLIB::Native_exp:
      case OpenCLLIB::Native_exp2:
      case OpenCLLIB::Native_exp10:
      case OpenCLLIB::Native_log:
      case OpenCLLIB::Native_log2:
      case OpenCLLIB::Native_log10:
      case OpenCLLIB::Native_powr:
      case OpenCLLIB::Native_recip:
      case OpenCLLIB::Native_rsqrt:
      case OpenCLLIB::Native_sin:
      case OpenCLLIB::Native_sqrt:
      case OpenCLLIB::Native_tan:
      case OpenCLLIB::FClamp:
      case OpenCLLIB::Degrees:
      case OpenCLLIB::FMax_common:
      case OpenCLLIB::FMin_common:
      case OpenCLLIB::Mix:
      case OpenCLLIB::Radians:
      case OpenCLLIB::Step:
      case OpenCLLIB::Smoothstep:
      case OpenCLLIB::Sign: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case OpenCLLIB::Fract:
      case OpenCLLIB::Modf:
      case OpenCLLIB::Sincos: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (result_type != x_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected type of operand X to be equal to Result Type";
        }

        const uint32_t p_type = _.GetOperandTypeId(inst, 5);
        spv::StorageClass p_storage_class;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected the last operand to be a pointer";
        }

        if (p_storage_class != spv::StorageClass::Generic &&
            p_storage_class != spv::StorageClass::CrossWorkgroup &&
            p_storage_class != spv::StorageClass::Workgroup &&
            p_storage_class != spv::StorageClass::Function) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected storage class of the pointer to be Generic, "
                    "CrossWorkgroup, Workgroup or Function";
        }

        if (result_type != p_data_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected data type of the pointer to be equal to Result "
                    "Type";
        }
        break;
      }

      case OpenCLLIB::Frexp:
      case OpenCLLIB::Lgamma_r:
      case OpenCLLIB::Remquo: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        uint32_t operand_index = 4;
        const uint32_t x_type = _.GetOperandTypeId(inst, operand_index++);
        if (result_type != x_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected type of operand X to be equal to Result Type";
        }

        if (ext_inst_key == OpenCLLIB::Remquo) {
          const uint32_t y_type = _.GetOperandTypeId(inst, operand_index++);
          if (result_type != y_type) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected type of operand Y to be equal to Result Type";
          }
        }

        const uint32_t p_type = _.GetOperandTypeId(inst, operand_index++);
        spv::StorageClass p_storage_class;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected the last operand to be a pointer";
        }

        if (p_storage_class != spv::StorageClass::Generic &&
            p_storage_class != spv::StorageClass::CrossWorkgroup &&
            p_storage_class != spv::StorageClass::Workgroup &&
            p_storage_class != spv::StorageClass::Function) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected storage class of the pointer to be Generic, "
                    "CrossWorkgroup, Workgroup or Function";
        }

        if (!_.IsIntScalarOrVectorType(p_data_type) ||
            _.GetBitWidth(p_data_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected data type of the pointer to be a 32-bit int "
                    "scalar or vector type";
        }

        if (_.GetDimension(p_data_type) != num_components) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected data type of the pointer to have the same number "
                    "of components as Result Type";
        }
        break;
      }

      case OpenCLLIB::Ilogb: {
        if (!_.IsIntScalarOrVectorType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit int scalar or vector "
                    "type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarOrVectorType(x_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X to be a float scalar or vector";
        }

        if (_.GetDimension(x_type) != num_components) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X to have the same number of components "
                    "as Result Type";
        }
        break;
      }

      case OpenCLLIB::Ldexp:
      case OpenCLLIB::Pown:
      case OpenCLLIB::Rootn: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (result_type != x_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected type of operand X to be equal to Result Type";
        }

        const uint32_t exp_type = _.GetOperandTypeId(inst, 5);
        if (!_.IsIntScalarOrVectorType(exp_type) ||
            _.GetBitWidth(exp_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected the exponent to be a 32-bit int scalar or vector";
        }

        if (_.GetDimension(exp_type) != num_components) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected the exponent to have the same number of "
                    "components as Result Type";
        }
        break;
      }

      case OpenCLLIB::Nan: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t nancode_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsIntScalarOrVectorType(nancode_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Nancode to be an int scalar or vector type";
        }

        if (_.GetDimension(nancode_type) != num_components) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Nancode to have the same number of components as "
                    "Result Type";
        }

        if (_.GetBitWidth(result_type) != _.GetBitWidth(nancode_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Nancode to have the same bit width as Result "
                    "Type";
        }
        break;
      }

      case OpenCLLIB::SAbs:
      case OpenCLLIB::SAbs_diff:
      case OpenCLLIB::SAdd_sat:
      case OpenCLLIB::UAdd_sat:
      case OpenCLLIB::SHadd:
      case OpenCLLIB::UHadd:
      case OpenCLLIB::SRhadd:
      case OpenCLLIB::URhadd:
      case OpenCLLIB::SClamp:
      case OpenCLLIB::UClamp:
      case OpenCLLIB::Clz:
      case OpenCLLIB::Ctz:
      case OpenCLLIB::SMad_hi:
      case OpenCLLIB::UMad_sat:
      case OpenCLLIB::SMad_sat:
      case OpenCLLIB::SMax:
      case OpenCLLIB::UMax:
      case OpenCLLIB::SMin:
      case OpenCLLIB::UMin:
      case OpenCLLIB::SMul_hi:
      case OpenCLLIB::Rotate:
      case OpenCLLIB::SSub_sat:
      case OpenCLLIB::USub_sat:
      case OpenCLLIB::Popcount:
      case OpenCLLIB::UAbs:
      case OpenCLLIB::UAbs_diff:
      case OpenCLLIB::UMul_hi:
      case OpenCLLIB::UMad_hi: {
        if (!_.IsIntScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case OpenCLLIB::U_Upsample:
      case OpenCLLIB::S_Upsample: {
        if (!_.IsIntScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int scalar or vector "
                    "type";
        }

        const uint32_t result_num_components = _.GetDimension(result_type);
        if (result_num_components > 4 && result_num_components != 8 &&
            result_num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t result_bit_width = _.GetBitWidth(result_type);
        if (result_bit_width != 16 && result_bit_width != 32 &&
            result_bit_width != 64) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected bit width of Result Type components to be 16, 32 "
                    "or 64";
        }

        const uint32_t hi_type = _.GetOperandTypeId(inst, 4);
        const uint32_t lo_type = _.GetOperandTypeId(inst, 5);

        if (hi_type != lo_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Hi and Lo operands to have the same type";
        }

        if (result_num_components != _.GetDimension(hi_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Hi and Lo operands to have the same number of "
                    "components as Result Type";
        }

        if (result_bit_width != 2 * _.GetBitWidth(hi_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected bit width of components of Hi and Lo operands to "
                    "be half of the bit width of components of Result Type";
        }
        break;
      }

      case OpenCLLIB::SMad24:
      case OpenCLLIB::UMad24:
      case OpenCLLIB::SMul24:
      case OpenCLLIB::UMul24: {
        if (!_.IsIntScalarOrVectorType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit int scalar or vector "
                    "type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case OpenCLLIB::Cross: {
        if (!_.IsFloatVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components != 3 && num_components != 4) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have 3 or 4 components";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t y_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        if (y_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Y type to be equal to Result Type";
        }
        break;
      }

      case OpenCLLIB::Distance:
      case OpenCLLIB::Fast_distance: {
        if (!_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar type";
        }

        const uint32_t p0_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarOrVectorType(p0_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P0 to be of float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(p0_type);
        if (num_components > 4) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P0 to have no more than 4 components";
        }

        if (result_type != _.GetComponentType(p0_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P0 component type to be equal to "
                 << "Result Type";
        }

        const uint32_t p1_type = _.GetOperandTypeId(inst, 5);
        if (p0_type != p1_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operands P0 and P1 to be of the same type";
        }
        break;
      }

      case OpenCLLIB::Length:
      case OpenCLLIB::Fast_length: {
        if (!_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar type";
        }

        const uint32_t p_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarOrVectorType(p_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a float scalar or vector";
        }

        const uint32_t num_components = _.GetDimension(p_type);
        if (num_components > 4) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to have no more than 4 components";
        }

        if (result_type != _.GetComponentType(p_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P component type to be equal to Result "
                    "Type";
        }
        break;
      }

      case OpenCLLIB::Normalize:
      case OpenCLLIB::Fast_normalize: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have no more than 4 components";
        }

        const uint32_t p_type = _.GetOperandTypeId(inst, 4);
        if (p_type != result_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P type to be equal to Result Type";
        }
        break;
      }

      case OpenCLLIB::Bitselect: {
        if (!_.IsFloatScalarOrVectorType(result_type) &&
            !_.IsIntScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int or float scalar or "
                    "vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case OpenCLLIB::Select: {
        if (!_.IsFloatScalarOrVectorType(result_type) &&
            !_.IsIntScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int or float scalar or "
                    "vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t a_type = _.GetOperandTypeId(inst, 4);
        const uint32_t b_type = _.GetOperandTypeId(inst, 5);
        const uint32_t c_type = _.GetOperandTypeId(inst, 6);

        if (result_type != a_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand A type to be equal to Result Type";
        }

        if (result_type != b_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand B type to be equal to Result Type";
        }

        if (!_.IsIntScalarOrVectorType(c_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand C to be an int scalar or vector";
        }

        if (num_components != _.GetDimension(c_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand C to have the same number of components "
                    "as Result Type";
        }

        if (_.GetBitWidth(result_type) != _.GetBitWidth(c_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand C to have the same bit width as Result "
                    "Type";
        }
        break;
      }

      case OpenCLLIB::Vloadn: {
        if (!_.IsFloatVectorType(result_type) &&
            !_.IsIntVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int or float vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have 2, 3, 4, 8 or 16 components";
        }

        const uint32_t offset_type = _.GetOperandTypeId(inst, 4);
        const uint32_t p_type = _.GetOperandTypeId(inst, 5);

        const uint32_t size_t_bit_width = GetSizeTBitWidth(_);
        if (!size_t_bit_width) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name()
                 << " can only be used with physical addressing models";
        }

        if (!_.IsIntScalarType(offset_type) ||
            _.GetBitWidth(offset_type) != size_t_bit_width) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Offset to be of type size_t ("
                 << size_t_bit_width
                 << "-bit integer for the addressing model used in the module)";
        }

        spv::StorageClass p_storage_class;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a pointer";
        }

        if (p_storage_class != spv::StorageClass::UniformConstant &&
            p_storage_class != spv::StorageClass::Generic &&
            p_storage_class != spv::StorageClass::CrossWorkgroup &&
            p_storage_class != spv::StorageClass::Workgroup &&
            p_storage_class != spv::StorageClass::Function) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P storage class to be UniformConstant, "
                    "Generic, CrossWorkgroup, Workgroup or Function";
        }

        if (_.GetComponentType(result_type) != p_data_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P data type to be equal to component "
                    "type of Result Type";
        }

        const uint32_t n_value = inst->word(7);
        if (num_components != n_value) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected literal N to be equal to the number of "
                    "components of Result Type";
        }
        break;
      }

      case OpenCLLIB::Vstoren: {
        if (_.GetIdOpcode(result_type) != spv::Op::OpTypeVoid) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": expected Result Type to be void";
        }

        const uint32_t data_type = _.GetOperandTypeId(inst, 4);
        const uint32_t offset_type = _.GetOperandTypeId(inst, 5);
        const uint32_t p_type = _.GetOperandTypeId(inst, 6);

        if (!_.IsFloatVectorType(data_type) && !_.IsIntVectorType(data_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Data to be an int or float vector";
        }

        const uint32_t num_components = _.GetDimension(data_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Data to have 2, 3, 4, 8 or 16 components";
        }

        const uint32_t size_t_bit_width = GetSizeTBitWidth(_);
        if (!size_t_bit_width) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name()
                 << " can only be used with physical addressing models";
        }

        if (!_.IsIntScalarType(offset_type) ||
            _.GetBitWidth(offset_type) != size_t_bit_width) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Offset to be of type size_t ("
                 << size_t_bit_width
                 << "-bit integer for the addressing model used in the module)";
        }

        spv::StorageClass p_storage_class;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a pointer";
        }

        if (p_storage_class != spv::StorageClass::Generic &&
            p_storage_class != spv::StorageClass::CrossWorkgroup &&
            p_storage_class != spv::StorageClass::Workgroup &&
            p_storage_class != spv::StorageClass::Function) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P storage class to be Generic, "
                    "CrossWorkgroup, Workgroup or Function";
        }

        if (_.GetComponentType(data_type) != p_data_type) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P data type to be equal to the type of "
                    "operand Data components";
        }
        break;
      }

      case OpenCLLIB::Vload_half: {
        if (!_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar type";
        }

        const uint32_t offset_type = _.GetOperandTypeId(inst, 4);
        const uint32_t p_type = _.GetOperandTypeId(inst, 5);

        const uint32_t size_t_bit_width = GetSizeTBitWidth(_);
        if (!size_t_bit_width) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name()
                 << " can only be used with physical addressing models";
        }

        if (!_.IsIntScalarType(offset_type) ||
            _.GetBitWidth(offset_type) != size_t_bit_width) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Offset to be of type size_t ("
                 << size_t_bit_width
                 << "-bit integer for the addressing model used in the module)";
        }

        spv::StorageClass p_storage_class;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a pointer";
        }

        if (p_storage_class != spv::StorageClass::UniformConstant &&
            p_storage_class != spv::StorageClass::Generic &&
            p_storage_class != spv::StorageClass::CrossWorkgroup &&
            p_storage_class != spv::StorageClass::Workgroup &&
            p_storage_class != spv::StorageClass::Function) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P storage class to be UniformConstant, "
                    "Generic, CrossWorkgroup, Workgroup or Function";
        }

        if (!_.IsFloatScalarType(p_data_type) ||
            _.GetBitWidth(p_data_type) != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P data type to be 16-bit float scalar";
        }
        break;
      }

      case OpenCLLIB::Vload_halfn:
      case OpenCLLIB::Vloada_halfn: {
        if (!_.IsFloatVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have 2, 3, 4, 8 or 16 components";
        }

        const uint32_t offset_type = _.GetOperandTypeId(inst, 4);
        const uint32_t p_type = _.GetOperandTypeId(inst, 5);

        const uint32_t size_t_bit_width = GetSizeTBitWidth(_);
        if (!size_t_bit_width) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name()
                 << " can only be used with physical addressing models";
        }

        if (!_.IsIntScalarType(offset_type) ||
            _.GetBitWidth(offset_type) != size_t_bit_width) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Offset to be of type size_t ("
                 << size_t_bit_width
                 << "-bit integer for the addressing model used in the module)";
        }

        spv::StorageClass p_storage_class;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a pointer";
        }

        if (p_storage_class != spv::StorageClass::UniformConstant &&
            p_storage_class != spv::StorageClass::Generic &&
            p_storage_class != spv::StorageClass::CrossWorkgroup &&
            p_storage_class != spv::StorageClass::Workgroup &&
            p_storage_class != spv::StorageClass::Function) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P storage class to be UniformConstant, "
                    "Generic, CrossWorkgroup, Workgroup or Function";
        }

        if (!_.IsFloatScalarType(p_data_type) ||
            _.GetBitWidth(p_data_type) != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P data type to be 16-bit float scalar";
        }

        const uint32_t n_value = inst->word(7);
        if (num_components != n_value) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected literal N to be equal to the number of "
                    "components of Result Type";
        }
        break;
      }

      case OpenCLLIB::Vstore_half:
      case OpenCLLIB::Vstore_half_r:
      case OpenCLLIB::Vstore_halfn:
      case OpenCLLIB::Vstore_halfn_r:
      case OpenCLLIB::Vstorea_halfn:
      case OpenCLLIB::Vstorea_halfn_r: {
        if (_.GetIdOpcode(result_type) != spv::Op::OpTypeVoid) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": expected Result Type to be void";
        }

        const uint32_t data_type = _.GetOperandTypeId(inst, 4);
        const uint32_t offset_type = _.GetOperandTypeId(inst, 5);
        const uint32_t p_type = _.GetOperandTypeId(inst, 6);
        const uint32_t data_type_bit_width = _.GetBitWidth(data_type);

        if (ext_inst_key == OpenCLLIB::Vstore_half ||
            ext_inst_key == OpenCLLIB::Vstore_half_r) {
          if (!_.IsFloatScalarType(data_type) ||
              (data_type_bit_width != 32 && data_type_bit_width != 64)) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected Data to be a 32 or 64-bit float scalar";
          }
        } else {
          if (!_.IsFloatVectorType(data_type) ||
              (data_type_bit_width != 32 && data_type_bit_width != 64)) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected Data to be a 32 or 64-bit float vector";
          }

          const uint32_t num_components = _.GetDimension(data_type);
          if (num_components > 4 && num_components != 8 &&
              num_components != 16) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected Data to have 2, 3, 4, 8 or 16 components";
          }
        }

        const uint32_t size_t_bit_width = GetSizeTBitWidth(_);
        if (!size_t_bit_width) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name()
                 << " can only be used with physical addressing models";
        }

        if (!_.IsIntScalarType(offset_type) ||
            _.GetBitWidth(offset_type) != size_t_bit_width) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Offset to be of type size_t ("
                 << size_t_bit_width
                 << "-bit integer for the addressing model used in the module)";
        }

        spv::StorageClass p_storage_class;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a pointer";
        }

        if (p_storage_class != spv::StorageClass::Generic &&
            p_storage_class != spv::StorageClass::CrossWorkgroup &&
            p_storage_class != spv::StorageClass::Workgroup &&
            p_storage_class != spv::StorageClass::Function) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P storage class to be Generic, "
                    "CrossWorkgroup, Workgroup or Function";
        }

        if (!_.IsFloatScalarType(p_data_type) ||
            _.GetBitWidth(p_data_type) != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P data type to be 16-bit float scalar";
        }

        // Rounding mode enum is checked by assembler.
        break;
      }

      case OpenCLLIB::Shuffle:
      case OpenCLLIB::Shuffle2: {
        if (!_.IsFloatVectorType(result_type) &&
            !_.IsIntVectorType(result_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int or float vector type";
        }

        const uint32_t result_num_components = _.GetDimension(result_type);
        if (result_num_components != 2 && result_num_components != 4 &&
            result_num_components != 8 && result_num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have 2, 4, 8 or 16 components";
        }

        uint32_t operand_index = 4;
        const uint32_t x_type = _.GetOperandTypeId(inst, operand_index++);

        if (ext_inst_key == OpenCLLIB::Shuffle2) {
          const uint32_t y_type = _.GetOperandTypeId(inst, operand_index++);
          if (x_type != y_type) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected operands X and Y to be of the same type";
          }
        }

        const uint32_t shuffle_mask_type =
            _.GetOperandTypeId(inst, operand_index++);

        if (!_.IsFloatVectorType(x_type) && !_.IsIntVectorType(x_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X to be an int or float vector";
        }

        const uint32_t x_num_components = _.GetDimension(x_type);
        if (x_num_components != 2 && x_num_components != 4 &&
            x_num_components != 8 && x_num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X to have 2, 4, 8 or 16 components";
        }

        const uint32_t result_component_type = _.GetComponentType(result_type);

        if (result_component_type != _.GetComponentType(x_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X and Result Type to have equal "
                    "component types";
        }

        if (!_.IsIntVectorType(shuffle_mask_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Shuffle Mask to be an int vector";
        }

        if (result_num_components != _.GetDimension(shuffle_mask_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Shuffle Mask to have the same number of "
                    "components as Result Type";
        }

        if (_.GetBitWidth(result_component_type) !=
            _.GetBitWidth(shuffle_mask_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Shuffle Mask components to have the same "
                    "bit width as Result Type components";
        }
        break;
      }

      case OpenCLLIB::Printf: {
        if (!_.IsIntScalarType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit int type";
        }

        const uint32_t format_type = _.GetOperandTypeId(inst, 4);
        spv::StorageClass format_storage_class;
        uint32_t format_data_type = 0;
        if (!_.GetPointerTypeInfo(format_type, &format_data_type,
                                  &format_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Format to be a pointer";
        }

        if (format_storage_class != spv::StorageClass::UniformConstant) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Format storage class to be UniformConstant";
        }

        if (!_.IsIntScalarType(format_data_type) ||
            _.GetBitWidth(format_data_type) != 8) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Format data type to be 8-bit int";
        }
        break;
      }

      case OpenCLLIB::Prefetch: {
        if (_.GetIdOpcode(result_type) != spv::Op::OpTypeVoid) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": expected Result Type to be void";
        }

        const uint32_t p_type = _.GetOperandTypeId(inst, 4);
        const uint32_t num_elements_type = _.GetOperandTypeId(inst, 5);

        spv::StorageClass p_storage_class;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Ptr to be a pointer";
        }

        if (p_storage_class != spv::StorageClass::CrossWorkgroup) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Ptr storage class to be CrossWorkgroup";
        }

        if (!_.IsFloatScalarOrVectorType(p_data_type) &&
            !_.IsIntScalarOrVectorType(p_data_type)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Ptr data type to be int or float scalar or "
                    "vector";
        }

        const uint32_t num_components = _.GetDimension(p_data_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t size_t_bit_width = GetSizeTBitWidth(_);
        if (!size_t_bit_width) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name()
                 << " can only be used with physical addressing models";
        }

        if (!_.IsIntScalarType(num_elements_type) ||
            _.GetBitWidth(num_elements_type) != size_t_bit_width) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Num Elements to be of type size_t ("
                 << size_t_bit_width
                 << "-bit integer for the addressing model used in the module)";
        }
        break;
      }
    }
  } else if (ext_inst_type == SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100 ||
             ext_inst_type ==
                 SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100) {
    if (!_.IsVoidType(result_type)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << ext_inst_name() << ": "
             << "expected result type must be a result id of "
             << "OpTypeVoid";
    }

    const bool vulkanDebugInfo =
        ext_inst_type == SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100;

    auto num_words = inst->words().size();

    // Handle any non-common NonSemanticShaderDebugInfo instructions.
    if (vulkanDebugInfo) {
      const NonSemanticShaderDebugInfo100Instructions ext_inst_key =
          NonSemanticShaderDebugInfo100Instructions(ext_inst_index);
      switch (ext_inst_key) {
        // The following block of instructions will be handled by the common
        // validation.
        case NonSemanticShaderDebugInfo100DebugInfoNone:
        case NonSemanticShaderDebugInfo100DebugCompilationUnit:
        case NonSemanticShaderDebugInfo100DebugTypeBasic:
        case NonSemanticShaderDebugInfo100DebugTypePointer:
        case NonSemanticShaderDebugInfo100DebugTypeQualifier:
        case NonSemanticShaderDebugInfo100DebugTypeArray:
        case NonSemanticShaderDebugInfo100DebugTypeVector:
        case NonSemanticShaderDebugInfo100DebugTypedef:
        case NonSemanticShaderDebugInfo100DebugTypeFunction:
        case NonSemanticShaderDebugInfo100DebugTypeEnum:
        case NonSemanticShaderDebugInfo100DebugTypeComposite:
        case NonSemanticShaderDebugInfo100DebugTypeMember:
        case NonSemanticShaderDebugInfo100DebugTypeInheritance:
        case NonSemanticShaderDebugInfo100DebugTypePtrToMember:
        case NonSemanticShaderDebugInfo100DebugTypeTemplate:
        case NonSemanticShaderDebugInfo100DebugTypeTemplateParameter:
        case NonSemanticShaderDebugInfo100DebugTypeTemplateTemplateParameter:
        case NonSemanticShaderDebugInfo100DebugTypeTemplateParameterPack:
        case NonSemanticShaderDebugInfo100DebugGlobalVariable:
        case NonSemanticShaderDebugInfo100DebugFunctionDeclaration:
        case NonSemanticShaderDebugInfo100DebugFunction:
        case NonSemanticShaderDebugInfo100DebugLexicalBlock:
        case NonSemanticShaderDebugInfo100DebugLexicalBlockDiscriminator:
        case NonSemanticShaderDebugInfo100DebugScope:
        case NonSemanticShaderDebugInfo100DebugNoScope:
        case NonSemanticShaderDebugInfo100DebugInlinedAt:
        case NonSemanticShaderDebugInfo100DebugLocalVariable:
        case NonSemanticShaderDebugInfo100DebugInlinedVariable:
        case NonSemanticShaderDebugInfo100DebugDeclare:
        case NonSemanticShaderDebugInfo100DebugValue:
        case NonSemanticShaderDebugInfo100DebugOperation:
        case NonSemanticShaderDebugInfo100DebugExpression:
        case NonSemanticShaderDebugInfo100DebugMacroDef:
        case NonSemanticShaderDebugInfo100DebugMacroUndef:
        case NonSemanticShaderDebugInfo100DebugImportedEntity:
        case NonSemanticShaderDebugInfo100DebugSource:
          break;
        case NonSemanticShaderDebugInfo100DebugTypeMatrix: {
          CHECK_DEBUG_OPERAND("Vector Type", CommonDebugInfoDebugTypeVector, 5);

          CHECK_CONST_UINT_OPERAND("Vector Count", 6);

          uint32_t vector_count = inst->word(6);
          uint64_t const_val;
          if (!_.EvalConstantValUint64(vector_count, &const_val)) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name()
                   << ": Vector Count must be 32-bit integer OpConstant";
          }

          vector_count = const_val & 0xffffffff;
          if (!vector_count || vector_count > 4) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": Vector Count must be positive "
                   << "integer less than or equal to 4";
          }
          break;
        }
        // TODO: Add validation rules for remaining cases as well.
        case NonSemanticShaderDebugInfo100DebugFunctionDefinition:
        case NonSemanticShaderDebugInfo100DebugSourceContinued:
        case NonSemanticShaderDebugInfo100DebugLine:
        case NonSemanticShaderDebugInfo100DebugNoLine:
        case NonSemanticShaderDebugInfo100DebugBuildIdentifier:
        case NonSemanticShaderDebugInfo100DebugStoragePath:
        case NonSemanticShaderDebugInfo100DebugEntryPoint:
          break;
        case NonSemanticShaderDebugInfo100InstructionsMax:
          assert(0);
          break;
      }
    }

    // Handle any non-common OpenCL insts, then common
    if (ext_inst_type != SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100 ||
        OpenCLDebugInfo100Instructions(ext_inst_index) !=
            OpenCLDebugInfo100DebugModuleINTEL) {
      const CommonDebugInfoInstructions ext_inst_key =
          CommonDebugInfoInstructions(ext_inst_index);
      switch (ext_inst_key) {
        case CommonDebugInfoDebugInfoNone:
        case CommonDebugInfoDebugNoScope:
          break;
          // The binary parser validates the opcode for DebugInfoNone,
          // DebugNoScope, DebugOperation. We just check the parameters to
          // DebugOperation are properly constants for vulkan debug info.
        case CommonDebugInfoDebugOperation: {
          CHECK_CONST_UINT_OPERAND("Operation", 5);
          for (uint32_t i = 6; i < num_words; ++i) {
            CHECK_CONST_UINT_OPERAND("Operand", i);
          }
          break;
        }
        case CommonDebugInfoDebugCompilationUnit: {
          CHECK_CONST_UINT_OPERAND("Version", 5);
          CHECK_CONST_UINT_OPERAND("DWARF Version", 6);
          CHECK_DEBUG_OPERAND("Source", CommonDebugInfoDebugSource, 7);
          CHECK_CONST_UINT_OPERAND("Language", 8);
          break;
        }
        case CommonDebugInfoDebugSource: {
          CHECK_OPERAND("File", spv::Op::OpString, 5);
          if (num_words == 7) CHECK_OPERAND("Text", spv::Op::OpString, 6);
          break;
        }
        case CommonDebugInfoDebugTypeBasic: {
          CHECK_OPERAND("Name", spv::Op::OpString, 5);
          CHECK_OPERAND("Size", spv::Op::OpConstant, 6);
          CHECK_CONST_UINT_OPERAND("Encoding", 7);
          break;
        }
        case CommonDebugInfoDebugTypePointer: {
          auto validate_base_type = ValidateOperandDebugType(
              _, "Base Type", inst, 5, ext_inst_name, false);
          if (validate_base_type != SPV_SUCCESS) return validate_base_type;
          CHECK_CONST_UINT_OPERAND("Storage Class", 6);
          CHECK_CONST_UINT_OPERAND("Flags", 7);
          break;
        }
        case CommonDebugInfoDebugTypeQualifier: {
          auto validate_base_type = ValidateOperandDebugType(
              _, "Base Type", inst, 5, ext_inst_name, false);
          if (validate_base_type != SPV_SUCCESS) return validate_base_type;
          CHECK_CONST_UINT_OPERAND("Type Qualifier", 6);
          break;
        }
        case CommonDebugInfoDebugTypeVector: {
          auto validate_base_type =
              ValidateOperandBaseType(_, inst, 5, ext_inst_name);
          if (validate_base_type != SPV_SUCCESS) return validate_base_type;

          CHECK_CONST_UINT_OPERAND("Component Count", 6);
          uint32_t component_count = inst->word(6);
          if (vulkanDebugInfo) {
            uint64_t const_val;
            if (!_.EvalConstantValUint64(component_count, &const_val)) {
              return _.diag(SPV_ERROR_INVALID_DATA, inst)
                     << ext_inst_name()
                     << ": Component Count must be 32-bit integer OpConstant";
            }
            component_count = const_val & 0xffffffff;
          }

          if (!component_count || component_count > 4) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": Component Count must be positive "
                   << "integer less than or equal to 4";
          }
          break;
        }
        case CommonDebugInfoDebugTypeArray: {
          auto validate_base_type = ValidateOperandDebugType(
              _, "Base Type", inst, 5, ext_inst_name, false);
          if (validate_base_type != SPV_SUCCESS) return validate_base_type;
          for (uint32_t i = 6; i < num_words; ++i) {
            bool invalid = false;
            auto* component_count = _.FindDef(inst->word(i));
            if (IsConstIntScalarTypeWith32Or64Bits(_, component_count)) {
              // TODO: We need a spec discussion for the runtime array for
              // OpenCL.
              if (!vulkanDebugInfo && !component_count->word(3)) {
                invalid = true;
              }
            } else if (component_count->words().size() > 6 &&
                       (CommonDebugInfoInstructions(component_count->word(4)) ==
                            CommonDebugInfoDebugLocalVariable ||
                        CommonDebugInfoInstructions(component_count->word(4)) ==
                            CommonDebugInfoDebugGlobalVariable)) {
              auto* component_count_type = _.FindDef(component_count->word(6));
              if (component_count_type->words().size() > 7) {
                uint32_t encoding = component_count_type->word(7);
                if (CommonDebugInfoInstructions(component_count_type->word(
                        4)) != CommonDebugInfoDebugTypeBasic ||
                    (vulkanDebugInfo && !IsUint32Constant(_, encoding)) ||
                    OpenCLDebugInfo100DebugBaseTypeAttributeEncoding(
                        vulkanDebugInfo
                            ? GetUint32Constant(_, encoding)
                            : encoding) != OpenCLDebugInfo100Unsigned) {
                  invalid = true;
                } else {
                  // DebugTypeBasic for DebugLocalVariable/DebugGlobalVariable
                  // must have Unsigned encoding and 32 or 64 as its size in
                  // bits.
                  Instruction* size_in_bits =
                      _.FindDef(component_count_type->word(6));
                  if (!_.IsIntScalarType(size_in_bits->type_id()) ||
                      (size_in_bits->word(3) != 32 &&
                       size_in_bits->word(3) != 64)) {
                    invalid = true;
                  }
                }
              } else {
                invalid = true;
              }
            } else {
              invalid = true;
            }
            if (invalid) {
              return _.diag(SPV_ERROR_INVALID_DATA, inst)
                     << ext_inst_name() << ": Component Count must be "
                     << "OpConstant with a 32- or 64-bits integer scalar type "
                        "or "
                     << "DebugGlobalVariable or DebugLocalVariable with a 32- "
                        "or "
                     << "64-bits unsigned integer scalar type";
            }
          }
          break;
        }
        case CommonDebugInfoDebugTypedef: {
          CHECK_OPERAND("Name", spv::Op::OpString, 5);
          auto validate_base_type =
              ValidateOperandBaseType(_, inst, 6, ext_inst_name);
          if (validate_base_type != SPV_SUCCESS) return validate_base_type;
          CHECK_DEBUG_OPERAND("Source", CommonDebugInfoDebugSource, 7);
          CHECK_CONST_UINT_OPERAND("Line", 8);
          CHECK_CONST_UINT_OPERAND("Column", 9);
          auto validate_parent =
              ValidateOperandLexicalScope(_, "Parent", inst, 10, ext_inst_name);
          if (validate_parent != SPV_SUCCESS) return validate_parent;
          break;
        }
        case CommonDebugInfoDebugTypeFunction: {
          CHECK_CONST_UINT_OPERAND("Flags", 5);
          auto* return_type = _.FindDef(inst->word(6));
          // TODO: We need a spec discussion that we have to allow return and
          // parameter types of a DebugTypeFunction to have template parameter.
          if (return_type->opcode() != spv::Op::OpTypeVoid) {
            auto validate_return = ValidateOperandDebugType(
                _, "Return Type", inst, 6, ext_inst_name, true);
            if (validate_return != SPV_SUCCESS) return validate_return;
          }
          for (uint32_t word_index = 7; word_index < num_words; ++word_index) {
            auto validate_param = ValidateOperandDebugType(
                _, "Parameter Types", inst, word_index, ext_inst_name, true);
            if (validate_param != SPV_SUCCESS) return validate_param;
          }
          break;
        }
        case CommonDebugInfoDebugTypeEnum: {
          CHECK_OPERAND("Name", spv::Op::OpString, 5);
          if (!DoesDebugInfoOperandMatchExpectation(
                  _,
                  [](CommonDebugInfoInstructions dbg_inst) {
                    return dbg_inst == CommonDebugInfoDebugInfoNone;
                  },
                  inst, 6)) {
            auto validate_underlying_type = ValidateOperandDebugType(
                _, "Underlying Types", inst, 6, ext_inst_name, false);
            if (validate_underlying_type != SPV_SUCCESS)
              return validate_underlying_type;
          }
          CHECK_DEBUG_OPERAND("Source", CommonDebugInfoDebugSource, 7);
          CHECK_CONST_UINT_OPERAND("Line", 8);
          CHECK_CONST_UINT_OPERAND("Column", 9);
          auto validate_parent =
              ValidateOperandLexicalScope(_, "Parent", inst, 10, ext_inst_name);
          if (validate_parent != SPV_SUCCESS) return validate_parent;
          CHECK_OPERAND("Size", spv::Op::OpConstant, 11);
          auto* size = _.FindDef(inst->word(11));
          if (!_.IsIntScalarType(size->type_id()) || !size->word(3)) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": expected operand Size is a "
                   << "positive integer";
          }
          CHECK_CONST_UINT_OPERAND("Flags", 12);
          for (uint32_t word_index = 13; word_index + 1 < num_words;
               word_index += 2) {
            CHECK_OPERAND("Value", spv::Op::OpConstant, word_index);
            CHECK_OPERAND("Name", spv::Op::OpString, word_index + 1);
          }
          break;
        }
        case CommonDebugInfoDebugTypeComposite: {
          CHECK_OPERAND("Name", spv::Op::OpString, 5);
          CHECK_DEBUG_OPERAND("Source", CommonDebugInfoDebugSource, 7);
          CHECK_CONST_UINT_OPERAND("Line", 8);
          CHECK_CONST_UINT_OPERAND("Column", 9);
          auto validate_parent =
              ValidateOperandLexicalScope(_, "Parent", inst, 10, ext_inst_name);
          if (validate_parent != SPV_SUCCESS) return validate_parent;
          CHECK_OPERAND("Linkage Name", spv::Op::OpString, 11);
          if (!DoesDebugInfoOperandMatchExpectation(
                  _,
                  [](CommonDebugInfoInstructions dbg_inst) {
                    return dbg_inst == CommonDebugInfoDebugInfoNone;
                  },
                  inst, 12)) {
            CHECK_OPERAND("Size", spv::Op::OpConstant, 12);
          }
          CHECK_CONST_UINT_OPERAND("Flags", 13);
          for (uint32_t word_index = 14; word_index < num_words; ++word_index) {
            if (!DoesDebugInfoOperandMatchExpectation(
                    _,
                    [](CommonDebugInfoInstructions dbg_inst) {
                      return dbg_inst == CommonDebugInfoDebugTypeMember ||
                             dbg_inst == CommonDebugInfoDebugFunction ||
                             dbg_inst == CommonDebugInfoDebugTypeInheritance;
                    },
                    inst, word_index)) {
              return _.diag(SPV_ERROR_INVALID_DATA, inst)
                     << ext_inst_name() << ": "
                     << "expected operand Members "
                     << "must be DebugTypeMember, DebugFunction, or "
                        "DebugTypeInheritance";
            }
          }
          break;
        }
        case CommonDebugInfoDebugTypeMember: {
          CHECK_OPERAND("Name", spv::Op::OpString, 5);
          // TODO: We need a spec discussion that we have to allow member types
          // to have template parameter.
          auto validate_type =
              ValidateOperandDebugType(_, "Type", inst, 6, ext_inst_name, true);
          if (validate_type != SPV_SUCCESS) return validate_type;
          CHECK_DEBUG_OPERAND("Source", CommonDebugInfoDebugSource, 7);
          CHECK_CONST_UINT_OPERAND("Line", 8);
          CHECK_CONST_UINT_OPERAND("Column", 9);
          // NonSemantic.Shader.DebugInfo doesn't have the Parent operand
          if (vulkanDebugInfo) {
            CHECK_OPERAND("Offset", spv::Op::OpConstant, 10);
            CHECK_OPERAND("Size", spv::Op::OpConstant, 11);
            CHECK_CONST_UINT_OPERAND("Flags", 12);
            if (num_words == 14)
              CHECK_OPERAND("Value", spv::Op::OpConstant, 13);
          } else {
            CHECK_DEBUG_OPERAND("Parent", CommonDebugInfoDebugTypeComposite,
                                10);
            CHECK_OPERAND("Offset", spv::Op::OpConstant, 11);
            CHECK_OPERAND("Size", spv::Op::OpConstant, 12);
            CHECK_CONST_UINT_OPERAND("Flags", 13);
            if (num_words == 15)
              CHECK_OPERAND("Value", spv::Op::OpConstant, 14);
          }
          break;
        }
        case CommonDebugInfoDebugTypeInheritance: {
          CHECK_DEBUG_OPERAND("Child", CommonDebugInfoDebugTypeComposite, 5);
          auto* debug_inst = _.FindDef(inst->word(5));
          auto composite_type =
              OpenCLDebugInfo100DebugCompositeType(debug_inst->word(6));
          if (composite_type != OpenCLDebugInfo100Class &&
              composite_type != OpenCLDebugInfo100Structure) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected operand Child must be class or struct debug "
                      "type";
          }
          CHECK_DEBUG_OPERAND("Parent", CommonDebugInfoDebugTypeComposite, 6);
          debug_inst = _.FindDef(inst->word(6));
          composite_type =
              OpenCLDebugInfo100DebugCompositeType(debug_inst->word(6));
          if (composite_type != OpenCLDebugInfo100Class &&
              composite_type != OpenCLDebugInfo100Structure) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected operand Parent must be class or struct debug "
                      "type";
          }
          CHECK_OPERAND("Offset", spv::Op::OpConstant, 7);
          CHECK_OPERAND("Size", spv::Op::OpConstant, 8);
          CHECK_CONST_UINT_OPERAND("Flags", 9);
          break;
        }
        case CommonDebugInfoDebugFunction: {
          CHECK_OPERAND("Name", spv::Op::OpString, 5);
          auto validate_type = ValidateOperandDebugType(_, "Type", inst, 6,
                                                        ext_inst_name, false);
          if (validate_type != SPV_SUCCESS) return validate_type;
          CHECK_DEBUG_OPERAND("Source", CommonDebugInfoDebugSource, 7);
          CHECK_CONST_UINT_OPERAND("Line", 8);
          CHECK_CONST_UINT_OPERAND("Column", 9);
          auto validate_parent =
              ValidateOperandLexicalScope(_, "Parent", inst, 10, ext_inst_name);
          if (validate_parent != SPV_SUCCESS) return validate_parent;
          CHECK_OPERAND("Linkage Name", spv::Op::OpString, 11);
          CHECK_CONST_UINT_OPERAND("Flags", 12);
          CHECK_CONST_UINT_OPERAND("Scope Line", 13);
          // NonSemantic.Shader.DebugInfo.100 doesn't include a reference to the
          // OpFunction
          if (vulkanDebugInfo) {
            if (num_words == 15) {
              CHECK_DEBUG_OPERAND("Declaration",
                                  CommonDebugInfoDebugFunctionDeclaration, 14);
            }
          } else {
            if (!DoesDebugInfoOperandMatchExpectation(
                    _,
                    [](CommonDebugInfoInstructions dbg_inst) {
                      return dbg_inst == CommonDebugInfoDebugInfoNone;
                    },
                    inst, 14)) {
              CHECK_OPERAND("Function", spv::Op::OpFunction, 14);
            }
            if (num_words == 16) {
              CHECK_DEBUG_OPERAND("Declaration",
                                  CommonDebugInfoDebugFunctionDeclaration, 15);
            }
          }
          break;
        }
        case CommonDebugInfoDebugFunctionDeclaration: {
          CHECK_OPERAND("Name", spv::Op::OpString, 5);
          auto validate_type = ValidateOperandDebugType(_, "Type", inst, 6,
                                                        ext_inst_name, false);
          if (validate_type != SPV_SUCCESS) return validate_type;
          CHECK_DEBUG_OPERAND("Source", CommonDebugInfoDebugSource, 7);
          CHECK_CONST_UINT_OPERAND("Line", 8);
          CHECK_CONST_UINT_OPERAND("Column", 9);
          auto validate_parent =
              ValidateOperandLexicalScope(_, "Parent", inst, 10, ext_inst_name);
          if (validate_parent != SPV_SUCCESS) return validate_parent;
          CHECK_OPERAND("Linkage Name", spv::Op::OpString, 11);
          CHECK_CONST_UINT_OPERAND("Flags", 12);
          break;
        }
        case CommonDebugInfoDebugLexicalBlock: {
          CHECK_DEBUG_OPERAND("Source", CommonDebugInfoDebugSource, 5);
          CHECK_CONST_UINT_OPERAND("Line", 6);
          CHECK_CONST_UINT_OPERAND("Column", 7);
          auto validate_parent =
              ValidateOperandLexicalScope(_, "Parent", inst, 8, ext_inst_name);
          if (validate_parent != SPV_SUCCESS) return validate_parent;
          if (num_words == 10) CHECK_OPERAND("Name", spv::Op::OpString, 9);
          break;
        }
        case CommonDebugInfoDebugScope: {
          auto validate_scope =
              ValidateOperandLexicalScope(_, "Scope", inst, 5, ext_inst_name);
          if (validate_scope != SPV_SUCCESS) return validate_scope;
          if (num_words == 7) {
            CHECK_DEBUG_OPERAND("Inlined At", CommonDebugInfoDebugInlinedAt, 6);
          }
          break;
        }
        case CommonDebugInfoDebugLocalVariable: {
          CHECK_OPERAND("Name", spv::Op::OpString, 5);
          // TODO: We need a spec discussion that we have to allow local
          // variable types to have template parameter.
          auto validate_type =
              ValidateOperandDebugType(_, "Type", inst, 6, ext_inst_name, true);
          if (validate_type != SPV_SUCCESS) return validate_type;
          CHECK_DEBUG_OPERAND("Source", CommonDebugInfoDebugSource, 7);
          CHECK_CONST_UINT_OPERAND("Line", 8);
          CHECK_CONST_UINT_OPERAND("Column", 9);
          auto validate_parent =
              ValidateOperandLexicalScope(_, "Parent", inst, 10, ext_inst_name);
          if (validate_parent != SPV_SUCCESS) return validate_parent;
          CHECK_CONST_UINT_OPERAND("Flags", 11);
          if (num_words == 13) {
            CHECK_CONST_UINT_OPERAND("ArgNumber", 12);
          }
          break;
        }
        case CommonDebugInfoDebugDeclare: {
          CHECK_DEBUG_OPERAND("Local Variable",
                              CommonDebugInfoDebugLocalVariable, 5);
          auto* operand = _.FindDef(inst->word(6));
          if (operand->opcode() != spv::Op::OpVariable &&
              operand->opcode() != spv::Op::OpFunctionParameter) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected operand Variable must be a result id of "
                      "OpVariable or OpFunctionParameter";
          }

          CHECK_DEBUG_OPERAND("Expression", CommonDebugInfoDebugExpression, 7);

          if (vulkanDebugInfo) {
            for (uint32_t word_index = 8; word_index < num_words;
                 ++word_index) {
              auto index_inst = _.FindDef(inst->word(word_index));
              auto type_id = index_inst != nullptr ? index_inst->type_id() : 0;
              if (type_id == 0 || !IsIntScalar(_, type_id, false, false))
                return _.diag(SPV_ERROR_INVALID_DATA, inst)
                       << ext_inst_name() << ": "
                       << "expected index must be scalar integer";
            }
          }
          break;
        }
        case CommonDebugInfoDebugExpression: {
          for (uint32_t word_index = 5; word_index < num_words; ++word_index) {
            CHECK_DEBUG_OPERAND("Operation", CommonDebugInfoDebugOperation,
                                word_index);
          }
          break;
        }
        case CommonDebugInfoDebugTypeTemplate: {
          if (!DoesDebugInfoOperandMatchExpectation(
                  _,
                  [](CommonDebugInfoInstructions dbg_inst) {
                    return dbg_inst == CommonDebugInfoDebugTypeComposite ||
                           dbg_inst == CommonDebugInfoDebugFunction;
                  },
                  inst, 5)) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected operand Target must be DebugTypeComposite "
                   << "or DebugFunction";
          }
          for (uint32_t word_index = 6; word_index < num_words; ++word_index) {
            if (!DoesDebugInfoOperandMatchExpectation(
                    _,
                    [](CommonDebugInfoInstructions dbg_inst) {
                      return dbg_inst ==
                                 CommonDebugInfoDebugTypeTemplateParameter ||
                             dbg_inst ==
                                 CommonDebugInfoDebugTypeTemplateTemplateParameter;
                    },
                    inst, word_index)) {
              return _.diag(SPV_ERROR_INVALID_DATA, inst)
                     << ext_inst_name() << ": "
                     << "expected operand Parameters must be "
                     << "DebugTypeTemplateParameter or "
                     << "DebugTypeTemplateTemplateParameter";
            }
          }
          break;
        }
        case CommonDebugInfoDebugTypeTemplateParameter: {
          CHECK_OPERAND("Name", spv::Op::OpString, 5);
          auto validate_actual_type = ValidateOperandDebugType(
              _, "Actual Type", inst, 6, ext_inst_name, false);
          if (validate_actual_type != SPV_SUCCESS) return validate_actual_type;
          if (!DoesDebugInfoOperandMatchExpectation(
                  _,
                  [](CommonDebugInfoInstructions dbg_inst) {
                    return dbg_inst == CommonDebugInfoDebugInfoNone;
                  },
                  inst, 7)) {
            CHECK_OPERAND("Value", spv::Op::OpConstant, 7);
          }
          CHECK_DEBUG_OPERAND("Source", CommonDebugInfoDebugSource, 8);
          CHECK_CONST_UINT_OPERAND("Line", 9);
          CHECK_CONST_UINT_OPERAND("Column", 10);
          break;
        }
        case CommonDebugInfoDebugGlobalVariable: {
          CHECK_OPERAND("Name", spv::Op::OpString, 5);
          auto validate_type = ValidateOperandDebugType(_, "Type", inst, 6,
                                                        ext_inst_name, false);
          if (validate_type != SPV_SUCCESS) return validate_type;
          CHECK_DEBUG_OPERAND("Source", CommonDebugInfoDebugSource, 7);
          CHECK_CONST_UINT_OPERAND("Line", 8);
          CHECK_CONST_UINT_OPERAND("Column", 9);
          auto validate_scope =
              ValidateOperandLexicalScope(_, "Scope", inst, 10, ext_inst_name);
          if (validate_scope != SPV_SUCCESS) return validate_scope;
          CHECK_OPERAND("Linkage Name", spv::Op::OpString, 11);
          if (!DoesDebugInfoOperandMatchExpectation(
                  _,
                  [](CommonDebugInfoInstructions dbg_inst) {
                    return dbg_inst == CommonDebugInfoDebugInfoNone;
                  },
                  inst, 12)) {
            auto* operand = _.FindDef(inst->word(12));
            if (operand->opcode() != spv::Op::OpVariable &&
                operand->opcode() != spv::Op::OpConstant) {
              return _.diag(SPV_ERROR_INVALID_DATA, inst)
                     << ext_inst_name() << ": "
                     << "expected operand Variable must be a result id of "
                        "OpVariable or OpConstant or DebugInfoNone";
            }
          }
          if (num_words == 15) {
            CHECK_DEBUG_OPERAND("Static Member Declaration",
                                CommonDebugInfoDebugTypeMember, 14);
          }
          break;
        }
        case CommonDebugInfoDebugInlinedAt: {
          CHECK_CONST_UINT_OPERAND("Line", 5);
          auto validate_scope =
              ValidateOperandLexicalScope(_, "Scope", inst, 6, ext_inst_name);
          if (validate_scope != SPV_SUCCESS) return validate_scope;
          if (num_words == 8) {
            CHECK_DEBUG_OPERAND("Inlined", CommonDebugInfoDebugInlinedAt, 7);
          }
          break;
        }
        case CommonDebugInfoDebugValue: {
          CHECK_DEBUG_OPERAND("Local Variable",
                              CommonDebugInfoDebugLocalVariable, 5);
          CHECK_DEBUG_OPERAND("Expression", CommonDebugInfoDebugExpression, 7);

          for (uint32_t word_index = 8; word_index < num_words; ++word_index) {
            // TODO: The following code simply checks if it is a const int
            // scalar or a DebugLocalVariable or DebugGlobalVariable, but we
            // have to check it using the same validation for Indexes of
            // OpAccessChain.
            if (!IsConstWithIntScalarType(_, inst, word_index) &&
                !IsDebugVariableWithIntScalarType(_, inst, word_index)) {
              return _.diag(SPV_ERROR_INVALID_DATA, inst)
                     << ext_inst_name() << ": expected operand Indexes is "
                     << "OpConstant, DebugGlobalVariable, or "
                     << "type is OpConstant with an integer scalar type";
            }
          }
          break;
        }

        // TODO: Add validation rules for remaining cases as well.
        case CommonDebugInfoDebugTypePtrToMember:
        case CommonDebugInfoDebugTypeTemplateTemplateParameter:
        case CommonDebugInfoDebugTypeTemplateParameterPack:
        case CommonDebugInfoDebugLexicalBlockDiscriminator:
        case CommonDebugInfoDebugInlinedVariable:
        case CommonDebugInfoDebugMacroDef:
        case CommonDebugInfoDebugMacroUndef:
        case CommonDebugInfoDebugImportedEntity:
          break;
        case CommonDebugInfoInstructionsMax:
          assert(0);
          break;
      }
    }
  } else if (ext_inst_type == SPV_EXT_INST_TYPE_NONSEMANTIC_CLSPVREFLECTION) {
    auto import_inst = _.FindDef(inst->GetOperandAs<uint32_t>(2));
    const std::string name = import_inst->GetOperandAs<std::string>(1);
    const std::string reflection = "NonSemantic.ClspvReflection.";
    char* end_ptr;
    auto version_string = name.substr(reflection.size());
    if (version_string.empty()) {
      return _.diag(SPV_ERROR_INVALID_DATA, import_inst)
             << "Missing NonSemantic.ClspvReflection import version";
    }
    uint32_t version = static_cast<uint32_t>(
        std::strtoul(version_string.c_str(), &end_ptr, 10));
    if (end_ptr && *end_ptr != '\0') {
      return _.diag(SPV_ERROR_INVALID_DATA, import_inst)
             << "NonSemantic.ClspvReflection import does not encode the "
                "version correctly";
    }
    if (version == 0 || version > NonSemanticClspvReflectionRevision) {
      return _.diag(SPV_ERROR_INVALID_DATA, import_inst)
             << "Unknown NonSemantic.ClspvReflection import version";
    }

    return ValidateClspvReflectionInstruction(_, inst, version);
  }

  return SPV_SUCCESS;
}

spv_result_t ExtensionPass(ValidationState_t& _, const Instruction* inst) {
  const spv::Op opcode = inst->opcode();
  if (opcode == spv::Op::OpExtension) return ValidateExtension(_, inst);
  if (opcode == spv::Op::OpExtInstImport) return ValidateExtInstImport(_, inst);
  if (opcode == spv::Op::OpExtInst) return ValidateExtInst(_, inst);

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
