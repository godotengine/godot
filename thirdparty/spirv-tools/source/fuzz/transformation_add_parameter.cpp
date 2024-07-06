// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/transformation_add_parameter.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddParameter::TransformationAddParameter(
    protobufs::TransformationAddParameter message)
    : message_(std::move(message)) {}

TransformationAddParameter::TransformationAddParameter(
    uint32_t function_id, uint32_t parameter_fresh_id,
    uint32_t parameter_type_id, std::map<uint32_t, uint32_t> call_parameter_ids,
    uint32_t function_type_fresh_id) {
  message_.set_function_id(function_id);
  message_.set_parameter_fresh_id(parameter_fresh_id);
  message_.set_parameter_type_id(parameter_type_id);
  *message_.mutable_call_parameter_ids() =
      fuzzerutil::MapToRepeatedUInt32Pair(call_parameter_ids);
  message_.set_function_type_fresh_id(function_type_fresh_id);
}

bool TransformationAddParameter::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Check that function exists.
  const auto* function =
      fuzzerutil::FindFunction(ir_context, message_.function_id());
  if (!function ||
      fuzzerutil::FunctionIsEntryPoint(ir_context, function->result_id())) {
    return false;
  }

  // The type must be supported.
  if (ir_context->get_def_use_mgr()->GetDef(message_.parameter_type_id()) ==
      nullptr) {
    return false;
  }
  if (!IsParameterTypeSupported(ir_context, message_.parameter_type_id())) {
    return false;
  }

  // Iterate over all callers.
  std::map<uint32_t, uint32_t> call_parameter_ids_map =
      fuzzerutil::RepeatedUInt32PairToMap(message_.call_parameter_ids());
  for (auto* instr :
       fuzzerutil::GetCallers(ir_context, message_.function_id())) {
    uint32_t caller_id = instr->result_id();

    // If there is no entry for this caller, return false.
    if (call_parameter_ids_map.find(caller_id) ==
        call_parameter_ids_map.end()) {
      return false;
    }
    uint32_t value_id = call_parameter_ids_map[caller_id];

    auto value_instr = ir_context->get_def_use_mgr()->GetDef(value_id);
    if (!value_instr) {
      return false;
    }
    // If the id of the value of the map is not available before the caller,
    // return false.
    if (!fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, instr,
                                                    value_id)) {
      return false;
    }

    // The type of the value must be defined.
    uint32_t value_type_id = fuzzerutil::GetTypeId(ir_context, value_id);
    if (!value_type_id) {
      return false;
    }

    // Type of every value of the map must be the same for all callers.
    if (message_.parameter_type_id() != value_type_id) {
      return false;
    }
  }
  return fuzzerutil::IsFreshId(ir_context, message_.parameter_fresh_id()) &&
         fuzzerutil::IsFreshId(ir_context, message_.function_type_fresh_id()) &&
         message_.parameter_fresh_id() != message_.function_type_fresh_id();
}

void TransformationAddParameter::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Find the function that will be transformed.
  auto* function = fuzzerutil::FindFunction(ir_context, message_.function_id());
  assert(function && "Can't find the function");

  std::map<uint32_t, uint32_t> call_parameter_ids_map =
      fuzzerutil::RepeatedUInt32PairToMap(message_.call_parameter_ids());

  uint32_t new_parameter_type_id = message_.parameter_type_id();
  auto new_parameter_type =
      ir_context->get_type_mgr()->GetType(new_parameter_type_id);
  assert(new_parameter_type && "New parameter has invalid type.");

  // Add new parameters to the function.
  function->AddParameter(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpFunctionParameter, new_parameter_type_id,
      message_.parameter_fresh_id(), opt::Instruction::OperandList()));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.parameter_fresh_id());

  // Fix all OpFunctionCall instructions.
  for (auto* inst : fuzzerutil::GetCallers(ir_context, function->result_id())) {
    inst->AddOperand(
        {SPV_OPERAND_TYPE_ID, {call_parameter_ids_map[inst->result_id()]}});
  }

  // Update function's type.
  {
    // We use a separate scope here since |old_function_type| might become a
    // dangling pointer after the call to the fuzzerutil::UpdateFunctionType.

    const auto* old_function_type =
        fuzzerutil::GetFunctionType(ir_context, function);
    assert(old_function_type && "Function must have a valid type");

    std::vector<uint32_t> parameter_type_ids;
    for (uint32_t i = 1; i < old_function_type->NumInOperands(); ++i) {
      parameter_type_ids.push_back(
          old_function_type->GetSingleWordInOperand(i));
    }

    parameter_type_ids.push_back(new_parameter_type_id);

    fuzzerutil::UpdateFunctionType(
        ir_context, function->result_id(), message_.function_type_fresh_id(),
        old_function_type->GetSingleWordInOperand(0), parameter_type_ids);
  }

  auto new_parameter_kind = new_parameter_type->kind();

  // Make sure our changes are analyzed.
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);

  // If the |new_parameter_type_id| is not a pointer type, mark id as
  // irrelevant so that we can replace its use with some other id. If the
  // |new_parameter_type_id| is a pointer type, we cannot mark it with
  // IdIsIrrelevant, because this pointer might be replaced by a pointer from
  // original shader. This would change the semantics of the module. In the case
  // of a pointer type we mark it with PointeeValueIsIrrelevant.
  if (new_parameter_kind != opt::analysis::Type::kPointer) {
    transformation_context->GetFactManager()->AddFactIdIsIrrelevant(
        message_.parameter_fresh_id());
  } else {
    transformation_context->GetFactManager()->AddFactValueOfPointeeIsIrrelevant(
        message_.parameter_fresh_id());
  }
}

protobufs::Transformation TransformationAddParameter::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_parameter() = message_;
  return result;
}

bool TransformationAddParameter::IsParameterTypeSupported(
    opt::IRContext* ir_context, uint32_t type_id) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3403):
  //  Think about other type instructions we can add here.
  opt::Instruction* type_inst = ir_context->get_def_use_mgr()->GetDef(type_id);
  switch (type_inst->opcode()) {
    case spv::Op::OpTypeBool:
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeVector:
      return true;
    case spv::Op::OpTypeArray:
      return IsParameterTypeSupported(ir_context,
                                      type_inst->GetSingleWordInOperand(0));
    case spv::Op::OpTypeStruct:
      if (fuzzerutil::HasBlockOrBufferBlockDecoration(ir_context, type_id)) {
        return false;
      }
      for (uint32_t i = 0; i < type_inst->NumInOperands(); i++) {
        if (!IsParameterTypeSupported(ir_context,
                                      type_inst->GetSingleWordInOperand(i))) {
          return false;
        }
      }
      return true;
    case spv::Op::OpTypePointer: {
      spv::StorageClass storage_class =
          static_cast<spv::StorageClass>(type_inst->GetSingleWordInOperand(0));
      switch (storage_class) {
        case spv::StorageClass::Private:
        case spv::StorageClass::Function:
        case spv::StorageClass::Workgroup: {
          return IsParameterTypeSupported(ir_context,
                                          type_inst->GetSingleWordInOperand(1));
        }
        default:
          return false;
      }
    }
    default:
      return false;
  }
}

std::unordered_set<uint32_t> TransformationAddParameter::GetFreshIds() const {
  return {message_.parameter_fresh_id(), message_.function_type_fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
