// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_add_local_variable.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddLocalVariable::TransformationAddLocalVariable(
    spvtools::fuzz::protobufs::TransformationAddLocalVariable message)
    : message_(std::move(message)) {}

TransformationAddLocalVariable::TransformationAddLocalVariable(
    uint32_t fresh_id, uint32_t type_id, uint32_t function_id,
    uint32_t initializer_id, bool value_is_irrelevant) {
  message_.set_fresh_id(fresh_id);
  message_.set_type_id(type_id);
  message_.set_function_id(function_id);
  message_.set_initializer_id(initializer_id);
  message_.set_value_is_irrelevant(value_is_irrelevant);
}

bool TransformationAddLocalVariable::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The provided id must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }
  // The pointer type id must indeed correspond to a pointer, and it must have
  // function storage class.
  auto type_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.type_id());
  if (!type_instruction ||
      type_instruction->opcode() != spv::Op::OpTypePointer ||
      spv::StorageClass(type_instruction->GetSingleWordInOperand(0)) !=
          spv::StorageClass::Function) {
    return false;
  }
  // The initializer must...
  auto initializer_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.initializer_id());
  // ... exist, ...
  if (!initializer_instruction) {
    return false;
  }
  // ... be a constant, ...
  if (!spvOpcodeIsConstant(initializer_instruction->opcode())) {
    return false;
  }
  // ... and have the same type as the pointee type.
  if (initializer_instruction->type_id() !=
      type_instruction->GetSingleWordInOperand(1)) {
    return false;
  }
  // The function to which the local variable is to be added must exist.
  return fuzzerutil::FindFunction(ir_context, message_.function_id());
}

void TransformationAddLocalVariable::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  opt::Instruction* new_instruction = fuzzerutil::AddLocalVariable(
      ir_context, message_.fresh_id(), message_.type_id(),
      message_.function_id(), message_.initializer_id());

  // Inform the def-use manager about the new instruction.
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction);
  ir_context->set_instr_block(
      new_instruction,
      fuzzerutil::FindFunction(ir_context, message_.function_id())
          ->entry()
          .get());

  if (message_.value_is_irrelevant()) {
    transformation_context->GetFactManager()->AddFactValueOfPointeeIsIrrelevant(
        message_.fresh_id());
  }
}

protobufs::Transformation TransformationAddLocalVariable::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_local_variable() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationAddLocalVariable::GetFreshIds()
    const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
