// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/transformation_push_id_through_variable.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationPushIdThroughVariable::TransformationPushIdThroughVariable(
    protobufs::TransformationPushIdThroughVariable message)
    : message_(std::move(message)) {}

TransformationPushIdThroughVariable::TransformationPushIdThroughVariable(
    uint32_t value_id, uint32_t value_synonym_id, uint32_t variable_id,
    uint32_t variable_storage_class, uint32_t initializer_id,
    const protobufs::InstructionDescriptor& instruction_descriptor) {
  message_.set_value_id(value_id);
  message_.set_value_synonym_id(value_synonym_id);
  message_.set_variable_id(variable_id);
  message_.set_variable_storage_class(variable_storage_class);
  message_.set_initializer_id(initializer_id);
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
}

bool TransformationPushIdThroughVariable::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.value_synonym_id| and |message_.variable_id| must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.value_synonym_id()) ||
      !fuzzerutil::IsFreshId(ir_context, message_.variable_id())) {
    return false;
  }

  // The instruction to insert before must be defined.
  auto instruction_to_insert_before =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  if (!instruction_to_insert_before) {
    return false;
  }

  // It must be valid to insert the OpStore and OpLoad instruction before it.
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
          spv::Op::OpStore, instruction_to_insert_before) ||
      !fuzzerutil::CanInsertOpcodeBeforeInstruction(
          spv::Op::OpLoad, instruction_to_insert_before)) {
    return false;
  }

  // The instruction to insert before must belong to a reachable block.
  auto basic_block = ir_context->get_instr_block(instruction_to_insert_before);
  if (!ir_context->IsReachable(*basic_block)) {
    return false;
  }

  // The value instruction must be defined and have a type.
  auto value_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.value_id());
  if (!value_instruction || !value_instruction->type_id()) {
    return false;
  }

  // A pointer type instruction pointing to the value type must be defined.
  auto pointer_type_id = fuzzerutil::MaybeGetPointerType(
      ir_context, value_instruction->type_id(),
      static_cast<spv::StorageClass>(message_.variable_storage_class()));
  if (!pointer_type_id) {
    return false;
  }

  // |message_.variable_storage_class| must be private or function.
  assert((message_.variable_storage_class() ==
              (uint32_t)spv::StorageClass::Private ||
          message_.variable_storage_class() ==
              (uint32_t)spv::StorageClass::Function) &&
         "The variable storage class must be private or function.");

  // Check that initializer is valid.
  const auto* constant_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.initializer_id());
  if (!constant_inst || !spvOpcodeIsConstant(constant_inst->opcode()) ||
      value_instruction->type_id() != constant_inst->type_id()) {
    return false;
  }

  // |message_.value_id| must be available at the insertion point.
  return fuzzerutil::IdIsAvailableBeforeInstruction(
      ir_context, instruction_to_insert_before, message_.value_id());
}

void TransformationPushIdThroughVariable::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  auto value_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.value_id());

  opt::Instruction* insert_before =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  opt::BasicBlock* enclosing_block = ir_context->get_instr_block(insert_before);

  // A pointer type instruction pointing to the value type must be defined.
  auto pointer_type_id = fuzzerutil::MaybeGetPointerType(
      ir_context, value_instruction->type_id(),
      static_cast<spv::StorageClass>(message_.variable_storage_class()));
  assert(pointer_type_id && "The required pointer type must be available.");

  // Adds whether a global or local variable.
  if (spv::StorageClass(message_.variable_storage_class()) ==
      spv::StorageClass::Private) {
    opt::Instruction* global_variable = fuzzerutil::AddGlobalVariable(
        ir_context, message_.variable_id(), pointer_type_id,
        spv::StorageClass::Private, message_.initializer_id());
    ir_context->get_def_use_mgr()->AnalyzeInstDefUse(global_variable);
  } else {
    opt::Function* function =
        ir_context
            ->get_instr_block(
                FindInstruction(message_.instruction_descriptor(), ir_context))
            ->GetParent();
    opt::Instruction* local_variable = fuzzerutil::AddLocalVariable(
        ir_context, message_.variable_id(), pointer_type_id,
        function->result_id(), message_.initializer_id());
    ir_context->get_def_use_mgr()->AnalyzeInstDefUse(local_variable);
    ir_context->set_instr_block(local_variable, &*function->entry());
  }

  // First, insert the OpLoad instruction before |instruction_descriptor| and
  // then insert the OpStore instruction before the OpLoad instruction.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.value_synonym_id());
  opt::Instruction* load_instruction =
      insert_before->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpLoad, value_instruction->type_id(),
          message_.value_synonym_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_id()}}})));
  opt::Instruction* store_instruction =
      load_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_id()}},
               {SPV_OPERAND_TYPE_ID, {message_.value_id()}}})));
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(store_instruction);
  ir_context->set_instr_block(store_instruction, enclosing_block);
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(load_instruction);
  ir_context->set_instr_block(load_instruction, enclosing_block);

  // We should be able to create a synonym of |value_id| if it's not irrelevant.
  if (fuzzerutil::CanMakeSynonymOf(ir_context, *transformation_context,
                                   *value_instruction) &&
      !transformation_context->GetFactManager()->IdIsIrrelevant(
          message_.value_synonym_id())) {
    // Adds the fact that |message_.value_synonym_id|
    // and |message_.value_id| are synonymous.
    transformation_context->GetFactManager()->AddFactDataSynonym(
        MakeDataDescriptor(message_.value_synonym_id(), {}),
        MakeDataDescriptor(message_.value_id(), {}));
  }
}

protobufs::Transformation TransformationPushIdThroughVariable::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_push_id_through_variable() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationPushIdThroughVariable::GetFreshIds()
    const {
  return {message_.value_synonym_id(), message_.variable_id()};
}

}  // namespace fuzz
}  // namespace spvtools
