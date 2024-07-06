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

#include "source/fuzz/transformation_mutate_pointer.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationMutatePointer::TransformationMutatePointer(
    protobufs::TransformationMutatePointer message)
    : message_(std::move(message)) {}

TransformationMutatePointer::TransformationMutatePointer(
    uint32_t pointer_id, uint32_t fresh_id,
    const protobufs::InstructionDescriptor& insert_before) {
  message_.set_pointer_id(pointer_id);
  message_.set_fresh_id(fresh_id);
  *message_.mutable_insert_before() = insert_before;
}

bool TransformationMutatePointer::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // Check that |fresh_id| is fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }

  auto* insert_before_inst =
      FindInstruction(message_.insert_before(), ir_context);

  // Check that |insert_before| is a valid instruction descriptor.
  if (!insert_before_inst) {
    return false;
  }

  // Check that it is possible to insert OpLoad and OpStore before
  // |insert_before_inst|. We are only using OpLoad here since the result does
  // not depend on the opcode.
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpLoad,
                                                    insert_before_inst)) {
    return false;
  }

  const auto* pointer_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.pointer_id());

  // Check that |pointer_id| is a result id of a valid pointer instruction.
  if (!pointer_inst || !IsValidPointerInstruction(ir_context, *pointer_inst)) {
    return false;
  }

  // Check that the module contains an irrelevant constant that will be used to
  // mutate |pointer_inst|. The constant is irrelevant so that the latter
  // transformation can change its value to something more interesting.
  auto constant_id = fuzzerutil::MaybeGetZeroConstant(
      ir_context, transformation_context,
      fuzzerutil::GetPointeeTypeIdFromPointerType(ir_context,
                                                  pointer_inst->type_id()),
      true);
  if (!constant_id) {
    return false;
  }

  assert(fuzzerutil::IdIsAvailableBeforeInstruction(
             ir_context, insert_before_inst, constant_id) &&
         "Global constant instruction is not available before "
         "|insert_before_inst|");

  // Check that |pointer_inst| is available before |insert_before_inst|.
  return fuzzerutil::IdIsAvailableBeforeInstruction(
      ir_context, insert_before_inst, pointer_inst->result_id());
}

void TransformationMutatePointer::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  auto* insert_before_inst =
      FindInstruction(message_.insert_before(), ir_context);
  assert(insert_before_inst && "|insert_before| descriptor is invalid");
  opt::BasicBlock* enclosing_block =
      ir_context->get_instr_block(insert_before_inst);

  auto pointee_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, fuzzerutil::GetTypeId(ir_context, message_.pointer_id()));

  // Back up the original value.
  auto backup_instruction = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpLoad, pointee_type_id, message_.fresh_id(),
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.pointer_id()}}});
  auto backup_instruction_ptr = backup_instruction.get();
  insert_before_inst->InsertBefore(std::move(backup_instruction));
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(backup_instruction_ptr);
  ir_context->set_instr_block(backup_instruction_ptr, enclosing_block);

  // Insert a new value.
  auto new_value_instruction = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpStore, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.pointer_id()}},
          {SPV_OPERAND_TYPE_ID,
           {fuzzerutil::MaybeGetZeroConstant(
               ir_context, *transformation_context, pointee_type_id, true)}}});
  auto new_value_instruction_ptr = new_value_instruction.get();
  insert_before_inst->InsertBefore(std::move(new_value_instruction));
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_value_instruction_ptr);
  ir_context->set_instr_block(new_value_instruction_ptr, enclosing_block);

  // Restore the original value.
  auto restore_instruction = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpStore, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.pointer_id()}},
          {SPV_OPERAND_TYPE_ID, {message_.fresh_id()}}});
  auto restore_instruction_ptr = restore_instruction.get();
  insert_before_inst->InsertBefore(std::move(restore_instruction));
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(restore_instruction_ptr);
  ir_context->set_instr_block(restore_instruction_ptr, enclosing_block);

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());
}

protobufs::Transformation TransformationMutatePointer::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_mutate_pointer() = message_;
  return result;
}

bool TransformationMutatePointer::IsValidPointerInstruction(
    opt::IRContext* ir_context, const opt::Instruction& inst) {
  // |inst| must have both result id and type id and it may not cause undefined
  // behaviour.
  if (!inst.result_id() || !inst.type_id() ||
      inst.opcode() == spv::Op::OpUndef ||
      inst.opcode() == spv::Op::OpConstantNull) {
    return false;
  }

  opt::Instruction* type_inst =
      ir_context->get_def_use_mgr()->GetDef(inst.type_id());
  assert(type_inst != nullptr && "|inst| has invalid type id");

  // |inst| must be a pointer.
  if (type_inst->opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  // |inst| must have a supported storage class.
  switch (
      static_cast<spv::StorageClass>(type_inst->GetSingleWordInOperand(0))) {
    case spv::StorageClass::Function:
    case spv::StorageClass::Private:
    case spv::StorageClass::Workgroup:
      break;
    default:
      return false;
  }

  // |inst|'s pointee must consist of scalars and/or composites.
  return fuzzerutil::CanCreateConstant(ir_context,
                                       type_inst->GetSingleWordInOperand(1));
}

std::unordered_set<uint32_t> TransformationMutatePointer::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
