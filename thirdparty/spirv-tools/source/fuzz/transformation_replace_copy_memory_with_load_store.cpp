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

#include "source/fuzz/transformation_replace_copy_memory_with_load_store.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceCopyMemoryWithLoadStore::
    TransformationReplaceCopyMemoryWithLoadStore(
        protobufs::TransformationReplaceCopyMemoryWithLoadStore message)
    : message_(std::move(message)) {}

TransformationReplaceCopyMemoryWithLoadStore::
    TransformationReplaceCopyMemoryWithLoadStore(
        uint32_t fresh_id, const protobufs::InstructionDescriptor&
                               copy_memory_instruction_descriptor) {
  message_.set_fresh_id(fresh_id);
  *message_.mutable_copy_memory_instruction_descriptor() =
      copy_memory_instruction_descriptor;
}

bool TransformationReplaceCopyMemoryWithLoadStore::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.fresh_id| must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }
  // The instruction to be replaced must be defined and have opcode
  // OpCopyMemory.
  auto copy_memory_instruction = FindInstruction(
      message_.copy_memory_instruction_descriptor(), ir_context);
  if (!copy_memory_instruction ||
      copy_memory_instruction->opcode() != spv::Op::OpCopyMemory) {
    return false;
  }
  return true;
}

void TransformationReplaceCopyMemoryWithLoadStore::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto copy_memory_instruction = FindInstruction(
      message_.copy_memory_instruction_descriptor(), ir_context);
  // |copy_memory_instruction| must be defined.
  assert(copy_memory_instruction &&
         copy_memory_instruction->opcode() == spv::Op::OpCopyMemory &&
         "The required OpCopyMemory instruction must be defined.");

  // Integrity check: Both operands must be pointers.

  // Get types of ids used as a source and target of |copy_memory_instruction|.
  auto target = ir_context->get_def_use_mgr()->GetDef(
      copy_memory_instruction->GetSingleWordInOperand(0));
  auto source = ir_context->get_def_use_mgr()->GetDef(
      copy_memory_instruction->GetSingleWordInOperand(1));
  auto target_type_opcode =
      ir_context->get_def_use_mgr()->GetDef(target->type_id())->opcode();
  auto source_type_opcode =
      ir_context->get_def_use_mgr()->GetDef(source->type_id())->opcode();

  // Keep release-mode compilers happy. (No unused variables.)
  (void)target;
  (void)source;
  (void)target_type_opcode;
  (void)source_type_opcode;

  assert(target_type_opcode == spv::Op::OpTypePointer &&
         source_type_opcode == spv::Op::OpTypePointer &&
         "Operands must be of type OpTypePointer");

  // Integrity check: |source| and |target| must point to the same type.
  uint32_t target_pointee_type = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, target->type_id());
  uint32_t source_pointee_type = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, source->type_id());

  // Keep release-mode compilers happy. (No unused variables.)
  (void)target_pointee_type;
  (void)source_pointee_type;

  assert(target_pointee_type == source_pointee_type &&
         "Operands must have the same type to which they point to.");

  // First, insert the OpStore instruction before the OpCopyMemory instruction
  // and then insert the OpLoad instruction before the OpStore instruction.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());
  FindInstruction(message_.copy_memory_instruction_descriptor(), ir_context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {target->result_id()}},
               {SPV_OPERAND_TYPE_ID, {message_.fresh_id()}}})))
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpLoad, target_pointee_type, message_.fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {source->result_id()}}})));

  // Remove the OpCopyMemory instruction.
  ir_context->KillInst(copy_memory_instruction);

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation
TransformationReplaceCopyMemoryWithLoadStore::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_copy_memory_with_load_store() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationReplaceCopyMemoryWithLoadStore::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
