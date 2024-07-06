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

#include "source/fuzz/transformation_store.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationStore::TransformationStore(protobufs::TransformationStore message)
    : message_(std::move(message)) {}

TransformationStore::TransformationStore(
    uint32_t pointer_id, bool is_atomic, uint32_t memory_scope,
    uint32_t memory_semantics, uint32_t value_id,
    const protobufs::InstructionDescriptor& instruction_to_insert_before) {
  message_.set_pointer_id(pointer_id);
  message_.set_is_atomic(is_atomic);
  message_.set_memory_scope_id(memory_scope);
  message_.set_memory_semantics_id(memory_semantics);
  message_.set_value_id(value_id);
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
}

bool TransformationStore::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // The pointer must exist and have a type.
  auto pointer = ir_context->get_def_use_mgr()->GetDef(message_.pointer_id());
  if (!pointer || !pointer->type_id()) {
    return false;
  }

  // The pointer type must indeed be a pointer.
  auto pointer_type = ir_context->get_def_use_mgr()->GetDef(pointer->type_id());
  assert(pointer_type && "Type id must be defined.");
  if (pointer_type->opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  // The pointer must not be read only.
  if (pointer->IsReadOnlyPointer()) {
    return false;
  }

  // We do not want to allow storing to null or undefined pointers.
  switch (pointer->opcode()) {
    case spv::Op::OpConstantNull:
    case spv::Op::OpUndef:
      return false;
    default:
      break;
  }

  // Determine which instruction we should be inserting before.
  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  // It must exist, ...
  if (!insert_before) {
    return false;
  }
  // ... and it must be legitimate to insert a store before it.
  if (!message_.is_atomic() && !fuzzerutil::CanInsertOpcodeBeforeInstruction(
                                   spv::Op::OpStore, insert_before)) {
    return false;
  }
  if (message_.is_atomic() && !fuzzerutil::CanInsertOpcodeBeforeInstruction(
                                  spv::Op::OpAtomicStore, insert_before)) {
    return false;
  }

  // The block we are inserting into needs to be dead, or else the pointee type
  // of the pointer we are storing to needs to be irrelevant (otherwise the
  // store could impact on the observable behaviour of the module).
  if (!transformation_context.GetFactManager()->BlockIsDead(
          ir_context->get_instr_block(insert_before)->id()) &&
      !transformation_context.GetFactManager()->PointeeValueIsIrrelevant(
          message_.pointer_id())) {
    return false;
  }

  // The value being stored needs to exist and have a type.
  auto value = ir_context->get_def_use_mgr()->GetDef(message_.value_id());
  if (!value || !value->type_id()) {
    return false;
  }

  // The type of the value must match the pointee type.
  if (pointer_type->GetSingleWordInOperand(1) != value->type_id()) {
    return false;
  }

  // The pointer needs to be available at the insertion point.
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                  message_.pointer_id())) {
    return false;
  }

  if (message_.is_atomic()) {
    // Check the exists of memory scope and memory semantics ids.
    auto memory_scope_instruction =
        ir_context->get_def_use_mgr()->GetDef(message_.memory_scope_id());
    auto memory_semantics_instruction =
        ir_context->get_def_use_mgr()->GetDef(message_.memory_semantics_id());

    if (!memory_scope_instruction) {
      return false;
    }
    if (!memory_semantics_instruction) {
      return false;
    }
    // The memory scope and memory semantics instructions must have the
    // 'OpConstant' opcode.
    if (memory_scope_instruction->opcode() != spv::Op::OpConstant) {
      return false;
    }
    if (memory_semantics_instruction->opcode() != spv::Op::OpConstant) {
      return false;
    }
    // The memory scope and memory semantics need to be available before
    // |insert_before|.
    if (!fuzzerutil::IdIsAvailableBeforeInstruction(
            ir_context, insert_before, message_.memory_scope_id())) {
      return false;
    }
    if (!fuzzerutil::IdIsAvailableBeforeInstruction(
            ir_context, insert_before, message_.memory_semantics_id())) {
      return false;
    }
    // The memory scope and memory semantics instructions must have an Integer
    // operand type with signedness does not matters.
    if (ir_context->get_def_use_mgr()
            ->GetDef(memory_scope_instruction->type_id())
            ->opcode() != spv::Op::OpTypeInt) {
      return false;
    }
    if (ir_context->get_def_use_mgr()
            ->GetDef(memory_semantics_instruction->type_id())
            ->opcode() != spv::Op::OpTypeInt) {
      return false;
    }

    // The size of the integer for memory scope and memory semantics
    // instructions must be equal to 32 bits.
    auto memory_scope_int_width =
        ir_context->get_def_use_mgr()
            ->GetDef(memory_scope_instruction->type_id())
            ->GetSingleWordInOperand(0);
    auto memory_semantics_int_width =
        ir_context->get_def_use_mgr()
            ->GetDef(memory_semantics_instruction->type_id())
            ->GetSingleWordInOperand(0);

    if (memory_scope_int_width != 32) {
      return false;
    }
    if (memory_semantics_int_width != 32) {
      return false;
    }

    // The memory scope constant value must be that of spv::Scope::Invocation.
    auto memory_scope_const_value =
        memory_scope_instruction->GetSingleWordInOperand(0);
    if (spv::Scope(memory_scope_const_value) != spv::Scope::Invocation) {
      return false;
    }

    // The memory semantics constant value must match the storage class of the
    // pointer being loaded from.
    auto memory_semantics_const_value = static_cast<spv::MemorySemanticsMask>(
        memory_semantics_instruction->GetSingleWordInOperand(0));
    if (memory_semantics_const_value !=
        fuzzerutil::GetMemorySemanticsForStorageClass(
            static_cast<spv::StorageClass>(
                pointer_type->GetSingleWordInOperand(0)))) {
      return false;
    }
  }

  // The value needs to be available at the insertion point.
  return fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                    message_.value_id());
}

void TransformationStore::Apply(opt::IRContext* ir_context,
                                TransformationContext* /*unused*/) const {
  if (message_.is_atomic()) {
    // OpAtomicStore instruction.
    auto insert_before =
        FindInstruction(message_.instruction_to_insert_before(), ir_context);
    auto new_instruction = MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpAtomicStore, 0, 0,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.pointer_id()}},
             {SPV_OPERAND_TYPE_SCOPE_ID, {message_.memory_scope_id()}},
             {SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID,
              {message_.memory_semantics_id()}},
             {SPV_OPERAND_TYPE_ID, {message_.value_id()}}}));
    auto new_instruction_ptr = new_instruction.get();
    insert_before->InsertBefore(std::move(new_instruction));
    // Inform the def-use manager about the new instruction and record its basic
    // block.
    ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
    ir_context->set_instr_block(new_instruction_ptr,
                                ir_context->get_instr_block(insert_before));

  } else {
    // OpStore instruction.
    auto insert_before =
        FindInstruction(message_.instruction_to_insert_before(), ir_context);
    auto new_instruction = MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpStore, 0, 0,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.pointer_id()}},
             {SPV_OPERAND_TYPE_ID, {message_.value_id()}}}));
    auto new_instruction_ptr = new_instruction.get();
    insert_before->InsertBefore(std::move(new_instruction));
    // Inform the def-use manager about the new instruction and record its basic
    // block.
    ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
    ir_context->set_instr_block(new_instruction_ptr,
                                ir_context->get_instr_block(insert_before));
  }
}

protobufs::Transformation TransformationStore::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_store() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationStore::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
