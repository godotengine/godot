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

#include "source/fuzz/transformation_add_copy_memory.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace fuzz {

TransformationAddCopyMemory::TransformationAddCopyMemory(
    protobufs::TransformationAddCopyMemory message)
    : message_(std::move(message)) {}

TransformationAddCopyMemory::TransformationAddCopyMemory(
    const protobufs::InstructionDescriptor& instruction_descriptor,
    uint32_t fresh_id, uint32_t source_id, spv::StorageClass storage_class,
    uint32_t initializer_id) {
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
  message_.set_fresh_id(fresh_id);
  message_.set_source_id(source_id);
  message_.set_storage_class(uint32_t(storage_class));
  message_.set_initializer_id(initializer_id);
}

bool TransformationAddCopyMemory::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Check that target id is fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }

  // Check that instruction descriptor is valid. This also checks that
  // |message_.instruction_descriptor| is not a global instruction.
  auto* inst = FindInstruction(message_.instruction_descriptor(), ir_context);
  if (!inst) {
    return false;
  }

  // Check that we can insert OpCopyMemory before |instruction_descriptor|.
  auto iter = fuzzerutil::GetIteratorForInstruction(
      ir_context->get_instr_block(inst), inst);
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpCopyMemory,
                                                    iter)) {
    return false;
  }

  // Check that source instruction exists and is valid.
  auto* source_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.source_id());
  if (!source_inst || !IsInstructionSupported(ir_context, source_inst)) {
    return false;
  }

  // |storage_class| is either Function or Private.
  if (spv::StorageClass(message_.storage_class()) !=
          spv::StorageClass::Function &&
      spv::StorageClass(message_.storage_class()) !=
          spv::StorageClass::Private) {
    return false;
  }

  auto pointee_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, source_inst->type_id());

  // OpTypePointer with |message_.storage_class| exists.
  if (!fuzzerutil::MaybeGetPointerType(
          ir_context, pointee_type_id,
          static_cast<spv::StorageClass>(message_.storage_class()))) {
    return false;
  }

  // Check that |initializer_id| exists and has valid type.
  const auto* initializer_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.initializer_id());
  if (!initializer_inst || initializer_inst->type_id() != pointee_type_id) {
    return false;
  }

  // Check that domination rules are satisfied.
  return fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, inst,
                                                    message_.source_id());
}

void TransformationAddCopyMemory::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Insert OpCopyMemory before |instruction_descriptor|.
  auto* insert_before_inst =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  assert(insert_before_inst);
  opt::BasicBlock* enclosing_block =
      ir_context->get_instr_block(insert_before_inst);

  // Add global or local variable to copy memory into.
  auto storage_class = static_cast<spv::StorageClass>(message_.storage_class());
  auto type_id = fuzzerutil::MaybeGetPointerType(
      ir_context,
      fuzzerutil::GetPointeeTypeIdFromPointerType(
          ir_context, fuzzerutil::GetTypeId(ir_context, message_.source_id())),
      storage_class);

  if (storage_class == spv::StorageClass::Private) {
    opt::Instruction* new_global =
        fuzzerutil::AddGlobalVariable(ir_context, message_.fresh_id(), type_id,
                                      storage_class, message_.initializer_id());
    ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_global);
  } else {
    assert(storage_class == spv::StorageClass::Function &&
           "Storage class can be either Private or Function");
    opt::Function* enclosing_function = enclosing_block->GetParent();
    opt::Instruction* new_local = fuzzerutil::AddLocalVariable(
        ir_context, message_.fresh_id(), type_id,
        enclosing_function->result_id(), message_.initializer_id());
    ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_local);
    ir_context->set_instr_block(new_local, &*enclosing_function->entry());
  }

  auto insert_before_iter = fuzzerutil::GetIteratorForInstruction(
      enclosing_block, insert_before_inst);

  auto new_instruction = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpCopyMemory, 0, 0,
      opt::Instruction::OperandList{
          {SPV_OPERAND_TYPE_ID, {message_.fresh_id()}},
          {SPV_OPERAND_TYPE_ID, {message_.source_id()}}});
  auto new_instruction_ptr = new_instruction.get();
  insert_before_iter.InsertBefore(std::move(new_instruction));
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
  ir_context->set_instr_block(new_instruction_ptr, enclosing_block);

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  // Even though the copy memory instruction will - at least temporarily - lead
  // to the destination and source pointers referring to identical values, this
  // fact is not guaranteed to hold throughout execution of the SPIR-V code
  // since the source pointer could be over-written. We thus assume nothing
  // about the destination pointer, and record this fact so that the destination
  // pointer can be used freely by other fuzzer passes.
  transformation_context->GetFactManager()->AddFactValueOfPointeeIsIrrelevant(
      message_.fresh_id());
}

protobufs::Transformation TransformationAddCopyMemory::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_copy_memory() = message_;
  return result;
}

bool TransformationAddCopyMemory::IsInstructionSupported(
    opt::IRContext* ir_context, opt::Instruction* inst) {
  if (!inst->result_id() || !inst->type_id() ||
      inst->opcode() == spv::Op::OpConstantNull ||
      inst->opcode() == spv::Op::OpUndef) {
    return false;
  }

  const auto* type = ir_context->get_type_mgr()->GetType(inst->type_id());
  assert(type && "Instruction must have a valid type");

  if (!type->AsPointer()) {
    return false;
  }

  // We do not support copying memory from a pointer to a block-/buffer
  // block-decorated struct.
  auto pointee_type_inst = ir_context->get_def_use_mgr()
                               ->GetDef(inst->type_id())
                               ->GetSingleWordInOperand(1);
  if (fuzzerutil::HasBlockOrBufferBlockDecoration(ir_context,
                                                  pointee_type_inst)) {
    return false;
  }

  return CanUsePointeeWithCopyMemory(*type->AsPointer()->pointee_type());
}

bool TransformationAddCopyMemory::CanUsePointeeWithCopyMemory(
    const opt::analysis::Type& type) {
  switch (type.kind()) {
    case opt::analysis::Type::kBool:
    case opt::analysis::Type::kInteger:
    case opt::analysis::Type::kFloat:
    case opt::analysis::Type::kArray:
      return true;
    case opt::analysis::Type::kVector:
      return CanUsePointeeWithCopyMemory(*type.AsVector()->element_type());
    case opt::analysis::Type::kMatrix:
      return CanUsePointeeWithCopyMemory(*type.AsMatrix()->element_type());
    case opt::analysis::Type::kStruct:
      return std::all_of(type.AsStruct()->element_types().begin(),
                         type.AsStruct()->element_types().end(),
                         [](const opt::analysis::Type* element) {
                           return CanUsePointeeWithCopyMemory(*element);
                         });
    default:
      return false;
  }
}

std::unordered_set<uint32_t> TransformationAddCopyMemory::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
