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

#include "source/fuzz/transformation_access_chain.h"

#include <vector>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationAccessChain::TransformationAccessChain(
    protobufs::TransformationAccessChain message)
    : message_(std::move(message)) {}

TransformationAccessChain::TransformationAccessChain(
    uint32_t fresh_id, uint32_t pointer_id,
    const std::vector<uint32_t>& index_id,
    const protobufs::InstructionDescriptor& instruction_to_insert_before,
    const std::vector<std::pair<uint32_t, uint32_t>>& fresh_ids_for_clamping) {
  message_.set_fresh_id(fresh_id);
  message_.set_pointer_id(pointer_id);
  for (auto id : index_id) {
    message_.add_index_id(id);
  }
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
  for (auto clamping_ids_pair : fresh_ids_for_clamping) {
    protobufs::UInt32Pair pair;
    pair.set_first(clamping_ids_pair.first);
    pair.set_second(clamping_ids_pair.second);
    *message_.add_fresh_ids_for_clamping() = pair;
  }
}

bool TransformationAccessChain::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Keep track of the fresh ids used to make sure that they are distinct.
  std::set<uint32_t> fresh_ids_used;

  // The result id must be fresh.
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.fresh_id(), ir_context, &fresh_ids_used)) {
    return false;
  }
  // The pointer id must exist and have a type.
  auto pointer = ir_context->get_def_use_mgr()->GetDef(message_.pointer_id());
  if (!pointer || !pointer->type_id()) {
    return false;
  }
  // The type must indeed be a pointer.
  auto pointer_type = ir_context->get_def_use_mgr()->GetDef(pointer->type_id());
  if (pointer_type->opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  // The described instruction to insert before must exist and be a suitable
  // point where an OpAccessChain instruction could be inserted.
  auto instruction_to_insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  if (!instruction_to_insert_before) {
    return false;
  }
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
          spv::Op::OpAccessChain, instruction_to_insert_before)) {
    return false;
  }

  // Do not allow making an access chain from a null or undefined pointer, as
  // we do not want to allow accessing such pointers.  This might be acceptable
  // in dead blocks, but we conservatively avoid it.
  switch (pointer->opcode()) {
    case spv::Op::OpConstantNull:
    case spv::Op::OpUndef:
      assert(
          false &&
          "Access chains should not be created from null/undefined pointers");
      return false;
    default:
      break;
  }

  // The pointer on which the access chain is to be based needs to be available
  // (according to dominance rules) at the insertion point.
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(
          ir_context, instruction_to_insert_before, message_.pointer_id())) {
    return false;
  }

  // We now need to use the given indices to walk the type structure of the
  // base type of the pointer, making sure that (a) the indices correspond to
  // integers, and (b) these integer values are in-bounds.

  // Start from the base type of the pointer.
  uint32_t subobject_type_id = pointer_type->GetSingleWordInOperand(1);

  int id_pairs_used = 0;

  // Consider the given index ids in turn.
  for (auto index_id : message_.index_id()) {
    // The index value will correspond to the value of the index if the object
    // is a struct, otherwise the value 0 will be used.
    uint32_t index_value;

    // Check whether the object is a struct.
    if (ir_context->get_def_use_mgr()->GetDef(subobject_type_id)->opcode() ==
        spv::Op::OpTypeStruct) {
      // It is a struct: we need to retrieve the integer value.

      bool successful;
      std::tie(successful, index_value) =
          GetStructIndexValue(ir_context, index_id, subobject_type_id);

      if (!successful) {
        return false;
      }
    } else {
      // It is not a struct: the index will need clamping.

      if (message_.fresh_ids_for_clamping().size() <= id_pairs_used) {
        // We don't have enough ids
        return false;
      }

      // Get two new ids to use and update the amount used.
      protobufs::UInt32Pair fresh_ids =
          message_.fresh_ids_for_clamping()[id_pairs_used++];

      // Valid ids need to have been given
      if (fresh_ids.first() == 0 || fresh_ids.second() == 0) {
        return false;
      }

      // Check that the ids are actually fresh and not already used by this
      // transformation.
      if (!CheckIdIsFreshAndNotUsedByThisTransformation(
              fresh_ids.first(), ir_context, &fresh_ids_used) ||
          !CheckIdIsFreshAndNotUsedByThisTransformation(
              fresh_ids.second(), ir_context, &fresh_ids_used)) {
        return false;
      }

      if (!ValidIndexToComposite(ir_context, index_id, subobject_type_id)) {
        return false;
      }

      // Perform the clamping using the fresh ids at our disposal.
      auto index_instruction = ir_context->get_def_use_mgr()->GetDef(index_id);

      uint32_t bound = fuzzerutil::GetBoundForCompositeIndex(
          *ir_context->get_def_use_mgr()->GetDef(subobject_type_id),
          ir_context);

      // The module must have an integer constant of value bound-1 of the same
      // type as the index.
      if (!fuzzerutil::MaybeGetIntegerConstantFromValueAndType(
              ir_context, bound - 1, index_instruction->type_id())) {
        return false;
      }

      // The module must have the definition of bool type to make a comparison.
      if (!fuzzerutil::MaybeGetBoolType(ir_context)) {
        return false;
      }

      // The index is not necessarily a constant, so we may not know its value.
      // We can use index 0 because the components of a non-struct composite
      // all have the same type, and index 0 is always in bounds.
      index_value = 0;
    }

    // Try to walk down the type using this index.  This will yield 0 if the
    // type is not a composite or the index is out of bounds, and the id of
    // the next type otherwise.
    subobject_type_id = fuzzerutil::WalkOneCompositeTypeIndex(
        ir_context, subobject_type_id, index_value);
    if (!subobject_type_id) {
      // Either the type was not a composite (so that too many indices were
      // provided), or the index was out of bounds.
      return false;
    }
  }
  // At this point, |subobject_type_id| is the type of the value targeted by
  // the new access chain.  The result type of the access chain should be a
  // pointer to this type, with the same storage class as for the original
  // pointer.  Such a pointer type needs to exist in the module.
  //
  // We do not use the type manager to look up this type, due to problems
  // associated with pointers to isomorphic structs being regarded as the same.
  return fuzzerutil::MaybeGetPointerType(
             ir_context, subobject_type_id,
             static_cast<spv::StorageClass>(
                 pointer_type->GetSingleWordInOperand(0))) != 0;
}

void TransformationAccessChain::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // The operands to the access chain are the pointer followed by the indices.
  // The result type of the access chain is determined by where the indices
  // lead.  We thus push the pointer to a sequence of operands, and then follow
  // the indices, pushing each to the operand list and tracking the type
  // obtained by following it.  Ultimately this yields the type of the
  // component reached by following all the indices, and the result type is
  // a pointer to this component type.
  opt::Instruction::OperandList operands;

  // Add the pointer id itself.
  operands.push_back({SPV_OPERAND_TYPE_ID, {message_.pointer_id()}});

  // Start walking the indices, starting with the pointer's base type.
  auto pointer_type = ir_context->get_def_use_mgr()->GetDef(
      ir_context->get_def_use_mgr()->GetDef(message_.pointer_id())->type_id());
  uint32_t subobject_type_id = pointer_type->GetSingleWordInOperand(1);

  uint32_t id_pairs_used = 0;

  opt::Instruction* instruction_to_insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  opt::BasicBlock* enclosing_block =
      ir_context->get_instr_block(instruction_to_insert_before);

  // Go through the index ids in turn.
  for (auto index_id : message_.index_id()) {
    uint32_t index_value;

    // Actual id to be used in the instruction: the original id
    // or the clamped one.
    uint32_t new_index_id;

    // Check whether the object is a struct.
    if (ir_context->get_def_use_mgr()->GetDef(subobject_type_id)->opcode() ==
        spv::Op::OpTypeStruct) {
      // It is a struct: we need to retrieve the integer value.

      index_value =
          GetStructIndexValue(ir_context, index_id, subobject_type_id).second;

      new_index_id = index_id;

    } else {
      // It is not a struct: the index will need clamping.

      // Get two new ids to use and update the amount used.
      protobufs::UInt32Pair fresh_ids =
          message_.fresh_ids_for_clamping()[id_pairs_used++];

      // Perform the clamping using the fresh ids at our disposal.
      // The module will not be changed if |add_clamping_instructions| is not
      // set.
      auto index_instruction = ir_context->get_def_use_mgr()->GetDef(index_id);

      uint32_t bound = fuzzerutil::GetBoundForCompositeIndex(
          *ir_context->get_def_use_mgr()->GetDef(subobject_type_id),
          ir_context);

      auto bound_minus_one_id =
          fuzzerutil::MaybeGetIntegerConstantFromValueAndType(
              ir_context, bound - 1, index_instruction->type_id());

      assert(bound_minus_one_id &&
             "A constant of value bound - 1 and the same type as the index "
             "must exist as a precondition.");

      uint32_t bool_type_id = fuzzerutil::MaybeGetBoolType(ir_context);

      assert(bool_type_id &&
             "An OpTypeBool instruction must exist as a precondition.");

      auto int_type_inst =
          ir_context->get_def_use_mgr()->GetDef(index_instruction->type_id());

      // Clamp the integer and add the corresponding instructions in the module
      // if |add_clamping_instructions| is set.

      // Compare the index with the bound via an instruction of the form:
      //   %fresh_ids.first = OpULessThanEqual %bool %int_id %bound_minus_one.
      fuzzerutil::UpdateModuleIdBound(ir_context, fresh_ids.first());
      auto comparison_instruction = MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpULessThanEqual, bool_type_id,
          fresh_ids.first(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {index_instruction->result_id()}},
               {SPV_OPERAND_TYPE_ID, {bound_minus_one_id}}}));
      auto comparison_instruction_ptr = comparison_instruction.get();
      instruction_to_insert_before->InsertBefore(
          std::move(comparison_instruction));
      ir_context->get_def_use_mgr()->AnalyzeInstDefUse(
          comparison_instruction_ptr);
      ir_context->set_instr_block(comparison_instruction_ptr, enclosing_block);

      // Select the index if in-bounds, otherwise one less than the bound:
      //   %fresh_ids.second = OpSelect %int_type %fresh_ids.first %int_id
      //                           %bound_minus_one
      fuzzerutil::UpdateModuleIdBound(ir_context, fresh_ids.second());
      auto select_instruction = MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpSelect, int_type_inst->result_id(),
          fresh_ids.second(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {fresh_ids.first()}},
               {SPV_OPERAND_TYPE_ID, {index_instruction->result_id()}},
               {SPV_OPERAND_TYPE_ID, {bound_minus_one_id}}}));
      auto select_instruction_ptr = select_instruction.get();
      instruction_to_insert_before->InsertBefore(std::move(select_instruction));
      ir_context->get_def_use_mgr()->AnalyzeInstDefUse(select_instruction_ptr);
      ir_context->set_instr_block(select_instruction_ptr, enclosing_block);

      new_index_id = fresh_ids.second();

      index_value = 0;
    }

    // Add the correct index id to the operands.
    operands.push_back({SPV_OPERAND_TYPE_ID, {new_index_id}});

    // Walk to the next type in the composite object using this index.
    subobject_type_id = fuzzerutil::WalkOneCompositeTypeIndex(
        ir_context, subobject_type_id, index_value);
  }
  // The access chain's result type is a pointer to the composite component
  // that was reached after following all indices.  The storage class is that
  // of the original pointer.
  uint32_t result_type = fuzzerutil::MaybeGetPointerType(
      ir_context, subobject_type_id,
      static_cast<spv::StorageClass>(pointer_type->GetSingleWordInOperand(0)));

  // Add the access chain instruction to the module, and update the module's
  // id bound.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());
  auto access_chain_instruction =
      MakeUnique<opt::Instruction>(ir_context, spv::Op::OpAccessChain,
                                   result_type, message_.fresh_id(), operands);
  auto access_chain_instruction_ptr = access_chain_instruction.get();
  instruction_to_insert_before->InsertBefore(
      std::move(access_chain_instruction));
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(
      access_chain_instruction_ptr);
  ir_context->set_instr_block(access_chain_instruction_ptr, enclosing_block);

  // If the base pointer's pointee value was irrelevant, the same is true of
  // the pointee value of the result of this access chain.
  if (transformation_context->GetFactManager()->PointeeValueIsIrrelevant(
          message_.pointer_id())) {
    transformation_context->GetFactManager()->AddFactValueOfPointeeIsIrrelevant(
        message_.fresh_id());
  }
}

protobufs::Transformation TransformationAccessChain::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_access_chain() = message_;
  return result;
}

std::pair<bool, uint32_t> TransformationAccessChain::GetStructIndexValue(
    opt::IRContext* ir_context, uint32_t index_id,
    uint32_t object_type_id) const {
  assert(ir_context->get_def_use_mgr()->GetDef(object_type_id)->opcode() ==
             spv::Op::OpTypeStruct &&
         "Precondition: the type must be a struct type.");
  if (!ValidIndexToComposite(ir_context, index_id, object_type_id)) {
    return {false, 0};
  }
  auto index_instruction = ir_context->get_def_use_mgr()->GetDef(index_id);

  uint32_t bound = fuzzerutil::GetBoundForCompositeIndex(
      *ir_context->get_def_use_mgr()->GetDef(object_type_id), ir_context);

  // Ensure that the index given must represent a constant.
  assert(spvOpcodeIsConstant(index_instruction->opcode()) &&
         "A non-constant index should already have been rejected.");

  // The index must be in bounds.
  uint32_t value = index_instruction->GetSingleWordInOperand(0);

  if (value >= bound) {
    return {false, 0};
  }

  return {true, value};
}

bool TransformationAccessChain::ValidIndexToComposite(
    opt::IRContext* ir_context, uint32_t index_id, uint32_t object_type_id) {
  auto object_type_def = ir_context->get_def_use_mgr()->GetDef(object_type_id);
  // The object being indexed must be a composite.
  if (!spvOpcodeIsComposite(object_type_def->opcode())) {
    return false;
  }

  // Get the defining instruction of the index.
  auto index_instruction = ir_context->get_def_use_mgr()->GetDef(index_id);
  if (!index_instruction) {
    return false;
  }

  // The index type must be 32-bit integer.
  auto index_type =
      ir_context->get_def_use_mgr()->GetDef(index_instruction->type_id());
  if (index_type->opcode() != spv::Op::OpTypeInt ||
      index_type->GetSingleWordInOperand(0) != 32) {
    return false;
  }

  // If the object being traversed is a struct, the id must correspond to an
  // in-bound constant.
  if (object_type_def->opcode() == spv::Op::OpTypeStruct) {
    if (!spvOpcodeIsConstant(index_instruction->opcode())) {
      return false;
    }
  }
  return true;
}

std::unordered_set<uint32_t> TransformationAccessChain::GetFreshIds() const {
  std::unordered_set<uint32_t> result = {message_.fresh_id()};
  for (auto& fresh_ids_for_clamping : message_.fresh_ids_for_clamping()) {
    result.insert(fresh_ids_for_clamping.first());
    result.insert(fresh_ids_for_clamping.second());
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
