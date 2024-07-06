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

#include "transformation_composite_insert.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationCompositeInsert::TransformationCompositeInsert(
    protobufs::TransformationCompositeInsert message)
    : message_(std::move(message)) {}

TransformationCompositeInsert::TransformationCompositeInsert(
    const protobufs::InstructionDescriptor& instruction_to_insert_before,
    uint32_t fresh_id, uint32_t composite_id, uint32_t object_id,
    const std::vector<uint32_t>& index) {
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
  message_.set_fresh_id(fresh_id);
  message_.set_composite_id(composite_id);
  message_.set_object_id(object_id);
  for (auto an_index : index) {
    message_.add_index(an_index);
  }
}

bool TransformationCompositeInsert::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.fresh_id| must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }

  // |message_.composite_id| must refer to an existing composite value.
  auto composite =
      ir_context->get_def_use_mgr()->GetDef(message_.composite_id());

  if (!IsCompositeInstructionSupported(ir_context, composite)) {
    return false;
  }

  // The indices in |message_.index| must be suitable for indexing into
  // |composite->type_id()|.
  auto component_to_be_replaced_type_id = fuzzerutil::WalkCompositeTypeIndices(
      ir_context, composite->type_id(), message_.index());
  if (component_to_be_replaced_type_id == 0) {
    return false;
  }

  // The instruction having the id of |message_.object_id| must be defined.
  auto object_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.object_id());
  if (object_instruction == nullptr || object_instruction->type_id() == 0) {
    return false;
  }

  // We ignore pointers for now.
  auto object_instruction_type =
      ir_context->get_type_mgr()->GetType(object_instruction->type_id());
  if (object_instruction_type->AsPointer() != nullptr) {
    return false;
  }

  // The type id of the object having |message_.object_id| and the type id of
  // the component of the composite at index |message_.index| must be the same.
  if (component_to_be_replaced_type_id != object_instruction->type_id()) {
    return false;
  }

  // |message_.instruction_to_insert_before| must be a defined instruction.
  auto instruction_to_insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  if (instruction_to_insert_before == nullptr) {
    return false;
  }

  // |message_.composite_id| and |message_.object_id| must be available before
  // the |message_.instruction_to_insert_before|.
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(
          ir_context, instruction_to_insert_before, message_.composite_id())) {
    return false;
  }
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(
          ir_context, instruction_to_insert_before, message_.object_id())) {
    return false;
  }

  // It must be possible to insert an OpCompositeInsert before this
  // instruction.
  return fuzzerutil::CanInsertOpcodeBeforeInstruction(
      spv::Op::OpCompositeInsert, instruction_to_insert_before);
}

void TransformationCompositeInsert::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // |message_.struct_fresh_id| must be fresh.
  assert(fuzzerutil::IsFreshId(ir_context, message_.fresh_id()) &&
         "|message_.fresh_id| must be fresh");

  std::vector<uint32_t> index =
      fuzzerutil::RepeatedFieldToVector(message_.index());
  opt::Instruction::OperandList in_operands;
  in_operands.push_back({SPV_OPERAND_TYPE_ID, {message_.object_id()}});
  in_operands.push_back({SPV_OPERAND_TYPE_ID, {message_.composite_id()}});
  for (auto i : index) {
    in_operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}});
  }
  auto composite_type_id =
      fuzzerutil::GetTypeId(ir_context, message_.composite_id());

  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  auto new_instruction = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpCompositeInsert, composite_type_id,
      message_.fresh_id(), std::move(in_operands));
  auto new_instruction_ptr = new_instruction.get();
  insert_before->InsertBefore(std::move(new_instruction));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  // Inform the def-use manager about the new instruction and record its basic
  // block.
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
  ir_context->set_instr_block(new_instruction_ptr,
                              ir_context->get_instr_block(insert_before));

  // Add data synonym facts that arise from the insertion.
  AddDataSynonymFacts(ir_context, transformation_context);
}

protobufs::Transformation TransformationCompositeInsert::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_composite_insert() = message_;
  return result;
}

bool TransformationCompositeInsert::IsCompositeInstructionSupported(
    opt::IRContext* ir_context, opt::Instruction* instruction) {
  if (instruction == nullptr) {
    return false;
  }
  if (instruction->result_id() == 0 || instruction->type_id() == 0) {
    return false;
  }
  auto composite_type =
      ir_context->get_type_mgr()->GetType(instruction->type_id());
  if (!fuzzerutil::IsCompositeType(composite_type)) {
    return false;
  }

  // Empty composites are not supported.
  auto instruction_type_inst =
      ir_context->get_def_use_mgr()->GetDef(instruction->type_id());
  if (fuzzerutil::GetBoundForCompositeIndex(*instruction_type_inst,
                                            ir_context) == 0) {
    return false;
  }
  return true;
}

std::unordered_set<uint32_t> TransformationCompositeInsert::GetFreshIds()
    const {
  return {message_.fresh_id()};
}

void TransformationCompositeInsert::AddDataSynonymFacts(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // If the result id arising from the insertion is irrelevant then do not add
  // any data synonym facts.  (The result id can be irrelevant if the insertion
  // occurs in a dead block.)
  if (transformation_context->GetFactManager()->IdIsIrrelevant(
          message_.fresh_id())) {
    return;
  }

  // So long as the |message_.composite_id| is suitable for participating in
  // synonyms, every every element of the insertion result except for at the
  // index being inserted into is synonymous with the corresponding element of
  // |message_.composite_id|.  In that case, for every index that is a prefix of
  // |index|, the components different from the one that contains the inserted
  // object are synonymous with corresponding elements in the original
  // composite.
  uint32_t current_node_type_id =
      fuzzerutil::GetTypeId(ir_context, message_.composite_id());
  std::vector<uint32_t> current_index;

  std::vector<uint32_t> index =
      fuzzerutil::RepeatedFieldToVector(message_.index());

  for (uint32_t current_level : index) {
    auto current_node_type_inst =
        ir_context->get_def_use_mgr()->GetDef(current_node_type_id);
    uint32_t index_to_skip = current_level;
    uint32_t num_of_components = fuzzerutil::GetBoundForCompositeIndex(
        *current_node_type_inst, ir_context);

    // Update the current_node_type_id.
    current_node_type_id = fuzzerutil::WalkOneCompositeTypeIndex(
        ir_context, current_node_type_id, index_to_skip);

    for (uint32_t i = 0; i < num_of_components; i++) {
      if (i == index_to_skip) {
        continue;
      }
      current_index.push_back(i);
      if (fuzzerutil::CanMakeSynonymOf(ir_context, *transformation_context,
                                       *ir_context->get_def_use_mgr()->GetDef(
                                           message_.composite_id()))) {
        transformation_context->GetFactManager()->AddFactDataSynonym(
            MakeDataDescriptor(message_.fresh_id(), current_index),
            MakeDataDescriptor(message_.composite_id(), current_index));
      }
      current_index.pop_back();
    }
    // Store the prefix of the |index|.
    current_index.push_back(current_level);
  }
  // If the object being inserted supports synonym creation then it is
  // synonymous with the result of the insert instruction at the given index.
  if (fuzzerutil::CanMakeSynonymOf(
          ir_context, *transformation_context,
          *ir_context->get_def_use_mgr()->GetDef(message_.object_id()))) {
    transformation_context->GetFactManager()->AddFactDataSynonym(
        MakeDataDescriptor(message_.object_id(), {}),
        MakeDataDescriptor(message_.fresh_id(), index));
  }
}

}  // namespace fuzz
}  // namespace spvtools
