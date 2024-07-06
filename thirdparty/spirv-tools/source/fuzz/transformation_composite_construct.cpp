// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_composite_construct.h"

#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace fuzz {

TransformationCompositeConstruct::TransformationCompositeConstruct(
    protobufs::TransformationCompositeConstruct message)
    : message_(std::move(message)) {}

TransformationCompositeConstruct::TransformationCompositeConstruct(
    uint32_t composite_type_id, std::vector<uint32_t> component,
    const protobufs::InstructionDescriptor& instruction_to_insert_before,
    uint32_t fresh_id) {
  message_.set_composite_type_id(composite_type_id);
  for (auto a_component : component) {
    message_.add_component(a_component);
  }
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
  message_.set_fresh_id(fresh_id);
}

bool TransformationCompositeConstruct::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    // We require the id for the composite constructor to be unused.
    return false;
  }

  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  if (!insert_before) {
    // The instruction before which the composite should be inserted was not
    // found.
    return false;
  }

  auto composite_type =
      ir_context->get_type_mgr()->GetType(message_.composite_type_id());

  if (!fuzzerutil::IsCompositeType(composite_type)) {
    // The type must actually be a composite.
    return false;
  }

  // If the type is an array, matrix, struct or vector, the components need to
  // be suitable for constructing something of that type.
  if (composite_type->AsArray() &&
      !ComponentsForArrayConstructionAreOK(ir_context,
                                           *composite_type->AsArray())) {
    return false;
  }
  if (composite_type->AsMatrix() &&
      !ComponentsForMatrixConstructionAreOK(ir_context,
                                            *composite_type->AsMatrix())) {
    return false;
  }
  if (composite_type->AsStruct() &&
      !ComponentsForStructConstructionAreOK(ir_context,
                                            *composite_type->AsStruct())) {
    return false;
  }
  if (composite_type->AsVector() &&
      !ComponentsForVectorConstructionAreOK(ir_context,
                                            *composite_type->AsVector())) {
    return false;
  }

  // Now check whether every component being used to initialize the composite is
  // available at the desired program point.
  for (auto component : message_.component()) {
    auto* inst = ir_context->get_def_use_mgr()->GetDef(component);
    if (!inst) {
      return false;
    }

    if (!fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                    component)) {
      return false;
    }
  }

  return true;
}

void TransformationCompositeConstruct::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Use the base and offset information from the transformation to determine
  // where in the module a new instruction should be inserted.
  auto insert_before_inst =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  auto destination_block = ir_context->get_instr_block(insert_before_inst);
  auto insert_before = fuzzerutil::GetIteratorForInstruction(
      destination_block, insert_before_inst);

  // Prepare the input operands for an OpCompositeConstruct instruction.
  opt::Instruction::OperandList in_operands;
  for (auto& component_id : message_.component()) {
    in_operands.push_back({SPV_OPERAND_TYPE_ID, {component_id}});
  }

  // Insert an OpCompositeConstruct instruction.
  auto new_instruction = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpCompositeConstruct, message_.composite_type_id(),
      message_.fresh_id(), in_operands);
  auto new_instruction_ptr = new_instruction.get();
  insert_before.InsertBefore(std::move(new_instruction));
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
  ir_context->set_instr_block(new_instruction_ptr, destination_block);

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  // No analyses need to be invalidated since the transformation is local to a
  // block and the def-use and instruction-to-block mappings have been updated.

  AddDataSynonymFacts(ir_context, transformation_context);
}

bool TransformationCompositeConstruct::ComponentsForArrayConstructionAreOK(
    opt::IRContext* ir_context, const opt::analysis::Array& array_type) const {
  if (array_type.length_info().words[0] !=
      opt::analysis::Array::LengthInfo::kConstant) {
    // We only handle constant-sized arrays.
    return false;
  }
  if (array_type.length_info().words.size() != 2) {
    // We only handle the case where the array size can be captured in a single
    // word.
    return false;
  }
  // Get the array size.
  auto array_size = array_type.length_info().words[1];
  if (static_cast<uint32_t>(message_.component().size()) != array_size) {
    // The number of components must match the array size.
    return false;
  }
  // Check that each component is the result id of an instruction whose type is
  // the array's element type.
  for (auto component_id : message_.component()) {
    auto inst = ir_context->get_def_use_mgr()->GetDef(component_id);
    if (inst == nullptr || !inst->type_id()) {
      // The component does not correspond to an instruction with a result
      // type.
      return false;
    }
    auto component_type = ir_context->get_type_mgr()->GetType(inst->type_id());
    assert(component_type);
    if (component_type != array_type.element_type()) {
      // The component's type does not match the array's element type.
      return false;
    }
  }
  return true;
}

bool TransformationCompositeConstruct::ComponentsForMatrixConstructionAreOK(
    opt::IRContext* ir_context,
    const opt::analysis::Matrix& matrix_type) const {
  if (static_cast<uint32_t>(message_.component().size()) !=
      matrix_type.element_count()) {
    // The number of components must match the number of columns of the matrix.
    return false;
  }
  // Check that each component is the result id of an instruction whose type is
  // the matrix's column type.
  for (auto component_id : message_.component()) {
    auto inst = ir_context->get_def_use_mgr()->GetDef(component_id);
    if (inst == nullptr || !inst->type_id()) {
      // The component does not correspond to an instruction with a result
      // type.
      return false;
    }
    auto component_type = ir_context->get_type_mgr()->GetType(inst->type_id());
    assert(component_type);
    if (component_type != matrix_type.element_type()) {
      // The component's type does not match the matrix's column type.
      return false;
    }
  }
  return true;
}

bool TransformationCompositeConstruct::ComponentsForStructConstructionAreOK(
    opt::IRContext* ir_context,
    const opt::analysis::Struct& struct_type) const {
  if (static_cast<uint32_t>(message_.component().size()) !=
      struct_type.element_types().size()) {
    // The number of components must match the number of fields of the struct.
    return false;
  }
  // Check that each component is the result id of an instruction those type
  // matches the associated field type.
  for (uint32_t field_index = 0;
       field_index < struct_type.element_types().size(); field_index++) {
    auto inst = ir_context->get_def_use_mgr()->GetDef(
        message_.component()[field_index]);
    if (inst == nullptr || !inst->type_id()) {
      // The component does not correspond to an instruction with a result
      // type.
      return false;
    }
    auto component_type = ir_context->get_type_mgr()->GetType(inst->type_id());
    assert(component_type);
    if (component_type != struct_type.element_types()[field_index]) {
      // The component's type does not match the corresponding field type.
      return false;
    }
  }
  return true;
}

bool TransformationCompositeConstruct::ComponentsForVectorConstructionAreOK(
    opt::IRContext* ir_context,
    const opt::analysis::Vector& vector_type) const {
  uint32_t base_element_count = 0;
  auto element_type = vector_type.element_type();
  for (auto& component_id : message_.component()) {
    auto inst = ir_context->get_def_use_mgr()->GetDef(component_id);
    if (inst == nullptr || !inst->type_id()) {
      // The component does not correspond to an instruction with a result
      // type.
      return false;
    }
    auto component_type = ir_context->get_type_mgr()->GetType(inst->type_id());
    assert(component_type);
    if (component_type == element_type) {
      base_element_count++;
    } else if (component_type->AsVector() &&
               component_type->AsVector()->element_type() == element_type) {
      base_element_count += component_type->AsVector()->element_count();
    } else {
      // The component was not appropriate; e.g. no type corresponding to the
      // given id was found, or the type that was found was not compatible
      // with the vector being constructed.
      return false;
    }
  }
  // The number of components provided (when vector components are flattened
  // out) needs to match the length of the vector being constructed.
  return base_element_count == vector_type.element_count();
}

protobufs::Transformation TransformationCompositeConstruct::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_composite_construct() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationCompositeConstruct::GetFreshIds()
    const {
  return {message_.fresh_id()};
}

void TransformationCompositeConstruct::AddDataSynonymFacts(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // If the result id of the composite we are constructing is irrelevant (e.g.
  // because it is in a dead block) then we do not make any synonyms.
  if (transformation_context->GetFactManager()->IdIsIrrelevant(
          message_.fresh_id())) {
    return;
  }

  // Inform the fact manager that we now have new synonyms: every component of
  // the composite is synonymous with the id used to construct that component
  // (so long as it is legitimate to create a synonym from that id), except in
  // the case of a vector where a single vector id can span multiple components.
  auto composite_type =
      ir_context->get_type_mgr()->GetType(message_.composite_type_id());
  uint32_t index = 0;
  for (auto component : message_.component()) {
    auto component_type = ir_context->get_type_mgr()->GetType(
        ir_context->get_def_use_mgr()->GetDef(component)->type_id());
    // Whether the component is a vector being packed into a vector determines
    // how we should keep track of the indices associated with components.
    const bool packing_vector_into_vector =
        composite_type->AsVector() && component_type->AsVector();
    if (!fuzzerutil::CanMakeSynonymOf(
            ir_context, *transformation_context,
            *ir_context->get_def_use_mgr()->GetDef(component))) {
      // We can't make a synonym of this component, so we skip on to the next
      // component.  In the case where we're packing a vector into a vector we
      // have to skip as many components of the resulting vectors as there are
      // elements of the component vector.
      index += packing_vector_into_vector
                   ? component_type->AsVector()->element_count()
                   : 1;
      continue;
    }
    if (packing_vector_into_vector) {
      // The case where the composite being constructed is a vector and the
      // component provided for construction is also a vector is special.  It
      // requires adding a synonym fact relating each element of the sub-vector
      // to the corresponding element of the composite being constructed.
      assert(component_type->AsVector()->element_type() ==
             composite_type->AsVector()->element_type());
      assert(component_type->AsVector()->element_count() <
             composite_type->AsVector()->element_count());
      for (uint32_t subvector_index = 0;
           subvector_index < component_type->AsVector()->element_count();
           subvector_index++) {
        transformation_context->GetFactManager()->AddFactDataSynonym(
            MakeDataDescriptor(component, {subvector_index}),
            MakeDataDescriptor(message_.fresh_id(), {index}));
        index++;
      }
    } else {
      // The other cases are simple: the component is made directly synonymous
      // with the element of the composite being constructed.
      transformation_context->GetFactManager()->AddFactDataSynonym(
          MakeDataDescriptor(component, {}),
          MakeDataDescriptor(message_.fresh_id(), {index}));
      index++;
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
