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

#include "source/fuzz/fuzzer_pass_construct_composites.h"

#include <memory>

#include "source/fuzz/available_instructions.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_composite_construct.h"

namespace spvtools {
namespace fuzz {

FuzzerPassConstructComposites::FuzzerPassConstructComposites(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassConstructComposites::Apply() {
  // Gather up the ids of all composite types, but skip block-/buffer
  // block-decorated struct types.
  std::vector<uint32_t> composite_type_ids;
  for (auto& inst : GetIRContext()->types_values()) {
    if (fuzzerutil::IsCompositeType(
            GetIRContext()->get_type_mgr()->GetType(inst.result_id())) &&
        !fuzzerutil::HasBlockOrBufferBlockDecoration(GetIRContext(),
                                                     inst.result_id())) {
      composite_type_ids.push_back(inst.result_id());
    }
  }

  if (composite_type_ids.empty()) {
    // There are no composite types, so this fuzzer pass cannot do anything.
    return;
  }

  AvailableInstructions available_composite_constituents(
      GetIRContext(),
      [this](opt::IRContext* ir_context, opt::Instruction* inst) -> bool {
        if (!inst->result_id() || !inst->type_id()) {
          return false;
        }

        // If the id is irrelevant, we can use it since it will not
        // participate in DataSynonym fact. Otherwise, we should be able
        // to produce a synonym out of the id.
        return GetTransformationContext()->GetFactManager()->IdIsIrrelevant(
                   inst->result_id()) ||
               fuzzerutil::CanMakeSynonymOf(ir_context,
                                            *GetTransformationContext(), *inst);
      });

  ForEachInstructionWithInstructionDescriptor(
      [this, &available_composite_constituents, &composite_type_ids](
          opt::Function* /*unused*/, opt::BasicBlock* /*unused*/,
          opt::BasicBlock::iterator inst_it,
          const protobufs::InstructionDescriptor& instruction_descriptor)
          -> void {
        // Randomly decide whether to try inserting a composite construction
        // here.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfConstructingComposite())) {
          return;
        }

        // Check whether it is legitimate to insert a composite construction
        // before the instruction.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                spv::Op::OpCompositeConstruct, inst_it)) {
          return;
        }

        // For each instruction that is available at this program point (i.e. an
        // instruction that is global or whose definition strictly dominates the
        // program point) and suitable for making a synonym of, associate it
        // with the id of its result type.
        TypeIdToInstructions type_id_to_available_instructions;
        auto available_instructions =
            available_composite_constituents.GetAvailableBeforeInstruction(
                &*inst_it);
        for (uint32_t available_instruction_index = 0;
             available_instruction_index < available_instructions.size();
             available_instruction_index++) {
          opt::Instruction* inst =
              available_instructions[available_instruction_index];
          type_id_to_available_instructions[inst->type_id()].push_back(
              inst->result_id());
        }

        // At this point, |composite_type_ids| captures all the composite types
        // we could try to create, while |type_id_to_available_instructions|
        // captures all the available result ids we might use, organized by
        // type.

        // Now we choose a composite type to construct, building it from
        // available constituent components and using zero constants if suitable
        // components are not available.

        std::vector<uint32_t> constructor_arguments;
        uint32_t chosen_composite_type =
            composite_type_ids[GetFuzzerContext()->RandomIndex(
                composite_type_ids)];

        // Construct a composite of this type, using an appropriate helper
        // method depending on the kind of composite type.
        auto composite_type_inst =
            GetIRContext()->get_def_use_mgr()->GetDef(chosen_composite_type);
        switch (composite_type_inst->opcode()) {
          case spv::Op::OpTypeArray:
            constructor_arguments = FindComponentsToConstructArray(
                *composite_type_inst, type_id_to_available_instructions);
            break;
          case spv::Op::OpTypeMatrix:
            constructor_arguments = FindComponentsToConstructMatrix(
                *composite_type_inst, type_id_to_available_instructions);
            break;
          case spv::Op::OpTypeStruct:
            constructor_arguments = FindComponentsToConstructStruct(
                *composite_type_inst, type_id_to_available_instructions);
            break;
          case spv::Op::OpTypeVector:
            constructor_arguments = FindComponentsToConstructVector(
                *composite_type_inst, type_id_to_available_instructions);
            break;
          default:
            assert(false &&
                   "The space of possible composite types should be covered "
                   "by the above cases.");
            break;
        }
        assert(!constructor_arguments.empty());

        // Make and apply a transformation.
        ApplyTransformation(TransformationCompositeConstruct(
            chosen_composite_type, constructor_arguments,
            instruction_descriptor, GetFuzzerContext()->GetFreshId()));
      });
}

std::vector<uint32_t>
FuzzerPassConstructComposites::FindComponentsToConstructArray(
    const opt::Instruction& array_type_instruction,
    const TypeIdToInstructions& type_id_to_available_instructions) {
  assert(array_type_instruction.opcode() == spv::Op::OpTypeArray &&
         "Precondition: instruction must be an array type.");

  // Get the element type for the array.
  auto element_type_id = array_type_instruction.GetSingleWordInOperand(0);

  // Get all instructions at our disposal that compute something of this element
  // type.
  auto available_instructions =
      type_id_to_available_instructions.find(element_type_id);

  uint32_t array_length =
      GetIRContext()
          ->get_def_use_mgr()
          ->GetDef(array_type_instruction.GetSingleWordInOperand(1))
          ->GetSingleWordInOperand(0);

  std::vector<uint32_t> result;
  for (uint32_t index = 0; index < array_length; index++) {
    if (available_instructions == type_id_to_available_instructions.cend()) {
      // No suitable instructions are available, so use a zero constant
      result.push_back(FindOrCreateZeroConstant(element_type_id, true));
    } else {
      result.push_back(
          available_instructions->second[GetFuzzerContext()->RandomIndex(
              available_instructions->second)]);
    }
  }
  return result;
}

std::vector<uint32_t>
FuzzerPassConstructComposites::FindComponentsToConstructMatrix(
    const opt::Instruction& matrix_type_instruction,
    const TypeIdToInstructions& type_id_to_available_instructions) {
  assert(matrix_type_instruction.opcode() == spv::Op::OpTypeMatrix &&
         "Precondition: instruction must be a matrix type.");

  // Get the element type for the matrix.
  auto element_type_id = matrix_type_instruction.GetSingleWordInOperand(0);

  // Get all instructions at our disposal that compute something of this element
  // type.
  auto available_instructions =
      type_id_to_available_instructions.find(element_type_id);

  std::vector<uint32_t> result;
  for (uint32_t index = 0;
       index < matrix_type_instruction.GetSingleWordInOperand(1); index++) {
    if (available_instructions == type_id_to_available_instructions.cend()) {
      // No suitable components are available, so use a zero constant.
      result.push_back(FindOrCreateZeroConstant(element_type_id, true));
    } else {
      result.push_back(
          available_instructions->second[GetFuzzerContext()->RandomIndex(
              available_instructions->second)]);
    }
  }
  return result;
}

std::vector<uint32_t>
FuzzerPassConstructComposites::FindComponentsToConstructStruct(
    const opt::Instruction& struct_type_instruction,
    const TypeIdToInstructions& type_id_to_available_instructions) {
  assert(struct_type_instruction.opcode() == spv::Op::OpTypeStruct &&
         "Precondition: instruction must be a struct type.");
  std::vector<uint32_t> result;
  // Consider the type of each field of the struct.
  for (uint32_t in_operand_index = 0;
       in_operand_index < struct_type_instruction.NumInOperands();
       in_operand_index++) {
    auto element_type_id =
        struct_type_instruction.GetSingleWordInOperand(in_operand_index);
    // Find the instructions at our disposal that compute something of the field
    // type.
    auto available_instructions =
        type_id_to_available_instructions.find(element_type_id);
    if (available_instructions == type_id_to_available_instructions.cend()) {
      // No suitable component is available for this element type, so use a zero
      // constant.
      result.push_back(FindOrCreateZeroConstant(element_type_id, true));
    } else {
      result.push_back(
          available_instructions->second[GetFuzzerContext()->RandomIndex(
              available_instructions->second)]);
    }
  }
  return result;
}

std::vector<uint32_t>
FuzzerPassConstructComposites::FindComponentsToConstructVector(
    const opt::Instruction& vector_type_instruction,
    const TypeIdToInstructions& type_id_to_available_instructions) {
  assert(vector_type_instruction.opcode() == spv::Op::OpTypeVector &&
         "Precondition: instruction must be a vector type.");

  // Get details of the type underlying the vector, and the width of the vector,
  // for convenience.
  auto element_type_id = vector_type_instruction.GetSingleWordInOperand(0);
  auto element_type = GetIRContext()->get_type_mgr()->GetType(element_type_id);
  auto element_count = vector_type_instruction.GetSingleWordInOperand(1);

  // Collect a mapping, from type id to width, for scalar/vector types that are
  // smaller in width than |vector_type|, but that have the same underlying
  // type.  For example, if |vector_type| is vec4, the mapping will be:
  //   { float -> 1, vec2 -> 2, vec3 -> 3 }
  // The mapping will have missing entries if some of these types do not exist.

  std::map<uint32_t, uint32_t> smaller_vector_type_id_to_width;
  // Add the underlying type.  This id must exist, in order for |vector_type| to
  // exist.
  smaller_vector_type_id_to_width[element_type_id] = 1;

  // Now add every vector type with width at least 2, and less than the width of
  // |vector_type|.
  for (uint32_t width = 2; width < element_count; width++) {
    opt::analysis::Vector smaller_vector_type(element_type, width);
    auto smaller_vector_type_id =
        GetIRContext()->get_type_mgr()->GetId(&smaller_vector_type);
    // We might find that there is no declared type of this smaller width.
    // For example, a module can declare vec4 without having declared vec2 or
    // vec3.
    if (smaller_vector_type_id) {
      smaller_vector_type_id_to_width[smaller_vector_type_id] = width;
    }
  }

  // Now we know the types that are available to us, we set about populating a
  // vector of the right length.  We do this by deciding, with no order in mind,
  // which instructions we will use to populate the vector, and subsequently
  // randomly choosing an order.  This is to avoid biasing construction of
  // vectors with smaller vectors to the left and scalars to the right.  That is
  // a concern because, e.g. in the case of populating a vec4, if we populate
  // the constructor instructions left-to-right, we can always choose a vec3 to
  // construct the first three elements, but can only choose a vec3 to construct
  // the last three elements if we chose a float to construct the first element
  // (otherwise there will not be space left for a vec3).

  uint32_t vector_slots_used = 0;

  // The instructions result ids we will use to construct the vector, in no
  // particular order at this stage.
  std::vector<uint32_t> result;

  while (vector_slots_used < element_count) {
    std::vector<uint32_t> instructions_to_choose_from;
    for (auto& entry : smaller_vector_type_id_to_width) {
      if (entry.second >
          std::min(element_count - 1, element_count - vector_slots_used)) {
        continue;
      }
      auto available_instructions =
          type_id_to_available_instructions.find(entry.first);
      if (available_instructions == type_id_to_available_instructions.cend()) {
        continue;
      }
      instructions_to_choose_from.insert(instructions_to_choose_from.end(),
                                         available_instructions->second.begin(),
                                         available_instructions->second.end());
    }
    // If there are no instructions to choose from then use a zero constant,
    // otherwise select one of the instructions at random.
    uint32_t id_of_instruction_to_use =
        instructions_to_choose_from.empty()
            ? FindOrCreateZeroConstant(element_type_id, true)
            : instructions_to_choose_from[GetFuzzerContext()->RandomIndex(
                  instructions_to_choose_from)];
    opt::Instruction* instruction_to_use =
        GetIRContext()->get_def_use_mgr()->GetDef(id_of_instruction_to_use);
    result.push_back(instruction_to_use->result_id());
    auto chosen_type =
        GetIRContext()->get_type_mgr()->GetType(instruction_to_use->type_id());
    if (chosen_type->AsVector()) {
      assert(chosen_type->AsVector()->element_type() == element_type);
      assert(chosen_type->AsVector()->element_count() < element_count);
      assert(chosen_type->AsVector()->element_count() <=
             element_count - vector_slots_used);
      vector_slots_used += chosen_type->AsVector()->element_count();
    } else {
      assert(chosen_type == element_type);
      vector_slots_used += 1;
    }
  }
  assert(vector_slots_used == element_count);

  GetFuzzerContext()->Shuffle(&result);
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
