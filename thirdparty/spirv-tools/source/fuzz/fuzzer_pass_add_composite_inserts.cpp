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

#include "source/fuzz/fuzzer_pass_add_composite_inserts.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/pseudo_random_generator.h"
#include "source/fuzz/transformation_composite_insert.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddCompositeInserts::FuzzerPassAddCompositeInserts(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddCompositeInserts::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator instruction_iterator,
             const protobufs::InstructionDescriptor& instruction_descriptor)
          -> void {
        assert(
            instruction_iterator->opcode() ==
                spv::Op(instruction_descriptor.target_instruction_opcode()) &&
            "The opcode of the instruction we might insert before must be "
            "the same as the opcode in the descriptor for the instruction");

        // Randomly decide whether to try adding an OpCompositeInsert
        // instruction.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingCompositeInsert())) {
          return;
        }

        // It must be possible to insert an OpCompositeInsert instruction
        // before |instruction_iterator|.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                spv::Op::OpCompositeInsert, instruction_iterator)) {
          return;
        }

        // Look for available values that have composite type.
        std::vector<opt::Instruction*> available_composites =
            FindAvailableInstructions(
                function, block, instruction_iterator,
                [instruction_descriptor](
                    opt::IRContext* ir_context,
                    opt::Instruction* instruction) -> bool {
                  // |instruction| must be a supported instruction of composite
                  // type.
                  if (!TransformationCompositeInsert::
                          IsCompositeInstructionSupported(ir_context,
                                                          instruction)) {
                    return false;
                  }

                  auto instruction_type = ir_context->get_type_mgr()->GetType(
                      instruction->type_id());

                  // No components of the composite can have type
                  // OpTypeRuntimeArray.
                  if (ContainsRuntimeArray(*instruction_type)) {
                    return false;
                  }

                  // No components of the composite can be pointers.
                  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3658):
                  //       Structs can have components of pointer type.
                  //       FindOrCreateZeroConstant cannot be called on a
                  //       pointer. We ignore pointers for now. Consider adding
                  //       support for pointer types.
                  if (ContainsPointer(*instruction_type)) {
                    return false;
                  }

                  return true;
                });

        // If there are no available values, then return.
        if (available_composites.empty()) {
          return;
        }

        // Choose randomly one available composite value.
        auto available_composite =
            available_composites[GetFuzzerContext()->RandomIndex(
                available_composites)];

        // Take a random component of the chosen composite value. If the chosen
        // component is itself a composite, then randomly decide whether to take
        // its component and repeat.
        uint32_t current_node_type_id = available_composite->type_id();
        std::vector<uint32_t> path_to_replaced;
        while (true) {
          auto current_node_type_inst =
              GetIRContext()->get_def_use_mgr()->GetDef(current_node_type_id);
          uint32_t num_of_components = fuzzerutil::GetBoundForCompositeIndex(
              *current_node_type_inst, GetIRContext());

          // If the composite is empty, then end the iteration.
          if (num_of_components == 0) {
            break;
          }
          uint32_t one_selected_index =
              GetFuzzerContext()->GetRandomIndexForCompositeInsert(
                  num_of_components);

          // Construct a final index by appending the current index.
          path_to_replaced.push_back(one_selected_index);
          current_node_type_id = fuzzerutil::WalkOneCompositeTypeIndex(
              GetIRContext(), current_node_type_id, one_selected_index);

          // If the component is not a composite then end the iteration.
          if (!fuzzerutil::IsCompositeType(
                  GetIRContext()->get_type_mgr()->GetType(
                      current_node_type_id))) {
            break;
          }

          // If the component is a composite, but we decide not to go deeper,
          // then end the iteration.
          if (!GetFuzzerContext()->ChoosePercentage(
                  GetFuzzerContext()
                      ->GetChanceOfGoingDeeperToInsertInComposite())) {
            break;
          }
        }

        // Look for available objects that have the type id
        // |current_node_type_id| and can be inserted.
        std::vector<opt::Instruction*> available_objects =
            FindAvailableInstructions(
                function, block, instruction_iterator,
                [instruction_descriptor, current_node_type_id](
                    opt::IRContext* /*unused*/,
                    opt::Instruction* instruction) -> bool {
                  if (instruction->result_id() == 0 ||
                      instruction->type_id() == 0) {
                    return false;
                  }
                  if (instruction->type_id() != current_node_type_id) {
                    return false;
                  }
                  return true;
                });

        // If there are no objects of the specific type available, check if
        // FindOrCreateZeroConstant can be called and create a zero constant of
        // this type.
        uint32_t available_object_id;
        if (available_objects.empty()) {
          if (!fuzzerutil::CanCreateConstant(GetIRContext(),
                                             current_node_type_id)) {
            return;
          }
          available_object_id =
              FindOrCreateZeroConstant(current_node_type_id, false);
        } else {
          available_object_id =
              available_objects[GetFuzzerContext()->RandomIndex(
                                    available_objects)]
                  ->result_id();
        }
        auto new_result_id = GetFuzzerContext()->GetFreshId();

        // Insert an OpCompositeInsert instruction which copies
        // |available_composite| and in the copy inserts the object
        // of type |available_object_id| at index |index_to_replace|.
        ApplyTransformation(TransformationCompositeInsert(
            instruction_descriptor, new_result_id,
            available_composite->result_id(), available_object_id,
            path_to_replaced));
      });
}

bool FuzzerPassAddCompositeInserts::ContainsPointer(
    const opt::analysis::Type& type) {
  switch (type.kind()) {
    case opt::analysis::Type::kPointer:
      return true;
    case opt::analysis::Type::kArray:
      return ContainsPointer(*type.AsArray()->element_type());
    case opt::analysis::Type::kMatrix:
      return ContainsPointer(*type.AsMatrix()->element_type());
    case opt::analysis::Type::kVector:
      return ContainsPointer(*type.AsVector()->element_type());
    case opt::analysis::Type::kStruct:
      return std::any_of(type.AsStruct()->element_types().begin(),
                         type.AsStruct()->element_types().end(),
                         [](const opt::analysis::Type* element_type) {
                           return ContainsPointer(*element_type);
                         });
    default:
      return false;
  }
}

bool FuzzerPassAddCompositeInserts::ContainsRuntimeArray(
    const opt::analysis::Type& type) {
  switch (type.kind()) {
    case opt::analysis::Type::kRuntimeArray:
      return true;
    case opt::analysis::Type::kStruct:
      // If any component of a struct is of type OpTypeRuntimeArray, return
      // true.
      return std::any_of(type.AsStruct()->element_types().begin(),
                         type.AsStruct()->element_types().end(),
                         [](const opt::analysis::Type* element_type) {
                           return ContainsRuntimeArray(*element_type);
                         });
    default:
      return false;
  }
}

}  // namespace fuzz
}  // namespace spvtools
