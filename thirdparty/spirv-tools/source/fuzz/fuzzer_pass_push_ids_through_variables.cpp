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

#include "source/fuzz/fuzzer_pass_push_ids_through_variables.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_push_id_through_variable.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPushIdsThroughVariables::FuzzerPassPushIdsThroughVariables(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassPushIdsThroughVariables::Apply() {
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

        // Randomly decide whether to try pushing an id through a variable.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfPushingIdThroughVariable())) {
          return;
        }

        // The block containing the instruction we are going to insert before
        // must be reachable.
        if (!GetIRContext()->IsReachable(*block)) {
          return;
        }

        // It must be valid to insert OpStore and OpLoad instructions
        // before the instruction to insert before.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                spv::Op::OpStore, instruction_iterator) ||
            !fuzzerutil::CanInsertOpcodeBeforeInstruction(
                spv::Op::OpLoad, instruction_iterator)) {
          return;
        }

        // Randomly decides whether a global or local variable will be added.
        auto variable_storage_class = GetFuzzerContext()->ChooseEven()
                                          ? spv::StorageClass::Private
                                          : spv::StorageClass::Function;

        // Gets the available basic and pointer types.
        auto basic_type_ids_and_pointers =
            GetAvailableBasicTypesAndPointers(variable_storage_class);
        auto& basic_types = basic_type_ids_and_pointers.first;

        // There must be at least some basic types.
        if (basic_types.empty()) {
          return;
        }

        uint32_t basic_type_id =
            basic_types[GetFuzzerContext()->RandomIndex(basic_types)];

        // Looks for ids that we might wish to consider pushing through a
        // variable.
        std::vector<opt::Instruction*> value_instructions =
            FindAvailableInstructions(
                function, block, instruction_iterator,
                [this, basic_type_id, instruction_descriptor](
                    opt::IRContext* ir_context,
                    opt::Instruction* instruction) -> bool {
                  if (!instruction->result_id() || !instruction->type_id()) {
                    return false;
                  }

                  if (instruction->type_id() != basic_type_id) {
                    return false;
                  }

                  // If the id is irrelevant, we can use it since it will not
                  // participate in DataSynonym fact. Otherwise, we should be
                  // able to produce a synonym out of the id.
                  if (!GetTransformationContext()
                           ->GetFactManager()
                           ->IdIsIrrelevant(instruction->result_id()) &&
                      !fuzzerutil::CanMakeSynonymOf(ir_context,
                                                    *GetTransformationContext(),
                                                    *instruction)) {
                    return false;
                  }

                  return fuzzerutil::IdIsAvailableBeforeInstruction(
                      ir_context,
                      FindInstruction(instruction_descriptor, ir_context),
                      instruction->result_id());
                });

        if (value_instructions.empty()) {
          return;
        }

        // If the pointer type does not exist, then create it.
        FindOrCreatePointerType(basic_type_id, variable_storage_class);

        // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3403):
        //  type support here is limited by the FindOrCreateZeroConstant
        //  function.
        const auto* type_inst =
            GetIRContext()->get_def_use_mgr()->GetDef(basic_type_id);
        assert(type_inst);
        switch (type_inst->opcode()) {
          case spv::Op::OpTypeBool:
          case spv::Op::OpTypeFloat:
          case spv::Op::OpTypeInt:
          case spv::Op::OpTypeArray:
          case spv::Op::OpTypeMatrix:
          case spv::Op::OpTypeVector:
          case spv::Op::OpTypeStruct:
            break;
          default:
            return;
        }

        // Create a constant to initialize the variable from. This might update
        // module's id bound so it must be done before any fresh ids are
        // computed.
        auto initializer_id = FindOrCreateZeroConstant(basic_type_id, false);

        // Applies the push id through variable transformation.
        ApplyTransformation(TransformationPushIdThroughVariable(
            value_instructions[GetFuzzerContext()->RandomIndex(
                                   value_instructions)]
                ->result_id(),
            GetFuzzerContext()->GetFreshId(), GetFuzzerContext()->GetFreshId(),
            uint32_t(variable_storage_class), initializer_id,
            instruction_descriptor));
      });
}

}  // namespace fuzz
}  // namespace spvtools
