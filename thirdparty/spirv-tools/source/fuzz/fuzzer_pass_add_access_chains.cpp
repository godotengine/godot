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

#include "source/fuzz/fuzzer_pass_add_access_chains.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_access_chain.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddAccessChains::FuzzerPassAddAccessChains(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddAccessChains::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor)
          -> void {
        assert(
            inst_it->opcode() ==
                spv::Op(instruction_descriptor.target_instruction_opcode()) &&
            "The opcode of the instruction we might insert before must be "
            "the same as the opcode in the descriptor for the instruction");

        // Check whether it is legitimate to insert an access chain
        // instruction before this instruction.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                spv::Op::OpAccessChain, inst_it)) {
          return;
        }

        // Randomly decide whether to try inserting a load here.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingAccessChain())) {
          return;
        }

        // Get all of the pointers that are currently in scope, excluding
        // explicitly null and undefined pointers.
        std::vector<opt::Instruction*> relevant_pointer_instructions =
            FindAvailableInstructions(
                function, block, inst_it,
                [](opt::IRContext* context,
                   opt::Instruction* instruction) -> bool {
                  if (!instruction->result_id() || !instruction->type_id()) {
                    // A pointer needs both a result and type id.
                    return false;
                  }
                  switch (instruction->opcode()) {
                    case spv::Op::OpConstantNull:
                    case spv::Op::OpUndef:
                      // Do not allow making an access chain from a null or
                      // undefined pointer.  (We can eliminate these cases
                      // before actually checking that the instruction is a
                      // pointer.)
                      return false;
                    default:
                      break;
                  }
                  // If the instruction has pointer type, we can legitimately
                  // make an access chain from it.
                  return context->get_def_use_mgr()
                             ->GetDef(instruction->type_id())
                             ->opcode() == spv::Op::OpTypePointer;
                });

        // At this point, |relevant_instructions| contains all the pointers
        // we might think of making an access chain from.
        if (relevant_pointer_instructions.empty()) {
          return;
        }

        auto chosen_pointer =
            relevant_pointer_instructions[GetFuzzerContext()->RandomIndex(
                relevant_pointer_instructions)];
        std::vector<uint32_t> index_ids;

        // Each index accessing a non-struct composite will be clamped, thus
        // needing a pair of fresh ids
        std::vector<std::pair<uint32_t, uint32_t>> fresh_ids_for_clamping;

        auto pointer_type = GetIRContext()->get_def_use_mgr()->GetDef(
            chosen_pointer->type_id());
        uint32_t subobject_type_id = pointer_type->GetSingleWordInOperand(1);
        while (true) {
          auto subobject_type =
              GetIRContext()->get_def_use_mgr()->GetDef(subobject_type_id);
          if (!spvOpcodeIsComposite(subobject_type->opcode())) {
            break;
          }
          if (!GetFuzzerContext()->ChoosePercentage(
                  GetFuzzerContext()
                      ->GetChanceOfGoingDeeperWhenMakingAccessChain())) {
            break;
          }
          uint32_t bound;
          switch (subobject_type->opcode()) {
            case spv::Op::OpTypeArray:
              bound = fuzzerutil::GetArraySize(*subobject_type, GetIRContext());
              break;
            case spv::Op::OpTypeMatrix:
            case spv::Op::OpTypeVector:
              bound = subobject_type->GetSingleWordInOperand(1);
              break;
            case spv::Op::OpTypeStruct:
              bound = fuzzerutil::GetNumberOfStructMembers(*subobject_type);
              break;
            default:
              assert(false && "Not a composite type opcode.");
              // Set the bound to a value in order to keep release compilers
              // happy.
              bound = 0;
              break;
          }
          if (bound == 0) {
            // It is possible for a composite type to legitimately have zero
            // sub-components, at least in the case of a struct, which
            // can have no fields.
            break;
          }

          uint32_t index_value =
              GetFuzzerContext()->GetRandomIndexForAccessChain(bound);

          switch (subobject_type->opcode()) {
            case spv::Op::OpTypeArray:
            case spv::Op::OpTypeMatrix:
            case spv::Op::OpTypeVector: {
              // The index will be clamped

              bool is_signed = GetFuzzerContext()->ChooseEven();

              // Make the constant ready for clamping. We need:
              // - an OpTypeBool to be present in the module
              // - an OpConstant with the same type as the index and value
              //   the maximum value for an index
              // - a new pair of fresh ids for the clamping instructions
              FindOrCreateBoolType();
              FindOrCreateIntegerConstant({bound - 1}, 32, is_signed, false);
              std::pair<uint32_t, uint32_t> fresh_pair_of_ids = {
                  GetFuzzerContext()->GetFreshId(),
                  GetFuzzerContext()->GetFreshId()};
              fresh_ids_for_clamping.emplace_back(fresh_pair_of_ids);

              index_ids.push_back(FindOrCreateIntegerConstant(
                  {index_value}, 32, is_signed, false));
              subobject_type_id = subobject_type->GetSingleWordInOperand(0);

            } break;
            case spv::Op::OpTypeStruct:
              index_ids.push_back(FindOrCreateIntegerConstant(
                  {index_value}, 32, GetFuzzerContext()->ChooseEven(), false));
              subobject_type_id =
                  subobject_type->GetSingleWordInOperand(index_value);
              break;
            default:
              assert(false && "Not a composite type opcode.");
          }
        }
        // The transformation we are about to create will only apply if a
        // pointer suitable for the access chain's result type exists, so we
        // create one if it does not.
        FindOrCreatePointerType(subobject_type_id,
                                static_cast<spv::StorageClass>(
                                    pointer_type->GetSingleWordInOperand(0)));
        // Apply the transformation to add an access chain.
        ApplyTransformation(TransformationAccessChain(
            GetFuzzerContext()->GetFreshId(), chosen_pointer->result_id(),
            index_ids, instruction_descriptor, fresh_ids_for_clamping));
      });
}

}  // namespace fuzz
}  // namespace spvtools
