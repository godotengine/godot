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

#include "source/fuzz/fuzzer_pass_add_loads.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_load.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddLoads::FuzzerPassAddLoads(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddLoads::Apply() {
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

        // Randomly decide whether to try inserting a load here.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingLoad())) {
          return;
        }

        // Check whether it is legitimate to insert a load or atomic load before
        // this instruction.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpLoad,
                                                          inst_it)) {
          return;
        }
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpAtomicLoad,
                                                          inst_it)) {
          return;
        }

        std::vector<opt::Instruction*> relevant_instructions =
            FindAvailableInstructions(
                function, block, inst_it,
                [](opt::IRContext* context,
                   opt::Instruction* instruction) -> bool {
                  if (!instruction->result_id() || !instruction->type_id()) {
                    return false;
                  }
                  switch (instruction->opcode()) {
                    case spv::Op::OpConstantNull:
                    case spv::Op::OpUndef:
                      // Do not allow loading from a null or undefined pointer;
                      // this might be OK if the block is dead, but for now we
                      // conservatively avoid it.
                      return false;
                    default:
                      break;
                  }
                  return context->get_def_use_mgr()
                             ->GetDef(instruction->type_id())
                             ->opcode() == spv::Op::OpTypePointer;
                });

        // At this point, |relevant_instructions| contains all the pointers
        // we might think of loading from.
        if (relevant_instructions.empty()) {
          return;
        }

        auto chosen_instruction =
            relevant_instructions[GetFuzzerContext()->RandomIndex(
                relevant_instructions)];

        bool is_atomic_load = false;
        uint32_t memory_scope_id = 0;
        uint32_t memory_semantics_id = 0;

        auto storage_class = static_cast<spv::StorageClass>(
            GetIRContext()
                ->get_def_use_mgr()
                ->GetDef(chosen_instruction->type_id())
                ->GetSingleWordInOperand(0));

        switch (storage_class) {
          case spv::StorageClass::StorageBuffer:
          case spv::StorageClass::PhysicalStorageBuffer:
          case spv::StorageClass::Workgroup:
          case spv::StorageClass::CrossWorkgroup:
          case spv::StorageClass::AtomicCounter:
          case spv::StorageClass::Image:
            if (GetFuzzerContext()->ChoosePercentage(
                    GetFuzzerContext()->GetChanceOfAddingAtomicLoad())) {
              is_atomic_load = true;

              memory_scope_id = FindOrCreateConstant(
                  {uint32_t(spv::Scope::Invocation)},
                  FindOrCreateIntegerType(32, GetFuzzerContext()->ChooseEven()),
                  false);

              memory_semantics_id = FindOrCreateConstant(
                  {static_cast<uint32_t>(
                      fuzzerutil::GetMemorySemanticsForStorageClass(
                          storage_class))},
                  FindOrCreateIntegerType(32, GetFuzzerContext()->ChooseEven()),
                  false);
            }
            break;

          default:
            break;
        }

        // Create and apply the transformation.
        ApplyTransformation(TransformationLoad(
            GetFuzzerContext()->GetFreshId(), chosen_instruction->result_id(),
            is_atomic_load, memory_scope_id, memory_semantics_id,
            instruction_descriptor));
      });
}

}  // namespace fuzz
}  // namespace spvtools
