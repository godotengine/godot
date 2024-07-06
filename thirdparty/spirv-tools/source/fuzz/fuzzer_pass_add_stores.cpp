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

#include "source/fuzz/fuzzer_pass_add_stores.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_store.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddStores::FuzzerPassAddStores(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddStores::Apply() {
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

        // Randomly decide whether to try inserting a store here.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingStore())) {
          return;
        }

        // Check whether it is legitimate to insert a store before this
        // instruction.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpStore,
                                                          inst_it)) {
          return;
        }
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                spv::Op::OpAtomicStore, inst_it)) {
          return;
        }

        // Look for pointers we might consider storing to.
        std::vector<opt::Instruction*> relevant_pointers =
            FindAvailableInstructions(
                function, block, inst_it,
                [this, block](opt::IRContext* context,
                              opt::Instruction* instruction) -> bool {
                  if (!instruction->result_id() || !instruction->type_id()) {
                    return false;
                  }
                  auto type_inst = context->get_def_use_mgr()->GetDef(
                      instruction->type_id());
                  if (type_inst->opcode() != spv::Op::OpTypePointer) {
                    // Not a pointer.
                    return false;
                  }
                  if (instruction->IsReadOnlyPointer()) {
                    // Read only: cannot store to it.
                    return false;
                  }
                  switch (instruction->opcode()) {
                    case spv::Op::OpConstantNull:
                    case spv::Op::OpUndef:
                      // Do not allow storing to a null or undefined pointer;
                      // this might be OK if the block is dead, but for now we
                      // conservatively avoid it.
                      return false;
                    default:
                      break;
                  }
                  return GetTransformationContext()
                             ->GetFactManager()
                             ->BlockIsDead(block->id()) ||
                         GetTransformationContext()
                             ->GetFactManager()
                             ->PointeeValueIsIrrelevant(
                                 instruction->result_id());
                });

        // At this point, |relevant_pointers| contains all the pointers we might
        // think of storing to.
        if (relevant_pointers.empty()) {
          return;
        }

        auto pointer = relevant_pointers[GetFuzzerContext()->RandomIndex(
            relevant_pointers)];

        std::vector<opt::Instruction*> relevant_values =
            FindAvailableInstructions(
                function, block, inst_it,
                [pointer](opt::IRContext* context,
                          opt::Instruction* instruction) -> bool {
                  if (!instruction->result_id() || !instruction->type_id()) {
                    return false;
                  }
                  return instruction->type_id() ==
                         context->get_def_use_mgr()
                             ->GetDef(pointer->type_id())
                             ->GetSingleWordInOperand(1);
                });

        if (relevant_values.empty()) {
          return;
        }

        bool is_atomic_store = false;
        uint32_t memory_scope_id = 0;
        uint32_t memory_semantics_id = 0;

        auto storage_class =
            static_cast<spv::StorageClass>(GetIRContext()
                                               ->get_def_use_mgr()
                                               ->GetDef(pointer->type_id())
                                               ->GetSingleWordInOperand(0));

        switch (storage_class) {
          case spv::StorageClass::StorageBuffer:
          case spv::StorageClass::PhysicalStorageBuffer:
          case spv::StorageClass::Workgroup:
          case spv::StorageClass::CrossWorkgroup:
          case spv::StorageClass::AtomicCounter:
          case spv::StorageClass::Image:
            if (GetFuzzerContext()->ChoosePercentage(
                    GetFuzzerContext()->GetChanceOfAddingAtomicStore())) {
              is_atomic_store = true;

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
        ApplyTransformation(TransformationStore(
            pointer->result_id(), is_atomic_store, memory_scope_id,
            memory_semantics_id,
            relevant_values[GetFuzzerContext()->RandomIndex(relevant_values)]
                ->result_id(),
            instruction_descriptor));
      });
}

}  // namespace fuzz
}  // namespace spvtools
