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

#include "source/fuzz/fuzzer_pass_replace_copy_objects_with_stores_loads.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_replace_copy_object_with_store_load.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceCopyObjectsWithStoresLoads::
    FuzzerPassReplaceCopyObjectsWithStoresLoads(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations,
        bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassReplaceCopyObjectsWithStoresLoads::Apply() {
  GetIRContext()->module()->ForEachInst([this](opt::Instruction* instruction) {
    // Randomly decide whether to replace OpCopyObject.
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()
                ->GetChanceOfReplacingCopyObjectWithStoreLoad())) {
      return;
    }
    // The instruction must be OpCopyObject.
    if (instruction->opcode() != spv::Op::OpCopyObject) {
      return;
    }
    // The opcode of the type_id instruction cannot be a OpTypePointer,
    // because we cannot define a pointer to pointer.
    if (GetIRContext()
            ->get_def_use_mgr()
            ->GetDef(instruction->type_id())
            ->opcode() == spv::Op::OpTypePointer) {
      return;
    }
    // It must be valid to insert OpStore and OpLoad instructions
    // before the instruction OpCopyObject.
    if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpStore,
                                                      instruction) ||
        !fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpLoad,
                                                      instruction)) {
      return;
    }

    // Randomly decides whether a global or local variable will be added.
    auto variable_storage_class = GetFuzzerContext()->ChooseEven()
                                      ? spv::StorageClass::Private
                                      : spv::StorageClass::Function;

    // Find or create a constant to initialize the variable from. The type of
    // |instruction| must be such that the function FindOrCreateConstant can be
    // called.
    if (!fuzzerutil::CanCreateConstant(GetIRContext(),
                                       instruction->type_id())) {
      return;
    }
    auto variable_initializer_id =
        FindOrCreateZeroConstant(instruction->type_id(), false);

    // Make sure that pointer type is defined.
    FindOrCreatePointerType(instruction->type_id(), variable_storage_class);
    // Apply the transformation replacing OpCopyObject with Store and Load.
    ApplyTransformation(TransformationReplaceCopyObjectWithStoreLoad(
        instruction->result_id(), GetFuzzerContext()->GetFreshId(),
        uint32_t(variable_storage_class), variable_initializer_id));
  });
}

}  // namespace fuzz
}  // namespace spvtools
