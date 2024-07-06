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

#include "source/fuzz/fuzzer_pass_replace_copy_memories_with_loads_stores.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_replace_copy_memory_with_load_store.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceCopyMemoriesWithLoadsStores::
    FuzzerPassReplaceCopyMemoriesWithLoadsStores(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations,
        bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassReplaceCopyMemoriesWithLoadsStores::Apply() {
  GetIRContext()->module()->ForEachInst([this](opt::Instruction* instruction) {
    // Randomly decide whether to replace the OpCopyMemory.
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()
                ->GetChanceOfReplacingCopyMemoryWithLoadStore())) {
      return;
    }

    // The instruction must be OpCopyMemory.
    if (instruction->opcode() != spv::Op::OpCopyMemory) {
      return;
    }

    // Apply the transformation replacing OpCopyMemory with OpLoad and OpStore.
    ApplyTransformation(TransformationReplaceCopyMemoryWithLoadStore(
        GetFuzzerContext()->GetFreshId(),
        MakeInstructionDescriptor(GetIRContext(), instruction)));
  });
}

}  // namespace fuzz
}  // namespace spvtools
