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

#include "source/fuzz/fuzzer_pass_replace_loads_stores_with_copy_memories.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_replace_load_store_with_copy_memory.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceLoadsStoresWithCopyMemories::
    FuzzerPassReplaceLoadsStoresWithCopyMemories(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations,
        bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassReplaceLoadsStoresWithCopyMemories::Apply() {
  // We look for matching pairs of instructions OpLoad and
  // OpStore within the same block. Potential instructions OpLoad to be matched
  // are stored in a hash map. If we encounter instructions that write to memory
  // or instructions of memory barriers that could operate on variables within
  // unsafe storage classes we need to erase the hash map to avoid unsafe
  // operations.

  // A vector of matching OpLoad and OpStore instructions.
  std::vector<std::pair<opt::Instruction*, opt::Instruction*>>
      op_load_store_pairs;

  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // A hash map storing potential OpLoad instructions.
      std::unordered_map<uint32_t, opt::Instruction*> current_op_loads;
      for (auto& instruction : block) {
        // Add a potential OpLoad instruction.
        if (instruction.opcode() == spv::Op::OpLoad) {
          current_op_loads[instruction.result_id()] = &instruction;
        } else if (instruction.opcode() == spv::Op::OpStore) {
          if (current_op_loads.find(instruction.GetSingleWordOperand(1)) !=
              current_op_loads.end()) {
            // We have found the matching OpLoad instruction to the current
            // OpStore instruction.
            op_load_store_pairs.push_back(std::make_pair(
                current_op_loads[instruction.GetSingleWordOperand(1)],
                &instruction));
          }
        }
        if (TransformationReplaceLoadStoreWithCopyMemory::IsMemoryWritingOpCode(
                instruction.opcode())) {
          current_op_loads.clear();
        } else if (TransformationReplaceLoadStoreWithCopyMemory::
                       IsMemoryBarrierOpCode(instruction.opcode())) {
          for (auto it = current_op_loads.begin();
               it != current_op_loads.end();) {
            // Get the storage class.
            opt::Instruction* source_id =
                GetIRContext()->get_def_use_mgr()->GetDef(
                    it->second->GetSingleWordOperand(2));
            spv::StorageClass storage_class =
                fuzzerutil::GetStorageClassFromPointerType(
                    GetIRContext(), source_id->type_id());
            if (!TransformationReplaceLoadStoreWithCopyMemory::
                    IsStorageClassSafeAcrossMemoryBarriers(storage_class)) {
              it = current_op_loads.erase(it);
            } else {
              it++;
            }
          }
        }
      }
    }
  }
  for (auto instr_pair : op_load_store_pairs) {
    // Randomly decide to apply the transformation for the
    // potential pairs.
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()
                ->GetChanceOfReplacingLoadStoreWithCopyMemory())) {
      ApplyTransformation(TransformationReplaceLoadStoreWithCopyMemory(
          MakeInstructionDescriptor(GetIRContext(), instr_pair.first),
          MakeInstructionDescriptor(GetIRContext(), instr_pair.second)));
    }
  }
}  // namespace fuzz
}  // namespace fuzz
}  // namespace spvtools
