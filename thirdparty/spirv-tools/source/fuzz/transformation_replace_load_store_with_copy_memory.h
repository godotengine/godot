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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_LOAD_STORE_WITH_COPY_MEMORY_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_LOAD_STORE_WITH_COPY_MEMORY_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceLoadStoreWithCopyMemory : public Transformation {
 public:
  explicit TransformationReplaceLoadStoreWithCopyMemory(
      protobufs::TransformationReplaceLoadStoreWithCopyMemory message);

  TransformationReplaceLoadStoreWithCopyMemory(
      const protobufs::InstructionDescriptor& load_instruction_descriptor,
      const protobufs::InstructionDescriptor& store_instruction_descriptor);

  // - |message_.load_instruction_descriptor| must identify an OpLoad
  //   instruction.
  // - |message_.store_instruction_descriptor| must identify an OpStore
  //   instruction.
  // - The OpStore must write the intermediate value loaded by the OpLoad.
  // - The OpLoad and the OpStore must not have certain instruction in between
  //   (checked by IsMemoryWritingOpCode(), IsMemoryBarrierOpCode(),
  //   IsStorageClassSafeAcrossMemoryBarriers()).
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Takes a pair of instruction descriptors to OpLoad and OpStore that have the
  // same intermediate value and replaces the OpStore with an equivalent
  // OpCopyMemory.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  // Checks if the instruction that has an |op_code| might write to
  // the source operand of the OpLoad instruction.
  static bool IsMemoryWritingOpCode(spv::Op op_code);

  // Checks if the instruction that has an |op_code| is a memory barrier that
  // could interfere with the source operand of the OpLoad instruction
  static bool IsMemoryBarrierOpCode(spv::Op op_code);

  // Checks if the |storage_class| of the source operand of the OpLoad
  // instruction implies that this variable cannot change (due to other threads)
  // across memory barriers.
  static bool IsStorageClassSafeAcrossMemoryBarriers(
      spv::StorageClass storage_class);

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationReplaceLoadStoreWithCopyMemory message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_LOAD_STORE_WITH_COPY_MEMORY_H_
