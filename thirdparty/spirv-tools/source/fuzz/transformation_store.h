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

#ifndef SOURCE_FUZZ_TRANSFORMATION_STORE_H_
#define SOURCE_FUZZ_TRANSFORMATION_STORE_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationStore : public Transformation {
 public:
  explicit TransformationStore(protobufs::TransformationStore message);

  TransformationStore(
      uint32_t pointer_id, bool is_atomic, uint32_t memory_scope,
      uint32_t memory_semantics, uint32_t value_id,
      const protobufs::InstructionDescriptor& instruction_to_insert_before);

  // - |message_.pointer_id| must be the id of a pointer
  // - The pointer type must not have read-only storage class
  // - The pointer must not be OpConstantNull or OpUndef
  // - |message_.is_atomic| must be true if want to work with OpAtomicStore.
  // - If |is_atomic| is true then |message_memory_scope_id| must be the id of
  //   an OpConstant 32 bit integer instruction with the value
  //   spv::Scope::Invocation.
  // - If |is_atomic| is true then |message_.memory_semantics_id| must be the id
  //   of an OpConstant 32 bit integer instruction with the values
  //   SpvMemorySemanticsWorkgroupMemoryMask or
  //   SpvMemorySemanticsUniformMemoryMask.
  // - |message_.value_id| must be an instruction result id that has the same
  //   type as the pointee type of |message_.pointer_id|
  // - |message_.instruction_to_insert_before| must identify an instruction
  //   before which it is valid to insert an OpStore, and where both
  //   |message_.pointer_id| and |message_.value_id| are available (according
  //   to dominance rules)
  // - Either the insertion point must be in a dead block, or it must be known
  //   that the pointee value of |message_.pointer_id| is irrelevant
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an instruction of the form:
  //   OpStore |pointer_id| |value_id|
  // before the instruction identified by
  // |message_.instruction_to_insert_before|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationStore message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_STORE_H_
