// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SET_MEMORY_OPERANDS_MASK_H_
#define SOURCE_FUZZ_TRANSFORMATION_SET_MEMORY_OPERANDS_MASK_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationSetMemoryOperandsMask : public Transformation {
 public:
  explicit TransformationSetMemoryOperandsMask(
      protobufs::TransformationSetMemoryOperandsMask message);

  TransformationSetMemoryOperandsMask(
      const protobufs::InstructionDescriptor& memory_access_instruction,
      uint32_t memory_operands_mask, uint32_t memory_operands_mask_index);

  // - |message_.memory_access_instruction| must describe a memory access
  //   instruction.
  // - |message_.memory_operands_mask_index| must be suitable for this memory
  //   access instruction, e.g. it must be 0 in the case of OpLoad, and may be
  //   1 in the case of OpCopyMemory if the SPIR-V version is 1.4 or higher.
  // - |message_.memory_operands_mask| must be identical to the original memory
  //   operands mask, except that Volatile may be added, and Nontemporal may be
  //   toggled.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Replaces the operands mask identified by
  // |message_.memory_operands_mask_index| in the instruction described by
  // |message_.memory_access_instruction| with |message_.memory_operands_mask|,
  // creating an input operand for the mask if no such operand was present.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Helper function that determines whether |instruction| is a memory
  // instruction (e.g. OpLoad).
  static bool IsMemoryAccess(const opt::Instruction& instruction);

  // Does the version of SPIR-V being used support multiple memory operand
  // masks on relevant memory access instructions?
  static bool MultipleMemoryOperandMasksAreSupported(
      opt::IRContext* ir_context);

  // Helper function to get the input operand index associated with mask number
  // |mask_index|. This is a bit tricky if there are multiple masks, because the
  // index associated with the second mask depends on whether the first mask
  // includes any flags such as Aligned that have corresponding operands.
  static uint32_t GetInOperandIndexForMask(const opt::Instruction& instruction,
                                           uint32_t mask_index);

 private:
  protobufs::TransformationSetMemoryOperandsMask message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_SET_MEMORY_OPERANDS_MASK_H_
