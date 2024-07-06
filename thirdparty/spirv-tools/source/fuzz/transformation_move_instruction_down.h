// Copyright (c) 2020 Vasyl Teliman
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_MOVE_INSTRUCTION_DOWN_H_
#define SOURCE_FUZZ_TRANSFORMATION_MOVE_INSTRUCTION_DOWN_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationMoveInstructionDown : public Transformation {
 public:
  explicit TransformationMoveInstructionDown(
      protobufs::TransformationMoveInstructionDown message);

  explicit TransformationMoveInstructionDown(
      const protobufs::InstructionDescriptor& instruction);

  // - |instruction| should be a descriptor of a valid instruction in the module
  // - |instruction|'s opcode should be supported by this transformation
  // - neither |instruction| nor its successor may be the last instruction in
  //   the block
  // - |instruction|'s successor may not be dependent on the |instruction|
  // - it should be possible to insert |instruction|'s opcode after its
  //   successor
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Swaps |instruction| with its successor.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  // Returns true if the |inst| is supported by this transformation.
  static bool IsInstructionSupported(opt::IRContext* ir_context,
                                     const opt::Instruction& inst);

  // Returns true if |inst| represents a "simple" instruction. That is, it
  // neither reads from nor writes to the memory and is not a barrier.
  static bool IsSimpleInstruction(opt::IRContext* ir_context,
                                  const opt::Instruction& inst);

  // Returns true if |inst| reads from memory.
  static bool IsMemoryReadInstruction(opt::IRContext* ir_context,
                                      const opt::Instruction& inst);

  // Returns id being used by |inst| to read from. |inst| must be a memory read
  // instruction (see IsMemoryReadInstruction). Returned id is not guaranteed to
  // have pointer type.
  static uint32_t GetMemoryReadTarget(opt::IRContext* ir_context,
                                      const opt::Instruction& inst);

  // Returns true if |inst| that writes to the memory.
  static bool IsMemoryWriteInstruction(opt::IRContext* ir_context,
                                       const opt::Instruction& inst);

  // Returns id being used by |inst| to write into. |inst| must be a memory
  // write instruction (see IsMemoryWriteInstruction). Returned id is not
  // guaranteed to have pointer type.
  static uint32_t GetMemoryWriteTarget(opt::IRContext* ir_context,
                                       const opt::Instruction& inst);

  // Returns true if |inst| either reads from or writes to the memory
  // (see IsMemoryReadInstruction and IsMemoryWriteInstruction accordingly).
  static bool IsMemoryInstruction(opt::IRContext* ir_context,
                                  const opt::Instruction& inst);

  // Returns true if |inst| is a barrier instruction.
  static bool IsBarrierInstruction(const opt::Instruction& inst);

  // Returns true if it is possible to swap |a| and |b| without changing the
  // module's semantics. |a| and |b| are required to be supported instructions
  // (see IsInstructionSupported). In particular, if either |a| or |b| are
  // memory or barrier instructions, some checks are used to only say that they
  // can be swapped if the swap is definitely semantics-preserving.
  static bool CanSafelySwapInstructions(opt::IRContext* ir_context,
                                        const opt::Instruction& a,
                                        const opt::Instruction& b,
                                        const FactManager& fact_manager);

  protobufs::TransformationMoveInstructionDown message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_MOVE_INSTRUCTION_DOWN_H_
