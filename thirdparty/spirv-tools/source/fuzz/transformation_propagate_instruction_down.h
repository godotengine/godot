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

#ifndef SOURCE_FUZZ_TRANSFORMATION_PROPAGATE_INSTRUCTION_DOWN_H_
#define SOURCE_FUZZ_TRANSFORMATION_PROPAGATE_INSTRUCTION_DOWN_H_

#include <map>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationPropagateInstructionDown : public Transformation {
 public:
  explicit TransformationPropagateInstructionDown(
      protobufs::TransformationPropagateInstructionDown message);

  TransformationPropagateInstructionDown(
      uint32_t block_id, uint32_t phi_fresh_id,
      const std::map<uint32_t, uint32_t>& successor_id_to_fresh_id);

  // - It should be possible to apply this transformation to |block_id| (see
  //   IsApplicableToBlock method).
  // - Every acceptable successor of |block_id| (see GetAcceptableSuccessors
  //   method) must have an entry in the |successor_id_to_fresh_id| map unless
  //   overflow ids are available.
  // - All values in |successor_id_to_fresh_id| and |phi_fresh_id| must be
  //   unique and fresh.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - Adds a clone of the propagated instruction into every acceptable
  //   successor of |block_id|.
  // - Removes the original instruction.
  // - Creates an OpPhi instruction if possible, that tries to group created
  //   clones.
  // - If the original instruction's id was irrelevant - marks created
  //   instructions as irrelevant. Otherwise, marks the created instructions as
  //   synonymous to each other if possible (i.e. skips instructions, copied
  //   into dead blocks).
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if this transformation can be applied to the block with id
  // |block_id|. Concretely, returns true iff:
  // - |block_id| is a result id of some reachable basic block in the module.
  // - the block has an instruction to propagate (see
  //   GetInstructionToPropagate method).
  // - the block has at least one acceptable successor (see
  //   GetAcceptableSuccessors method).
  // - none of the acceptable successors have OpPhi instructions that use the
  //   original instruction.
  // - it is possible to replace every use of the original instruction with some
  //   of the propagated instructions (or an OpPhi if we can create it - see
  //   GetOpPhiBlockId method).
  static bool IsApplicableToBlock(opt::IRContext* ir_context,
                                  uint32_t block_id);

  // Returns ids of successors of |block_id|, that can be used to propagate an
  // instruction into. Concretely, a successor block is acceptable if all
  // dependencies of the propagated instruction dominate it. Note that this
  // implies that an acceptable successor must be reachable in the CFG.
  // For example:
  //    %1 = OpLabel
  //         OpSelectionMerge %2 None
  //         OpBranchConditional %cond %2 %3
  //    %3 = OpLabel
  //    %4 = OpUndef %int
  //    %5 = OpCopyObject %int %4
  //         OpBranch %2
  //    %2 = OpLabel
  //    ...
  // In this example, %2 is not an acceptable successor of %3 since one of the
  // dependencies (%4) of the propagated instruction (%5) does not dominate it.
  static std::unordered_set<uint32_t> GetAcceptableSuccessors(
      opt::IRContext* ir_context, uint32_t block_id);

  std::unordered_set<uint32_t> GetFreshIds() const override;

 private:
  // Returns the last possible instruction in the |block_id| that satisfies the
  // following properties:
  // - has result id
  // - has type id
  // - has supported opcode (see IsOpcodeSupported method)
  // - has no users in its basic block.
  // Returns nullptr if no such an instruction exists. For example:
  //    %1 = OpLabel
  //    %2 = OpUndef %int
  //    %3 = OpUndef %int
  //         OpStore %var %3
  //         OpBranch %some_block
  // In this example:
  // - We cannot propagate OpBranch nor OpStore since they both have unsupported
  //   opcodes and have neither result ids nor type ids.
  // - We cannot propagate %3 either since it is used by OpStore.
  // - We can propagate %2 since it satisfies all our conditions.
  // The basic idea behind this method it to make sure that the returned
  // instruction will not break domination rules in its original block when
  // propagated.
  static opt::Instruction* GetInstructionToPropagate(opt::IRContext* ir_context,
                                                     uint32_t block_id);

  // Returns true if |opcode| is supported by this transformation.
  static bool IsOpcodeSupported(spv::Op opcode);

  // Returns the first instruction in the |block| that allows us to insert
  // |opcode| above itself. Returns nullptr is no such instruction exists.
  static opt::Instruction* GetFirstInsertBeforeInstruction(
      opt::IRContext* ir_context, uint32_t block_id, spv::Op opcode);

  // Returns a result id of a basic block, where an OpPhi instruction can be
  // inserted. Returns nullptr if it's not possible to create an OpPhi. The
  // created OpPhi instruction groups all the propagated clones of the original
  // instruction. |block_id| is a result id of the block we propagate the
  // instruction from. |successor_ids| contains result ids of the successors we
  // propagate the instruction into. Concretely, returns a non-null value if:
  // - |block_id| is in some construct.
  // - The merge block of that construct is reachable.
  // - |block_id| dominates that merge block.
  // - That merge block may not be an acceptable successor of |block_id|.
  // - There must be at least one |block_id|'s acceptable successor for every
  //   predecessor of the merge block, dominating that predecessor.
  // - We can't create an OpPhi if the module has neither VariablePointers nor
  //   VariablePointersStorageBuffer capabilities.
  // A simple example of when we can insert an OpPhi instruction is:
  // - This snippet of code:
  //    %1 = OpLabel
  //    %2 = OpUndef %int
  //         OpSelectionMerge %5 None
  //         OpBranchConditional %cond %3 %4
  //    %3 = OpLabel
  //         OpBranch %5
  //    %4 = OpLabel
  //         OpBranch %5
  //    %5 = OpLabel
  //         ...
  //   will be transformed into the following one (if %2 is propagated):
  //    %1 = OpLabel
  //         OpSelectionMerge %5 None
  //         OpBranchConditional %cond %3 %4
  //    %3 = OpLabel
  //    %6 = OpUndef %int
  //         OpBranch %5
  //    %4 = OpLabel
  //    %7 = OpUndef %int
  //         OpBranch %5
  //    %5 = OpLabel
  //    %8 = OpPhi %int %6 %3 %7 %4
  //         ...
  // The fact that we introduce an OpPhi allows us to increase the applicability
  // of the transformation. Concretely, we wouldn't be able to apply it in the
  // example above if %2 were used in %5. Some more complicated examples can be
  // found in unit tests.
  static uint32_t GetOpPhiBlockId(
      opt::IRContext* ir_context, uint32_t block_id,
      const opt::Instruction& inst_to_propagate,
      const std::unordered_set<uint32_t>& successor_ids);

  protobufs::TransformationPropagateInstructionDown message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_PROPAGATE_INSTRUCTION_DOWN_H_
