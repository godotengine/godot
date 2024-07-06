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

#ifndef SOURCE_FUZZ_TRANSFORMATION_PROPAGATE_INSTRUCTION_UP_H_
#define SOURCE_FUZZ_TRANSFORMATION_PROPAGATE_INSTRUCTION_UP_H_

#include <map>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationPropagateInstructionUp : public Transformation {
 public:
  explicit TransformationPropagateInstructionUp(
      protobufs::TransformationPropagateInstructionUp message);

  TransformationPropagateInstructionUp(
      uint32_t block_id,
      const std::map<uint32_t, uint32_t>& predecessor_id_to_fresh_id);

  // - |block_id| must be a valid result id of some OpLabel instruction.
  // - |block_id| must have at least one predecessor
  // - |block_id| must contain an instruction that can be propagated using this
  //   transformation
  // - the instruction can be propagated if:
  //   - it's not an OpPhi
  //   - it is supported by this transformation
  //   - it depends only on instructions from different basic blocks or on
  //     OpPhi instructions from the same basic block
  // - it should be possible to insert the propagated instruction at the end of
  //   each |block_id|'s predecessor
  // - |predecessor_id_to_fresh_id| must have an entry for at least every
  //   predecessor of |block_id|
  // - each value in the |predecessor_id_to_fresh_id| map must be a fresh id
  // - all fresh ids in the |predecessor_id_to_fresh_id| must be unique
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Inserts a copy of the propagated instruction into each |block_id|'s
  // predecessor. Replaces the original instruction with an OpPhi referring
  // inserted copies.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if this transformation can be applied to the block with id
  // |block_id|. Concretely, returns true iff:
  // - |block_id| is a valid id of some block in the module
  // - |block_id| has predecessors
  // - |block_id| contains an instruction that can be propagated
  // - it is possible to insert the propagated instruction into every
  //   |block_id|'s predecessor
  static bool IsApplicableToBlock(opt::IRContext* ir_context,
                                  uint32_t block_id);

 private:
  // Returns the instruction that will be propagated into the predecessors of
  // the |block_id|. Returns nullptr if no such an instruction exists.
  static opt::Instruction* GetInstructionToPropagate(opt::IRContext* ir_context,
                                                     uint32_t block_id);

  // Returns true if |opcode| is supported by this transformation.
  static bool IsOpcodeSupported(spv::Op opcode);

  protobufs::TransformationPropagateInstructionUp message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_PROPAGATE_INSTRUCTION_UP_H_
