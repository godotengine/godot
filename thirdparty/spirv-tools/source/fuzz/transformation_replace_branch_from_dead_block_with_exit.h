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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_BRANCH_FROM_DEAD_BLOCK_WITH_EXIT_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_BRANCH_FROM_DEAD_BLOCK_WITH_EXIT_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/basic_block.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceBranchFromDeadBlockWithExit : public Transformation {
 public:
  explicit TransformationReplaceBranchFromDeadBlockWithExit(
      protobufs::TransformationReplaceBranchFromDeadBlockWithExit message);

  TransformationReplaceBranchFromDeadBlockWithExit(uint32_t block_id,
                                                   spv::Op opcode,
                                                   uint32_t return_value_id);

  // - |message_.block_id| must be the id of a dead block that is not part of
  //   a continue construct
  // - |message_.block_id| must end with OpBranch
  // - The successor of |message_.block_id| must have at least one other
  //   predecessor
  // - |message_.opcode()| must be one of OpKill, OpReturn, OpReturnValue and
  //   OpUnreachable
  // - |message_.opcode()| can only be OpKill if the module's entry points all
  //   have Fragment execution mode
  // - |message_.opcode()| can only be OpReturn if the return type of the
  //   function containing the block is void
  // - If |message_.opcode()| is OpReturnValue then |message_.return_value_id|
  //   must be an id that is available at the block terminator and that matches
  //   the return type of the enclosing function
  // - Domination rules should be preserved when we apply this transformation.
  //   In particular, if some block appears after the |block_id|'s successor in
  //   the CFG, then that block cannot dominate |block_id|'s successor when this
  //   transformation is applied.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Changes the terminator of |message_.block_id| to have opcode
  // |message_.opcode|, additionally with input operand
  // |message_.return_value_id| in the case that |message_.opcode| is
  // OpReturnValue.
  //
  // If |message_.block_id|'s successor starts with OpPhi instructions these are
  // updated so that they no longer refer to |message_.block_id|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if and only if |block| meets the criteria for having its
  // terminator replaced with an early exit (see IsApplicable for details of the
  // criteria.)
  static bool BlockIsSuitable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context,
      const opt::BasicBlock& block);

 private:
  protobufs::TransformationReplaceBranchFromDeadBlockWithExit message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_BRANCH_FROM_DEAD_BLOCK_WITH_EXIT_H_
