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

#ifndef SOURCE_FUZZ_FUZZER_PASS_OUTLINE_FUNCTIONS_H_
#define SOURCE_FUZZ_FUZZER_PASS_OUTLINE_FUNCTIONS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass for outlining single-entry single-exit regions of a  control
// flow graph into their own functions.
class FuzzerPassOutlineFunctions : public FuzzerPass {
 public:
  FuzzerPassOutlineFunctions(opt::IRContext* ir_context,
                             TransformationContext* transformation_context,
                             FuzzerContext* fuzzer_context,
                             protobufs::TransformationSequence* transformations,
                             bool ignore_inapplicable_transformations);

  void Apply() override;

  // Returns a block suitable to be an entry block for a region that can be
  // outlined, i.e. a block that is not a loop header and that does not start
  // with OpPhi or OpVariable. In particular, it returns:
  // - |entry_block| if it is suitable
  // - otherwise, a block found by:
  //   - looking for or creating a new preheader, if |entry_block| is a loop
  //     header
  //   - splitting the candidate entry block, if it starts with OpPhi or
  //     OpVariable.
  // Returns nullptr if a suitable block cannot be found following the
  // instructions above.
  opt::BasicBlock* MaybeGetEntryBlockSuitableForOutlining(
      opt::BasicBlock* entry_block);

  // Returns:
  // - |exit_block| if it is not a merge block
  // - the second block obtained by splitting |exit_block|, if |exit_block| is a
  //   merge block.
  // Assumes that |exit_block| is not a continue target.
  // The block returned by this function should be suitable to be the exit block
  // of a region that can be outlined.
  // Returns nullptr if |exit_block| is a merge block and it cannot be split.
  opt::BasicBlock* MaybeGetExitBlockSuitableForOutlining(
      opt::BasicBlock* exit_block);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_OUTLINE_FUNCTIONS_H_
