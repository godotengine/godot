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

#ifndef SOURCE_FUZZ_FUZZER_PASS_WRAP_REGIONS_IN_SELECTIONS_H_
#define SOURCE_FUZZ_FUZZER_PASS_WRAP_REGIONS_IN_SELECTIONS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Randomly wraps a region of blocks in every function into a selection
// construct.
class FuzzerPassWrapRegionsInSelections : public FuzzerPass {
 public:
  FuzzerPassWrapRegionsInSelections(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;

 private:
  // Tries to adjust |header_block_candidate| such that
  // TransformationWrapRegionInSelection has higher chances of being
  // applied. In particular, tries to split |header_block_candidate| if it's
  // already a header block of some other construct.
  opt::BasicBlock* MaybeGetHeaderBlockCandidate(
      opt::BasicBlock* header_block_candidate);

  // Tries to adjust |merge_block_candidate| such that
  // TransformationWrapRegionInSelection has higher chances of being
  // applied. In particular, tries to split |merge_block_candidate| if it's
  // already a merge block of some other construct.
  opt::BasicBlock* MaybeGetMergeBlockCandidate(
      opt::BasicBlock* merge_block_candidate);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_WRAP_REGIONS_IN_SELECTIONS_H_
