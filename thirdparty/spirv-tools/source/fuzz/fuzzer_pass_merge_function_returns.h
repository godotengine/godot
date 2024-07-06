// Copyright (c) 2020 Stefano Milizia
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

#ifndef SOURCE_FUZZ_FUZZER_PASS_MERGE_FUNCTION_RETURNS_H_
#define SOURCE_FUZZ_FUZZER_PASS_MERGE_FUNCTION_RETURNS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass for changing functions in the module so that they don't have an
// early return.  When handling a function the pass first eliminates early
// terminator instructions, such as OpKill, by wrapping them in functions and
// replacing them with a function call followed by a return.  The return
// instructions that arise are then modified so that the function does not have
// early returns.
class FuzzerPassMergeFunctionReturns : public FuzzerPass {
 public:
  FuzzerPassMergeFunctionReturns(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;

 private:
  // Returns a map from type ids to a list of ids with that type and which are
  // available at the end of the entry block of |function|.
  std::map<uint32_t, std::vector<uint32_t>>
  GetTypesToIdsAvailableAfterEntryBlock(opt::Function* function) const;

  // Returns the set of all the loop merge blocks whose corresponding loops
  // contain at least one of the blocks in |blocks|.
  std::set<uint32_t> GetMergeBlocksOfLoopsContainingBlocks(
      const std::set<uint32_t>& blocks) const;

  // Returns a list of ReturnMergingInfo messages, containing the information
  // needed by the transformation for each of the relevant merge blocks.
  // If a new id is created (because |ids_available_after_entry_block| does not
  // have an entry for the corresponding type), a new entry is added to
  // |ids_available_after_entry_block|, mapping its type to a singleton set
  // containing it.
  std::vector<protobufs::ReturnMergingInfo> GetInfoNeededForMergeBlocks(
      const std::vector<uint32_t>& merge_blocks,
      std::map<uint32_t, std::vector<uint32_t>>*
          ids_available_after_entry_block);

  // Returns true if and only if |function| is a wrapper for an early terminator
  // instruction such as OpKill.
  bool IsEarlyTerminatorWrapper(const opt::Function& function) const;
};
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_MERGE_FUNCTION_RETURNS_H_
