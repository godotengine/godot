// Copyright (c) 2018 Google LLC.
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

#ifndef SOURCE_OPT_STRUCT_CFG_ANALYSIS_H_
#define SOURCE_OPT_STRUCT_CFG_ANALYSIS_H_

#include <unordered_map>
#include <unordered_set>

#include "source/opt/function.h"
#include "source/util/bit_vector.h"

namespace spvtools {
namespace opt {

class IRContext;

// An analysis that, for each basic block, finds the constructs in which it is
// contained, so we can easily get headers and merge nodes.
class StructuredCFGAnalysis {
 public:
  explicit StructuredCFGAnalysis(IRContext* ctx);

  // Returns the id of the header of the innermost merge construct
  // that contains |bb_id|.  Returns |0| if |bb_id| is not contained in any
  // merge construct.
  uint32_t ContainingConstruct(uint32_t bb_id) {
    auto it = bb_to_construct_.find(bb_id);
    if (it == bb_to_construct_.end()) {
      return 0;
    }
    return it->second.containing_construct;
  }

  // Returns the id of the header of the innermost merge construct
  // that contains |inst|.  Returns |0| if |inst| is not contained in any
  // merge construct.
  uint32_t ContainingConstruct(Instruction* inst);

  // Returns the id of the merge block of the innermost merge construct
  // that contains |bb_id|.  Returns |0| if |bb_id| is not contained in any
  // merge construct.
  uint32_t MergeBlock(uint32_t bb_id);

  // Returns the nesting depth of the given block, i.e. the number of merge
  // constructs containing it. Headers and merge blocks are not considered part
  // of the corresponding merge constructs.
  uint32_t NestingDepth(uint32_t block_id);

  // Returns the id of the header of the innermost loop construct
  // that contains |bb_id|.  Return |0| if |bb_id| is not contained in any loop
  // construct.
  uint32_t ContainingLoop(uint32_t bb_id) {
    auto it = bb_to_construct_.find(bb_id);
    if (it == bb_to_construct_.end()) {
      return 0;
    }
    return it->second.containing_loop;
  }

  // Returns the id of the merge block of the innermost loop construct
  // that contains |bb_id|.  Return |0| if |bb_id| is not contained in any loop
  // construct.
  uint32_t LoopMergeBlock(uint32_t bb_id);

  // Returns the id of the continue block of the innermost loop construct
  // that contains |bb_id|.  Return |0| if |bb_id| is not contained in any loop
  // construct.
  uint32_t LoopContinueBlock(uint32_t bb_id);

  // Returns the loop nesting depth of |bb_id| within its function, i.e. the
  // number of loop constructs in which |bb_id| is contained. As per other
  // functions in StructuredCFGAnalysis, a loop header is not regarded as being
  // part of the loop that it heads, so that e.g. the nesting depth of an
  // outer-most loop header is 0.
  uint32_t LoopNestingDepth(uint32_t bb_id);

  // Returns the id of the header of the innermost switch construct
  // that contains |bb_id| as long as there is no intervening loop.  Returns |0|
  // if no such construct exists.
  uint32_t ContainingSwitch(uint32_t bb_id) {
    auto it = bb_to_construct_.find(bb_id);
    if (it == bb_to_construct_.end()) {
      return 0;
    }
    return it->second.containing_switch;
  }
  // Returns the id of the merge block of the innermost switch construct
  // that contains |bb_id| as long as there is no intervening loop.  Return |0|
  // if no such block exists.
  uint32_t SwitchMergeBlock(uint32_t bb_id);

  // Returns true if |bb_id| is the continue block for a loop.
  bool IsContinueBlock(uint32_t bb_id);

  // Returns true if |bb_id| is in the continue construct for its inner most
  // containing loop.
  bool IsInContainingLoopsContinueConstruct(uint32_t bb_id);

  // Returns true if |bb_id| is in the continue construct for any loop in its
  // function.
  bool IsInContinueConstruct(uint32_t bb_id);

  // Return true if |bb_id| is the merge block for a construct.
  bool IsMergeBlock(uint32_t bb_id);

  // Returns the set of function ids that are called directly or indirectly from
  // a continue construct.
  std::unordered_set<uint32_t> FindFuncsCalledFromContinue();

 private:
  // Struct used to hold the information for a basic block.
  // |containing_construct| is the header for the innermost containing
  // construct, or 0 if no such construct exists.  It could be a selection
  // construct or a loop construct.
  //
  // |containing_loop| is the innermost containing loop construct, or 0 if the
  // basic bloc is not in a loop.  If the basic block is in a selection
  // construct that is contained in a loop construct, then these two values will
  // not be the same.
  //
  // |containing_switch| is the innermost contain selection construct with an
  // |OpSwitch| for the branch, as long as there is not intervening loop.  This
  // is used to identify the selection construct from which it can break.
  //
  // |in_continue| is true of the block is in the continue construct for its
  // innermost containing loop.
  struct ConstructInfo {
    uint32_t containing_construct;
    uint32_t containing_loop;
    uint32_t containing_switch;
    bool in_continue;
  };

  // Populates |bb_to_construct_| with the innermost containing merge and loop
  // constructs for each basic block in |func|.
  void AddBlocksInFunction(Function* func);

  IRContext* context_;

  // A map from a basic block to the headers of its inner most containing
  // constructs.
  std::unordered_map<uint32_t, ConstructInfo> bb_to_construct_;
  utils::BitVector merge_blocks_;
};

}  // namespace opt
}  // namespace spvtools
#endif  // SOURCE_OPT_STRUCT_CFG_ANALYSIS_H_
