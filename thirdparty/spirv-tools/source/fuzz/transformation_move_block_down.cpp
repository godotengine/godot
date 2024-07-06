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

#include "source/fuzz/transformation_move_block_down.h"

#include "source/opt/basic_block.h"

namespace spvtools {
namespace fuzz {

TransformationMoveBlockDown::TransformationMoveBlockDown(
    protobufs::TransformationMoveBlockDown message)
    : message_(std::move(message)) {}

TransformationMoveBlockDown::TransformationMoveBlockDown(uint32_t id) {
  message_.set_block_id(id);
}

bool TransformationMoveBlockDown::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Go through every block in every function, looking for a block whose id
  // matches that of the block we want to consider moving down.
  for (auto& function : *ir_context->module()) {
    for (auto block_it = function.begin(); block_it != function.end();
         ++block_it) {
      if (block_it->id() == message_.block_id()) {
        // We have found a match.
        if (block_it == function.begin()) {
          // The block is the first one appearing in the function.  We are not
          // allowed to move this block down.
          return false;
        }
        // Record the block we would like to consider moving down.
        opt::BasicBlock* block_matching_id = &*block_it;
        if (!ir_context->GetDominatorAnalysis(&function)->IsReachable(
                block_matching_id)) {
          // The block is not reachable.  We are not allowed to move it down.
          return false;
        }
        // Now see whether there is some block following that block in program
        // order.
        ++block_it;
        if (block_it == function.end()) {
          // There is no such block; i.e., the block we are considering moving
          // is the last one in the function.  The transformation thus does not
          // apply.
          return false;
        }
        opt::BasicBlock* next_block_in_program_order = &*block_it;
        // We can move the block of interest down if and only if it does not
        // dominate the block that comes next.
        return !ir_context->GetDominatorAnalysis(&function)->Dominates(
            block_matching_id, next_block_in_program_order);
      }
    }
  }

  // We did not find a matching block, so the transformation is not applicable:
  // there is no relevant block to move.
  return false;
}

void TransformationMoveBlockDown::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Go through every block in every function, looking for a block whose id
  // matches that of the block we want to move down.
  for (auto& function : *ir_context->module()) {
    for (auto block_it = function.begin(); block_it != function.end();
         ++block_it) {
      if (block_it->id() == message_.block_id()) {
        ++block_it;
        assert(block_it != function.end() &&
               "To be able to move a block down, it needs to have a "
               "program-order successor.");
        function.MoveBasicBlockToAfter(message_.block_id(), &*block_it);
        // For performance, it is vital to keep the dominator analysis valid
        // (which due to https://github.com/KhronosGroup/SPIRV-Tools/issues/2889
        // requires keeping the CFG analysis valid).
        ir_context->InvalidateAnalysesExceptFor(
            opt::IRContext::Analysis::kAnalysisDefUse |
            opt::IRContext::Analysis::kAnalysisCFG |
            opt::IRContext::Analysis::kAnalysisDominatorAnalysis);

        return;
      }
    }
  }
  assert(false && "No block was found to move down.");
}

protobufs::Transformation TransformationMoveBlockDown::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_move_block_down() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationMoveBlockDown::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
