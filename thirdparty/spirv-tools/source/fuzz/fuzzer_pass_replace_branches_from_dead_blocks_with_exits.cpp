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

#include "source/fuzz/fuzzer_pass_replace_branches_from_dead_blocks_with_exits.h"

#include <algorithm>
#include <vector>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_replace_branch_from_dead_block_with_exit.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceBranchesFromDeadBlocksWithExits::
    FuzzerPassReplaceBranchesFromDeadBlocksWithExits(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations,
        bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassReplaceBranchesFromDeadBlocksWithExits::Apply() {
  // OpKill can only be used as a terminator in a function that is guaranteed
  // to be executed with the Fragment execution model.  We conservatively only
  // allow OpKill if every entry point in the module has the Fragment execution
  // model.
  auto fragment_execution_model_guaranteed = std::all_of(
      GetIRContext()->module()->entry_points().begin(),
      GetIRContext()->module()->entry_points().end(),
      [](const opt::Instruction& entry_point) -> bool {
        return spv::ExecutionModel(entry_point.GetSingleWordInOperand(0)) ==
               spv::ExecutionModel::Fragment;
      });

  // Transformations of this type can disable one another.  To avoid ordering
  // bias, we therefore build a set of candidate transformations to apply, and
  // subsequently apply them in a random order, skipping any that cease to be
  // applicable.
  std::vector<TransformationReplaceBranchFromDeadBlockWithExit>
      candidate_transformations;

  // Consider every block in every function.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // Probabilistically decide whether to skip this block.
      if (GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()
                  ->GetChanceOfReplacingBranchFromDeadBlockWithExit())) {
        continue;
      }
      // Check whether the block is suitable for having its terminator replaced.
      if (!TransformationReplaceBranchFromDeadBlockWithExit::BlockIsSuitable(
              GetIRContext(), *GetTransformationContext(), block)) {
        continue;
      }
      // We can always use OpUnreachable to replace a block's terminator.
      // Whether we can use OpKill depends on the execution model, and which of
      // OpReturn and OpReturnValue we can use depends on the return type of the
      // enclosing function.
      std::vector<spv::Op> opcodes = {spv::Op::OpUnreachable};
      if (fragment_execution_model_guaranteed) {
        opcodes.emplace_back(spv::Op::OpKill);
      }
      auto function_return_type =
          GetIRContext()->get_type_mgr()->GetType(function.type_id());
      if (function_return_type->AsVoid()) {
        opcodes.emplace_back(spv::Op::OpReturn);
      } else if (fuzzerutil::CanCreateConstant(GetIRContext(),
                                               function.type_id())) {
        // For simplicity we only allow OpReturnValue if the function return
        // type is a type for which we can create a constant.  This allows us a
        // zero of the given type as a default return value.
        opcodes.emplace_back(spv::Op::OpReturnValue);
      }
      // Choose one of the available terminator opcodes at random and create a
      // candidate transformation.
      auto opcode = opcodes[GetFuzzerContext()->RandomIndex(opcodes)];
      candidate_transformations.emplace_back(
          TransformationReplaceBranchFromDeadBlockWithExit(
              block.id(), opcode,
              opcode == spv::Op::OpReturnValue
                  ? FindOrCreateZeroConstant(function.type_id(), true)
                  : 0));
    }
  }

  // Process the candidate transformations in a random order.
  while (!candidate_transformations.empty()) {
    // Transformations of this type can disable one another.  For example,
    // suppose we have dead blocks A, B, C, D arranged as follows:
    //
    //         A         |
    //        / \        |
    //       B   C       |
    //        \ /        |
    //         D         |
    //
    // Here we can replace the terminator of either B or C with an early exit,
    // because D has two predecessors.  But if we replace the terminator of B,
    // say, we get:
    //
    //         A         |
    //        / \        |
    //       B   C       |
    //          /        |
    //         D         |
    //
    // and now it is no longer OK to replace the terminator of C as D only has
    // one predecessor and we do not want to make D unreachable in the control
    // flow graph.
    MaybeApplyTransformation(
        GetFuzzerContext()->RemoveAtRandomIndex(&candidate_transformations));
  }
}

}  // namespace fuzz
}  // namespace spvtools
