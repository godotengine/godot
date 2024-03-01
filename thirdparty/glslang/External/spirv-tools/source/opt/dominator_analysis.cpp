// Copyright (c) 2018 Google LLC
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

#include "source/opt/dominator_analysis.h"

#include <unordered_set>

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

BasicBlock* DominatorAnalysisBase::CommonDominator(BasicBlock* b1,
                                                   BasicBlock* b2) const {
  if (!b1 || !b2) return nullptr;

  std::unordered_set<BasicBlock*> seen;
  BasicBlock* block = b1;
  while (block && seen.insert(block).second) {
    block = ImmediateDominator(block);
  }

  block = b2;
  while (block && !seen.count(block)) {
    block = ImmediateDominator(block);
  }

  return block;
}

bool DominatorAnalysisBase::Dominates(Instruction* a, Instruction* b) const {
  if (!a || !b) {
    return false;
  }

  if (a == b) {
    return true;
  }

  BasicBlock* bb_a = a->context()->get_instr_block(a);
  BasicBlock* bb_b = b->context()->get_instr_block(b);

  if (bb_a != bb_b) {
    return tree_.Dominates(bb_a, bb_b);
  }

  const Instruction* current = a;
  const Instruction* other = b;

  if (tree_.IsPostDominator()) {
    std::swap(current, other);
  }

  // We handle OpLabel instructions explicitly since they are not stored in the
  // instruction list.
  if (current->opcode() == spv::Op::OpLabel) {
    return true;
  }

  while ((current = current->NextNode())) {
    if (current == other) {
      return true;
    }
  }

  return false;
}

}  // namespace opt
}  // namespace spvtools
