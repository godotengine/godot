// Copyright (c) 2017 Google Inc.
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

#include "source/opt/redundancy_elimination.h"

#include "source/opt/value_number_table.h"

namespace spvtools {
namespace opt {

Pass::Status RedundancyEliminationPass::Process() {
  bool modified = false;
  ValueNumberTable vnTable(context());

  for (auto& func : *get_module()) {
    if (func.IsDeclaration()) {
      continue;
    }

    // Build the dominator tree for this function. It is how the code is
    // traversed.
    DominatorTree& dom_tree =
        context()->GetDominatorAnalysis(&func)->GetDomTree();

    // Keeps track of all ids that contain a given value number. We keep
    // track of multiple values because they could have the same value, but
    // different decorations.
    std::map<uint32_t, uint32_t> value_to_ids;

    if (EliminateRedundanciesFrom(dom_tree.GetRoot(), vnTable, value_to_ids)) {
      modified = true;
    }
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool RedundancyEliminationPass::EliminateRedundanciesFrom(
    DominatorTreeNode* bb, const ValueNumberTable& vnTable,
    std::map<uint32_t, uint32_t> value_to_ids) {
  bool modified = EliminateRedundanciesInBB(bb->bb_, vnTable, &value_to_ids);

  for (auto dominated_bb : bb->children_) {
    modified |= EliminateRedundanciesFrom(dominated_bb, vnTable, value_to_ids);
  }

  return modified;
}
}  // namespace opt
}  // namespace spvtools
