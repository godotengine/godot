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

#include "source/opt/local_redundancy_elimination.h"

#include "source/opt/value_number_table.h"

namespace spvtools {
namespace opt {

Pass::Status LocalRedundancyEliminationPass::Process() {
  bool modified = false;
  ValueNumberTable vnTable(context());

  for (auto& func : *get_module()) {
    for (auto& bb : func) {
      // Keeps track of all ids that contain a given value number. We keep
      // track of multiple values because they could have the same value, but
      // different decorations.
      std::map<uint32_t, uint32_t> value_to_ids;
      if (EliminateRedundanciesInBB(&bb, vnTable, &value_to_ids))
        modified = true;
    }
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool LocalRedundancyEliminationPass::EliminateRedundanciesInBB(
    BasicBlock* block, const ValueNumberTable& vnTable,
    std::map<uint32_t, uint32_t>* value_to_ids) {
  bool modified = false;

  auto func = [this, &vnTable, &modified, value_to_ids](Instruction* inst) {
    if (inst->result_id() == 0) {
      return;
    }

    uint32_t value = vnTable.GetValueNumber(inst);

    if (value == 0) {
      return;
    }

    auto candidate = value_to_ids->insert({value, inst->result_id()});
    if (!candidate.second) {
      context()->KillNamesAndDecorates(inst);
      context()->ReplaceAllUsesWith(inst->result_id(), candidate.first->second);
      context()->KillInst(inst);
      modified = true;
    }
  };
  block->ForEachInst(func);
  return modified;
}
}  // namespace opt
}  // namespace spvtools
