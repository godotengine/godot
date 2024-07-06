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

#ifndef SOURCE_OPT_REDUNDANCY_ELIMINATION_H_
#define SOURCE_OPT_REDUNDANCY_ELIMINATION_H_

#include <map>

#include "source/opt/ir_context.h"
#include "source/opt/local_redundancy_elimination.h"
#include "source/opt/pass.h"
#include "source/opt/value_number_table.h"

namespace spvtools {
namespace opt {

// This pass implements total redundancy elimination.  This is the same as
// local redundancy elimination except it looks across basic block boundaries.
// An instruction, inst, is totally redundant if there is another instruction
// that dominates inst, and also computes the same value.
class RedundancyEliminationPass : public LocalRedundancyEliminationPass {
 public:
  const char* name() const override { return "redundancy-elimination"; }
  Status Process() override;

 protected:
  // Removes for all total redundancies in the function starting at |bb|.
  //
  // |vnTable| must have computed a value number for every result id defined
  // in the function containing |bb|.
  //
  // |value_to_ids| is a map from value number to ids.  If {vn, id} is in
  // |value_to_ids| then vn is the value number of id, and the definition of id
  // dominates |bb|.
  //
  // Returns true if at least one instruction is deleted.
  bool EliminateRedundanciesFrom(DominatorTreeNode* bb,
                                 const ValueNumberTable& vnTable,
                                 std::map<uint32_t, uint32_t> value_to_ids);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_REDUNDANCY_ELIMINATION_H_
