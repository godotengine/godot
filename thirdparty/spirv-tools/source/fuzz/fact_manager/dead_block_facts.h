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

#ifndef SOURCE_FUZZ_FACT_MANAGER_DEAD_BLOCK_FACTS_H_
#define SOURCE_FUZZ_FACT_MANAGER_DEAD_BLOCK_FACTS_H_

#include <unordered_set>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {
namespace fact_manager {

// The purpose of this class is to group the fields and data used to represent
// facts about data blocks.
class DeadBlockFacts {
 public:
  explicit DeadBlockFacts(opt::IRContext* ir_context);

  // Marks |fact.block_id()| as being dead. Returns true if |fact.block_id()|
  // represents a result id of some OpLabel instruction in |ir_context_|.
  // Returns false otherwise.
  bool MaybeAddFact(const protobufs::FactBlockIsDead& fact);

  // See method in FactManager which delegates to this method.
  bool BlockIsDead(uint32_t block_id) const;

  // Returns a set of all the block ids that have been declared dead.
  const std::unordered_set<uint32_t>& GetDeadBlocks() const;

 private:
  std::unordered_set<uint32_t> dead_block_ids_;
  opt::IRContext* ir_context_;
};

}  // namespace fact_manager
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FACT_MANAGER_DEAD_BLOCK_FACTS_H_
