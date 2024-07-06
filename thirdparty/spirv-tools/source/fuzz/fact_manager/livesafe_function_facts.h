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

#ifndef SOURCE_FUZZ_FACT_MANAGER_LIVESAFE_FUNCTION_FACTS_H_
#define SOURCE_FUZZ_FACT_MANAGER_LIVESAFE_FUNCTION_FACTS_H_

#include <unordered_set>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {
namespace fact_manager {

// The purpose of this class is to group the fields and data used to represent
// facts about livesafe functions.
class LivesafeFunctionFacts {
 public:
  explicit LivesafeFunctionFacts(opt::IRContext* ir_context);

  // See method in FactManager which delegates to this method. Returns true if
  // |fact.function_id()| is a result id of some non-entry-point function in
  // |ir_context_|. Returns false otherwise.
  bool MaybeAddFact(const protobufs::FactFunctionIsLivesafe& fact);

  // See method in FactManager which delegates to this method.
  bool FunctionIsLivesafe(uint32_t function_id) const;

 private:
  std::unordered_set<uint32_t> livesafe_function_ids_;
  opt::IRContext* ir_context_;
};

}  // namespace fact_manager
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FACT_MANAGER_LIVESAFE_FUNCTION_FACTS_H_
