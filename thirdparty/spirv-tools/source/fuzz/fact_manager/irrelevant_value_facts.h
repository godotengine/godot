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

#ifndef SOURCE_FUZZ_FACT_MANAGER_IRRELEVANT_VALUE_FACTS_H_
#define SOURCE_FUZZ_FACT_MANAGER_IRRELEVANT_VALUE_FACTS_H_

#include <unordered_set>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {
namespace fact_manager {

// Forward reference to the DataSynonymAndIdEquationFacts class.
class DataSynonymAndIdEquationFacts;
// Forward reference to the DeadBlockFacts class.
class DeadBlockFacts;

// The purpose of this class is to group the fields and data used to represent
// facts about various irrelevant values in the module.
class IrrelevantValueFacts {
 public:
  explicit IrrelevantValueFacts(opt::IRContext* ir_context);

  // See method in FactManager which delegates to this method. Returns true if
  // |fact.pointer_id()| is a result id of pointer type in the |ir_context_| and
  // |fact.pointer_id()| does not participate in DataSynonym facts. Returns
  // false otherwise. |data_synonym_and_id_equation_facts| and |context| are
  // passed for consistency checks.
  bool MaybeAddFact(
      const protobufs::FactPointeeValueIsIrrelevant& fact,
      const DataSynonymAndIdEquationFacts& data_synonym_and_id_equation_facts);

  // See method in FactManager which delegates to this method. Returns true if
  // |fact.result_id()| is a result id of non-pointer type in the |ir_context_|
  // and |fact.result_id()| does not participate in DataSynonym facts. Returns
  // false otherwise. |data_synonym_and_id_equation_facts| and |context| are
  // passed for consistency checks.
  bool MaybeAddFact(
      const protobufs::FactIdIsIrrelevant& fact,
      const DataSynonymAndIdEquationFacts& data_synonym_and_id_equation_facts);

  // See method in FactManager which delegates to this method.
  bool PointeeValueIsIrrelevant(uint32_t pointer_id) const;

  // See method in FactManager which delegates to this method.
  // |dead_block_facts| and |context| are passed to check whether |result_id| is
  // declared inside a dead block, in which case it is irrelevant.
  bool IdIsIrrelevant(uint32_t result_id,
                      const DeadBlockFacts& dead_block_facts) const;

  // See method in FactManager which delegates to this method.
  // |dead_block_facts| and |context| are passed to also add all the ids
  // declared in dead blocks to the set of irrelevant ids.
  std::unordered_set<uint32_t> GetIrrelevantIds(
      const DeadBlockFacts& dead_block_facts) const;

 private:
  std::unordered_set<uint32_t> pointers_to_irrelevant_pointees_ids_;
  std::unordered_set<uint32_t> irrelevant_ids_;
  opt::IRContext* ir_context_;
};

}  // namespace fact_manager
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FACT_MANAGER_IRRELEVANT_VALUE_FACTS_H_
