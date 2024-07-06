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

#ifndef SOURCE_FUZZ_FUZZER_PASS_ADD_OPPHI_SYNONYMS_H_
#define SOURCE_FUZZ_FUZZER_PASS_ADD_OPPHI_SYNONYMS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass to add OpPhi instructions which can take the values of ids that
// have been marked as synonymous. This instruction will itself be marked as
// synonymous with the others.
class FuzzerPassAddOpPhiSynonyms : public FuzzerPass {
 public:
  FuzzerPassAddOpPhiSynonyms(opt::IRContext* ir_context,
                             TransformationContext* transformation_context,
                             FuzzerContext* fuzzer_context,
                             protobufs::TransformationSequence* transformations,
                             bool ignore_inapplicable_transformations);

  void Apply() override;

  // Computes the equivalence classes for the non-pointer and non-irrelevant ids
  // in the module, where two ids are considered equivalent iff they have been
  // declared synonymous and they have the same type.
  std::vector<std::set<uint32_t>> GetIdEquivalenceClasses();

  // Returns true iff |equivalence_class| contains at least
  // |distinct_ids_required| ids so that all of these ids are available at the
  // end of at least one predecessor of the block with label |block_id|.
  // Assumes that the block has at least one predecessor.
  bool EquivalenceClassIsSuitableForBlock(
      const std::set<uint32_t>& equivalence_class, uint32_t block_id,
      uint32_t distinct_ids_required);

  // Returns a vector with the ids that are available to use at the end of the
  // block with id |pred_id|, selected among the given |ids|. Assumes that
  // |pred_id| is the label of a block and all ids in |ids| exist in the module.
  std::vector<uint32_t> GetSuitableIds(const std::set<uint32_t>& ids,
                                       uint32_t pred_id);

 private:
  // Randomly chooses one of the equivalence classes in |candidates|, so that it
  // satisfies all of the following conditions:
  // - For each of the predecessors of the |block_id| block, there is at least
  //   one id in the chosen equivalence class that is available at the end of
  //   it.
  // - There are at least |distinct_ids_required| ids available at the end of
  //   some predecessor.
  // Returns nullptr if no equivalence class in |candidates| satisfies the
  // requirements.
  std::set<uint32_t>* MaybeFindSuitableEquivalenceClassRandomly(
      const std::vector<std::set<uint32_t>*>& candidates, uint32_t block_id,
      uint32_t distinct_ids_required);
};
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_ADD_OPPHI_SYNONYMS_H_
