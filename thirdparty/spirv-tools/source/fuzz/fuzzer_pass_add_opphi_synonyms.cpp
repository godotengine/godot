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

#include "source/fuzz/fuzzer_pass_add_opphi_synonyms.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_add_opphi_synonym.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddOpPhiSynonyms::FuzzerPassAddOpPhiSynonyms(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddOpPhiSynonyms::Apply() {
  // Get a list of synonymous ids with the same type that can be used in the
  // same OpPhi instruction.
  auto equivalence_classes = GetIdEquivalenceClasses();

  // Make a list of references, to avoid copying sets unnecessarily.
  std::vector<std::set<uint32_t>*> equivalence_class_pointers;
  for (auto& set : equivalence_classes) {
    equivalence_class_pointers.push_back(&set);
  }

  // Keep a list of transformations to apply at the end.
  std::vector<TransformationAddOpPhiSynonym> transformations_to_apply;

  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // Randomly decide whether to consider this block.
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfAddingOpPhiSynonym())) {
        continue;
      }

      // The block must not be dead.
      if (GetTransformationContext()->GetFactManager()->BlockIsDead(
              block.id())) {
        continue;
      }

      // The block must have at least one predecessor.
      size_t num_preds = GetIRContext()->cfg()->preds(block.id()).size();
      if (num_preds == 0) {
        continue;
      }

      std::set<uint32_t>* chosen_equivalence_class = nullptr;

      if (num_preds > 1) {
        // If the block has more than one predecessor, prioritise sets with at
        // least 2 ids available at some predecessor.
        chosen_equivalence_class = MaybeFindSuitableEquivalenceClassRandomly(
            equivalence_class_pointers, block.id(), 2);
      }

      // If a set was not already chosen, choose one with at least one available
      // id.
      if (!chosen_equivalence_class) {
        chosen_equivalence_class = MaybeFindSuitableEquivalenceClassRandomly(
            equivalence_class_pointers, block.id(), 1);
      }

      // If no suitable set was found, we cannot apply the transformation to
      // this block.
      if (!chosen_equivalence_class) {
        continue;
      }

      // Initialise the map from predecessor labels to ids.
      std::map<uint32_t, uint32_t> preds_to_ids;

      // Keep track of the ids used and of the id of a predecessor with at least
      // two ids to choose from. This is to ensure that, if possible, at least
      // two distinct ids will be used.
      std::set<uint32_t> ids_chosen;
      uint32_t pred_with_alternatives = 0;

      // Choose an id for each predecessor.
      for (uint32_t pred_id : GetIRContext()->cfg()->preds(block.id())) {
        auto suitable_ids = GetSuitableIds(*chosen_equivalence_class, pred_id);
        assert(!suitable_ids.empty() &&
               "We must be able to find at least one suitable id because the "
               "equivalence class was chosen among suitable ones.");

        // If this predecessor has more than one id to choose from and it is the
        // first one of this kind that we found, remember its id.
        if (suitable_ids.size() > 1 && !pred_with_alternatives) {
          pred_with_alternatives = pred_id;
        }

        uint32_t chosen_id =
            suitable_ids[GetFuzzerContext()->RandomIndex(suitable_ids)];

        // Add this id to the set of ids chosen.
        ids_chosen.emplace(chosen_id);

        // Add the pair (predecessor, chosen id) to the map.
        preds_to_ids[pred_id] = chosen_id;
      }

      // If:
      // - the block has more than one predecessor
      // - at least one predecessor has more than one alternative
      // - the same id has been chosen by all the predecessors
      // then choose another one for the predecessor with more than one
      // alternative.
      if (num_preds > 1 && pred_with_alternatives != 0 &&
          ids_chosen.size() == 1) {
        auto suitable_ids =
            GetSuitableIds(*chosen_equivalence_class, pred_with_alternatives);
        uint32_t chosen_id =
            GetFuzzerContext()->RemoveAtRandomIndex(&suitable_ids);
        if (chosen_id == preds_to_ids[pred_with_alternatives]) {
          chosen_id = GetFuzzerContext()->RemoveAtRandomIndex(&suitable_ids);
        }

        preds_to_ids[pred_with_alternatives] = chosen_id;
      }

      // Add the transformation to the list of transformations to apply.
      transformations_to_apply.emplace_back(block.id(), preds_to_ids,
                                            GetFuzzerContext()->GetFreshId());
    }
  }

  // Apply the transformations.
  for (const auto& transformation : transformations_to_apply) {
    ApplyTransformation(transformation);
  }
}

std::vector<std::set<uint32_t>>
FuzzerPassAddOpPhiSynonyms::GetIdEquivalenceClasses() {
  std::vector<std::set<uint32_t>> id_equivalence_classes;

  // Keep track of all the ids that have already be assigned to a class.
  std::set<uint32_t> already_in_a_class;

  for (const auto& pair : GetIRContext()->get_def_use_mgr()->id_to_defs()) {
    // Exclude ids that have already been assigned to a class.
    if (already_in_a_class.count(pair.first)) {
      continue;
    }

    // Exclude irrelevant ids.
    if (GetTransformationContext()->GetFactManager()->IdIsIrrelevant(
            pair.first)) {
      continue;
    }

    // Exclude ids having a type that is not allowed by the transformation.
    if (!TransformationAddOpPhiSynonym::CheckTypeIsAllowed(
            GetIRContext(), pair.second->type_id())) {
      continue;
    }

    // Exclude OpFunction and OpUndef instructions, because:
    // - OpFunction does not yield a value;
    // - OpUndef yields an undefined value at each use, so it should never be a
    //   synonym of another id.
    if (pair.second->opcode() == spv::Op::OpFunction ||
        pair.second->opcode() == spv::Op::OpUndef) {
      continue;
    }

    // We need a new equivalence class for this id.
    std::set<uint32_t> new_equivalence_class;

    // Add this id to the class.
    new_equivalence_class.emplace(pair.first);
    already_in_a_class.emplace(pair.first);

    // Add all the synonyms with the same type to this class.
    for (auto synonym :
         GetTransformationContext()->GetFactManager()->GetSynonymsForId(
             pair.first)) {
      // The synonym must be a plain id - it cannot be an indexed access into a
      // composite.
      if (synonym->index_size() > 0) {
        continue;
      }

      // The synonym must not be irrelevant.
      if (GetTransformationContext()->GetFactManager()->IdIsIrrelevant(
              synonym->object())) {
        continue;
      }

      auto synonym_def =
          GetIRContext()->get_def_use_mgr()->GetDef(synonym->object());
      // The synonym must exist and have the same type as the id we are
      // considering.
      if (!synonym_def || synonym_def->type_id() != pair.second->type_id()) {
        continue;
      }

      // We can add this synonym to the new equivalence class.
      new_equivalence_class.emplace(synonym->object());
      already_in_a_class.emplace(synonym->object());
    }

    // Add the new equivalence class to the list of equivalence classes.
    id_equivalence_classes.emplace_back(std::move(new_equivalence_class));
  }

  return id_equivalence_classes;
}

bool FuzzerPassAddOpPhiSynonyms::EquivalenceClassIsSuitableForBlock(
    const std::set<uint32_t>& equivalence_class, uint32_t block_id,
    uint32_t distinct_ids_required) {
  bool at_least_one_id_for_each_pred = true;

  // Keep a set of the suitable ids found.
  std::set<uint32_t> suitable_ids_found;

  // Loop through all the predecessors of the block.
  for (auto pred_id : GetIRContext()->cfg()->preds(block_id)) {
    // Find the last instruction in the predecessor block.
    auto last_instruction =
        GetIRContext()->get_instr_block(pred_id)->terminator();

    // Initially assume that there is not a suitable id for this predecessor.
    bool at_least_one_suitable_id_found = false;
    for (uint32_t id : equivalence_class) {
      if (fuzzerutil::IdIsAvailableBeforeInstruction(GetIRContext(),
                                                     last_instruction, id)) {
        // We have found a suitable id.
        at_least_one_suitable_id_found = true;
        suitable_ids_found.emplace(id);

        // If we have already found enough distinct suitable ids, we don't need
        // to check the remaining ones for this predecessor.
        if (suitable_ids_found.size() >= distinct_ids_required) {
          break;
        }
      }
    }
    // If no suitable id was found for this predecessor, this equivalence class
    // is not suitable and we don't need to check the other predecessors.
    if (!at_least_one_suitable_id_found) {
      at_least_one_id_for_each_pred = false;
      break;
    }
  }

  // The equivalence class is suitable if at least one suitable id was found for
  // each predecessor and we have found at least |distinct_ids_required|
  // distinct suitable ids in general.
  return at_least_one_id_for_each_pred &&
         suitable_ids_found.size() >= distinct_ids_required;
}

std::vector<uint32_t> FuzzerPassAddOpPhiSynonyms::GetSuitableIds(
    const std::set<uint32_t>& ids, uint32_t pred_id) {
  // Initialise an empty vector of suitable ids.
  std::vector<uint32_t> suitable_ids;

  // Get the predecessor block.
  auto predecessor = fuzzerutil::MaybeFindBlock(GetIRContext(), pred_id);

  // Loop through the ids to find the suitable ones.
  for (uint32_t id : ids) {
    if (fuzzerutil::IdIsAvailableBeforeInstruction(
            GetIRContext(), predecessor->terminator(), id)) {
      suitable_ids.push_back(id);
    }
  }

  return suitable_ids;
}

std::set<uint32_t>*
FuzzerPassAddOpPhiSynonyms::MaybeFindSuitableEquivalenceClassRandomly(
    const std::vector<std::set<uint32_t>*>& candidates, uint32_t block_id,
    uint32_t distinct_ids_required) {
  auto remaining_candidates = candidates;
  while (!remaining_candidates.empty()) {
    // Choose one set randomly and return it if it is suitable.
    auto chosen =
        GetFuzzerContext()->RemoveAtRandomIndex(&remaining_candidates);
    if (EquivalenceClassIsSuitableForBlock(*chosen, block_id,
                                           distinct_ids_required)) {
      return chosen;
    }
  }

  // No suitable sets were found.
  return nullptr;
}

}  // namespace fuzz
}  // namespace spvtools
