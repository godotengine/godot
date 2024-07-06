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

#include "source/fuzz/fact_manager/data_synonym_and_id_equation_facts.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {
namespace fact_manager {

size_t DataSynonymAndIdEquationFacts::OperationHash::operator()(
    const Operation& operation) const {
  std::u32string hash;
  hash.push_back(uint32_t(operation.opcode));
  for (auto operand : operation.operands) {
    hash.push_back(static_cast<uint32_t>(DataDescriptorHash()(operand)));
  }
  return std::hash<std::u32string>()(hash);
}

bool DataSynonymAndIdEquationFacts::OperationEquals::operator()(
    const Operation& first, const Operation& second) const {
  // Equal operations require...
  //
  // Equal opcodes.
  if (first.opcode != second.opcode) {
    return false;
  }
  // Matching operand counts.
  if (first.operands.size() != second.operands.size()) {
    return false;
  }
  // Equal operands.
  for (uint32_t i = 0; i < first.operands.size(); i++) {
    if (!DataDescriptorEquals()(first.operands[i], second.operands[i])) {
      return false;
    }
  }
  return true;
}

DataSynonymAndIdEquationFacts::DataSynonymAndIdEquationFacts(
    opt::IRContext* ir_context)
    : ir_context_(ir_context) {}

bool DataSynonymAndIdEquationFacts::MaybeAddFact(
    const protobufs::FactDataSynonym& fact,
    const DeadBlockFacts& dead_block_facts,
    const IrrelevantValueFacts& irrelevant_value_facts) {
  if (irrelevant_value_facts.IdIsIrrelevant(fact.data1().object(),
                                            dead_block_facts) ||
      irrelevant_value_facts.IdIsIrrelevant(fact.data2().object(),
                                            dead_block_facts)) {
    // Irrelevant ids cannot be synonymous with other ids.
    return false;
  }

  // Add the fact, including all facts relating sub-components of the data
  // descriptors that are involved.
  AddDataSynonymFactRecursive(fact.data1(), fact.data2());
  return true;
}

bool DataSynonymAndIdEquationFacts::MaybeAddFact(
    const protobufs::FactIdEquation& fact,
    const DeadBlockFacts& dead_block_facts,
    const IrrelevantValueFacts& irrelevant_value_facts) {
  if (irrelevant_value_facts.IdIsIrrelevant(fact.lhs_id(), dead_block_facts)) {
    // Irrelevant ids cannot participate in IdEquation facts.
    return false;
  }

  for (auto id : fact.rhs_id()) {
    if (irrelevant_value_facts.IdIsIrrelevant(id, dead_block_facts)) {
      // Irrelevant ids cannot participate in IdEquation facts.
      return false;
    }
  }

  protobufs::DataDescriptor lhs_dd = MakeDataDescriptor(fact.lhs_id(), {});

  // Register the LHS in the equivalence relation if needed.
  RegisterDataDescriptor(lhs_dd);

  // Get equivalence class representatives for all ids used on the RHS of the
  // equation.
  std::vector<const protobufs::DataDescriptor*> rhs_dds;
  for (auto rhs_id : fact.rhs_id()) {
    // Register a data descriptor based on this id in the equivalence relation
    // if needed, and then record the equivalence class representative.
    rhs_dds.push_back(RegisterDataDescriptor(MakeDataDescriptor(rhs_id, {})));
  }

  // Now add the fact.
  AddEquationFactRecursive(lhs_dd, static_cast<spv::Op>(fact.opcode()),
                           rhs_dds);
  return true;
}

DataSynonymAndIdEquationFacts::OperationSet
DataSynonymAndIdEquationFacts::GetEquations(
    const protobufs::DataDescriptor* lhs) const {
  auto existing = id_equations_.find(lhs);
  if (existing == id_equations_.end()) {
    return OperationSet();
  }
  return existing->second;
}

void DataSynonymAndIdEquationFacts::AddEquationFactRecursive(
    const protobufs::DataDescriptor& lhs_dd, spv::Op opcode,
    const std::vector<const protobufs::DataDescriptor*>& rhs_dds) {
  assert(synonymous_.Exists(lhs_dd) &&
         "The LHS must be known to the equivalence relation.");
  for (auto rhs_dd : rhs_dds) {
    // Keep release compilers happy.
    (void)(rhs_dd);
    assert(synonymous_.Exists(*rhs_dd) &&
           "The RHS operands must be known to the equivalence relation.");
  }

  auto lhs_dd_representative = synonymous_.Find(&lhs_dd);

  if (id_equations_.count(lhs_dd_representative) == 0) {
    // We have not seen an equation with this LHS before, so associate the LHS
    // with an initially empty set.
    id_equations_.insert({lhs_dd_representative, OperationSet()});
  }

  {
    auto existing_equations = id_equations_.find(lhs_dd_representative);
    assert(existing_equations != id_equations_.end() &&
           "A set of operations should be present, even if empty.");

    Operation new_operation = {opcode, rhs_dds};
    if (existing_equations->second.count(new_operation)) {
      // This equation is known, so there is nothing further to be done.
      return;
    }
    // Add the equation to the set of known equations.
    existing_equations->second.insert(new_operation);
  }

  // Now try to work out corollaries implied by the new equation and existing
  // facts.
  switch (opcode) {
    case spv::Op::OpConvertSToF:
    case spv::Op::OpConvertUToF:
      ComputeConversionDataSynonymFacts(*rhs_dds[0]);
      break;
    case spv::Op::OpBitcast: {
      assert(DataDescriptorsAreWellFormedAndComparable(lhs_dd, *rhs_dds[0]) &&
             "Operands of OpBitcast equation fact must have compatible types");
      if (!synonymous_.IsEquivalent(lhs_dd, *rhs_dds[0])) {
        AddDataSynonymFactRecursive(lhs_dd, *rhs_dds[0]);
      }
    } break;
    case spv::Op::OpIAdd: {
      // Equation form: "a = b + c"
      for (const auto& equation : GetEquations(rhs_dds[0])) {
        if (equation.opcode == spv::Op::OpISub) {
          // Equation form: "a = (d - e) + c"
          if (synonymous_.IsEquivalent(*equation.operands[1], *rhs_dds[1])) {
            // Equation form: "a = (d - c) + c"
            // We can thus infer "a = d"
            AddDataSynonymFactRecursive(lhs_dd, *equation.operands[0]);
          }
        }
      }
      for (const auto& equation : GetEquations(rhs_dds[1])) {
        if (equation.opcode == spv::Op::OpISub) {
          // Equation form: "a = b + (d - e)"
          if (synonymous_.IsEquivalent(*equation.operands[1], *rhs_dds[0])) {
            // Equation form: "a = b + (d - b)"
            // We can thus infer "a = d"
            AddDataSynonymFactRecursive(lhs_dd, *equation.operands[0]);
          }
        }
      }
      break;
    }
    case spv::Op::OpISub: {
      // Equation form: "a = b - c"
      for (const auto& equation : GetEquations(rhs_dds[0])) {
        if (equation.opcode == spv::Op::OpIAdd) {
          // Equation form: "a = (d + e) - c"
          if (synonymous_.IsEquivalent(*equation.operands[0], *rhs_dds[1])) {
            // Equation form: "a = (c + e) - c"
            // We can thus infer "a = e"
            AddDataSynonymFactRecursive(lhs_dd, *equation.operands[1]);
          }
          if (synonymous_.IsEquivalent(*equation.operands[1], *rhs_dds[1])) {
            // Equation form: "a = (d + c) - c"
            // We can thus infer "a = d"
            AddDataSynonymFactRecursive(lhs_dd, *equation.operands[0]);
          }
        }

        if (equation.opcode == spv::Op::OpISub) {
          // Equation form: "a = (d - e) - c"
          if (synonymous_.IsEquivalent(*equation.operands[0], *rhs_dds[1])) {
            // Equation form: "a = (c - e) - c"
            // We can thus infer "a = -e"
            AddEquationFactRecursive(lhs_dd, spv::Op::OpSNegate,
                                     {equation.operands[1]});
          }
        }
      }

      for (const auto& equation : GetEquations(rhs_dds[1])) {
        if (equation.opcode == spv::Op::OpIAdd) {
          // Equation form: "a = b - (d + e)"
          if (synonymous_.IsEquivalent(*equation.operands[0], *rhs_dds[0])) {
            // Equation form: "a = b - (b + e)"
            // We can thus infer "a = -e"
            AddEquationFactRecursive(lhs_dd, spv::Op::OpSNegate,
                                     {equation.operands[1]});
          }
          if (synonymous_.IsEquivalent(*equation.operands[1], *rhs_dds[0])) {
            // Equation form: "a = b - (d + b)"
            // We can thus infer "a = -d"
            AddEquationFactRecursive(lhs_dd, spv::Op::OpSNegate,
                                     {equation.operands[0]});
          }
        }
        if (equation.opcode == spv::Op::OpISub) {
          // Equation form: "a = b - (d - e)"
          if (synonymous_.IsEquivalent(*equation.operands[0], *rhs_dds[0])) {
            // Equation form: "a = b - (b - e)"
            // We can thus infer "a = e"
            AddDataSynonymFactRecursive(lhs_dd, *equation.operands[1]);
          }
        }
      }
      break;
    }
    case spv::Op::OpLogicalNot:
    case spv::Op::OpSNegate: {
      // Equation form: "a = !b" or "a = -b"
      for (const auto& equation : GetEquations(rhs_dds[0])) {
        if (equation.opcode == opcode) {
          // Equation form: "a = !!b" or "a = -(-b)"
          // We can thus infer "a = b"
          AddDataSynonymFactRecursive(lhs_dd, *equation.operands[0]);
        }
      }
      break;
    }
    default:
      break;
  }
}

void DataSynonymAndIdEquationFacts::AddDataSynonymFactRecursive(
    const protobufs::DataDescriptor& dd1,
    const protobufs::DataDescriptor& dd2) {
  assert((!ObjectStillExists(dd1) || !ObjectStillExists(dd2) ||
          DataDescriptorsAreWellFormedAndComparable(dd1, dd2)) &&
         "Mismatched data descriptors.");

  // Record that the data descriptors provided in the fact are equivalent.
  MakeEquivalent(dd1, dd2);
  assert(synonymous_.Find(&dd1) == synonymous_.Find(&dd2) &&
         "|dd1| and |dd2| must have a single representative");

  // Compute various corollary facts.

  // |dd1| and |dd2| belong to the same equivalence class so it doesn't matter
  // which one we use here.
  ComputeConversionDataSynonymFacts(dd1);

  ComputeCompositeDataSynonymFacts(dd1, dd2);
}

void DataSynonymAndIdEquationFacts::ComputeConversionDataSynonymFacts(
    const protobufs::DataDescriptor& dd) {
  assert(synonymous_.Exists(dd) &&
         "|dd| should've been registered in the equivalence relation");

  if (!ObjectStillExists(dd)) {
    // The object is gone from the module, so we cannot proceed.
    return;
  }

  const auto* type =
      ir_context_->get_type_mgr()->GetType(fuzzerutil::WalkCompositeTypeIndices(
          ir_context_, fuzzerutil::GetTypeId(ir_context_, dd.object()),
          dd.index()));
  assert(type && "Data descriptor has invalid type");

  if ((type->AsVector() && type->AsVector()->element_type()->AsInteger()) ||
      type->AsInteger()) {
    // If there exist equation facts of the form |%a = opcode %representative|
    // and |%b = opcode %representative| where |opcode| is either OpConvertSToF
    // or OpConvertUToF, then |a| and |b| are synonymous.
    std::vector<const protobufs::DataDescriptor*> convert_s_to_f_lhs;
    std::vector<const protobufs::DataDescriptor*> convert_u_to_f_lhs;

    for (const auto& fact : id_equations_) {
      auto equivalence_class = synonymous_.GetEquivalenceClass(*fact.first);
      auto dd_it =
          std::find_if(equivalence_class.begin(), equivalence_class.end(),
                       [this](const protobufs::DataDescriptor* a) {
                         return ObjectStillExists(*a);
                       });
      if (dd_it == equivalence_class.end()) {
        // Skip |equivalence_class| if it has no valid ids.
        continue;
      }

      for (const auto& equation : fact.second) {
        if (synonymous_.IsEquivalent(*equation.operands[0], dd)) {
          if (equation.opcode == spv::Op::OpConvertSToF) {
            convert_s_to_f_lhs.push_back(*dd_it);
          } else if (equation.opcode == spv::Op::OpConvertUToF) {
            convert_u_to_f_lhs.push_back(*dd_it);
          }
        }
      }
    }

    // We use pointers in the initializer list here since otherwise we would
    // copy memory from these vectors.
    for (const auto* synonyms : {&convert_s_to_f_lhs, &convert_u_to_f_lhs}) {
      for (const auto* synonym_a : *synonyms) {
        for (const auto* synonym_b : *synonyms) {
          // DataDescriptorsAreWellFormedAndComparable will be called in the
          // AddDataSynonymFactRecursive method.
          if (!synonymous_.IsEquivalent(*synonym_a, *synonym_b)) {
            // |synonym_a| and |synonym_b| have compatible types - they are
            // synonymous.
            AddDataSynonymFactRecursive(*synonym_a, *synonym_b);
          }
        }
      }
    }
  }
}

void DataSynonymAndIdEquationFacts::ComputeCompositeDataSynonymFacts(
    const protobufs::DataDescriptor& dd1,
    const protobufs::DataDescriptor& dd2) {
  // Check whether this is a synonym about composite objects. If it is,
  // we can recursively add synonym facts about their associated sub-components.

  // Get the type of the object referred to by the first data descriptor in the
  // synonym fact.
  uint32_t type_id = fuzzerutil::WalkCompositeTypeIndices(
      ir_context_,
      ir_context_->get_def_use_mgr()->GetDef(dd1.object())->type_id(),
      dd1.index());
  auto type = ir_context_->get_type_mgr()->GetType(type_id);
  auto type_instruction = ir_context_->get_def_use_mgr()->GetDef(type_id);
  assert(type != nullptr &&
         "Invalid data synonym fact: one side has an unknown type.");

  // Check whether the type is composite, recording the number of elements
  // associated with the composite if so.
  uint32_t num_composite_elements;
  if (type->AsArray()) {
    num_composite_elements =
        fuzzerutil::GetArraySize(*type_instruction, ir_context_);
  } else if (type->AsMatrix()) {
    num_composite_elements = type->AsMatrix()->element_count();
  } else if (type->AsStruct()) {
    num_composite_elements =
        fuzzerutil::GetNumberOfStructMembers(*type_instruction);
  } else if (type->AsVector()) {
    num_composite_elements = type->AsVector()->element_count();
  } else {
    // The type is not a composite, so return.
    return;
  }

  // If the fact has the form:
  //   obj_1[a_1, ..., a_m] == obj_2[b_1, ..., b_n]
  // then for each composite index i, we add a fact of the form:
  //   obj_1[a_1, ..., a_m, i] == obj_2[b_1, ..., b_n, i]
  //
  // However, to avoid adding a large number of synonym facts e.g. in the case
  // of arrays, we bound the number of composite elements to which this is
  // applied.  Nevertheless, we always add a synonym fact for the final
  // components, as this may be an interesting edge case.

  // The bound on the number of indices of the composite pair to note as being
  // synonymous.
  const uint32_t kCompositeElementBound = 10;

  for (uint32_t i = 0; i < num_composite_elements;) {
    std::vector<uint32_t> extended_indices1 =
        fuzzerutil::RepeatedFieldToVector(dd1.index());
    extended_indices1.push_back(i);
    std::vector<uint32_t> extended_indices2 =
        fuzzerutil::RepeatedFieldToVector(dd2.index());
    extended_indices2.push_back(i);
    AddDataSynonymFactRecursive(
        MakeDataDescriptor(dd1.object(), extended_indices1),
        MakeDataDescriptor(dd2.object(), extended_indices2));

    if (i < kCompositeElementBound - 1 || i == num_composite_elements - 1) {
      // We have not reached the bound yet, or have already skipped ahead to the
      // last element, so increment the loop counter as standard.
      i++;
    } else {
      // We have reached the bound, so skip ahead to the last element.
      assert(i == kCompositeElementBound - 1);
      i = num_composite_elements - 1;
    }
  }
}

void DataSynonymAndIdEquationFacts::ComputeClosureOfFacts(
    uint32_t maximum_equivalence_class_size) {
  // Suppose that obj_1[a_1, ..., a_m] and obj_2[b_1, ..., b_n] are distinct
  // data descriptors that describe objects of the same composite type, and that
  // the composite type is comprised of k components.
  //
  // For example, if m is a mat4x4 and v a vec4, we might consider:
  //   m[2]: describes the 2nd column of m, a vec4
  //   v[]: describes all of v, a vec4
  //
  // Suppose that we know, for every 0 <= i < k, that the fact:
  //   obj_1[a_1, ..., a_m, i] == obj_2[b_1, ..., b_n, i]
  // holds - i.e. that the children of the two data descriptors are synonymous.
  //
  // Then we can conclude that:
  //   obj_1[a_1, ..., a_m] == obj_2[b_1, ..., b_n]
  // holds.
  //
  // For instance, if we have the facts:
  //   m[2, 0] == v[0]
  //   m[2, 1] == v[1]
  //   m[2, 2] == v[2]
  //   m[2, 3] == v[3]
  // then we can conclude that:
  //   m[2] == v.
  //
  // This method repeatedly searches the equivalence relation of data
  // descriptors, deducing and adding such facts, until a pass over the
  // relation leads to no further facts being deduced.

  // The method relies on working with pairs of data descriptors, and in
  // particular being able to hash and compare such pairs.

  using DataDescriptorPair =
      std::pair<protobufs::DataDescriptor, protobufs::DataDescriptor>;

  struct DataDescriptorPairHash {
    std::size_t operator()(const DataDescriptorPair& pair) const {
      return DataDescriptorHash()(&pair.first) ^
             DataDescriptorHash()(&pair.second);
    }
  };

  struct DataDescriptorPairEquals {
    bool operator()(const DataDescriptorPair& first,
                    const DataDescriptorPair& second) const {
      return (DataDescriptorEquals()(&first.first, &second.first) &&
              DataDescriptorEquals()(&first.second, &second.second)) ||
             (DataDescriptorEquals()(&first.first, &second.second) &&
              DataDescriptorEquals()(&first.second, &second.first));
    }
  };

  // This map records, for a given pair of composite data descriptors of the
  // same type, all the indices at which the data descriptors are known to be
  // synonymous.  A pair is a key to this map only if we have observed that
  // the pair are synonymous at *some* index, but not at *all* indices.
  // Once we find that a pair of data descriptors are equivalent at all indices
  // we record the fact that they are synonymous and remove them from the map.
  //
  // Using the m and v example from above, initially the pair (m[2], v) would
  // not be a key to the map.  If we find that m[2, 2] == v[2] holds, we would
  // add an entry:
  //   (m[2], v) -> [false, false, true, false]
  // to record that they are synonymous at index 2.  If we then find that
  // m[2, 0] == v[0] holds, we would update this entry to:
  //   (m[2], v) -> [true, false, true, false]
  // If we then find that m[2, 3] == v[3] holds, we would update this entry to:
  //   (m[2], v) -> [true, false, true, true]
  // Finally, if we then find that m[2, 1] == v[1] holds, which would make the
  // boolean vector true at every index, we would add the fact:
  //   m[2] == v
  // to the equivalence relation and remove (m[2], v) from the map.
  std::unordered_map<DataDescriptorPair, std::vector<bool>,
                     DataDescriptorPairHash, DataDescriptorPairEquals>
      candidate_composite_synonyms;

  // We keep looking for new facts until we perform a complete pass over the
  // equivalence relation without finding any new facts.
  while (closure_computation_required_) {
    // We have not found any new facts yet during this pass; we set this to
    // 'true' if we do find a new fact.
    closure_computation_required_ = false;

    // Consider each class in the equivalence relation.
    for (auto representative :
         synonymous_.GetEquivalenceClassRepresentatives()) {
      auto equivalence_class = synonymous_.GetEquivalenceClass(*representative);

      if (equivalence_class.size() > maximum_equivalence_class_size) {
        // This equivalence class is larger than the maximum size we are willing
        // to consider, so we skip it.  This potentially leads to missed fact
        // deductions, but avoids excessive runtime for closure computation.
        continue;
      }

      // Consider every data descriptor in the equivalence class.
      for (auto dd1_it = equivalence_class.begin();
           dd1_it != equivalence_class.end(); ++dd1_it) {
        // If this data descriptor has no indices then it does not have the form
        // obj_1[a_1, ..., a_m, i], so move on.
        auto dd1 = *dd1_it;
        if (dd1->index_size() == 0) {
          continue;
        }

        // Consider every other data descriptor later in the equivalence class
        // (due to symmetry, there is no need to compare with previous data
        // descriptors).
        auto dd2_it = dd1_it;
        for (++dd2_it; dd2_it != equivalence_class.end(); ++dd2_it) {
          auto dd2 = *dd2_it;
          // If this data descriptor has no indices then it does not have the
          // form obj_2[b_1, ..., b_n, i], so move on.
          if (dd2->index_size() == 0) {
            continue;
          }

          // At this point we know that:
          // - |dd1| has the form obj_1[a_1, ..., a_m, i]
          // - |dd2| has the form obj_2[b_1, ..., b_n, j]
          assert(dd1->index_size() > 0 && dd2->index_size() > 0 &&
                 "Control should not reach here if either data descriptor has "
                 "no indices.");

          // We are only interested if i == j.
          if (dd1->index(dd1->index_size() - 1) !=
              dd2->index(dd2->index_size() - 1)) {
            continue;
          }

          const uint32_t common_final_index = dd1->index(dd1->index_size() - 1);

          // Make data descriptors |dd1_prefix| and |dd2_prefix| for
          //   obj_1[a_1, ..., a_m]
          // and
          //   obj_2[b_1, ..., b_n]
          // These are the two data descriptors we might be getting closer to
          // deducing as being synonymous, due to knowing that they are
          // synonymous when extended by a particular index.
          protobufs::DataDescriptor dd1_prefix;
          dd1_prefix.set_object(dd1->object());
          for (uint32_t i = 0; i < static_cast<uint32_t>(dd1->index_size() - 1);
               i++) {
            dd1_prefix.add_index(dd1->index(i));
          }
          protobufs::DataDescriptor dd2_prefix;
          dd2_prefix.set_object(dd2->object());
          for (uint32_t i = 0; i < static_cast<uint32_t>(dd2->index_size() - 1);
               i++) {
            dd2_prefix.add_index(dd2->index(i));
          }
          assert(!DataDescriptorEquals()(&dd1_prefix, &dd2_prefix) &&
                 "By construction these prefixes should be different.");

          // If we already know that these prefixes are synonymous, move on.
          if (synonymous_.Exists(dd1_prefix) &&
              synonymous_.Exists(dd2_prefix) &&
              synonymous_.IsEquivalent(dd1_prefix, dd2_prefix)) {
            continue;
          }
          if (!ObjectStillExists(*dd1) || !ObjectStillExists(*dd2)) {
            // The objects are not both available in the module, so we cannot
            // investigate the types of the associated data descriptors; we need
            // to move on.
            continue;
          }
          // Get the type of obj_1
          auto dd1_root_type_id =
              fuzzerutil::GetTypeId(ir_context_, dd1->object());
          // Use this type, together with a_1, ..., a_m, to get the type of
          // obj_1[a_1, ..., a_m].
          auto dd1_prefix_type = fuzzerutil::WalkCompositeTypeIndices(
              ir_context_, dd1_root_type_id, dd1_prefix.index());

          // Similarly, get the type of obj_2 and use it to get the type of
          // obj_2[b_1, ..., b_n].
          auto dd2_root_type_id =
              fuzzerutil::GetTypeId(ir_context_, dd2->object());
          auto dd2_prefix_type = fuzzerutil::WalkCompositeTypeIndices(
              ir_context_, dd2_root_type_id, dd2_prefix.index());

          // If the types of dd1_prefix and dd2_prefix are not the same, they
          // cannot be synonymous.
          if (dd1_prefix_type != dd2_prefix_type) {
            continue;
          }

          // At this point, we know we have synonymous data descriptors of the
          // form:
          //   obj_1[a_1, ..., a_m, i]
          //   obj_2[b_1, ..., b_n, i]
          // with the same last_index i, such that:
          //   obj_1[a_1, ..., a_m]
          // and
          //   obj_2[b_1, ..., b_n]
          // have the same type.

          // Work out how many components there are in the (common) commposite
          // type associated with obj_1[a_1, ..., a_m] and obj_2[b_1, ..., b_n].
          // This depends on whether the composite type is array, matrix, struct
          // or vector.
          uint32_t num_components_in_composite;
          auto composite_type =
              ir_context_->get_type_mgr()->GetType(dd1_prefix_type);
          auto composite_type_instruction =
              ir_context_->get_def_use_mgr()->GetDef(dd1_prefix_type);
          if (composite_type->AsArray()) {
            num_components_in_composite = fuzzerutil::GetArraySize(
                *composite_type_instruction, ir_context_);
            if (num_components_in_composite == 0) {
              // This indicates that the array has an unknown size, in which
              // case we cannot be sure we have matched all of its elements with
              // synonymous elements of another array.
              continue;
            }
          } else if (composite_type->AsMatrix()) {
            num_components_in_composite =
                composite_type->AsMatrix()->element_count();
          } else if (composite_type->AsStruct()) {
            num_components_in_composite = fuzzerutil::GetNumberOfStructMembers(
                *composite_type_instruction);
          } else {
            assert(composite_type->AsVector());
            num_components_in_composite =
                composite_type->AsVector()->element_count();
          }

          // We are one step closer to being able to say that |dd1_prefix| and
          // |dd2_prefix| are synonymous.
          DataDescriptorPair candidate_composite_synonym(dd1_prefix,
                                                         dd2_prefix);

          // We look up what we already know about this pair.
          auto existing_entry =
              candidate_composite_synonyms.find(candidate_composite_synonym);

          if (existing_entry == candidate_composite_synonyms.end()) {
            // If this is the first time we have seen the pair, we make a vector
            // of size |num_components_in_composite| that is 'true' at the
            // common final index associated with |dd1| and |dd2|, and 'false'
            // everywhere else, and register this vector as being associated
            // with the pair.
            std::vector<bool> entry;
            for (uint32_t i = 0; i < num_components_in_composite; i++) {
              entry.push_back(i == common_final_index);
            }
            candidate_composite_synonyms[candidate_composite_synonym] = entry;
            existing_entry =
                candidate_composite_synonyms.find(candidate_composite_synonym);
          } else {
            // We have seen this pair of data descriptors before, and we now
            // know that they are synonymous at one further index, so we
            // update the entry to record that.
            existing_entry->second[common_final_index] = true;
          }
          assert(existing_entry != candidate_composite_synonyms.end());

          // Check whether |dd1_prefix| and |dd2_prefix| are now known to match
          // at every sub-component.
          bool all_components_match = true;
          for (uint32_t i = 0; i < num_components_in_composite; i++) {
            if (!existing_entry->second[i]) {
              all_components_match = false;
              break;
            }
          }
          if (all_components_match) {
            // The two prefixes match on all sub-components, so we know that
            // they are synonymous.  We add this fact *non-recursively*, as we
            // have deduced that |dd1_prefix| and |dd2_prefix| are synonymous
            // by observing that all their sub-components are already
            // synonymous.
            assert(DataDescriptorsAreWellFormedAndComparable(dd1_prefix,
                                                             dd2_prefix));
            MakeEquivalent(dd1_prefix, dd2_prefix);
            // Now that we know this pair of data descriptors are synonymous,
            // there is no point recording how close they are to being
            // synonymous.
            candidate_composite_synonyms.erase(candidate_composite_synonym);
          }
        }
      }
    }
  }
}

void DataSynonymAndIdEquationFacts::MakeEquivalent(
    const protobufs::DataDescriptor& dd1,
    const protobufs::DataDescriptor& dd2) {
  // Register the data descriptors if they are not already known to the
  // equivalence relation.
  RegisterDataDescriptor(dd1);
  RegisterDataDescriptor(dd2);

  if (synonymous_.IsEquivalent(dd1, dd2)) {
    // The data descriptors are already known to be equivalent, so there is
    // nothing to do.
    return;
  }

  // We must make the data descriptors equivalent, and also make sure any
  // equation facts known about their representatives are merged.

  // Record the original equivalence class representatives of the data
  // descriptors.
  auto dd1_original_representative = synonymous_.Find(&dd1);
  auto dd2_original_representative = synonymous_.Find(&dd2);

  // Make the data descriptors equivalent.
  synonymous_.MakeEquivalent(dd1, dd2);
  // As we have updated the equivalence relation, we might be able to deduce
  // more facts by performing a closure computation, so we record that such a
  // computation is required.
  closure_computation_required_ = true;

  // At this point, exactly one of |dd1_original_representative| and
  // |dd2_original_representative| will be the representative of the combined
  // equivalence class.  We work out which one of them is still the class
  // representative and which one is no longer the class representative.

  auto still_representative = synonymous_.Find(dd1_original_representative) ==
                                      dd1_original_representative
                                  ? dd1_original_representative
                                  : dd2_original_representative;
  auto no_longer_representative =
      still_representative == dd1_original_representative
          ? dd2_original_representative
          : dd1_original_representative;

  assert(no_longer_representative != still_representative &&
         "The current and former representatives cannot be the same.");

  // We now need to add all equations about |no_longer_representative| to the
  // set of equations known about |still_representative|.

  // Get the equations associated with |no_longer_representative|.
  auto no_longer_representative_id_equations =
      id_equations_.find(no_longer_representative);
  if (no_longer_representative_id_equations != id_equations_.end()) {
    // There are some equations to transfer.  There might not yet be any
    // equations about |still_representative|; create an empty set of equations
    // if this is the case.
    if (!id_equations_.count(still_representative)) {
      id_equations_.insert({still_representative, OperationSet()});
    }
    auto still_representative_id_equations =
        id_equations_.find(still_representative);
    assert(still_representative_id_equations != id_equations_.end() &&
           "At this point there must be a set of equations.");
    // Add all the equations known about |no_longer_representative| to the set
    // of equations known about |still_representative|.
    still_representative_id_equations->second.insert(
        no_longer_representative_id_equations->second.begin(),
        no_longer_representative_id_equations->second.end());
  }
  // Delete the no longer-relevant equations about |no_longer_representative|.
  id_equations_.erase(no_longer_representative);
}

const protobufs::DataDescriptor*
DataSynonymAndIdEquationFacts::RegisterDataDescriptor(
    const protobufs::DataDescriptor& dd) {
  return synonymous_.Exists(dd) ? synonymous_.Find(&dd)
                                : synonymous_.Register(dd);
}

bool DataSynonymAndIdEquationFacts::DataDescriptorsAreWellFormedAndComparable(
    const protobufs::DataDescriptor& dd1,
    const protobufs::DataDescriptor& dd2) const {
  if (!ObjectStillExists(dd1) || !ObjectStillExists(dd2)) {
    // We trivially return true if one or other of the objects associated with
    // the data descriptors is gone.
    return true;
  }

  auto end_type_id_1 = fuzzerutil::WalkCompositeTypeIndices(
      ir_context_, fuzzerutil::GetTypeId(ir_context_, dd1.object()),
      dd1.index());
  auto end_type_id_2 = fuzzerutil::WalkCompositeTypeIndices(
      ir_context_, fuzzerutil::GetTypeId(ir_context_, dd2.object()),
      dd2.index());
  // The end types of the data descriptors must exist.
  if (end_type_id_1 == 0 || end_type_id_2 == 0) {
    return false;
  }
  // Neither end type is allowed to be void.
  if (ir_context_->get_def_use_mgr()->GetDef(end_type_id_1)->opcode() ==
          spv::Op::OpTypeVoid ||
      ir_context_->get_def_use_mgr()->GetDef(end_type_id_2)->opcode() ==
          spv::Op::OpTypeVoid) {
    return false;
  }
  // If the end types are the same, the data descriptors are comparable.
  if (end_type_id_1 == end_type_id_2) {
    return true;
  }
  // Otherwise they are only comparable if they are integer scalars or integer
  // vectors that differ only in signedness.

  // Get both types.
  const auto* type_a = ir_context_->get_type_mgr()->GetType(end_type_id_1);
  const auto* type_b = ir_context_->get_type_mgr()->GetType(end_type_id_2);
  assert(type_a && type_b && "Data descriptors have invalid type(s)");

  // If both types are numerical or vectors of numerical components, then they
  // are compatible if they have the same number of components and the same bit
  // count per component.

  if (type_a->AsVector() && type_b->AsVector()) {
    const auto* vector_a = type_a->AsVector();
    const auto* vector_b = type_b->AsVector();

    if (vector_a->element_count() != vector_b->element_count() ||
        vector_a->element_type()->AsBool() ||
        vector_b->element_type()->AsBool()) {
      // The case where both vectors have boolean elements and the same number
      // of components is handled by the direct equality check earlier.
      // You can't have multiple identical boolean vector types.
      return false;
    }

    type_a = vector_a->element_type();
    type_b = vector_b->element_type();
  }

  auto get_bit_count_for_numeric_type =
      [](const opt::analysis::Type& type) -> uint32_t {
    if (const auto* integer = type.AsInteger()) {
      return integer->width();
    } else if (const auto* floating = type.AsFloat()) {
      return floating->width();
    } else {
      assert(false && "|type| must be a numerical type");
      return 0;
    }
  };

  // Checks that both |type_a| and |type_b| are either numerical or vectors of
  // numerical components and have the same number of bits.
  return (type_a->AsInteger() || type_a->AsFloat()) &&
         (type_b->AsInteger() || type_b->AsFloat()) &&
         (get_bit_count_for_numeric_type(*type_a) ==
          get_bit_count_for_numeric_type(*type_b));
}

std::vector<const protobufs::DataDescriptor*>
DataSynonymAndIdEquationFacts::GetSynonymsForId(uint32_t id) const {
  return GetSynonymsForDataDescriptor(MakeDataDescriptor(id, {}));
}

std::vector<const protobufs::DataDescriptor*>
DataSynonymAndIdEquationFacts::GetSynonymsForDataDescriptor(
    const protobufs::DataDescriptor& data_descriptor) const {
  std::vector<const protobufs::DataDescriptor*> result;
  if (synonymous_.Exists(data_descriptor)) {
    for (auto dd : synonymous_.GetEquivalenceClass(data_descriptor)) {
      // There may be data descriptors in the equivalence class whose base
      // objects have been removed from the module.  We do not expose these
      // data descriptors to clients of the fact manager.
      if (ObjectStillExists(*dd)) {
        result.push_back(dd);
      }
    }
  }
  return result;
}

std::vector<uint32_t>
DataSynonymAndIdEquationFacts::GetIdsForWhichSynonymsAreKnown() const {
  std::vector<uint32_t> result;
  for (auto& data_descriptor : synonymous_.GetAllKnownValues()) {
    // We skip any data descriptors whose base objects no longer exist in the
    // module, and we restrict attention to data descriptors for plain ids,
    // which have no indices.
    if (ObjectStillExists(*data_descriptor) &&
        data_descriptor->index().empty()) {
      result.push_back(data_descriptor->object());
    }
  }
  return result;
}

std::vector<const protobufs::DataDescriptor*>
DataSynonymAndIdEquationFacts::GetAllKnownSynonyms() const {
  std::vector<const protobufs::DataDescriptor*> result;
  for (const auto* dd : synonymous_.GetAllKnownValues()) {
    if (ObjectStillExists(*dd)) {
      result.push_back(dd);
    }
  }
  return result;
}

bool DataSynonymAndIdEquationFacts::IsSynonymous(
    const protobufs::DataDescriptor& data_descriptor1,
    const protobufs::DataDescriptor& data_descriptor2) const {
  return synonymous_.Exists(data_descriptor1) &&
         synonymous_.Exists(data_descriptor2) &&
         synonymous_.IsEquivalent(data_descriptor1, data_descriptor2);
}

bool DataSynonymAndIdEquationFacts::ObjectStillExists(
    const protobufs::DataDescriptor& dd) const {
  return ir_context_->get_def_use_mgr()->GetDef(dd.object()) != nullptr;
}

}  // namespace fact_manager
}  // namespace fuzz
}  // namespace spvtools
