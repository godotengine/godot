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

#ifndef SOURCE_FUZZ_FACT_MANAGER_DATA_SYNONYM_AND_ID_EQUATION_FACTS_H_
#define SOURCE_FUZZ_FACT_MANAGER_DATA_SYNONYM_AND_ID_EQUATION_FACTS_H_

#include <unordered_set>
#include <vector>

#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/equivalence_relation.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {
namespace fact_manager {

// Forward reference to the DeadBlockFacts class.
class DeadBlockFacts;
// Forward reference to the IrrelevantValueFacts class.
class IrrelevantValueFacts;

// The purpose of this class is to group the fields and data used to represent
// facts about data synonyms and id equations.
class DataSynonymAndIdEquationFacts {
 public:
  explicit DataSynonymAndIdEquationFacts(opt::IRContext* ir_context);

  // See method in FactManager which delegates to this method. Returns true if
  // neither |fact.data1()| nor |fact.data2()| contain an
  // irrelevant id. Otherwise, returns false. |dead_block_facts| and
  // |irrelevant_value_facts| are passed for consistency checks.
  bool MaybeAddFact(const protobufs::FactDataSynonym& fact,
                    const DeadBlockFacts& dead_block_facts,
                    const IrrelevantValueFacts& irrelevant_value_facts);

  // See method in FactManager which delegates to this method. Returns true if
  // neither |fact.lhs_id()| nor any of |fact.rhs_id()| is irrelevant. Returns
  // false otherwise. |dead_block_facts| and |irrelevant_value_facts| are passed
  // for consistency checks.
  bool MaybeAddFact(const protobufs::FactIdEquation& fact,
                    const DeadBlockFacts& dead_block_facts,
                    const IrrelevantValueFacts& irrelevant_value_facts);

  // See method in FactManager which delegates to this method.
  std::vector<const protobufs::DataDescriptor*> GetSynonymsForId(
      uint32_t id) const;

  // See method in FactManager which delegates to this method.
  std::vector<const protobufs::DataDescriptor*> GetSynonymsForDataDescriptor(
      const protobufs::DataDescriptor& data_descriptor) const;

  // See method in FactManager which delegates to this method.
  std::vector<uint32_t> GetIdsForWhichSynonymsAreKnown() const;

  // See method in FactManager which delegates to this method.
  std::vector<const protobufs::DataDescriptor*> GetAllKnownSynonyms() const;

  // See method in FactManager which delegates to this method.
  bool IsSynonymous(const protobufs::DataDescriptor& data_descriptor1,
                    const protobufs::DataDescriptor& data_descriptor2) const;

  // See method in FactManager which delegates to this method.
  void ComputeClosureOfFacts(uint32_t maximum_equivalence_class_size);

 private:
  // This helper struct represents the right hand side of an equation as an
  // operator applied to a number of data descriptor operands.
  struct Operation {
    spv::Op opcode;
    std::vector<const protobufs::DataDescriptor*> operands;
  };

  // Hashing for operations, to allow deterministic unordered sets.
  struct OperationHash {
    size_t operator()(const Operation& operation) const;
  };

  // Equality for operations, to allow deterministic unordered sets.
  struct OperationEquals {
    bool operator()(const Operation& first, const Operation& second) const;
  };

  using OperationSet =
      std::unordered_set<Operation, OperationHash, OperationEquals>;

  // Adds the synonym |dd1| = |dd2| to the set of managed facts, and recurses
  // into sub-components of the data descriptors, if they are composites, to
  // record that their components are pairwise-synonymous.
  void AddDataSynonymFactRecursive(const protobufs::DataDescriptor& dd1,
                                   const protobufs::DataDescriptor& dd2);

  // Computes various corollary facts from the data descriptor |dd| if members
  // of its equivalence class participate in equation facts with OpConvert*
  // opcodes. The descriptor should be registered in the equivalence relation.
  void ComputeConversionDataSynonymFacts(const protobufs::DataDescriptor& dd);

  // Recurses into sub-components of the data descriptors, if they are
  // composites, to record that their components are pairwise-synonymous.
  void ComputeCompositeDataSynonymFacts(const protobufs::DataDescriptor& dd1,
                                        const protobufs::DataDescriptor& dd2);

  // Records the fact that |dd1| and |dd2| are equivalent, and merges the sets
  // of equations that are known about them.
  void MakeEquivalent(const protobufs::DataDescriptor& dd1,
                      const protobufs::DataDescriptor& dd2);

  // Registers a data descriptor in the equivalence relation if it hasn't been
  // registered yet, and returns its representative.
  const protobufs::DataDescriptor* RegisterDataDescriptor(
      const protobufs::DataDescriptor& dd);

  // Trivially returns true if either |dd1| or |dd2|'s objects are not present
  // in the module.
  //
  // Otherwise, returns true if and only if |dd1| and |dd2| are valid data
  // descriptors whose associated data have compatible types. Two types are
  // compatible if:
  // - they are the same
  // - they both are numerical or vectors of numerical components with the same
  //   number of components and the same bit count per component
  bool DataDescriptorsAreWellFormedAndComparable(
      const protobufs::DataDescriptor& dd1,
      const protobufs::DataDescriptor& dd2) const;

  OperationSet GetEquations(const protobufs::DataDescriptor* lhs) const;

  // Requires that |lhs_dd| and every element of |rhs_dds| is present in the
  // |synonymous_| equivalence relation, but is not necessarily its own
  // representative.  Records the fact that the equation
  // "|lhs_dd| |opcode| |rhs_dds_non_canonical|" holds, and adds any
  // corollaries, in the form of data synonym or equation facts, that follow
  // from this and other known facts.
  void AddEquationFactRecursive(
      const protobufs::DataDescriptor& lhs_dd, spv::Op opcode,
      const std::vector<const protobufs::DataDescriptor*>& rhs_dds);

  // Returns true if and only if |dd.object()| still exists in the module.
  bool ObjectStillExists(const protobufs::DataDescriptor& dd) const;

  // The data descriptors that are known to be synonymous with one another are
  // captured by this equivalence relation.
  EquivalenceRelation<protobufs::DataDescriptor, DataDescriptorHash,
                      DataDescriptorEquals>
      synonymous_;

  // When a new synonym fact is added, it may be possible to deduce further
  // synonym facts by computing a closure of all known facts.  However, this is
  // an expensive operation, so it should be performed sparingly and only there
  // is some chance of new facts being deduced.  This boolean tracks whether a
  // closure computation is required - i.e., whether a new fact has been added
  // since the last time such a computation was performed.
  bool closure_computation_required_ = false;

  // Represents a set of equations on data descriptors as a map indexed by
  // left-hand-side, mapping a left-hand-side to a set of operations, each of
  // which (together with the left-hand-side) defines an equation.
  //
  // All data descriptors occurring in equations are required to be present in
  // the |synonymous_| equivalence relation, and to be their own representatives
  // in that relation.
  std::unordered_map<const protobufs::DataDescriptor*, OperationSet>
      id_equations_;

  // Pointer to the SPIR-V module we store facts about.
  opt::IRContext* ir_context_;
};

}  // namespace fact_manager
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FACT_MANAGER_DATA_SYNONYM_AND_ID_EQUATION_FACTS_H_
