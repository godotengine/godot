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

#ifndef SOURCE_FUZZ_FACT_MANAGER_FACT_MANAGER_H_
#define SOURCE_FUZZ_FACT_MANAGER_FACT_MANAGER_H_

#include <set>
#include <utility>
#include <vector>

#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/fact_manager/constant_uniform_facts.h"
#include "source/fuzz/fact_manager/data_synonym_and_id_equation_facts.h"
#include "source/fuzz/fact_manager/dead_block_facts.h"
#include "source/fuzz/fact_manager/irrelevant_value_facts.h"
#include "source/fuzz/fact_manager/livesafe_function_facts.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/constants.h"

namespace spvtools {
namespace fuzz {

// Keeps track of facts about the module being transformed on which the fuzzing
// process can depend. Some initial facts can be provided, for example about
// guarantees on the values of inputs to SPIR-V entry points. Transformations
// may then rely on these facts, can add further facts that they establish.
// Facts are intended to be simple properties that either cannot be deduced from
// the module (such as properties that are guaranteed to hold for entry point
// inputs), or that are established by transformations, likely to be useful for
// future transformations, and not completely trivial to deduce straight from
// the module.
class FactManager {
 public:
  explicit FactManager(opt::IRContext* ir_context);

  // Adds all the facts from |facts|, checking them for validity with respect to
  // |ir_context_|. Warnings about invalid facts are communicated via
  // |message_consumer|; such facts are otherwise ignored.
  void AddInitialFacts(const MessageConsumer& message_consumer,
                       const protobufs::FactSequence& facts);

  // Checks the fact for validity with respect to |ir_context_|. Returns false,
  // with no side effects, if the fact is invalid. Otherwise adds |fact| to the
  // fact manager.
  bool MaybeAddFact(const protobufs::Fact& fact);

  // Record the fact that |data1| and |data2| are synonymous. Neither |data1|
  // nor |data2| may contain an irrelevant id.
  void AddFactDataSynonym(const protobufs::DataDescriptor& data1,
                          const protobufs::DataDescriptor& data2);

  // Records the fact that |block_id| is dead. |block_id| must be a result id
  // of some OpLabel instruction in the |ir_context_|.
  void AddFactBlockIsDead(uint32_t block_id);

  // Records the fact that |function_id| is livesafe. |function_id| must be a
  // result id of some non-entry-point function in the module.
  void AddFactFunctionIsLivesafe(uint32_t function_id);

  // Records the fact that the value of the pointee associated with |pointer_id|
  // is irrelevant: it does not affect the observable behaviour of the module.
  // |pointer_id| must exist in the module and actually be a pointer.
  void AddFactValueOfPointeeIsIrrelevant(uint32_t pointer_id);

  // Records a fact that the |result_id| is irrelevant (i.e. it doesn't affect
  // the semantics of the module).
  // |result_id| must exist in the module and it may not be a pointer.
  void AddFactIdIsIrrelevant(uint32_t result_id);

  // Records the fact that |lhs_id| is defined by the equation:
  //
  //   |lhs_id| = |opcode| |rhs_id[0]| ... |rhs_id[N-1]|
  //
  // Neither |lhs_id| nor any of |rhs_id| may be irrelevant.
  void AddFactIdEquation(uint32_t lhs_id, spv::Op opcode,
                         const std::vector<uint32_t>& rhs_id);

  // Inspects all known facts and adds corollary facts; e.g. if we know that
  // a.x == b.x and a.y == b.y, where a and b have vec2 type, we can record
  // that a == b holds.
  //
  // This method is expensive, and should only be called (by applying a
  // transformation) at the start of a fuzzer pass that depends on data
  // synonym facts, rather than calling it every time a new data synonym fact
  // is added.
  //
  // The parameter |maximum_equivalence_class_size| specifies the size beyond
  // which equivalence classes should not be mined for new facts, to avoid
  // excessively-long closure computations.
  void ComputeClosureOfFacts(uint32_t maximum_equivalence_class_size);

  // The fact manager is responsible for managing a few distinct categories of
  // facts. In principle there could be different fact managers for each kind
  // of fact, but in practice providing one 'go to' place for facts is
  // convenient.  To keep some separation, the public methods of the fact
  // manager should be grouped according to the kind of fact to which they
  // relate.

  //==============================
  // Querying facts about uniform constants

  // Provides the distinct type ids for which at least one  "constant ==
  // uniform element" fact is known.
  std::vector<uint32_t> GetTypesForWhichUniformValuesAreKnown() const;

  // Provides distinct constant ids with type |type_id| for which at least one
  // "constant == uniform element" fact is known.  If multiple identically-
  // valued constants are relevant, only one will appear in the sequence.
  std::vector<uint32_t> GetConstantsAvailableFromUniformsForType(
      uint32_t type_id) const;

  // Provides details of all uniform elements that are known to be equal to the
  // constant associated with |constant_id| in |ir_context_|.
  std::vector<protobufs::UniformBufferElementDescriptor>
  GetUniformDescriptorsForConstant(uint32_t constant_id) const;

  // Returns the id of a constant whose value is known to match that of
  // |uniform_descriptor|, and whose type matches the type of the uniform
  // element.  If multiple such constant is exist, the one that is returned
  // is arbitrary.  Returns 0 if no such constant id exists.
  uint32_t GetConstantFromUniformDescriptor(
      const protobufs::UniformBufferElementDescriptor& uniform_descriptor)
      const;

  // Returns all "constant == uniform element" facts known to the fact
  // manager, pairing each fact with id of the type that is associated with
  // both the constant and the uniform element.
  const std::vector<std::pair<protobufs::FactConstantUniform, uint32_t>>&
  GetConstantUniformFactsAndTypes() const;

  // End of uniform constant facts
  //==============================

  //==============================
  // Querying facts about id synonyms

  // Returns every id for which a fact of the form "this id is synonymous with
  // this piece of data" is known.
  std::vector<uint32_t> GetIdsForWhichSynonymsAreKnown() const;

  // Returns a vector of all data descriptors that participate in DataSynonym
  // facts. All descriptors are guaranteed to exist in the |ir_context_|.
  std::vector<const protobufs::DataDescriptor*> GetAllSynonyms() const;

  // Returns the equivalence class of all known synonyms of |id|, or an empty
  // set if no synonyms are known.
  std::vector<const protobufs::DataDescriptor*> GetSynonymsForId(
      uint32_t id) const;

  // Returns the equivalence class of all known synonyms of |data_descriptor|,
  // or empty if no synonyms are known.
  std::vector<const protobufs::DataDescriptor*> GetSynonymsForDataDescriptor(
      const protobufs::DataDescriptor& data_descriptor) const;

  // Returns true if and only if |data_descriptor1| and |data_descriptor2| are
  // known to be synonymous.
  bool IsSynonymous(const protobufs::DataDescriptor& data_descriptor1,
                    const protobufs::DataDescriptor& data_descriptor2) const;

  // End of id synonym facts
  //==============================

  //==============================
  // Querying facts about dead blocks

  // Returns true if and only if |block_id| is the id of a block known to be
  // dynamically unreachable.
  bool BlockIsDead(uint32_t block_id) const;

  // End of dead block facts
  //==============================

  //==============================
  // Querying facts about livesafe function

  // Returns true if and only if |function_id| is the id of a function known
  // to be livesafe.
  bool FunctionIsLivesafe(uint32_t function_id) const;

  // End of dead livesafe function facts
  //==============================

  //==============================
  // Querying facts about irrelevant values

  // Returns true if and only if the value of the pointee associated with
  // |pointer_id| is irrelevant.
  bool PointeeValueIsIrrelevant(uint32_t pointer_id) const;

  // Returns true if there exists a fact that the |result_id| is irrelevant or
  // if |result_id| is declared in a block that has been declared dead.
  bool IdIsIrrelevant(uint32_t result_id) const;

  // Returns a set of all the ids which have been declared irrelevant, or which
  // have been declared inside a dead block.
  std::unordered_set<uint32_t> GetIrrelevantIds() const;

  // End of irrelevant value facts
  //==============================

 private:
  // Keep these in alphabetical order.
  fact_manager::ConstantUniformFacts constant_uniform_facts_;
  fact_manager::DataSynonymAndIdEquationFacts
      data_synonym_and_id_equation_facts_;
  fact_manager::DeadBlockFacts dead_block_facts_;
  fact_manager::LivesafeFunctionFacts livesafe_function_facts_;
  fact_manager::IrrelevantValueFacts irrelevant_value_facts_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FACT_MANAGER_FACT_MANAGER_H_
