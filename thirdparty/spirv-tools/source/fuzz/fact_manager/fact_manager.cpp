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

#include "fact_manager.h"

#include <sstream>
#include <unordered_map>

#include "source/fuzz/uniform_buffer_element_descriptor.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {
namespace {

std::string ToString(const protobufs::FactConstantUniform& fact) {
  std::stringstream stream;
  stream << "(" << fact.uniform_buffer_element_descriptor().descriptor_set()
         << ", " << fact.uniform_buffer_element_descriptor().binding() << ")[";

  bool first = true;
  for (auto index : fact.uniform_buffer_element_descriptor().index()) {
    if (first) {
      first = false;
    } else {
      stream << ", ";
    }
    stream << index;
  }

  stream << "] == [";

  first = true;
  for (auto constant_word : fact.constant_word()) {
    if (first) {
      first = false;
    } else {
      stream << ", ";
    }
    stream << constant_word;
  }

  stream << "]";
  return stream.str();
}

std::string ToString(const protobufs::FactDataSynonym& fact) {
  std::stringstream stream;
  stream << fact.data1() << " = " << fact.data2();
  return stream.str();
}

std::string ToString(const protobufs::FactIdEquation& fact) {
  std::stringstream stream;
  stream << fact.lhs_id();
  stream << " " << fact.opcode();
  for (auto rhs_id : fact.rhs_id()) {
    stream << " " << rhs_id;
  }
  return stream.str();
}

std::string ToString(const protobufs::Fact& fact) {
  switch (fact.fact_case()) {
    case protobufs::Fact::kConstantUniformFact:
      return ToString(fact.constant_uniform_fact());
    case protobufs::Fact::kDataSynonymFact:
      return ToString(fact.data_synonym_fact());
    case protobufs::Fact::kIdEquationFact:
      return ToString(fact.id_equation_fact());
    default:
      assert(false && "Stringification not supported for this fact.");
      return "";
  }
}

}  // namespace

FactManager::FactManager(opt::IRContext* ir_context)
    : constant_uniform_facts_(ir_context),
      data_synonym_and_id_equation_facts_(ir_context),
      dead_block_facts_(ir_context),
      livesafe_function_facts_(ir_context),
      irrelevant_value_facts_(ir_context) {}

void FactManager::AddInitialFacts(const MessageConsumer& message_consumer,
                                  const protobufs::FactSequence& facts) {
  for (auto& fact : facts.fact()) {
    if (!MaybeAddFact(fact)) {
      auto message = "Invalid fact " + ToString(fact) + " ignored.";
      message_consumer(SPV_MSG_WARNING, nullptr, {}, message.c_str());
    }
  }
}

bool FactManager::MaybeAddFact(const fuzz::protobufs::Fact& fact) {
  switch (fact.fact_case()) {
    case protobufs::Fact::kBlockIsDeadFact:
      return dead_block_facts_.MaybeAddFact(fact.block_is_dead_fact());
    case protobufs::Fact::kConstantUniformFact:
      return constant_uniform_facts_.MaybeAddFact(fact.constant_uniform_fact());
    case protobufs::Fact::kDataSynonymFact:
      return data_synonym_and_id_equation_facts_.MaybeAddFact(
          fact.data_synonym_fact(), dead_block_facts_, irrelevant_value_facts_);
    case protobufs::Fact::kFunctionIsLivesafeFact:
      return livesafe_function_facts_.MaybeAddFact(
          fact.function_is_livesafe_fact());
    case protobufs::Fact::kIdEquationFact:
      return data_synonym_and_id_equation_facts_.MaybeAddFact(
          fact.id_equation_fact(), dead_block_facts_, irrelevant_value_facts_);
    case protobufs::Fact::kIdIsIrrelevant:
      return irrelevant_value_facts_.MaybeAddFact(
          fact.id_is_irrelevant(), data_synonym_and_id_equation_facts_);
    case protobufs::Fact::kPointeeValueIsIrrelevantFact:
      return irrelevant_value_facts_.MaybeAddFact(
          fact.pointee_value_is_irrelevant_fact(),
          data_synonym_and_id_equation_facts_);
    case protobufs::Fact::FACT_NOT_SET:
      assert(false && "The fact must be set");
      return false;
  }

  assert(false && "Unreachable");
  return false;
}

void FactManager::AddFactDataSynonym(const protobufs::DataDescriptor& data1,
                                     const protobufs::DataDescriptor& data2) {
  protobufs::FactDataSynonym fact;
  *fact.mutable_data1() = data1;
  *fact.mutable_data2() = data2;
  auto success = data_synonym_and_id_equation_facts_.MaybeAddFact(
      fact, dead_block_facts_, irrelevant_value_facts_);
  (void)success;  // Keep compilers happy in release mode.
  assert(success && "Unable to create DataSynonym fact");
}

std::vector<uint32_t> FactManager::GetConstantsAvailableFromUniformsForType(
    uint32_t type_id) const {
  return constant_uniform_facts_.GetConstantsAvailableFromUniformsForType(
      type_id);
}

std::vector<protobufs::UniformBufferElementDescriptor>
FactManager::GetUniformDescriptorsForConstant(uint32_t constant_id) const {
  return constant_uniform_facts_.GetUniformDescriptorsForConstant(constant_id);
}

uint32_t FactManager::GetConstantFromUniformDescriptor(
    const protobufs::UniformBufferElementDescriptor& uniform_descriptor) const {
  return constant_uniform_facts_.GetConstantFromUniformDescriptor(
      uniform_descriptor);
}

std::vector<uint32_t> FactManager::GetTypesForWhichUniformValuesAreKnown()
    const {
  return constant_uniform_facts_.GetTypesForWhichUniformValuesAreKnown();
}

const std::vector<std::pair<protobufs::FactConstantUniform, uint32_t>>&
FactManager::GetConstantUniformFactsAndTypes() const {
  return constant_uniform_facts_.GetConstantUniformFactsAndTypes();
}

std::vector<uint32_t> FactManager::GetIdsForWhichSynonymsAreKnown() const {
  return data_synonym_and_id_equation_facts_.GetIdsForWhichSynonymsAreKnown();
}

std::vector<const protobufs::DataDescriptor*> FactManager::GetAllSynonyms()
    const {
  return data_synonym_and_id_equation_facts_.GetAllKnownSynonyms();
}

std::vector<const protobufs::DataDescriptor*>
FactManager::GetSynonymsForDataDescriptor(
    const protobufs::DataDescriptor& data_descriptor) const {
  return data_synonym_and_id_equation_facts_.GetSynonymsForDataDescriptor(
      data_descriptor);
}

std::vector<const protobufs::DataDescriptor*> FactManager::GetSynonymsForId(
    uint32_t id) const {
  return data_synonym_and_id_equation_facts_.GetSynonymsForId(id);
}

bool FactManager::IsSynonymous(
    const protobufs::DataDescriptor& data_descriptor1,
    const protobufs::DataDescriptor& data_descriptor2) const {
  return data_synonym_and_id_equation_facts_.IsSynonymous(data_descriptor1,
                                                          data_descriptor2);
}

bool FactManager::BlockIsDead(uint32_t block_id) const {
  return dead_block_facts_.BlockIsDead(block_id);
}

void FactManager::AddFactBlockIsDead(uint32_t block_id) {
  protobufs::FactBlockIsDead fact;
  fact.set_block_id(block_id);
  auto success = dead_block_facts_.MaybeAddFact(fact);
  (void)success;  // Keep compilers happy in release mode.
  assert(success && "|block_id| is invalid");
}

bool FactManager::FunctionIsLivesafe(uint32_t function_id) const {
  return livesafe_function_facts_.FunctionIsLivesafe(function_id);
}

void FactManager::AddFactFunctionIsLivesafe(uint32_t function_id) {
  protobufs::FactFunctionIsLivesafe fact;
  fact.set_function_id(function_id);
  auto success = livesafe_function_facts_.MaybeAddFact(fact);
  (void)success;  // Keep compilers happy in release mode.
  assert(success && "|function_id| is invalid");
}

bool FactManager::PointeeValueIsIrrelevant(uint32_t pointer_id) const {
  return irrelevant_value_facts_.PointeeValueIsIrrelevant(pointer_id);
}

bool FactManager::IdIsIrrelevant(uint32_t result_id) const {
  return irrelevant_value_facts_.IdIsIrrelevant(result_id, dead_block_facts_);
}

std::unordered_set<uint32_t> FactManager::GetIrrelevantIds() const {
  return irrelevant_value_facts_.GetIrrelevantIds(dead_block_facts_);
}

void FactManager::AddFactValueOfPointeeIsIrrelevant(uint32_t pointer_id) {
  protobufs::FactPointeeValueIsIrrelevant fact;
  fact.set_pointer_id(pointer_id);
  auto success = irrelevant_value_facts_.MaybeAddFact(
      fact, data_synonym_and_id_equation_facts_);
  (void)success;  // Keep compilers happy in release mode.
  assert(success && "|pointer_id| is invalid");
}

void FactManager::AddFactIdIsIrrelevant(uint32_t result_id) {
  protobufs::FactIdIsIrrelevant fact;
  fact.set_result_id(result_id);
  auto success = irrelevant_value_facts_.MaybeAddFact(
      fact, data_synonym_and_id_equation_facts_);
  (void)success;  // Keep compilers happy in release mode.
  assert(success && "|result_id| is invalid");
}

void FactManager::AddFactIdEquation(uint32_t lhs_id, spv::Op opcode,
                                    const std::vector<uint32_t>& rhs_id) {
  protobufs::FactIdEquation fact;
  fact.set_lhs_id(lhs_id);
  fact.set_opcode(uint32_t(opcode));
  for (auto an_rhs_id : rhs_id) {
    fact.add_rhs_id(an_rhs_id);
  }
  auto success = data_synonym_and_id_equation_facts_.MaybeAddFact(
      fact, dead_block_facts_, irrelevant_value_facts_);
  (void)success;  // Keep compilers happy in release mode.
  assert(success && "Can't create IdIsIrrelevant fact");
}

void FactManager::ComputeClosureOfFacts(
    uint32_t maximum_equivalence_class_size) {
  data_synonym_and_id_equation_facts_.ComputeClosureOfFacts(
      maximum_equivalence_class_size);
}

}  // namespace fuzz
}  // namespace spvtools
