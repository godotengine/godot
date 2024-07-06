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

#include "source/fuzz/fact_manager/constant_uniform_facts.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/uniform_buffer_element_descriptor.h"

namespace spvtools {
namespace fuzz {
namespace fact_manager {

ConstantUniformFacts::ConstantUniformFacts(opt::IRContext* ir_context)
    : ir_context_(ir_context) {}

uint32_t ConstantUniformFacts::GetConstantId(
    const protobufs::FactConstantUniform& constant_uniform_fact,
    uint32_t type_id) const {
  auto type = ir_context_->get_type_mgr()->GetType(type_id);
  assert(type != nullptr && "Unknown type id.");
  const opt::analysis::Constant* known_constant;
  if (type->AsInteger()) {
    opt::analysis::IntConstant candidate_constant(
        type->AsInteger(), GetConstantWords(constant_uniform_fact));
    known_constant =
        ir_context_->get_constant_mgr()->FindConstant(&candidate_constant);
  } else {
    assert(
        type->AsFloat() &&
        "Uniform constant facts are only supported for int and float types.");
    opt::analysis::FloatConstant candidate_constant(
        type->AsFloat(), GetConstantWords(constant_uniform_fact));
    known_constant =
        ir_context_->get_constant_mgr()->FindConstant(&candidate_constant);
  }
  if (!known_constant) {
    return 0;
  }
  return ir_context_->get_constant_mgr()->FindDeclaredConstant(known_constant,
                                                               type_id);
}

std::vector<uint32_t> ConstantUniformFacts::GetConstantWords(
    const protobufs::FactConstantUniform& constant_uniform_fact) {
  std::vector<uint32_t> result;
  for (auto constant_word : constant_uniform_fact.constant_word()) {
    result.push_back(constant_word);
  }
  return result;
}

bool ConstantUniformFacts::DataMatches(
    const opt::Instruction& constant_instruction,
    const protobufs::FactConstantUniform& constant_uniform_fact) {
  assert(constant_instruction.opcode() == spv::Op::OpConstant);
  std::vector<uint32_t> data_in_constant;
  for (uint32_t i = 0; i < constant_instruction.NumInOperands(); i++) {
    data_in_constant.push_back(constant_instruction.GetSingleWordInOperand(i));
  }
  return data_in_constant == GetConstantWords(constant_uniform_fact);
}

std::vector<uint32_t>
ConstantUniformFacts::GetConstantsAvailableFromUniformsForType(
    uint32_t type_id) const {
  std::vector<uint32_t> result;
  std::set<uint32_t> already_seen;
  for (auto& fact_and_type_id : facts_and_type_ids_) {
    if (fact_and_type_id.second != type_id) {
      continue;
    }
    if (auto constant_id = GetConstantId(fact_and_type_id.first, type_id)) {
      if (already_seen.find(constant_id) == already_seen.end()) {
        result.push_back(constant_id);
        already_seen.insert(constant_id);
      }
    }
  }
  return result;
}

std::vector<protobufs::UniformBufferElementDescriptor>
ConstantUniformFacts::GetUniformDescriptorsForConstant(
    uint32_t constant_id) const {
  std::vector<protobufs::UniformBufferElementDescriptor> result;
  auto constant_inst = ir_context_->get_def_use_mgr()->GetDef(constant_id);
  assert(constant_inst->opcode() == spv::Op::OpConstant &&
         "The given id must be that of a constant");
  auto type_id = constant_inst->type_id();
  for (auto& fact_and_type_id : facts_and_type_ids_) {
    if (fact_and_type_id.second != type_id) {
      continue;
    }
    if (DataMatches(*constant_inst, fact_and_type_id.first)) {
      result.emplace_back(
          fact_and_type_id.first.uniform_buffer_element_descriptor());
    }
  }
  return result;
}

uint32_t ConstantUniformFacts::GetConstantFromUniformDescriptor(
    const protobufs::UniformBufferElementDescriptor& uniform_descriptor) const {
  // Consider each fact.
  for (auto& fact_and_type : facts_and_type_ids_) {
    // Check whether the uniform descriptor associated with the fact matches
    // |uniform_descriptor|.
    if (UniformBufferElementDescriptorEquals()(
            &uniform_descriptor,
            &fact_and_type.first.uniform_buffer_element_descriptor())) {
      return GetConstantId(fact_and_type.first, fact_and_type.second);
    }
  }
  // No fact associated with the given uniform descriptor was found.
  return 0;
}

std::vector<uint32_t>
ConstantUniformFacts::GetTypesForWhichUniformValuesAreKnown() const {
  std::vector<uint32_t> result;
  for (auto& fact_and_type : facts_and_type_ids_) {
    if (std::find(result.begin(), result.end(), fact_and_type.second) ==
        result.end()) {
      result.push_back(fact_and_type.second);
    }
  }
  return result;
}

bool ConstantUniformFacts::FloatingPointValueIsSuitable(
    const protobufs::FactConstantUniform& fact, uint32_t width) {
  const uint32_t kFloatWidth = 32;
  const uint32_t kDoubleWidth = 64;
  if (width != kFloatWidth && width != kDoubleWidth) {
    // Only 32- and 64-bit floating-point types are handled.
    return false;
  }
  std::vector<uint32_t> words = GetConstantWords(fact);
  if (width == 32) {
    float value;
    memcpy(&value, words.data(), sizeof(float));
    if (!std::isfinite(value)) {
      return false;
    }
  } else {
    double value;
    memcpy(&value, words.data(), sizeof(double));
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

bool ConstantUniformFacts::MaybeAddFact(
    const protobufs::FactConstantUniform& fact) {
  // Try to find a unique instruction that declares a variable such that the
  // variable is decorated with the descriptor set and binding associated with
  // the constant uniform fact.
  opt::Instruction* uniform_variable = FindUniformVariable(
      fact.uniform_buffer_element_descriptor(), ir_context_, true);

  if (!uniform_variable) {
    return false;
  }

  assert(spv::Op::OpVariable == uniform_variable->opcode());
  assert(spv::StorageClass::Uniform ==
         spv::StorageClass(uniform_variable->GetSingleWordInOperand(0)));

  auto should_be_uniform_pointer_type =
      ir_context_->get_type_mgr()->GetType(uniform_variable->type_id());
  if (!should_be_uniform_pointer_type->AsPointer()) {
    return false;
  }
  if (should_be_uniform_pointer_type->AsPointer()->storage_class() !=
      spv::StorageClass::Uniform) {
    return false;
  }
  auto should_be_uniform_pointer_instruction =
      ir_context_->get_def_use_mgr()->GetDef(uniform_variable->type_id());
  auto composite_type =
      should_be_uniform_pointer_instruction->GetSingleWordInOperand(1);

  auto final_element_type_id = fuzzerutil::WalkCompositeTypeIndices(
      ir_context_, composite_type,
      fact.uniform_buffer_element_descriptor().index());
  if (!final_element_type_id) {
    return false;
  }
  auto final_element_type =
      ir_context_->get_type_mgr()->GetType(final_element_type_id);
  assert(final_element_type &&
         "There should be a type corresponding to this id.");

  if (!(final_element_type->AsFloat() || final_element_type->AsInteger())) {
    return false;
  }
  auto width = final_element_type->AsFloat()
                   ? final_element_type->AsFloat()->width()
                   : final_element_type->AsInteger()->width();

  if (final_element_type->AsFloat() &&
      !FloatingPointValueIsSuitable(fact, width)) {
    return false;
  }

  auto required_words = (width + 32 - 1) / 32;
  if (static_cast<uint32_t>(fact.constant_word().size()) != required_words) {
    return false;
  }
  facts_and_type_ids_.emplace_back(
      std::pair<protobufs::FactConstantUniform, uint32_t>(
          fact, final_element_type_id));
  return true;
}

const std::vector<std::pair<protobufs::FactConstantUniform, uint32_t>>&
ConstantUniformFacts::GetConstantUniformFactsAndTypes() const {
  return facts_and_type_ids_;
}

}  // namespace fact_manager
}  // namespace fuzz
}  // namespace spvtools
