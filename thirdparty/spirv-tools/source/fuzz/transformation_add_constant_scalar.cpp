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

#include "source/fuzz/transformation_add_constant_scalar.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddConstantScalar::TransformationAddConstantScalar(
    spvtools::fuzz::protobufs::TransformationAddConstantScalar message)
    : message_(std::move(message)) {}

TransformationAddConstantScalar::TransformationAddConstantScalar(
    uint32_t fresh_id, uint32_t type_id, const std::vector<uint32_t>& words,
    bool is_irrelevant) {
  message_.set_fresh_id(fresh_id);
  message_.set_type_id(type_id);
  message_.set_is_irrelevant(is_irrelevant);
  for (auto word : words) {
    message_.add_word(word);
  }
}

bool TransformationAddConstantScalar::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The id needs to be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }
  // The type id for the scalar must exist and be a type.
  auto type = ir_context->get_type_mgr()->GetType(message_.type_id());
  if (!type) {
    return false;
  }
  uint32_t width;
  if (type->AsFloat()) {
    width = type->AsFloat()->width();
  } else if (type->AsInteger()) {
    width = type->AsInteger()->width();
  } else {
    return false;
  }
  // The number of words is the integer floor of the width.
  auto words = (width + 32 - 1) / 32;

  // The number of words provided by the transformation needs to match the
  // width of the type.
  return static_cast<uint32_t>(message_.word().size()) == words;
}

void TransformationAddConstantScalar::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  auto new_instruction = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpConstant, message_.type_id(), message_.fresh_id(),
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_LITERAL_INTEGER,
            std::vector<uint32_t>(message_.word().begin(),
                                  message_.word().end())}}));
  auto new_instruction_ptr = new_instruction.get();
  ir_context->module()->AddGlobalValue(std::move(new_instruction));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  // Inform the def-use manager about the new instruction. Invalidate the
  // constant manager as we have added a new constant.
  ir_context->get_def_use_mgr()->AnalyzeInstDef(new_instruction_ptr);
  ir_context->InvalidateAnalyses(opt::IRContext::kAnalysisConstants);

  if (message_.is_irrelevant()) {
    transformation_context->GetFactManager()->AddFactIdIsIrrelevant(
        message_.fresh_id());
  }
}

protobufs::Transformation TransformationAddConstantScalar::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_constant_scalar() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationAddConstantScalar::GetFreshIds()
    const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
