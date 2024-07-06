// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/transformation_adjust_branch_weights.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

namespace {

const uint32_t kBranchWeightForTrueLabelIndex = 3;
const uint32_t kBranchWeightForFalseLabelIndex = 4;

}  // namespace

TransformationAdjustBranchWeights::TransformationAdjustBranchWeights(
    protobufs::TransformationAdjustBranchWeights message)
    : message_(std::move(message)) {}

TransformationAdjustBranchWeights::TransformationAdjustBranchWeights(
    const protobufs::InstructionDescriptor& instruction_descriptor,
    const std::pair<uint32_t, uint32_t>& branch_weights) {
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
  message_.mutable_branch_weights()->set_first(branch_weights.first);
  message_.mutable_branch_weights()->set_second(branch_weights.second);
}

bool TransformationAdjustBranchWeights::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  auto instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  if (instruction == nullptr) {
    return false;
  }

  spv::Op opcode = static_cast<spv::Op>(
      message_.instruction_descriptor().target_instruction_opcode());

  assert(instruction->opcode() == opcode &&
         "The located instruction must have the same opcode as in the "
         "descriptor.");

  // Must be an OpBranchConditional instruction.
  if (opcode != spv::Op::OpBranchConditional) {
    return false;
  }

  assert((message_.branch_weights().first() != 0 ||
          message_.branch_weights().second() != 0) &&
         "At least one weight must be non-zero.");

  assert(message_.branch_weights().first() <=
             UINT32_MAX - message_.branch_weights().second() &&
         "The sum of the two weights must not be greater than UINT32_MAX.");

  return true;
}

void TransformationAdjustBranchWeights::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  if (instruction->HasBranchWeights()) {
    instruction->SetOperand(kBranchWeightForTrueLabelIndex,
                            {message_.branch_weights().first()});
    instruction->SetOperand(kBranchWeightForFalseLabelIndex,
                            {message_.branch_weights().second()});
  } else {
    instruction->AddOperand({SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER,
                             {message_.branch_weights().first()}});
    instruction->AddOperand({SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER,
                             {message_.branch_weights().second()}});
  }
}

protobufs::Transformation TransformationAdjustBranchWeights::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_adjust_branch_weights() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationAdjustBranchWeights::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
