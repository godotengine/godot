// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/transformation_permute_phi_operands.h"

#include <vector>

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationPermutePhiOperands::TransformationPermutePhiOperands(
    protobufs::TransformationPermutePhiOperands message)
    : message_(std::move(message)) {}

TransformationPermutePhiOperands::TransformationPermutePhiOperands(
    uint32_t result_id, const std::vector<uint32_t>& permutation) {
  message_.set_result_id(result_id);

  for (auto index : permutation) {
    message_.add_permutation(index);
  }
}

bool TransformationPermutePhiOperands::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Check that |message_.result_id| is valid.
  const auto* inst =
      ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  if (!inst || inst->opcode() != spv::Op::OpPhi) {
    return false;
  }

  // Check that |message_.permutation| has expected size.
  auto expected_permutation_size = inst->NumInOperands() / 2;
  if (static_cast<uint32_t>(message_.permutation().size()) !=
      expected_permutation_size) {
    return false;
  }

  // Check that |message_.permutation| has elements in range
  // [0, expected_permutation_size - 1].
  std::vector<uint32_t> permutation(message_.permutation().begin(),
                                    message_.permutation().end());
  assert(!fuzzerutil::HasDuplicates(permutation) &&
         "Permutation has duplicates");

  // We must check whether the permutation is empty first because in that case
  // |expected_permutation_size - 1| will produce
  // |std::numeric_limits<uint32_t>::max()| since it's an unsigned integer.
  return permutation.empty() ||
         fuzzerutil::IsPermutationOfRange(permutation, 0,
                                          expected_permutation_size - 1);
}

void TransformationPermutePhiOperands::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto* inst = ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  assert(inst);

  opt::Instruction::OperandList permuted_operands;
  permuted_operands.reserve(inst->NumInOperands());

  for (auto index : message_.permutation()) {
    permuted_operands.push_back(std::move(inst->GetInOperand(2 * index)));
    permuted_operands.push_back(std::move(inst->GetInOperand(2 * index + 1)));
  }

  inst->SetInOperands(std::move(permuted_operands));

  // Update the def-use manager.
  ir_context->UpdateDefUse(inst);
}

protobufs::Transformation TransformationPermutePhiOperands::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_permute_phi_operands() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationPermutePhiOperands::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
