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

#include "source/fuzz/transformation_permute_function_parameters.h"

#include <vector>

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationPermuteFunctionParameters::
    TransformationPermuteFunctionParameters(
        protobufs::TransformationPermuteFunctionParameters message)
    : message_(std::move(message)) {}

TransformationPermuteFunctionParameters::
    TransformationPermuteFunctionParameters(
        uint32_t function_id, uint32_t function_type_fresh_id,
        const std::vector<uint32_t>& permutation) {
  message_.set_function_id(function_id);
  message_.set_function_type_fresh_id(function_type_fresh_id);

  for (auto index : permutation) {
    message_.add_permutation(index);
  }
}

bool TransformationPermuteFunctionParameters::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Check that function exists
  const auto* function =
      fuzzerutil::FindFunction(ir_context, message_.function_id());
  if (!function || function->DefInst().opcode() != spv::Op::OpFunction ||
      fuzzerutil::FunctionIsEntryPoint(ir_context, function->result_id())) {
    return false;
  }

  // Check that permutation has valid indices
  const auto* function_type = fuzzerutil::GetFunctionType(ir_context, function);
  assert(function_type && "Function type is null");

  std::vector<uint32_t> permutation(message_.permutation().begin(),
                                    message_.permutation().end());

  // Don't take return type into account
  auto arg_size = function_type->NumInOperands() - 1;

  // |permutation| vector should be equal to the number of arguments
  if (static_cast<uint32_t>(permutation.size()) != arg_size) {
    return false;
  }

  // Check that permutation doesn't have duplicated values.
  assert(!fuzzerutil::HasDuplicates(permutation) &&
         "Permutation has duplicates");

  // Check that elements in permutation are in range [0, arg_size - 1].
  //
  // We must check whether the permutation is empty first because in that case
  // |arg_size - 1| will produce |std::numeric_limits<uint32_t>::max()| since
  // it's an unsigned integer.
  if (!permutation.empty() &&
      !fuzzerutil::IsPermutationOfRange(permutation, 0, arg_size - 1)) {
    return false;
  }

  return fuzzerutil::IsFreshId(ir_context, message_.function_type_fresh_id());
}

void TransformationPermuteFunctionParameters::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Find the function that will be transformed
  auto* function = fuzzerutil::FindFunction(ir_context, message_.function_id());
  assert(function && "Can't find the function");

  // Adjust OpFunctionParameter instructions

  // Collect ids and types from OpFunctionParameter instructions
  std::vector<uint32_t> param_id, param_type;
  function->ForEachParam(
      [&param_id, &param_type](const opt::Instruction* param) {
        param_id.push_back(param->result_id());
        param_type.push_back(param->type_id());
      });

  // Permute parameters' ids and types
  std::vector<uint32_t> permuted_param_id, permuted_param_type;
  for (auto index : message_.permutation()) {
    permuted_param_id.push_back(param_id[index]);
    permuted_param_type.push_back(param_type[index]);
  }

  // Set OpFunctionParameter instructions to point to new parameters
  size_t i = 0;
  function->ForEachParam(
      [&i, &permuted_param_id, &permuted_param_type](opt::Instruction* param) {
        param->SetResultType(permuted_param_type[i]);
        param->SetResultId(permuted_param_id[i]);
        ++i;
      });

  // Fix all OpFunctionCall instructions
  for (auto* call : fuzzerutil::GetCallers(ir_context, function->result_id())) {
    opt::Instruction::OperandList call_operands = {
        call->GetInOperand(0)  // Function id
    };

    for (auto index : message_.permutation()) {
      // Take function id into account
      call_operands.push_back(call->GetInOperand(index + 1));
    }

    call->SetInOperands(std::move(call_operands));
  }

  // Update function type.
  {
    // We use a separate scope here since |old_function_type_inst| might become
    // a dangling pointer after the call to the fuzzerutil::UpdateFunctionType.

    auto* old_function_type_inst =
        fuzzerutil::GetFunctionType(ir_context, function);
    assert(old_function_type_inst && "Function must have a valid type");

    std::vector<uint32_t> parameter_type_ids;
    for (auto index : message_.permutation()) {
      // +1 since the first operand to OpTypeFunction is a return type.
      parameter_type_ids.push_back(
          old_function_type_inst->GetSingleWordInOperand(index + 1));
    }

    // Change function's type.
    fuzzerutil::UpdateFunctionType(
        ir_context, function->result_id(), message_.function_type_fresh_id(),
        old_function_type_inst->GetSingleWordInOperand(0), parameter_type_ids);
  }

  // Make sure our changes are analyzed
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationPermuteFunctionParameters::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_permute_function_parameters() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationPermuteFunctionParameters::GetFreshIds() const {
  return {message_.function_type_fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
