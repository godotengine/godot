// Copyright (c) 2021 Mostafa Ashraf
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

#include "source/fuzz/transformation_swap_function_variables.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationSwapFunctionVariables::TransformationSwapFunctionVariables(
    protobufs::TransformationSwapFunctionVariables message)
    : message_(std::move(message)) {}

TransformationSwapFunctionVariables::TransformationSwapFunctionVariables(
    uint32_t result_id1, uint32_t result_id2) {
  message_.set_result_id1(result_id1);
  message_.set_result_id2(result_id2);
}

bool TransformationSwapFunctionVariables::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  uint32_t result_id1 = message_.result_id1();
  uint32_t result_id2 = message_.result_id2();

  assert((result_id1 != result_id2) && "Two results ids are equal");

  // The result ids used in the message must refer to instructions.
  auto instruction1 = ir_context->get_def_use_mgr()->GetDef(result_id1);
  auto instruction2 = ir_context->get_def_use_mgr()->GetDef(result_id2);
  if (instruction1 == nullptr || instruction2 == nullptr) {
    return false;
  }
  // Both instructions must be variables.
  if (instruction1->opcode() != spv::Op::OpVariable ||
      instruction2->opcode() != spv::Op::OpVariable) {
    return false;
  }

  // Both variable instructions must be in some basic block (as they are
  // function-local variables), and they must be in the same block (as they need
  // to be variables of the same function).
  auto* block_1 = ir_context->get_instr_block(result_id1);
  auto* block_2 = ir_context->get_instr_block(result_id2);
  if (block_1 == nullptr || block_2 == nullptr) {
    return false;
  }

  return block_1 == block_2;
}

void TransformationSwapFunctionVariables::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // The result ids used in the message must refer to instructions.
  auto instruction1 =
      ir_context->get_def_use_mgr()->GetDef(message_.result_id1());
  auto instruction2 =
      ir_context->get_def_use_mgr()->GetDef(message_.result_id2());

  std::unique_ptr<opt::Instruction> temp_instruction =
      MakeUnique<opt::Instruction>();

  temp_instruction->InsertBefore(instruction1);
  instruction1->InsertAfter(instruction2);
  instruction2->InsertAfter(temp_instruction.get());
  temp_instruction->RemoveFromList();
}

protobufs::Transformation TransformationSwapFunctionVariables::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_swap_function_variables() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationSwapFunctionVariables::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
