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

#include "source/fuzz/transformation_swap_commutable_operands.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationSwapCommutableOperands::TransformationSwapCommutableOperands(
    protobufs::TransformationSwapCommutableOperands message)
    : message_(std::move(message)) {}

TransformationSwapCommutableOperands::TransformationSwapCommutableOperands(
    const protobufs::InstructionDescriptor& instruction_descriptor) {
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
}

bool TransformationSwapCommutableOperands::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  auto instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  if (instruction == nullptr) return false;

  spv::Op opcode = static_cast<spv::Op>(
      message_.instruction_descriptor().target_instruction_opcode());
  assert(spv::Op(instruction->opcode()) == opcode &&
         "The located instruction must have the same opcode as in the "
         "descriptor.");
  return spvOpcodeIsCommutativeBinaryOperator(opcode);
}

void TransformationSwapCommutableOperands::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  // By design, the instructions defined to be commutative have exactly two
  // input parameters.
  std::swap(instruction->GetInOperand(0), instruction->GetInOperand(1));
}

protobufs::Transformation TransformationSwapCommutableOperands::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_swap_commutable_operands() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationSwapCommutableOperands::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
