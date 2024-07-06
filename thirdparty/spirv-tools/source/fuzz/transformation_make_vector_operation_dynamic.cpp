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

#include "source/fuzz/transformation_make_vector_operation_dynamic.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationMakeVectorOperationDynamic::
    TransformationMakeVectorOperationDynamic(
        protobufs::TransformationMakeVectorOperationDynamic message)
    : message_(std::move(message)) {}

TransformationMakeVectorOperationDynamic::
    TransformationMakeVectorOperationDynamic(uint32_t instruction_result_id,
                                             uint32_t constant_index_id) {
  message_.set_instruction_result_id(instruction_result_id);
  message_.set_constant_index_id(constant_index_id);
}

bool TransformationMakeVectorOperationDynamic::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |instruction| must be a vector operation.
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.instruction_result_id());
  if (!IsVectorOperation(ir_context, instruction)) {
    return false;
  }

  // |constant_index_instruction| must be defined as an integer instruction.
  auto constant_index_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.constant_index_id());
  if (!constant_index_instruction || !constant_index_instruction->type_id() ||
      !ir_context->get_type_mgr()
           ->GetType(constant_index_instruction->type_id())
           ->AsInteger()) {
    return false;
  }

  return true;
}

void TransformationMakeVectorOperationDynamic::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.instruction_result_id());

  // The OpVectorInsertDynamic instruction has the vector and component operands
  // in reverse order in relation to the OpCompositeInsert corresponding
  // operands.
  if (instruction->opcode() == spv::Op::OpCompositeInsert) {
    std::swap(instruction->GetInOperand(0), instruction->GetInOperand(1));
  }

  // Sets the literal operand to the equivalent constant.
  instruction->SetInOperand(
      instruction->opcode() == spv::Op::OpCompositeExtract ? 1 : 2,
      {message_.constant_index_id()});

  // Sets the |instruction| opcode to the corresponding vector dynamic opcode.
  instruction->SetOpcode(instruction->opcode() == spv::Op::OpCompositeExtract
                             ? spv::Op::OpVectorExtractDynamic
                             : spv::Op::OpVectorInsertDynamic);
}

protobufs::Transformation TransformationMakeVectorOperationDynamic::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_make_vector_operation_dynamic() = message_;
  return result;
}

bool TransformationMakeVectorOperationDynamic::IsVectorOperation(
    opt::IRContext* ir_context, opt::Instruction* instruction) {
  // |instruction| must be defined and must be an OpCompositeExtract/Insert
  // instruction.
  if (!instruction || (instruction->opcode() != spv::Op::OpCompositeExtract &&
                       instruction->opcode() != spv::Op::OpCompositeInsert)) {
    return false;
  }

  // The composite must be a vector.
  auto composite_instruction =
      ir_context->get_def_use_mgr()->GetDef(instruction->GetSingleWordInOperand(
          instruction->opcode() == spv::Op::OpCompositeExtract ? 0 : 1));
  if (!ir_context->get_type_mgr()
           ->GetType(composite_instruction->type_id())
           ->AsVector()) {
    return false;
  }

  return true;
}

std::unordered_set<uint32_t>
TransformationMakeVectorOperationDynamic::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
