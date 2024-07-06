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

#include "source/fuzz/transformation_replace_linear_algebra_instruction.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceLinearAlgebraInstruction::
    TransformationReplaceLinearAlgebraInstruction(
        protobufs::TransformationReplaceLinearAlgebraInstruction message)
    : message_(std::move(message)) {}

TransformationReplaceLinearAlgebraInstruction::
    TransformationReplaceLinearAlgebraInstruction(
        const std::vector<uint32_t>& fresh_ids,
        const protobufs::InstructionDescriptor& instruction_descriptor) {
  for (auto fresh_id : fresh_ids) {
    message_.add_fresh_ids(fresh_id);
  }
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
}

bool TransformationReplaceLinearAlgebraInstruction::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  auto instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);

  // It must be a linear algebra instruction.
  if (!spvOpcodeIsLinearAlgebra(instruction->opcode())) {
    return false;
  }

  // |message_.fresh_ids.size| must be the exact number of fresh ids needed to
  // apply the transformation.
  if (static_cast<uint32_t>(message_.fresh_ids().size()) !=
      GetRequiredFreshIdCount(ir_context, instruction)) {
    return false;
  }

  // All ids in |message_.fresh_ids| must be fresh.
  for (uint32_t fresh_id : message_.fresh_ids()) {
    if (!fuzzerutil::IsFreshId(ir_context, fresh_id)) {
      return false;
    }
  }

  return true;
}

void TransformationReplaceLinearAlgebraInstruction::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto linear_algebra_instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);

  switch (linear_algebra_instruction->opcode()) {
    case spv::Op::OpTranspose:
      ReplaceOpTranspose(ir_context, linear_algebra_instruction);
      break;
    case spv::Op::OpVectorTimesScalar:
      ReplaceOpVectorTimesScalar(ir_context, linear_algebra_instruction);
      break;
    case spv::Op::OpMatrixTimesScalar:
      ReplaceOpMatrixTimesScalar(ir_context, linear_algebra_instruction);
      break;
    case spv::Op::OpVectorTimesMatrix:
      ReplaceOpVectorTimesMatrix(ir_context, linear_algebra_instruction);
      break;
    case spv::Op::OpMatrixTimesVector:
      ReplaceOpMatrixTimesVector(ir_context, linear_algebra_instruction);
      break;
    case spv::Op::OpMatrixTimesMatrix:
      ReplaceOpMatrixTimesMatrix(ir_context, linear_algebra_instruction);
      break;
    case spv::Op::OpOuterProduct:
      ReplaceOpOuterProduct(ir_context, linear_algebra_instruction);
      break;
    case spv::Op::OpDot:
      ReplaceOpDot(ir_context, linear_algebra_instruction);
      break;
    default:
      assert(false && "Should be unreachable.");
      break;
  }

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation
TransformationReplaceLinearAlgebraInstruction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_linear_algebra_instruction() = message_;
  return result;
}

uint32_t TransformationReplaceLinearAlgebraInstruction::GetRequiredFreshIdCount(
    opt::IRContext* ir_context, opt::Instruction* instruction) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3354):
  // Right now we only support certain operations.
  switch (instruction->opcode()) {
    case spv::Op::OpTranspose: {
      // For each matrix row, |2 * matrix_column_count| OpCompositeExtract and 1
      // OpCompositeConstruct will be inserted.
      auto matrix_instruction = ir_context->get_def_use_mgr()->GetDef(
          instruction->GetSingleWordInOperand(0));
      uint32_t matrix_column_count =
          ir_context->get_type_mgr()
              ->GetType(matrix_instruction->type_id())
              ->AsMatrix()
              ->element_count();
      uint32_t matrix_row_count = ir_context->get_type_mgr()
                                      ->GetType(matrix_instruction->type_id())
                                      ->AsMatrix()
                                      ->element_type()
                                      ->AsVector()
                                      ->element_count();
      return matrix_row_count * (2 * matrix_column_count + 1);
    }
    case spv::Op::OpVectorTimesScalar:
      // For each vector component, 1 OpCompositeExtract and 1 OpFMul will be
      // inserted.
      return 2 *
             ir_context->get_type_mgr()
                 ->GetType(ir_context->get_def_use_mgr()
                               ->GetDef(instruction->GetSingleWordInOperand(0))
                               ->type_id())
                 ->AsVector()
                 ->element_count();
    case spv::Op::OpMatrixTimesScalar: {
      // For each matrix column, |1 + column.size| OpCompositeExtract,
      // |column.size| OpFMul and 1 OpCompositeConstruct instructions will be
      // inserted.
      auto matrix_instruction = ir_context->get_def_use_mgr()->GetDef(
          instruction->GetSingleWordInOperand(0));
      auto matrix_type =
          ir_context->get_type_mgr()->GetType(matrix_instruction->type_id());
      return 2 * matrix_type->AsMatrix()->element_count() *
             (1 + matrix_type->AsMatrix()
                      ->element_type()
                      ->AsVector()
                      ->element_count());
    }
    case spv::Op::OpVectorTimesMatrix: {
      // For each vector component, 1 OpCompositeExtract instruction will be
      // inserted. For each matrix column, |1 + vector_component_count|
      // OpCompositeExtract, |vector_component_count| OpFMul and
      // |vector_component_count - 1| OpFAdd instructions will be inserted.
      auto vector_instruction = ir_context->get_def_use_mgr()->GetDef(
          instruction->GetSingleWordInOperand(0));
      auto matrix_instruction = ir_context->get_def_use_mgr()->GetDef(
          instruction->GetSingleWordInOperand(1));
      uint32_t vector_component_count =
          ir_context->get_type_mgr()
              ->GetType(vector_instruction->type_id())
              ->AsVector()
              ->element_count();
      uint32_t matrix_column_count =
          ir_context->get_type_mgr()
              ->GetType(matrix_instruction->type_id())
              ->AsMatrix()
              ->element_count();
      return vector_component_count * (3 * matrix_column_count + 1);
    }
    case spv::Op::OpMatrixTimesVector: {
      // For each matrix column, |1 + matrix_row_count| OpCompositeExtract
      // will be inserted. For each matrix row, |matrix_column_count| OpFMul and
      // |matrix_column_count - 1| OpFAdd instructions will be inserted. For
      // each vector component, 1 OpCompositeExtract instruction will be
      // inserted.
      auto matrix_instruction = ir_context->get_def_use_mgr()->GetDef(
          instruction->GetSingleWordInOperand(0));
      uint32_t matrix_column_count =
          ir_context->get_type_mgr()
              ->GetType(matrix_instruction->type_id())
              ->AsMatrix()
              ->element_count();
      uint32_t matrix_row_count = ir_context->get_type_mgr()
                                      ->GetType(matrix_instruction->type_id())
                                      ->AsMatrix()
                                      ->element_type()
                                      ->AsVector()
                                      ->element_count();
      return 3 * matrix_column_count * matrix_row_count +
             2 * matrix_column_count - matrix_row_count;
    }
    case spv::Op::OpMatrixTimesMatrix: {
      // For each matrix 2 column, 1 OpCompositeExtract, 1 OpCompositeConstruct,
      // |3 * matrix_1_row_count * matrix_1_column_count| OpCompositeExtract,
      // |matrix_1_row_count * matrix_1_column_count| OpFMul,
      // |matrix_1_row_count * (matrix_1_column_count - 1)| OpFAdd instructions
      // will be inserted.
      auto matrix_1_instruction = ir_context->get_def_use_mgr()->GetDef(
          instruction->GetSingleWordInOperand(0));
      uint32_t matrix_1_column_count =
          ir_context->get_type_mgr()
              ->GetType(matrix_1_instruction->type_id())
              ->AsMatrix()
              ->element_count();
      uint32_t matrix_1_row_count =
          ir_context->get_type_mgr()
              ->GetType(matrix_1_instruction->type_id())
              ->AsMatrix()
              ->element_type()
              ->AsVector()
              ->element_count();

      auto matrix_2_instruction = ir_context->get_def_use_mgr()->GetDef(
          instruction->GetSingleWordInOperand(1));
      uint32_t matrix_2_column_count =
          ir_context->get_type_mgr()
              ->GetType(matrix_2_instruction->type_id())
              ->AsMatrix()
              ->element_count();
      return matrix_2_column_count *
             (2 + matrix_1_row_count * (5 * matrix_1_column_count - 1));
    }
    case spv::Op::OpOuterProduct: {
      // For each |vector_2| component, |vector_1_component_count + 1|
      // OpCompositeExtract, |vector_1_component_count| OpFMul and 1
      // OpCompositeConstruct instructions will be inserted.
      auto vector_1_instruction = ir_context->get_def_use_mgr()->GetDef(
          instruction->GetSingleWordInOperand(0));
      auto vector_2_instruction = ir_context->get_def_use_mgr()->GetDef(
          instruction->GetSingleWordInOperand(1));
      uint32_t vector_1_component_count =
          ir_context->get_type_mgr()
              ->GetType(vector_1_instruction->type_id())
              ->AsVector()
              ->element_count();
      uint32_t vector_2_component_count =
          ir_context->get_type_mgr()
              ->GetType(vector_2_instruction->type_id())
              ->AsVector()
              ->element_count();
      return 2 * vector_2_component_count * (vector_1_component_count + 1);
    }
    case spv::Op::OpDot:
      // For each pair of vector components, 2 OpCompositeExtract and 1 OpFMul
      // will be inserted. The first two OpFMul instructions will result the
      // first OpFAdd instruction to be inserted. For each remaining OpFMul, 1
      // OpFAdd will be inserted. The last OpFAdd instruction is got by changing
      // the OpDot instruction.
      return 4 * ir_context->get_type_mgr()
                     ->GetType(
                         ir_context->get_def_use_mgr()
                             ->GetDef(instruction->GetSingleWordInOperand(0))
                             ->type_id())
                     ->AsVector()
                     ->element_count() -
             2;
    default:
      assert(false && "Unsupported linear algebra instruction.");
      return 0;
  }
}

void TransformationReplaceLinearAlgebraInstruction::ReplaceOpTranspose(
    opt::IRContext* ir_context,
    opt::Instruction* linear_algebra_instruction) const {
  // Gets OpTranspose instruction information.
  auto matrix_instruction = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(0));
  uint32_t matrix_column_count = ir_context->get_type_mgr()
                                     ->GetType(matrix_instruction->type_id())
                                     ->AsMatrix()
                                     ->element_count();
  auto matrix_column_type = ir_context->get_type_mgr()
                                ->GetType(matrix_instruction->type_id())
                                ->AsMatrix()
                                ->element_type();
  auto matrix_column_component_type =
      matrix_column_type->AsVector()->element_type();
  uint32_t matrix_row_count = matrix_column_type->AsVector()->element_count();
  auto resulting_matrix_column_type =
      ir_context->get_type_mgr()
          ->GetType(linear_algebra_instruction->type_id())
          ->AsMatrix()
          ->element_type();

  uint32_t fresh_id_index = 0;
  std::vector<uint32_t> result_column_ids(matrix_row_count);
  for (uint32_t i = 0; i < matrix_row_count; i++) {
    std::vector<uint32_t> column_component_ids(matrix_column_count);
    for (uint32_t j = 0; j < matrix_column_count; j++) {
      // Extracts the matrix column.
      uint32_t matrix_column_id = message_.fresh_ids(fresh_id_index++);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpCompositeExtract,
          ir_context->get_type_mgr()->GetId(matrix_column_type),
          matrix_column_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {matrix_instruction->result_id()}},
               {SPV_OPERAND_TYPE_LITERAL_INTEGER, {j}}})));

      // Extracts the matrix column component.
      column_component_ids[j] = message_.fresh_ids(fresh_id_index++);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpCompositeExtract,
          ir_context->get_type_mgr()->GetId(matrix_column_component_type),
          column_component_ids[j],
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {matrix_column_id}},
               {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}})));
    }

    // Inserts the resulting matrix column.
    opt::Instruction::OperandList in_operands;
    for (auto& column_component_id : column_component_ids) {
      in_operands.push_back({SPV_OPERAND_TYPE_ID, {column_component_id}});
    }
    result_column_ids[i] = message_.fresh_ids(fresh_id_index++);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeConstruct,
        ir_context->get_type_mgr()->GetId(resulting_matrix_column_type),
        result_column_ids[i], opt::Instruction::OperandList(in_operands)));
  }

  // The OpTranspose instruction is changed to an OpCompositeConstruct
  // instruction.
  linear_algebra_instruction->SetOpcode(spv::Op::OpCompositeConstruct);
  linear_algebra_instruction->SetInOperand(0, {result_column_ids[0]});
  for (uint32_t i = 1; i < result_column_ids.size(); i++) {
    linear_algebra_instruction->AddOperand(
        {SPV_OPERAND_TYPE_ID, {result_column_ids[i]}});
  }

  fuzzerutil::UpdateModuleIdBound(
      ir_context, message_.fresh_ids(message_.fresh_ids().size() - 1));
}

void TransformationReplaceLinearAlgebraInstruction::ReplaceOpVectorTimesScalar(
    opt::IRContext* ir_context,
    opt::Instruction* linear_algebra_instruction) const {
  // Gets OpVectorTimesScalar in operands.
  auto vector = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(0));
  auto scalar = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(1));

  uint32_t vector_component_count = ir_context->get_type_mgr()
                                        ->GetType(vector->type_id())
                                        ->AsVector()
                                        ->element_count();
  std::vector<uint32_t> float_multiplication_ids(vector_component_count);
  uint32_t fresh_id_index = 0;

  for (uint32_t i = 0; i < vector_component_count; i++) {
    // Extracts |vector| component.
    uint32_t vector_extract_id = message_.fresh_ids(fresh_id_index++);
    fuzzerutil::UpdateModuleIdBound(ir_context, vector_extract_id);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeExtract, scalar->type_id(),
        vector_extract_id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {vector->result_id()}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}})));

    // Multiplies the |vector| component with the |scalar|.
    uint32_t float_multiplication_id = message_.fresh_ids(fresh_id_index++);
    float_multiplication_ids[i] = float_multiplication_id;
    fuzzerutil::UpdateModuleIdBound(ir_context, float_multiplication_id);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpFMul, scalar->type_id(), float_multiplication_id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {vector_extract_id}},
             {SPV_OPERAND_TYPE_ID, {scalar->result_id()}}})));
  }

  // The OpVectorTimesScalar instruction is changed to an OpCompositeConstruct
  // instruction.
  linear_algebra_instruction->SetOpcode(spv::Op::OpCompositeConstruct);
  linear_algebra_instruction->SetInOperand(0, {float_multiplication_ids[0]});
  linear_algebra_instruction->SetInOperand(1, {float_multiplication_ids[1]});
  for (uint32_t i = 2; i < float_multiplication_ids.size(); i++) {
    linear_algebra_instruction->AddOperand(
        {SPV_OPERAND_TYPE_ID, {float_multiplication_ids[i]}});
  }
}

void TransformationReplaceLinearAlgebraInstruction::ReplaceOpMatrixTimesScalar(
    opt::IRContext* ir_context,
    opt::Instruction* linear_algebra_instruction) const {
  // Gets OpMatrixTimesScalar in operands.
  auto matrix_instruction = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(0));
  auto scalar_instruction = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(1));

  // Gets matrix information.
  uint32_t matrix_column_count = ir_context->get_type_mgr()
                                     ->GetType(matrix_instruction->type_id())
                                     ->AsMatrix()
                                     ->element_count();
  auto matrix_column_type = ir_context->get_type_mgr()
                                ->GetType(matrix_instruction->type_id())
                                ->AsMatrix()
                                ->element_type();
  uint32_t matrix_column_size = matrix_column_type->AsVector()->element_count();

  std::vector<uint32_t> composite_construct_ids(matrix_column_count);
  uint32_t fresh_id_index = 0;

  for (uint32_t i = 0; i < matrix_column_count; i++) {
    // Extracts |matrix| column.
    uint32_t matrix_extract_id = message_.fresh_ids(fresh_id_index++);
    fuzzerutil::UpdateModuleIdBound(ir_context, matrix_extract_id);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeExtract,
        ir_context->get_type_mgr()->GetId(matrix_column_type),
        matrix_extract_id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {matrix_instruction->result_id()}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}})));

    std::vector<uint32_t> float_multiplication_ids(matrix_column_size);

    for (uint32_t j = 0; j < matrix_column_size; j++) {
      // Extracts |column| component.
      uint32_t column_extract_id = message_.fresh_ids(fresh_id_index++);
      fuzzerutil::UpdateModuleIdBound(ir_context, column_extract_id);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpCompositeExtract,
          scalar_instruction->type_id(), column_extract_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {matrix_extract_id}},
               {SPV_OPERAND_TYPE_LITERAL_INTEGER, {j}}})));

      // Multiplies the |column| component with the |scalar|.
      float_multiplication_ids[j] = message_.fresh_ids(fresh_id_index++);
      fuzzerutil::UpdateModuleIdBound(ir_context, float_multiplication_ids[j]);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpFMul, scalar_instruction->type_id(),
          float_multiplication_ids[j],
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {column_extract_id}},
               {SPV_OPERAND_TYPE_ID, {scalar_instruction->result_id()}}})));
    }

    // Constructs a new column multiplied by |scalar|.
    opt::Instruction::OperandList composite_construct_in_operands;
    for (uint32_t& float_multiplication_id : float_multiplication_ids) {
      composite_construct_in_operands.push_back(
          {SPV_OPERAND_TYPE_ID, {float_multiplication_id}});
    }
    composite_construct_ids[i] = message_.fresh_ids(fresh_id_index++);
    fuzzerutil::UpdateModuleIdBound(ir_context, composite_construct_ids[i]);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeConstruct,
        ir_context->get_type_mgr()->GetId(matrix_column_type),
        composite_construct_ids[i], composite_construct_in_operands));
  }

  // The OpMatrixTimesScalar instruction is changed to an OpCompositeConstruct
  // instruction.
  linear_algebra_instruction->SetOpcode(spv::Op::OpCompositeConstruct);
  linear_algebra_instruction->SetInOperand(0, {composite_construct_ids[0]});
  linear_algebra_instruction->SetInOperand(1, {composite_construct_ids[1]});
  for (uint32_t i = 2; i < composite_construct_ids.size(); i++) {
    linear_algebra_instruction->AddOperand(
        {SPV_OPERAND_TYPE_ID, {composite_construct_ids[i]}});
  }
}

void TransformationReplaceLinearAlgebraInstruction::ReplaceOpVectorTimesMatrix(
    opt::IRContext* ir_context,
    opt::Instruction* linear_algebra_instruction) const {
  // Gets vector information.
  auto vector_instruction = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(0));
  uint32_t vector_component_count = ir_context->get_type_mgr()
                                        ->GetType(vector_instruction->type_id())
                                        ->AsVector()
                                        ->element_count();
  auto vector_component_type = ir_context->get_type_mgr()
                                   ->GetType(vector_instruction->type_id())
                                   ->AsVector()
                                   ->element_type();

  // Extracts vector components.
  uint32_t fresh_id_index = 0;
  std::vector<uint32_t> vector_component_ids(vector_component_count);
  for (uint32_t i = 0; i < vector_component_count; i++) {
    vector_component_ids[i] = message_.fresh_ids(fresh_id_index++);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeExtract,
        ir_context->get_type_mgr()->GetId(vector_component_type),
        vector_component_ids[i],
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {vector_instruction->result_id()}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}})));
  }

  // Gets matrix information.
  auto matrix_instruction = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(1));
  uint32_t matrix_column_count = ir_context->get_type_mgr()
                                     ->GetType(matrix_instruction->type_id())
                                     ->AsMatrix()
                                     ->element_count();
  auto matrix_column_type = ir_context->get_type_mgr()
                                ->GetType(matrix_instruction->type_id())
                                ->AsMatrix()
                                ->element_type();

  std::vector<uint32_t> result_component_ids(matrix_column_count);
  for (uint32_t i = 0; i < matrix_column_count; i++) {
    // Extracts matrix column.
    uint32_t matrix_extract_id = message_.fresh_ids(fresh_id_index++);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeExtract,
        ir_context->get_type_mgr()->GetId(matrix_column_type),
        matrix_extract_id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {matrix_instruction->result_id()}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}})));

    std::vector<uint32_t> float_multiplication_ids(vector_component_count);
    for (uint32_t j = 0; j < vector_component_count; j++) {
      // Extracts column component.
      uint32_t column_extract_id = message_.fresh_ids(fresh_id_index++);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpCompositeExtract,
          ir_context->get_type_mgr()->GetId(vector_component_type),
          column_extract_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {matrix_extract_id}},
               {SPV_OPERAND_TYPE_LITERAL_INTEGER, {j}}})));

      // Multiplies corresponding vector and column components.
      float_multiplication_ids[j] = message_.fresh_ids(fresh_id_index++);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpFMul,
          ir_context->get_type_mgr()->GetId(vector_component_type),
          float_multiplication_ids[j],
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {vector_component_ids[j]}},
               {SPV_OPERAND_TYPE_ID, {column_extract_id}}})));
    }

    // Adds the multiplication results.
    std::vector<uint32_t> float_add_ids;
    uint32_t float_add_id = message_.fresh_ids(fresh_id_index++);
    float_add_ids.push_back(float_add_id);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpFAdd,
        ir_context->get_type_mgr()->GetId(vector_component_type), float_add_id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {float_multiplication_ids[0]}},
             {SPV_OPERAND_TYPE_ID, {float_multiplication_ids[1]}}})));
    for (uint32_t j = 2; j < float_multiplication_ids.size(); j++) {
      float_add_id = message_.fresh_ids(fresh_id_index++);
      float_add_ids.push_back(float_add_id);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpFAdd,
          ir_context->get_type_mgr()->GetId(vector_component_type),
          float_add_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {float_multiplication_ids[j]}},
               {SPV_OPERAND_TYPE_ID, {float_add_ids[j - 2]}}})));
    }

    result_component_ids[i] = float_add_ids.back();
  }

  // The OpVectorTimesMatrix instruction is changed to an OpCompositeConstruct
  // instruction.
  linear_algebra_instruction->SetOpcode(spv::Op::OpCompositeConstruct);
  linear_algebra_instruction->SetInOperand(0, {result_component_ids[0]});
  linear_algebra_instruction->SetInOperand(1, {result_component_ids[1]});
  for (uint32_t i = 2; i < result_component_ids.size(); i++) {
    linear_algebra_instruction->AddOperand(
        {SPV_OPERAND_TYPE_ID, {result_component_ids[i]}});
  }

  fuzzerutil::UpdateModuleIdBound(
      ir_context, message_.fresh_ids(message_.fresh_ids().size() - 1));
}

void TransformationReplaceLinearAlgebraInstruction::ReplaceOpMatrixTimesVector(
    opt::IRContext* ir_context,
    opt::Instruction* linear_algebra_instruction) const {
  // Gets matrix information.
  auto matrix_instruction = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(0));
  uint32_t matrix_column_count = ir_context->get_type_mgr()
                                     ->GetType(matrix_instruction->type_id())
                                     ->AsMatrix()
                                     ->element_count();
  auto matrix_column_type = ir_context->get_type_mgr()
                                ->GetType(matrix_instruction->type_id())
                                ->AsMatrix()
                                ->element_type();
  uint32_t matrix_row_count = matrix_column_type->AsVector()->element_count();

  // Extracts matrix columns.
  uint32_t fresh_id_index = 0;
  std::vector<uint32_t> matrix_column_ids(matrix_column_count);
  for (uint32_t i = 0; i < matrix_column_count; i++) {
    matrix_column_ids[i] = message_.fresh_ids(fresh_id_index++);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeExtract,
        ir_context->get_type_mgr()->GetId(matrix_column_type),
        matrix_column_ids[i],
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {matrix_instruction->result_id()}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}})));
  }

  // Gets vector information.
  auto vector_instruction = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(1));
  auto vector_component_type = ir_context->get_type_mgr()
                                   ->GetType(vector_instruction->type_id())
                                   ->AsVector()
                                   ->element_type();

  // Extracts vector components.
  std::vector<uint32_t> vector_component_ids(matrix_column_count);
  for (uint32_t i = 0; i < matrix_column_count; i++) {
    vector_component_ids[i] = message_.fresh_ids(fresh_id_index++);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeExtract,
        ir_context->get_type_mgr()->GetId(vector_component_type),
        vector_component_ids[i],
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {vector_instruction->result_id()}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}})));
  }

  std::vector<uint32_t> result_component_ids(matrix_row_count);
  for (uint32_t i = 0; i < matrix_row_count; i++) {
    std::vector<uint32_t> float_multiplication_ids(matrix_column_count);
    for (uint32_t j = 0; j < matrix_column_count; j++) {
      // Extracts column component.
      uint32_t column_extract_id = message_.fresh_ids(fresh_id_index++);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpCompositeExtract,
          ir_context->get_type_mgr()->GetId(vector_component_type),
          column_extract_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {matrix_column_ids[j]}},
               {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}})));

      // Multiplies corresponding vector and column components.
      float_multiplication_ids[j] = message_.fresh_ids(fresh_id_index++);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpFMul,
          ir_context->get_type_mgr()->GetId(vector_component_type),
          float_multiplication_ids[j],
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {column_extract_id}},
               {SPV_OPERAND_TYPE_ID, {vector_component_ids[j]}}})));
    }

    // Adds the multiplication results.
    std::vector<uint32_t> float_add_ids;
    uint32_t float_add_id = message_.fresh_ids(fresh_id_index++);
    float_add_ids.push_back(float_add_id);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpFAdd,
        ir_context->get_type_mgr()->GetId(vector_component_type), float_add_id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {float_multiplication_ids[0]}},
             {SPV_OPERAND_TYPE_ID, {float_multiplication_ids[1]}}})));
    for (uint32_t j = 2; j < float_multiplication_ids.size(); j++) {
      float_add_id = message_.fresh_ids(fresh_id_index++);
      float_add_ids.push_back(float_add_id);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpFAdd,
          ir_context->get_type_mgr()->GetId(vector_component_type),
          float_add_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {float_multiplication_ids[j]}},
               {SPV_OPERAND_TYPE_ID, {float_add_ids[j - 2]}}})));
    }

    result_component_ids[i] = float_add_ids.back();
  }

  // The OpMatrixTimesVector instruction is changed to an OpCompositeConstruct
  // instruction.
  linear_algebra_instruction->SetOpcode(spv::Op::OpCompositeConstruct);
  linear_algebra_instruction->SetInOperand(0, {result_component_ids[0]});
  linear_algebra_instruction->SetInOperand(1, {result_component_ids[1]});
  for (uint32_t i = 2; i < result_component_ids.size(); i++) {
    linear_algebra_instruction->AddOperand(
        {SPV_OPERAND_TYPE_ID, {result_component_ids[i]}});
  }

  fuzzerutil::UpdateModuleIdBound(
      ir_context, message_.fresh_ids(message_.fresh_ids().size() - 1));
}

void TransformationReplaceLinearAlgebraInstruction::ReplaceOpMatrixTimesMatrix(
    opt::IRContext* ir_context,
    opt::Instruction* linear_algebra_instruction) const {
  // Gets matrix 1 information.
  auto matrix_1_instruction = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(0));
  uint32_t matrix_1_column_count =
      ir_context->get_type_mgr()
          ->GetType(matrix_1_instruction->type_id())
          ->AsMatrix()
          ->element_count();
  auto matrix_1_column_type = ir_context->get_type_mgr()
                                  ->GetType(matrix_1_instruction->type_id())
                                  ->AsMatrix()
                                  ->element_type();
  auto matrix_1_column_component_type =
      matrix_1_column_type->AsVector()->element_type();
  uint32_t matrix_1_row_count =
      matrix_1_column_type->AsVector()->element_count();

  // Gets matrix 2 information.
  auto matrix_2_instruction = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(1));
  uint32_t matrix_2_column_count =
      ir_context->get_type_mgr()
          ->GetType(matrix_2_instruction->type_id())
          ->AsMatrix()
          ->element_count();
  auto matrix_2_column_type = ir_context->get_type_mgr()
                                  ->GetType(matrix_2_instruction->type_id())
                                  ->AsMatrix()
                                  ->element_type();

  uint32_t fresh_id_index = 0;
  std::vector<uint32_t> result_column_ids(matrix_2_column_count);
  for (uint32_t i = 0; i < matrix_2_column_count; i++) {
    // Extracts matrix 2 column.
    uint32_t matrix_2_column_id = message_.fresh_ids(fresh_id_index++);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeExtract,
        ir_context->get_type_mgr()->GetId(matrix_2_column_type),
        matrix_2_column_id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {matrix_2_instruction->result_id()}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}})));

    std::vector<uint32_t> column_component_ids(matrix_1_row_count);
    for (uint32_t j = 0; j < matrix_1_row_count; j++) {
      std::vector<uint32_t> float_multiplication_ids(matrix_1_column_count);
      for (uint32_t k = 0; k < matrix_1_column_count; k++) {
        // Extracts matrix 1 column.
        uint32_t matrix_1_column_id = message_.fresh_ids(fresh_id_index++);
        linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
            ir_context, spv::Op::OpCompositeExtract,
            ir_context->get_type_mgr()->GetId(matrix_1_column_type),
            matrix_1_column_id,
            opt::Instruction::OperandList(
                {{SPV_OPERAND_TYPE_ID, {matrix_1_instruction->result_id()}},
                 {SPV_OPERAND_TYPE_LITERAL_INTEGER, {k}}})));

        // Extracts matrix 1 column component.
        uint32_t matrix_1_column_component_id =
            message_.fresh_ids(fresh_id_index++);
        linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
            ir_context, spv::Op::OpCompositeExtract,
            ir_context->get_type_mgr()->GetId(matrix_1_column_component_type),
            matrix_1_column_component_id,
            opt::Instruction::OperandList(
                {{SPV_OPERAND_TYPE_ID, {matrix_1_column_id}},
                 {SPV_OPERAND_TYPE_LITERAL_INTEGER, {j}}})));

        // Extracts matrix 2 column component.
        uint32_t matrix_2_column_component_id =
            message_.fresh_ids(fresh_id_index++);
        linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
            ir_context, spv::Op::OpCompositeExtract,
            ir_context->get_type_mgr()->GetId(matrix_1_column_component_type),
            matrix_2_column_component_id,
            opt::Instruction::OperandList(
                {{SPV_OPERAND_TYPE_ID, {matrix_2_column_id}},
                 {SPV_OPERAND_TYPE_LITERAL_INTEGER, {k}}})));

        // Multiplies corresponding matrix 1 and matrix 2 column components.
        float_multiplication_ids[k] = message_.fresh_ids(fresh_id_index++);
        linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
            ir_context, spv::Op::OpFMul,
            ir_context->get_type_mgr()->GetId(matrix_1_column_component_type),
            float_multiplication_ids[k],
            opt::Instruction::OperandList(
                {{SPV_OPERAND_TYPE_ID, {matrix_1_column_component_id}},
                 {SPV_OPERAND_TYPE_ID, {matrix_2_column_component_id}}})));
      }

      // Adds the multiplication results.
      std::vector<uint32_t> float_add_ids;
      uint32_t float_add_id = message_.fresh_ids(fresh_id_index++);
      float_add_ids.push_back(float_add_id);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpFAdd,
          ir_context->get_type_mgr()->GetId(matrix_1_column_component_type),
          float_add_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {float_multiplication_ids[0]}},
               {SPV_OPERAND_TYPE_ID, {float_multiplication_ids[1]}}})));
      for (uint32_t k = 2; k < float_multiplication_ids.size(); k++) {
        float_add_id = message_.fresh_ids(fresh_id_index++);
        float_add_ids.push_back(float_add_id);
        linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
            ir_context, spv::Op::OpFAdd,
            ir_context->get_type_mgr()->GetId(matrix_1_column_component_type),
            float_add_id,
            opt::Instruction::OperandList(
                {{SPV_OPERAND_TYPE_ID, {float_multiplication_ids[k]}},
                 {SPV_OPERAND_TYPE_ID, {float_add_ids[k - 2]}}})));
      }

      column_component_ids[j] = float_add_ids.back();
    }

    // Inserts the resulting matrix column.
    opt::Instruction::OperandList in_operands;
    for (auto& column_component_id : column_component_ids) {
      in_operands.push_back({SPV_OPERAND_TYPE_ID, {column_component_id}});
    }
    result_column_ids[i] = message_.fresh_ids(fresh_id_index++);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeConstruct,
        ir_context->get_type_mgr()->GetId(matrix_1_column_type),
        result_column_ids[i], opt::Instruction::OperandList(in_operands)));
  }

  // The OpMatrixTimesMatrix instruction is changed to an OpCompositeConstruct
  // instruction.
  linear_algebra_instruction->SetOpcode(spv::Op::OpCompositeConstruct);
  linear_algebra_instruction->SetInOperand(0, {result_column_ids[0]});
  linear_algebra_instruction->SetInOperand(1, {result_column_ids[1]});
  for (uint32_t i = 2; i < result_column_ids.size(); i++) {
    linear_algebra_instruction->AddOperand(
        {SPV_OPERAND_TYPE_ID, {result_column_ids[i]}});
  }

  fuzzerutil::UpdateModuleIdBound(
      ir_context, message_.fresh_ids(message_.fresh_ids().size() - 1));
}

void TransformationReplaceLinearAlgebraInstruction::ReplaceOpOuterProduct(
    opt::IRContext* ir_context,
    opt::Instruction* linear_algebra_instruction) const {
  // Gets vector 1 information.
  auto vector_1_instruction = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(0));
  uint32_t vector_1_component_count =
      ir_context->get_type_mgr()
          ->GetType(vector_1_instruction->type_id())
          ->AsVector()
          ->element_count();
  auto vector_1_component_type = ir_context->get_type_mgr()
                                     ->GetType(vector_1_instruction->type_id())
                                     ->AsVector()
                                     ->element_type();

  // Gets vector 2 information.
  auto vector_2_instruction = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(1));
  uint32_t vector_2_component_count =
      ir_context->get_type_mgr()
          ->GetType(vector_2_instruction->type_id())
          ->AsVector()
          ->element_count();

  uint32_t fresh_id_index = 0;
  std::vector<uint32_t> result_column_ids(vector_2_component_count);
  for (uint32_t i = 0; i < vector_2_component_count; i++) {
    // Extracts |vector_2| component.
    uint32_t vector_2_component_id = message_.fresh_ids(fresh_id_index++);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeExtract,
        ir_context->get_type_mgr()->GetId(vector_1_component_type),
        vector_2_component_id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {vector_2_instruction->result_id()}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}})));

    std::vector<uint32_t> column_component_ids(vector_1_component_count);
    for (uint32_t j = 0; j < vector_1_component_count; j++) {
      // Extracts |vector_1| component.
      uint32_t vector_1_component_id = message_.fresh_ids(fresh_id_index++);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpCompositeExtract,
          ir_context->get_type_mgr()->GetId(vector_1_component_type),
          vector_1_component_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {vector_1_instruction->result_id()}},
               {SPV_OPERAND_TYPE_LITERAL_INTEGER, {j}}})));

      // Multiplies |vector_1| and |vector_2| components.
      column_component_ids[j] = message_.fresh_ids(fresh_id_index++);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpFMul,
          ir_context->get_type_mgr()->GetId(vector_1_component_type),
          column_component_ids[j],
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {vector_2_component_id}},
               {SPV_OPERAND_TYPE_ID, {vector_1_component_id}}})));
    }

    // Inserts the resulting matrix column.
    opt::Instruction::OperandList in_operands;
    for (auto& column_component_id : column_component_ids) {
      in_operands.push_back({SPV_OPERAND_TYPE_ID, {column_component_id}});
    }
    result_column_ids[i] = message_.fresh_ids(fresh_id_index++);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeConstruct,
        vector_1_instruction->type_id(), result_column_ids[i], in_operands));
  }

  // The OpOuterProduct instruction is changed to an OpCompositeConstruct
  // instruction.
  linear_algebra_instruction->SetOpcode(spv::Op::OpCompositeConstruct);
  linear_algebra_instruction->SetInOperand(0, {result_column_ids[0]});
  linear_algebra_instruction->SetInOperand(1, {result_column_ids[1]});
  for (uint32_t i = 2; i < result_column_ids.size(); i++) {
    linear_algebra_instruction->AddOperand(
        {SPV_OPERAND_TYPE_ID, {result_column_ids[i]}});
  }

  fuzzerutil::UpdateModuleIdBound(
      ir_context, message_.fresh_ids(message_.fresh_ids().size() - 1));
}

void TransformationReplaceLinearAlgebraInstruction::ReplaceOpDot(
    opt::IRContext* ir_context,
    opt::Instruction* linear_algebra_instruction) const {
  // Gets OpDot in operands.
  auto vector_1 = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(0));
  auto vector_2 = ir_context->get_def_use_mgr()->GetDef(
      linear_algebra_instruction->GetSingleWordInOperand(1));

  uint32_t vectors_component_count = ir_context->get_type_mgr()
                                         ->GetType(vector_1->type_id())
                                         ->AsVector()
                                         ->element_count();
  std::vector<uint32_t> float_multiplication_ids(vectors_component_count);
  uint32_t fresh_id_index = 0;

  for (uint32_t i = 0; i < vectors_component_count; i++) {
    // Extracts |vector_1| component.
    uint32_t vector_1_extract_id = message_.fresh_ids(fresh_id_index++);
    fuzzerutil::UpdateModuleIdBound(ir_context, vector_1_extract_id);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeExtract,
        linear_algebra_instruction->type_id(), vector_1_extract_id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {vector_1->result_id()}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}})));

    // Extracts |vector_2| component.
    uint32_t vector_2_extract_id = message_.fresh_ids(fresh_id_index++);
    fuzzerutil::UpdateModuleIdBound(ir_context, vector_2_extract_id);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeExtract,
        linear_algebra_instruction->type_id(), vector_2_extract_id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {vector_2->result_id()}},
             {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}})));

    // Multiplies the pair of components.
    float_multiplication_ids[i] = message_.fresh_ids(fresh_id_index++);
    fuzzerutil::UpdateModuleIdBound(ir_context, float_multiplication_ids[i]);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpFMul, linear_algebra_instruction->type_id(),
        float_multiplication_ids[i],
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {vector_1_extract_id}},
             {SPV_OPERAND_TYPE_ID, {vector_2_extract_id}}})));
  }

  // If the vector has 2 components, then there will be 2 float multiplication
  // instructions.
  if (vectors_component_count == 2) {
    linear_algebra_instruction->SetOpcode(spv::Op::OpFAdd);
    linear_algebra_instruction->SetInOperand(0, {float_multiplication_ids[0]});
    linear_algebra_instruction->SetInOperand(1, {float_multiplication_ids[1]});
  } else {
    // The first OpFAdd instruction has as operands the first two OpFMul
    // instructions.
    std::vector<uint32_t> float_add_ids;
    uint32_t float_add_id = message_.fresh_ids(fresh_id_index++);
    float_add_ids.push_back(float_add_id);
    fuzzerutil::UpdateModuleIdBound(ir_context, float_add_id);
    linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpFAdd, linear_algebra_instruction->type_id(),
        float_add_id,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {float_multiplication_ids[0]}},
             {SPV_OPERAND_TYPE_ID, {float_multiplication_ids[1]}}})));

    // The remaining OpFAdd instructions has as operands an OpFMul and an OpFAdd
    // instruction.
    for (uint32_t i = 2; i < float_multiplication_ids.size() - 1; i++) {
      float_add_id = message_.fresh_ids(fresh_id_index++);
      fuzzerutil::UpdateModuleIdBound(ir_context, float_add_id);
      float_add_ids.push_back(float_add_id);
      linear_algebra_instruction->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpFAdd, linear_algebra_instruction->type_id(),
          float_add_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {float_multiplication_ids[i]}},
               {SPV_OPERAND_TYPE_ID, {float_add_ids[i - 2]}}})));
    }

    // The last OpFAdd instruction is got by changing some of the OpDot
    // instruction attributes.
    linear_algebra_instruction->SetOpcode(spv::Op::OpFAdd);
    linear_algebra_instruction->SetInOperand(
        0, {float_multiplication_ids[float_multiplication_ids.size() - 1]});
    linear_algebra_instruction->SetInOperand(
        1, {float_add_ids[float_add_ids.size() - 1]});
  }
}

std::unordered_set<uint32_t>
TransformationReplaceLinearAlgebraInstruction::GetFreshIds() const {
  std::unordered_set<uint32_t> result;
  for (auto id : message_.fresh_ids()) {
    result.insert(id);
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
