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

#include "source/fuzz/transformation_expand_vector_reduction.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationExpandVectorReduction::TransformationExpandVectorReduction(
    protobufs::TransformationExpandVectorReduction message)
    : message_(std::move(message)) {}

TransformationExpandVectorReduction::TransformationExpandVectorReduction(
    const uint32_t instruction_result_id,
    const std::vector<uint32_t>& fresh_ids) {
  message_.set_instruction_result_id(instruction_result_id);
  *message_.mutable_fresh_ids() =
      google::protobuf::RepeatedField<google::protobuf::uint32>(
          fresh_ids.begin(), fresh_ids.end());
}

bool TransformationExpandVectorReduction::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  auto* instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.instruction_result_id());

  // |instruction| must be defined.
  if (!instruction) {
    return false;
  }

  // |instruction| must be OpAny or OpAll.
  if (instruction->opcode() != spv::Op::OpAny &&
      instruction->opcode() != spv::Op::OpAll) {
    return false;
  }

  // |message_.fresh_ids.size| must have the exact number of fresh ids required
  // to apply the transformation.
  if (static_cast<uint32_t>(message_.fresh_ids().size()) !=
      GetRequiredFreshIdCount(ir_context, instruction)) {
    return false;
  }

  std::set<uint32_t> ids_used_by_this_transformation;
  for (uint32_t fresh_id : message_.fresh_ids()) {
    // All ids in |message_.fresh_ids| must be fresh.
    if (!fuzzerutil::IsFreshId(ir_context, fresh_id)) {
      return false;
    }

    // All fresh ids need to be distinct.
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(
            fresh_id, ir_context, &ids_used_by_this_transformation)) {
      return false;
    }
  }

  return true;
}

void TransformationExpandVectorReduction::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  auto* instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.instruction_result_id());
  auto* vector = ir_context->get_def_use_mgr()->GetDef(
      instruction->GetSingleWordInOperand(0));
  uint32_t vector_component_count = ir_context->get_type_mgr()
                                        ->GetType(vector->type_id())
                                        ->AsVector()
                                        ->element_count();

  // Fresh id iterator.
  auto fresh_id = message_.fresh_ids().begin();

  // |vector_components| are the ids of the extracted components from |vector|.
  std::vector<uint32_t> vector_components;

  for (uint32_t i = 0; i < vector_component_count; i++) {
    // Extracts the i-th |vector| component.
    auto vector_component =
        opt::Instruction(ir_context, spv::Op::OpCompositeExtract,
                         instruction->type_id(), *fresh_id++,
                         {{SPV_OPERAND_TYPE_ID, {vector->result_id()}},
                          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {i}}});
    instruction->InsertBefore(MakeUnique<opt::Instruction>(vector_component));
    fuzzerutil::UpdateModuleIdBound(ir_context, vector_component.result_id());
    vector_components.push_back(vector_component.result_id());
  }

  // The first two |vector| components are used in the first logical operation.
  auto logical_instruction = opt::Instruction(
      ir_context,
      instruction->opcode() == spv::Op::OpAny ? spv::Op::OpLogicalOr
                                              : spv::Op::OpLogicalAnd,
      instruction->type_id(), *fresh_id++,
      {{SPV_OPERAND_TYPE_ID, {vector_components[0]}},
       {SPV_OPERAND_TYPE_ID, {vector_components[1]}}});
  instruction->InsertBefore(MakeUnique<opt::Instruction>(logical_instruction));
  fuzzerutil::UpdateModuleIdBound(ir_context, logical_instruction.result_id());

  // Evaluates the remaining components.
  for (uint32_t i = 2; i < vector_components.size(); i++) {
    logical_instruction = opt::Instruction(
        ir_context, logical_instruction.opcode(), instruction->type_id(),
        *fresh_id++,
        {{SPV_OPERAND_TYPE_ID, {vector_components[i]}},
         {SPV_OPERAND_TYPE_ID, {logical_instruction.result_id()}}});
    instruction->InsertBefore(
        MakeUnique<opt::Instruction>(logical_instruction));
    fuzzerutil::UpdateModuleIdBound(ir_context,
                                    logical_instruction.result_id());
  }

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

  // If it's possible to make a synonym of |instruction|, then add the fact that
  // the last |logical_instruction| is a synonym of |instruction|.
  if (fuzzerutil::CanMakeSynonymOf(ir_context, *transformation_context,
                                   *instruction)) {
    transformation_context->GetFactManager()->AddFactDataSynonym(
        MakeDataDescriptor(logical_instruction.result_id(), {}),
        MakeDataDescriptor(instruction->result_id(), {}));
  }
}

protobufs::Transformation TransformationExpandVectorReduction::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_expand_vector_reduction() = message_;
  return result;
}

uint32_t TransformationExpandVectorReduction::GetRequiredFreshIdCount(
    opt::IRContext* ir_context, opt::Instruction* instruction) {
  // For each vector component, 1 OpCompositeExtract and 1 OpLogical* (except
  // for the first component) instructions will be inserted.
  return 2 * ir_context->get_type_mgr()
                 ->GetType(ir_context->get_def_use_mgr()
                               ->GetDef(instruction->GetSingleWordInOperand(0))
                               ->type_id())
                 ->AsVector()
                 ->element_count() -
         1;
}

std::unordered_set<uint32_t> TransformationExpandVectorReduction::GetFreshIds()
    const {
  std::unordered_set<uint32_t> result;
  for (auto id : message_.fresh_ids()) {
    result.insert(id);
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
