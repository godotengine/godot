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

#include "source/fuzz/transformation.h"

#include <cassert>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_access_chain.h"
#include "source/fuzz/transformation_add_bit_instruction_synonym.h"
#include "source/fuzz/transformation_add_constant_boolean.h"
#include "source/fuzz/transformation_add_constant_composite.h"
#include "source/fuzz/transformation_add_constant_null.h"
#include "source/fuzz/transformation_add_constant_scalar.h"
#include "source/fuzz/transformation_add_copy_memory.h"
#include "source/fuzz/transformation_add_dead_block.h"
#include "source/fuzz/transformation_add_dead_break.h"
#include "source/fuzz/transformation_add_dead_continue.h"
#include "source/fuzz/transformation_add_early_terminator_wrapper.h"
#include "source/fuzz/transformation_add_function.h"
#include "source/fuzz/transformation_add_global_undef.h"
#include "source/fuzz/transformation_add_global_variable.h"
#include "source/fuzz/transformation_add_image_sample_unused_components.h"
#include "source/fuzz/transformation_add_local_variable.h"
#include "source/fuzz/transformation_add_loop_preheader.h"
#include "source/fuzz/transformation_add_loop_to_create_int_constant_synonym.h"
#include "source/fuzz/transformation_add_no_contraction_decoration.h"
#include "source/fuzz/transformation_add_opphi_synonym.h"
#include "source/fuzz/transformation_add_parameter.h"
#include "source/fuzz/transformation_add_relaxed_decoration.h"
#include "source/fuzz/transformation_add_spec_constant_op.h"
#include "source/fuzz/transformation_add_synonym.h"
#include "source/fuzz/transformation_add_type_array.h"
#include "source/fuzz/transformation_add_type_boolean.h"
#include "source/fuzz/transformation_add_type_float.h"
#include "source/fuzz/transformation_add_type_function.h"
#include "source/fuzz/transformation_add_type_int.h"
#include "source/fuzz/transformation_add_type_matrix.h"
#include "source/fuzz/transformation_add_type_pointer.h"
#include "source/fuzz/transformation_add_type_struct.h"
#include "source/fuzz/transformation_add_type_vector.h"
#include "source/fuzz/transformation_adjust_branch_weights.h"
#include "source/fuzz/transformation_composite_construct.h"
#include "source/fuzz/transformation_composite_extract.h"
#include "source/fuzz/transformation_composite_insert.h"
#include "source/fuzz/transformation_compute_data_synonym_fact_closure.h"
#include "source/fuzz/transformation_duplicate_region_with_selection.h"
#include "source/fuzz/transformation_equation_instruction.h"
#include "source/fuzz/transformation_expand_vector_reduction.h"
#include "source/fuzz/transformation_flatten_conditional_branch.h"
#include "source/fuzz/transformation_function_call.h"
#include "source/fuzz/transformation_inline_function.h"
#include "source/fuzz/transformation_invert_comparison_operator.h"
#include "source/fuzz/transformation_load.h"
#include "source/fuzz/transformation_make_vector_operation_dynamic.h"
#include "source/fuzz/transformation_merge_blocks.h"
#include "source/fuzz/transformation_merge_function_returns.h"
#include "source/fuzz/transformation_move_block_down.h"
#include "source/fuzz/transformation_move_instruction_down.h"
#include "source/fuzz/transformation_mutate_pointer.h"
#include "source/fuzz/transformation_outline_function.h"
#include "source/fuzz/transformation_permute_function_parameters.h"
#include "source/fuzz/transformation_permute_phi_operands.h"
#include "source/fuzz/transformation_propagate_instruction_down.h"
#include "source/fuzz/transformation_propagate_instruction_up.h"
#include "source/fuzz/transformation_push_id_through_variable.h"
#include "source/fuzz/transformation_record_synonymous_constants.h"
#include "source/fuzz/transformation_replace_add_sub_mul_with_carrying_extended.h"
#include "source/fuzz/transformation_replace_boolean_constant_with_constant_binary.h"
#include "source/fuzz/transformation_replace_branch_from_dead_block_with_exit.h"
#include "source/fuzz/transformation_replace_constant_with_uniform.h"
#include "source/fuzz/transformation_replace_copy_memory_with_load_store.h"
#include "source/fuzz/transformation_replace_copy_object_with_store_load.h"
#include "source/fuzz/transformation_replace_id_with_synonym.h"
#include "source/fuzz/transformation_replace_irrelevant_id.h"
#include "source/fuzz/transformation_replace_linear_algebra_instruction.h"
#include "source/fuzz/transformation_replace_load_store_with_copy_memory.h"
#include "source/fuzz/transformation_replace_opphi_id_from_dead_predecessor.h"
#include "source/fuzz/transformation_replace_opselect_with_conditional_branch.h"
#include "source/fuzz/transformation_replace_parameter_with_global.h"
#include "source/fuzz/transformation_replace_params_with_struct.h"
#include "source/fuzz/transformation_set_function_control.h"
#include "source/fuzz/transformation_set_loop_control.h"
#include "source/fuzz/transformation_set_memory_operands_mask.h"
#include "source/fuzz/transformation_set_selection_control.h"
#include "source/fuzz/transformation_split_block.h"
#include "source/fuzz/transformation_store.h"
#include "source/fuzz/transformation_swap_commutable_operands.h"
#include "source/fuzz/transformation_swap_conditional_branch_operands.h"
#include "source/fuzz/transformation_swap_function_variables.h"
#include "source/fuzz/transformation_swap_two_functions.h"
#include "source/fuzz/transformation_toggle_access_chain_instruction.h"
#include "source/fuzz/transformation_vector_shuffle.h"
#include "source/fuzz/transformation_wrap_early_terminator_in_function.h"
#include "source/fuzz/transformation_wrap_region_in_selection.h"
#include "source/fuzz/transformation_wrap_vector_synonym.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

Transformation::~Transformation() = default;

std::unique_ptr<Transformation> Transformation::FromMessage(
    const protobufs::Transformation& message) {
  switch (message.transformation_case()) {
    case protobufs::Transformation::TransformationCase::kAccessChain:
      return MakeUnique<TransformationAccessChain>(message.access_chain());
    case protobufs::Transformation::TransformationCase::
        kAddBitInstructionSynonym:
      return MakeUnique<TransformationAddBitInstructionSynonym>(
          message.add_bit_instruction_synonym());
    case protobufs::Transformation::TransformationCase::kAddConstantBoolean:
      return MakeUnique<TransformationAddConstantBoolean>(
          message.add_constant_boolean());
    case protobufs::Transformation::TransformationCase::kAddConstantComposite:
      return MakeUnique<TransformationAddConstantComposite>(
          message.add_constant_composite());
    case protobufs::Transformation::TransformationCase::kAddConstantNull:
      return MakeUnique<TransformationAddConstantNull>(
          message.add_constant_null());
    case protobufs::Transformation::TransformationCase::kAddConstantScalar:
      return MakeUnique<TransformationAddConstantScalar>(
          message.add_constant_scalar());
    case protobufs::Transformation::TransformationCase::kAddCopyMemory:
      return MakeUnique<TransformationAddCopyMemory>(message.add_copy_memory());
    case protobufs::Transformation::TransformationCase::kAddDeadBlock:
      return MakeUnique<TransformationAddDeadBlock>(message.add_dead_block());
    case protobufs::Transformation::TransformationCase::kAddDeadBreak:
      return MakeUnique<TransformationAddDeadBreak>(message.add_dead_break());
    case protobufs::Transformation::TransformationCase::kAddDeadContinue:
      return MakeUnique<TransformationAddDeadContinue>(
          message.add_dead_continue());
    case protobufs::Transformation::TransformationCase::
        kAddEarlyTerminatorWrapper:
      return MakeUnique<TransformationAddEarlyTerminatorWrapper>(
          message.add_early_terminator_wrapper());
    case protobufs::Transformation::TransformationCase::kAddFunction:
      return MakeUnique<TransformationAddFunction>(message.add_function());
    case protobufs::Transformation::TransformationCase::kAddGlobalUndef:
      return MakeUnique<TransformationAddGlobalUndef>(
          message.add_global_undef());
    case protobufs::Transformation::TransformationCase::kAddGlobalVariable:
      return MakeUnique<TransformationAddGlobalVariable>(
          message.add_global_variable());
    case protobufs::Transformation::TransformationCase::
        kAddImageSampleUnusedComponents:
      return MakeUnique<TransformationAddImageSampleUnusedComponents>(
          message.add_image_sample_unused_components());
    case protobufs::Transformation::TransformationCase::kAddLocalVariable:
      return MakeUnique<TransformationAddLocalVariable>(
          message.add_local_variable());
    case protobufs::Transformation::TransformationCase::kAddLoopPreheader:
      return MakeUnique<TransformationAddLoopPreheader>(
          message.add_loop_preheader());
    case protobufs::Transformation::TransformationCase::
        kAddLoopToCreateIntConstantSynonym:
      return MakeUnique<TransformationAddLoopToCreateIntConstantSynonym>(
          message.add_loop_to_create_int_constant_synonym());
    case protobufs::Transformation::TransformationCase::
        kAddNoContractionDecoration:
      return MakeUnique<TransformationAddNoContractionDecoration>(
          message.add_no_contraction_decoration());
    case protobufs::Transformation::TransformationCase::kAddOpphiSynonym:
      return MakeUnique<TransformationAddOpPhiSynonym>(
          message.add_opphi_synonym());
    case protobufs::Transformation::TransformationCase::kAddParameter:
      return MakeUnique<TransformationAddParameter>(message.add_parameter());
    case protobufs::Transformation::TransformationCase::kAddRelaxedDecoration:
      return MakeUnique<TransformationAddRelaxedDecoration>(
          message.add_relaxed_decoration());
    case protobufs::Transformation::TransformationCase::kAddSpecConstantOp:
      return MakeUnique<TransformationAddSpecConstantOp>(
          message.add_spec_constant_op());
    case protobufs::Transformation::TransformationCase::kAddSynonym:
      return MakeUnique<TransformationAddSynonym>(message.add_synonym());
    case protobufs::Transformation::TransformationCase::kAddTypeArray:
      return MakeUnique<TransformationAddTypeArray>(message.add_type_array());
    case protobufs::Transformation::TransformationCase::kAddTypeBoolean:
      return MakeUnique<TransformationAddTypeBoolean>(
          message.add_type_boolean());
    case protobufs::Transformation::TransformationCase::kAddTypeFloat:
      return MakeUnique<TransformationAddTypeFloat>(message.add_type_float());
    case protobufs::Transformation::TransformationCase::kAddTypeFunction:
      return MakeUnique<TransformationAddTypeFunction>(
          message.add_type_function());
    case protobufs::Transformation::TransformationCase::kAddTypeInt:
      return MakeUnique<TransformationAddTypeInt>(message.add_type_int());
    case protobufs::Transformation::TransformationCase::kAddTypeMatrix:
      return MakeUnique<TransformationAddTypeMatrix>(message.add_type_matrix());
    case protobufs::Transformation::TransformationCase::kAddTypePointer:
      return MakeUnique<TransformationAddTypePointer>(
          message.add_type_pointer());
    case protobufs::Transformation::TransformationCase::kAddTypeStruct:
      return MakeUnique<TransformationAddTypeStruct>(message.add_type_struct());
    case protobufs::Transformation::TransformationCase::kAddTypeVector:
      return MakeUnique<TransformationAddTypeVector>(message.add_type_vector());
    case protobufs::Transformation::TransformationCase::kAdjustBranchWeights:
      return MakeUnique<TransformationAdjustBranchWeights>(
          message.adjust_branch_weights());
    case protobufs::Transformation::TransformationCase::kCompositeConstruct:
      return MakeUnique<TransformationCompositeConstruct>(
          message.composite_construct());
    case protobufs::Transformation::TransformationCase::kCompositeExtract:
      return MakeUnique<TransformationCompositeExtract>(
          message.composite_extract());
    case protobufs::Transformation::TransformationCase::kCompositeInsert:
      return MakeUnique<TransformationCompositeInsert>(
          message.composite_insert());
    case protobufs::Transformation::TransformationCase::
        kComputeDataSynonymFactClosure:
      return MakeUnique<TransformationComputeDataSynonymFactClosure>(
          message.compute_data_synonym_fact_closure());
    case protobufs::Transformation::TransformationCase::
        kDuplicateRegionWithSelection:
      return MakeUnique<TransformationDuplicateRegionWithSelection>(
          message.duplicate_region_with_selection());
    case protobufs::Transformation::TransformationCase::kEquationInstruction:
      return MakeUnique<TransformationEquationInstruction>(
          message.equation_instruction());
    case protobufs::Transformation::TransformationCase::kExpandVectorReduction:
      return MakeUnique<TransformationExpandVectorReduction>(
          message.expand_vector_reduction());
    case protobufs::Transformation::TransformationCase::
        kFlattenConditionalBranch:
      return MakeUnique<TransformationFlattenConditionalBranch>(
          message.flatten_conditional_branch());
    case protobufs::Transformation::TransformationCase::kFunctionCall:
      return MakeUnique<TransformationFunctionCall>(message.function_call());
    case protobufs::Transformation::TransformationCase::kInlineFunction:
      return MakeUnique<TransformationInlineFunction>(
          message.inline_function());
    case protobufs::Transformation::TransformationCase::
        kInvertComparisonOperator:
      return MakeUnique<TransformationInvertComparisonOperator>(
          message.invert_comparison_operator());
    case protobufs::Transformation::TransformationCase::kLoad:
      return MakeUnique<TransformationLoad>(message.load());
    case protobufs::Transformation::TransformationCase::
        kMakeVectorOperationDynamic:
      return MakeUnique<TransformationMakeVectorOperationDynamic>(
          message.make_vector_operation_dynamic());
    case protobufs::Transformation::TransformationCase::kMergeBlocks:
      return MakeUnique<TransformationMergeBlocks>(message.merge_blocks());
    case protobufs::Transformation::TransformationCase::kMergeFunctionReturns:
      return MakeUnique<TransformationMergeFunctionReturns>(
          message.merge_function_returns());
    case protobufs::Transformation::TransformationCase::kMoveBlockDown:
      return MakeUnique<TransformationMoveBlockDown>(message.move_block_down());
    case protobufs::Transformation::TransformationCase::kMoveInstructionDown:
      return MakeUnique<TransformationMoveInstructionDown>(
          message.move_instruction_down());
    case protobufs::Transformation::TransformationCase::kMutatePointer:
      return MakeUnique<TransformationMutatePointer>(message.mutate_pointer());
    case protobufs::Transformation::TransformationCase::kOutlineFunction:
      return MakeUnique<TransformationOutlineFunction>(
          message.outline_function());
    case protobufs::Transformation::TransformationCase::
        kPermuteFunctionParameters:
      return MakeUnique<TransformationPermuteFunctionParameters>(
          message.permute_function_parameters());
    case protobufs::Transformation::TransformationCase::kPermutePhiOperands:
      return MakeUnique<TransformationPermutePhiOperands>(
          message.permute_phi_operands());
    case protobufs::Transformation::TransformationCase::
        kPropagateInstructionDown:
      return MakeUnique<TransformationPropagateInstructionDown>(
          message.propagate_instruction_down());
    case protobufs::Transformation::TransformationCase::kPropagateInstructionUp:
      return MakeUnique<TransformationPropagateInstructionUp>(
          message.propagate_instruction_up());
    case protobufs::Transformation::TransformationCase::kPushIdThroughVariable:
      return MakeUnique<TransformationPushIdThroughVariable>(
          message.push_id_through_variable());
    case protobufs::Transformation::TransformationCase::
        kRecordSynonymousConstants:
      return MakeUnique<TransformationRecordSynonymousConstants>(
          message.record_synonymous_constants());
    case protobufs::Transformation::TransformationCase::
        kReplaceAddSubMulWithCarryingExtended:
      return MakeUnique<TransformationReplaceAddSubMulWithCarryingExtended>(
          message.replace_add_sub_mul_with_carrying_extended());
    case protobufs::Transformation::TransformationCase::
        kReplaceBooleanConstantWithConstantBinary:
      return MakeUnique<TransformationReplaceBooleanConstantWithConstantBinary>(
          message.replace_boolean_constant_with_constant_binary());
    case protobufs::Transformation::TransformationCase::
        kReplaceBranchFromDeadBlockWithExit:
      return MakeUnique<TransformationReplaceBranchFromDeadBlockWithExit>(
          message.replace_branch_from_dead_block_with_exit());
    case protobufs::Transformation::TransformationCase::
        kReplaceConstantWithUniform:
      return MakeUnique<TransformationReplaceConstantWithUniform>(
          message.replace_constant_with_uniform());
    case protobufs::Transformation::TransformationCase::
        kReplaceCopyMemoryWithLoadStore:
      return MakeUnique<TransformationReplaceCopyMemoryWithLoadStore>(
          message.replace_copy_memory_with_load_store());
    case protobufs::Transformation::TransformationCase::
        kReplaceCopyObjectWithStoreLoad:
      return MakeUnique<TransformationReplaceCopyObjectWithStoreLoad>(
          message.replace_copy_object_with_store_load());
    case protobufs::Transformation::TransformationCase::kReplaceIdWithSynonym:
      return MakeUnique<TransformationReplaceIdWithSynonym>(
          message.replace_id_with_synonym());
    case protobufs::Transformation::TransformationCase::kReplaceIrrelevantId:
      return MakeUnique<TransformationReplaceIrrelevantId>(
          message.replace_irrelevant_id());
    case protobufs::Transformation::TransformationCase::
        kReplaceLinearAlgebraInstruction:
      return MakeUnique<TransformationReplaceLinearAlgebraInstruction>(
          message.replace_linear_algebra_instruction());
    case protobufs::Transformation::TransformationCase::
        kReplaceLoadStoreWithCopyMemory:
      return MakeUnique<TransformationReplaceLoadStoreWithCopyMemory>(
          message.replace_load_store_with_copy_memory());
    case protobufs::Transformation::TransformationCase::
        kReplaceOpselectWithConditionalBranch:
      return MakeUnique<TransformationReplaceOpSelectWithConditionalBranch>(
          message.replace_opselect_with_conditional_branch());
    case protobufs::Transformation::TransformationCase::
        kReplaceParameterWithGlobal:
      return MakeUnique<TransformationReplaceParameterWithGlobal>(
          message.replace_parameter_with_global());
    case protobufs::Transformation::TransformationCase::
        kReplaceParamsWithStruct:
      return MakeUnique<TransformationReplaceParamsWithStruct>(
          message.replace_params_with_struct());
    case protobufs::Transformation::TransformationCase::
        kReplaceOpphiIdFromDeadPredecessor:
      return MakeUnique<TransformationReplaceOpPhiIdFromDeadPredecessor>(
          message.replace_opphi_id_from_dead_predecessor());
    case protobufs::Transformation::TransformationCase::kSetFunctionControl:
      return MakeUnique<TransformationSetFunctionControl>(
          message.set_function_control());
    case protobufs::Transformation::TransformationCase::kSetLoopControl:
      return MakeUnique<TransformationSetLoopControl>(
          message.set_loop_control());
    case protobufs::Transformation::TransformationCase::kSetMemoryOperandsMask:
      return MakeUnique<TransformationSetMemoryOperandsMask>(
          message.set_memory_operands_mask());
    case protobufs::Transformation::TransformationCase::kSetSelectionControl:
      return MakeUnique<TransformationSetSelectionControl>(
          message.set_selection_control());
    case protobufs::Transformation::TransformationCase::kSplitBlock:
      return MakeUnique<TransformationSplitBlock>(message.split_block());
    case protobufs::Transformation::TransformationCase::kStore:
      return MakeUnique<TransformationStore>(message.store());
    case protobufs::Transformation::TransformationCase::kSwapCommutableOperands:
      return MakeUnique<TransformationSwapCommutableOperands>(
          message.swap_commutable_operands());
    case protobufs::Transformation::TransformationCase::
        kSwapConditionalBranchOperands:
      return MakeUnique<TransformationSwapConditionalBranchOperands>(
          message.swap_conditional_branch_operands());
    case protobufs::Transformation::TransformationCase::kSwapFunctionVariables:
      return MakeUnique<TransformationSwapFunctionVariables>(
          message.swap_function_variables());
    case protobufs::Transformation::TransformationCase::kSwapTwoFunctions:
      return MakeUnique<TransformationSwapTwoFunctions>(
          message.swap_two_functions());
    case protobufs::Transformation::TransformationCase::
        kToggleAccessChainInstruction:
      return MakeUnique<TransformationToggleAccessChainInstruction>(
          message.toggle_access_chain_instruction());
    case protobufs::Transformation::TransformationCase::kVectorShuffle:
      return MakeUnique<TransformationVectorShuffle>(message.vector_shuffle());
    case protobufs::Transformation::TransformationCase::
        kWrapEarlyTerminatorInFunction:
      return MakeUnique<TransformationWrapEarlyTerminatorInFunction>(
          message.wrap_early_terminator_in_function());
    case protobufs::Transformation::TransformationCase::kWrapRegionInSelection:
      return MakeUnique<TransformationWrapRegionInSelection>(
          message.wrap_region_in_selection());
    case protobufs::Transformation::TransformationCase::kWrapVectorSynonym:
      return MakeUnique<TransformationWrapVectorSynonym>(
          message.wrap_vector_synonym());
    case protobufs::Transformation::TRANSFORMATION_NOT_SET:
      assert(false && "An unset transformation was encountered.");
      return nullptr;
  }
  assert(false && "Should be unreachable as all cases must be handled above.");
  return nullptr;
}

bool Transformation::CheckIdIsFreshAndNotUsedByThisTransformation(
    uint32_t id, opt::IRContext* ir_context,
    std::set<uint32_t>* ids_used_by_this_transformation) {
  if (!fuzzerutil::IsFreshId(ir_context, id)) {
    return false;
  }
  if (ids_used_by_this_transformation->count(id) != 0) {
    return false;
  }
  ids_used_by_this_transformation->insert(id);
  return true;
}

}  // namespace fuzz
}  // namespace spvtools
