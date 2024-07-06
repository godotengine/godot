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

#include "source/fuzz/transformation_outline_function.h"

#include <set>

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationOutlineFunction::TransformationOutlineFunction(
    protobufs::TransformationOutlineFunction message)
    : message_(std::move(message)) {}

TransformationOutlineFunction::TransformationOutlineFunction(
    uint32_t entry_block, uint32_t exit_block,
    uint32_t new_function_struct_return_type_id, uint32_t new_function_type_id,
    uint32_t new_function_id, uint32_t new_function_region_entry_block,
    uint32_t new_caller_result_id, uint32_t new_callee_result_id,
    const std::map<uint32_t, uint32_t>& input_id_to_fresh_id,
    const std::map<uint32_t, uint32_t>& output_id_to_fresh_id) {
  message_.set_entry_block(entry_block);
  message_.set_exit_block(exit_block);
  message_.set_new_function_struct_return_type_id(
      new_function_struct_return_type_id);
  message_.set_new_function_type_id(new_function_type_id);
  message_.set_new_function_id(new_function_id);
  message_.set_new_function_region_entry_block(new_function_region_entry_block);
  message_.set_new_caller_result_id(new_caller_result_id);
  message_.set_new_callee_result_id(new_callee_result_id);
  *message_.mutable_input_id_to_fresh_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(input_id_to_fresh_id);
  *message_.mutable_output_id_to_fresh_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(output_id_to_fresh_id);
}

bool TransformationOutlineFunction::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  std::set<uint32_t> ids_used_by_this_transformation;

  // The various new ids used by the transformation must be fresh and distinct.

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_function_struct_return_type_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_function_type_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_function_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_function_region_entry_block(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_caller_result_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  if (!CheckIdIsFreshAndNotUsedByThisTransformation(
          message_.new_callee_result_id(), ir_context,
          &ids_used_by_this_transformation)) {
    return false;
  }

  for (auto& pair : message_.input_id_to_fresh_id()) {
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(
            pair.second(), ir_context, &ids_used_by_this_transformation)) {
      return false;
    }
  }

  for (auto& pair : message_.output_id_to_fresh_id()) {
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(
            pair.second(), ir_context, &ids_used_by_this_transformation)) {
      return false;
    }
  }

  // The entry and exit block ids must indeed refer to blocks.
  for (auto block_id : {message_.entry_block(), message_.exit_block()}) {
    auto block_label = ir_context->get_def_use_mgr()->GetDef(block_id);
    if (!block_label || block_label->opcode() != spv::Op::OpLabel) {
      return false;
    }
  }

  auto entry_block = ir_context->cfg()->block(message_.entry_block());
  auto exit_block = ir_context->cfg()->block(message_.exit_block());

  // The entry block cannot start with OpVariable - this would mean that
  // outlining would remove a variable from the function containing the region
  // being outlined.
  if (entry_block->begin()->opcode() == spv::Op::OpVariable) {
    return false;
  }

  // For simplicity, we do not allow the entry block to be a loop header.
  if (entry_block->GetLoopMergeInst()) {
    return false;
  }

  // For simplicity, we do not allow the exit block to be a merge block or
  // continue target.
  if (fuzzerutil::IsMergeOrContinue(ir_context, exit_block->id())) {
    return false;
  }

  // The entry block cannot start with OpPhi.  This is to keep the
  // transformation logic simple.  (Another transformation to split the OpPhis
  // from a block could be applied to avoid this scenario.)
  if (entry_block->begin()->opcode() == spv::Op::OpPhi) {
    return false;
  }

  // The block must be in the same function.
  if (entry_block->GetParent() != exit_block->GetParent()) {
    return false;
  }

  // The entry block must dominate the exit block.
  auto dominator_analysis =
      ir_context->GetDominatorAnalysis(entry_block->GetParent());
  if (!dominator_analysis->Dominates(entry_block, exit_block)) {
    return false;
  }

  // The exit block must post-dominate the entry block.
  auto postdominator_analysis =
      ir_context->GetPostDominatorAnalysis(entry_block->GetParent());
  if (!postdominator_analysis->Dominates(exit_block, entry_block)) {
    return false;
  }

  // Find all the blocks dominated by |message_.entry_block| and post-dominated
  // by |message_.exit_block|.
  auto region_set = GetRegionBlocks(
      ir_context,
      entry_block = ir_context->cfg()->block(message_.entry_block()),
      exit_block = ir_context->cfg()->block(message_.exit_block()));

  // Check whether |region_set| really is a single-entry single-exit region, and
  // also check whether structured control flow constructs and their merge
  // and continue constructs are either wholly in or wholly out of the region -
  // e.g. avoid the situation where the region contains the head of a loop but
  // not the loop's continue construct.
  //
  // This is achieved by going through every block in the function that contains
  // the region.
  for (auto& block : *entry_block->GetParent()) {
    if (region_set.count(&block) != 0) {
      // The block is in the region. Check that it does not have any unreachable
      // predecessors. If it does, then we do not regard the region as single-
      // entry-single-exit and hence do not outline it.
      for (auto pred : ir_context->cfg()->preds(block.id())) {
        if (!ir_context->IsReachable(*ir_context->cfg()->block(pred))) {
          // The predecessor is unreachable.
          return false;
        }
      }
    }

    if (&block == exit_block) {
      // It is OK (and typically expected) for the exit block of the region to
      // have successors outside the region.
      //
      // It is also OK for the exit block to head a selection construct: the
      // block containing the call to the outlined function will end up heading
      // this construct if outlining takes place.  However, it is not OK for
      // the exit block to head a loop construct.
      if (block.GetLoopMergeInst()) {
        return false;
      }
      continue;
    }

    if (region_set.count(&block) != 0) {
      // The block is in the region and is not the region's exit block.  Let's
      // see whether all of the block's successors are in the region.  If they
      // are not, the region is not single-entry single-exit.
      bool all_successors_in_region = true;
      block.WhileEachSuccessorLabel([&all_successors_in_region, ir_context,
                                     &region_set](uint32_t successor) -> bool {
        if (region_set.count(ir_context->cfg()->block(successor)) == 0) {
          all_successors_in_region = false;
          return false;
        }
        return true;
      });
      if (!all_successors_in_region) {
        return false;
      }
    }

    if (auto merge = block.GetMergeInst()) {
      // The block is a loop or selection header -- the header and its
      // associated merge block had better both be in the region or both be
      // outside the region.
      auto merge_block =
          ir_context->cfg()->block(merge->GetSingleWordOperand(0));
      if (region_set.count(&block) != region_set.count(merge_block)) {
        return false;
      }
    }

    if (auto loop_merge = block.GetLoopMergeInst()) {
      // Similar to the above, but for the continue target of a loop.
      auto continue_target =
          ir_context->cfg()->block(loop_merge->GetSingleWordOperand(1));
      if (continue_target != exit_block &&
          region_set.count(&block) != region_set.count(continue_target)) {
        return false;
      }
    }
  }

  // For each region input id, i.e. every id defined outside the region but
  // used inside the region, ...
  auto input_id_to_fresh_id_map =
      fuzzerutil::RepeatedUInt32PairToMap(message_.input_id_to_fresh_id());
  for (auto id : GetRegionInputIds(ir_context, region_set, exit_block)) {
    // There needs to be a corresponding fresh id to be used as a function
    // parameter, or overflow ids need to be available.
    if (input_id_to_fresh_id_map.count(id) == 0 &&
        !transformation_context.GetOverflowIdSource()->HasOverflowIds()) {
      return false;
    }
    // Furthermore, if the input id has pointer type it must be an OpVariable
    // or OpFunctionParameter.
    auto input_id_inst = ir_context->get_def_use_mgr()->GetDef(id);
    if (ir_context->get_def_use_mgr()
            ->GetDef(input_id_inst->type_id())
            ->opcode() == spv::Op::OpTypePointer) {
      switch (input_id_inst->opcode()) {
        case spv::Op::OpFunctionParameter:
        case spv::Op::OpVariable:
          // These are OK.
          break;
        default:
          // Anything else is not OK.
          return false;
      }
    }
  }

  // For each region output id -- i.e. every id defined inside the region but
  // used outside the region, ...
  auto output_id_to_fresh_id_map =
      fuzzerutil::RepeatedUInt32PairToMap(message_.output_id_to_fresh_id());
  for (auto id : GetRegionOutputIds(ir_context, region_set, exit_block)) {
    if (
        // ... there needs to be a corresponding fresh id that can hold the
        // value for this id computed in the outlined function (or overflow ids
        // must be available), and ...
        (output_id_to_fresh_id_map.count(id) == 0 &&
         !transformation_context.GetOverflowIdSource()->HasOverflowIds())
        // ... the output id must not have pointer type (to avoid creating a
        // struct with pointer members to pass data out of the outlined
        // function)
        || ir_context->get_def_use_mgr()
                   ->GetDef(fuzzerutil::GetTypeId(ir_context, id))
                   ->opcode() == spv::Op::OpTypePointer) {
      return false;
    }
  }

  return true;
}

void TransformationOutlineFunction::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // The entry block for the region before outlining.
  auto original_region_entry_block =
      ir_context->cfg()->block(message_.entry_block());

  // The exit block for the region before outlining.
  auto original_region_exit_block =
      ir_context->cfg()->block(message_.exit_block());

  // The single-entry single-exit region defined by |message_.entry_block| and
  // |message_.exit_block|.
  std::set<opt::BasicBlock*> region_blocks = GetRegionBlocks(
      ir_context, original_region_entry_block, original_region_exit_block);

  // Input and output ids for the region being outlined.
  std::vector<uint32_t> region_input_ids =
      GetRegionInputIds(ir_context, region_blocks, original_region_exit_block);
  std::vector<uint32_t> region_output_ids =
      GetRegionOutputIds(ir_context, region_blocks, original_region_exit_block);

  // Maps from input and output ids to fresh ids.
  auto input_id_to_fresh_id_map =
      fuzzerutil::RepeatedUInt32PairToMap(message_.input_id_to_fresh_id());
  auto output_id_to_fresh_id_map =
      fuzzerutil::RepeatedUInt32PairToMap(message_.output_id_to_fresh_id());

  // Use overflow ids to augment these maps at any locations where fresh ids are
  // required but not provided.
  for (uint32_t id : region_input_ids) {
    if (input_id_to_fresh_id_map.count(id) == 0) {
      input_id_to_fresh_id_map.insert(
          {id,
           transformation_context->GetOverflowIdSource()->GetNextOverflowId()});
    }
  }
  for (uint32_t id : region_output_ids) {
    if (output_id_to_fresh_id_map.count(id) == 0) {
      output_id_to_fresh_id_map.insert(
          {id,
           transformation_context->GetOverflowIdSource()->GetNextOverflowId()});
    }
  }

  UpdateModuleIdBoundForFreshIds(ir_context, input_id_to_fresh_id_map,
                                 output_id_to_fresh_id_map);

  // Construct a map that associates each output id with its type id.
  std::map<uint32_t, uint32_t> output_id_to_type_id;
  for (uint32_t output_id : region_output_ids) {
    output_id_to_type_id[output_id] =
        ir_context->get_def_use_mgr()->GetDef(output_id)->type_id();
  }

  // The region will be collapsed to a single block that calls a function
  // containing the outlined region.  This block needs to end with whatever
  // the exit block of the region ended with before outlining.  We thus clone
  // the terminator of the region's exit block, and the merge instruction for
  // the block if there is one, so that we can append them to the end of the
  // collapsed block later.
  std::unique_ptr<opt::Instruction> cloned_exit_block_terminator =
      std::unique_ptr<opt::Instruction>(
          original_region_exit_block->terminator()->Clone(ir_context));
  std::unique_ptr<opt::Instruction> cloned_exit_block_merge =
      original_region_exit_block->GetMergeInst()
          ? std::unique_ptr<opt::Instruction>(
                original_region_exit_block->GetMergeInst()->Clone(ir_context))
          : nullptr;

  // Make a function prototype for the outlined function, which involves
  // figuring out its required type.
  std::unique_ptr<opt::Function> outlined_function = PrepareFunctionPrototype(
      region_input_ids, region_output_ids, input_id_to_fresh_id_map, ir_context,
      transformation_context);

  // Adapt the region to be outlined so that its input ids are replaced with the
  // ids of the outlined function's input parameters, and so that output ids
  // are similarly remapped.
  RemapInputAndOutputIdsInRegion(
      ir_context, *original_region_exit_block, region_blocks, region_input_ids,
      region_output_ids, input_id_to_fresh_id_map, output_id_to_fresh_id_map);

  // Fill out the body of the outlined function according to the region that is
  // being outlined.
  PopulateOutlinedFunction(
      *original_region_entry_block, *original_region_exit_block, region_blocks,
      region_output_ids, output_id_to_type_id, output_id_to_fresh_id_map,
      ir_context, outlined_function.get());

  // Collapse the region that has been outlined into a function down to a single
  // block that calls said function.
  ShrinkOriginalRegion(
      ir_context, region_blocks, region_input_ids, region_output_ids,
      output_id_to_type_id, outlined_function->type_id(),
      std::move(cloned_exit_block_merge),
      std::move(cloned_exit_block_terminator), original_region_entry_block);

  // Add the outlined function to the module.
  const auto* outlined_function_ptr = outlined_function.get();
  ir_context->module()->AddFunction(std::move(outlined_function));

  // Major surgery has been conducted on the module, so invalidate all analyses.
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);

  // If the original function was livesafe, the new function should also be
  // livesafe.
  if (transformation_context->GetFactManager()->FunctionIsLivesafe(
          original_region_entry_block->GetParent()->result_id())) {
    transformation_context->GetFactManager()->AddFactFunctionIsLivesafe(
        message_.new_function_id());
  }

  // Record the fact that all blocks in the outlined region are dead if the
  // first block is dead.
  if (transformation_context->GetFactManager()->BlockIsDead(
          original_region_entry_block->id())) {
    transformation_context->GetFactManager()->AddFactBlockIsDead(
        outlined_function_ptr->entry()->id());
  }
}

protobufs::Transformation TransformationOutlineFunction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_outline_function() = message_;
  return result;
}

std::vector<uint32_t> TransformationOutlineFunction::GetRegionInputIds(
    opt::IRContext* ir_context, const std::set<opt::BasicBlock*>& region_set,
    opt::BasicBlock* region_exit_block) {
  std::vector<uint32_t> result;

  auto enclosing_function = region_exit_block->GetParent();

  // Consider each parameter of the function containing the region.
  enclosing_function->ForEachParam(
      [ir_context, &region_set, &result](opt::Instruction* function_parameter) {
        // Consider every use of the parameter.
        ir_context->get_def_use_mgr()->WhileEachUse(
            function_parameter,
            [ir_context, function_parameter, &region_set, &result](
                opt::Instruction* use, uint32_t /*unused*/) {
              // Get the block, if any, in which the parameter is used.
              auto use_block = ir_context->get_instr_block(use);
              // If the use is in a block that lies within the region, the
              // parameter is an input id for the region.
              if (use_block && region_set.count(use_block) != 0) {
                result.push_back(function_parameter->result_id());
                return false;
              }
              return true;
            });
      });

  // Consider all definitions in the function that might turn out to be input
  // ids.
  for (auto& block : *enclosing_function) {
    std::vector<opt::Instruction*> candidate_input_ids_for_block;
    if (region_set.count(&block) == 0) {
      // All instructions in blocks outside the region are candidate's for
      // generating input ids.
      for (auto& inst : block) {
        candidate_input_ids_for_block.push_back(&inst);
      }
    } else {
      // Blocks in the region cannot generate input ids.
      continue;
    }

    // Consider each candidate input id to check whether it is used in the
    // region.
    for (auto& inst : candidate_input_ids_for_block) {
      ir_context->get_def_use_mgr()->WhileEachUse(
          inst,
          [ir_context, &inst, region_exit_block, &region_set, &result](
              opt::Instruction* use, uint32_t /*unused*/) -> bool {
            // Find the block in which this id use occurs, recording the id as
            // an input id if the block is outside the region, with some
            // exceptions detailed below.
            auto use_block = ir_context->get_instr_block(use);

            if (!use_block) {
              // There might be no containing block, e.g. if the use is in a
              // decoration.
              return true;
            }

            if (region_set.count(use_block) == 0) {
              // The use is not in the region: this does not make it an input
              // id.
              return true;
            }

            if (use_block == region_exit_block && use->IsBlockTerminator()) {
              // We do not regard uses in the exit block terminator as input
              // ids, as this terminator does not get outlined.
              return true;
            }

            result.push_back(inst->result_id());
            return false;
          });
    }
  }
  return result;
}

std::vector<uint32_t> TransformationOutlineFunction::GetRegionOutputIds(
    opt::IRContext* ir_context, const std::set<opt::BasicBlock*>& region_set,
    opt::BasicBlock* region_exit_block) {
  std::vector<uint32_t> result;

  // Consider each block in the function containing the region.
  for (auto& block : *region_exit_block->GetParent()) {
    if (region_set.count(&block) == 0) {
      // Skip blocks that are not in the region.
      continue;
    }
    // Consider each use of each instruction defined in the block.
    for (auto& inst : block) {
      ir_context->get_def_use_mgr()->WhileEachUse(
          &inst,
          [&region_set, ir_context, &inst, region_exit_block, &result](
              opt::Instruction* use, uint32_t /*unused*/) -> bool {
            // Find the block in which this id use occurs, recording the id as
            // an output id if the block is outside the region, with some
            // exceptions detailed below.
            auto use_block = ir_context->get_instr_block(use);

            if (!use_block) {
              // There might be no containing block, e.g. if the use is in a
              // decoration.
              return true;
            }

            if (region_set.count(use_block) != 0) {
              // The use is in the region.
              if (use_block != region_exit_block || !use->IsBlockTerminator()) {
                // Furthermore, the use is not in the terminator of the region's
                // exit block.
                return true;
              }
            }

            result.push_back(inst.result_id());
            return false;
          });
    }
  }
  return result;
}

std::set<opt::BasicBlock*> TransformationOutlineFunction::GetRegionBlocks(
    opt::IRContext* ir_context, opt::BasicBlock* entry_block,
    opt::BasicBlock* exit_block) {
  auto enclosing_function = entry_block->GetParent();
  auto dominator_analysis =
      ir_context->GetDominatorAnalysis(enclosing_function);
  auto postdominator_analysis =
      ir_context->GetPostDominatorAnalysis(enclosing_function);

  std::set<opt::BasicBlock*> result;
  for (auto& block : *enclosing_function) {
    if (dominator_analysis->Dominates(entry_block, &block) &&
        postdominator_analysis->Dominates(exit_block, &block)) {
      result.insert(&block);
    }
  }
  return result;
}

std::unique_ptr<opt::Function>
TransformationOutlineFunction::PrepareFunctionPrototype(
    const std::vector<uint32_t>& region_input_ids,
    const std::vector<uint32_t>& region_output_ids,
    const std::map<uint32_t, uint32_t>& input_id_to_fresh_id_map,
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  uint32_t return_type_id = 0;
  uint32_t function_type_id = 0;

  // First, try to find an existing function type that is suitable.  This is
  // only possible if the region generates no output ids; if it generates output
  // ids we are going to make a new struct for those, and since that struct does
  // not exist there cannot already be a function type with this struct as its
  // return type.
  if (region_output_ids.empty()) {
    std::vector<uint32_t> return_and_parameter_types;
    opt::analysis::Void void_type;
    return_type_id = ir_context->get_type_mgr()->GetId(&void_type);
    return_and_parameter_types.push_back(return_type_id);
    for (auto id : region_input_ids) {
      return_and_parameter_types.push_back(
          ir_context->get_def_use_mgr()->GetDef(id)->type_id());
    }
    function_type_id =
        fuzzerutil::FindFunctionType(ir_context, return_and_parameter_types);
  }

  // If no existing function type was found, we need to create one.
  if (function_type_id == 0) {
    assert(
        ((return_type_id == 0) == !region_output_ids.empty()) &&
        "We should only have set the return type if there are no output ids.");
    // If the region generates output ids, we need to make a struct with one
    // field per output id.
    if (!region_output_ids.empty()) {
      opt::Instruction::OperandList struct_member_types;
      for (uint32_t output_id : region_output_ids) {
        auto output_id_type =
            ir_context->get_def_use_mgr()->GetDef(output_id)->type_id();
        if (ir_context->get_def_use_mgr()->GetDef(output_id_type)->opcode() ==
            spv::Op::OpTypeVoid) {
          // We cannot add a void field to a struct.  We instead use OpUndef to
          // handle void output ids.
          continue;
        }
        struct_member_types.push_back({SPV_OPERAND_TYPE_ID, {output_id_type}});
      }
      // Add a new struct type to the module.
      ir_context->module()->AddType(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpTypeStruct, 0,
          message_.new_function_struct_return_type_id(),
          std::move(struct_member_types)));
      // The return type for the function is the newly-created struct.
      return_type_id = message_.new_function_struct_return_type_id();
    }
    assert(
        return_type_id != 0 &&
        "We should either have a void return type, or have created a struct.");

    // The region's input ids dictate the parameter types to the function.
    opt::Instruction::OperandList function_type_operands;
    function_type_operands.push_back({SPV_OPERAND_TYPE_ID, {return_type_id}});
    for (auto id : region_input_ids) {
      function_type_operands.push_back(
          {SPV_OPERAND_TYPE_ID,
           {ir_context->get_def_use_mgr()->GetDef(id)->type_id()}});
    }
    // Add a new function type to the module, and record that this is the type
    // id for the new function.
    ir_context->module()->AddType(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpTypeFunction, 0, message_.new_function_type_id(),
        function_type_operands));
    function_type_id = message_.new_function_type_id();
  }

  // Create a new function with |message_.new_function_id| as the function id,
  // and the return type and function type prepared above.
  std::unique_ptr<opt::Function> outlined_function =
      MakeUnique<opt::Function>(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpFunction, return_type_id,
          message_.new_function_id(),
          opt::Instruction::OperandList(
              {{spv_operand_type_t ::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                {uint32_t(spv::FunctionControlMask::MaskNone)}},
               {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                {function_type_id}}})));

  // Add one parameter to the function for each input id, using the fresh ids
  // provided in |input_id_to_fresh_id_map|, or overflow ids if needed.
  for (auto id : region_input_ids) {
    uint32_t fresh_id = input_id_to_fresh_id_map.at(id);
    outlined_function->AddParameter(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpFunctionParameter,
        ir_context->get_def_use_mgr()->GetDef(id)->type_id(), fresh_id,
        opt::Instruction::OperandList()));

    // Analyse the use of the new parameter instruction.
    outlined_function->ForEachParam(
        [fresh_id, ir_context](opt::Instruction* inst) {
          if (inst->result_id() == fresh_id) {
            ir_context->AnalyzeDefUse(inst);
          }
        });

    // If the input id is an irrelevant-valued variable, the same should be true
    // of the corresponding parameter.
    if (transformation_context->GetFactManager()->PointeeValueIsIrrelevant(
            id)) {
      transformation_context->GetFactManager()
          ->AddFactValueOfPointeeIsIrrelevant(input_id_to_fresh_id_map.at(id));
    }
  }

  return outlined_function;
}

void TransformationOutlineFunction::UpdateModuleIdBoundForFreshIds(
    opt::IRContext* ir_context,
    const std::map<uint32_t, uint32_t>& input_id_to_fresh_id_map,
    const std::map<uint32_t, uint32_t>& output_id_to_fresh_id_map) const {
  // Enlarge the module's id bound as needed to accommodate the various fresh
  // ids associated with the transformation.
  fuzzerutil::UpdateModuleIdBound(
      ir_context, message_.new_function_struct_return_type_id());
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.new_function_type_id());
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.new_function_id());
  fuzzerutil::UpdateModuleIdBound(ir_context,
                                  message_.new_function_region_entry_block());
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.new_caller_result_id());
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.new_callee_result_id());

  for (auto& entry : input_id_to_fresh_id_map) {
    fuzzerutil::UpdateModuleIdBound(ir_context, entry.second);
  }

  for (auto& entry : output_id_to_fresh_id_map) {
    fuzzerutil::UpdateModuleIdBound(ir_context, entry.second);
  }
}

void TransformationOutlineFunction::RemapInputAndOutputIdsInRegion(
    opt::IRContext* ir_context,
    const opt::BasicBlock& original_region_exit_block,
    const std::set<opt::BasicBlock*>& region_blocks,
    const std::vector<uint32_t>& region_input_ids,
    const std::vector<uint32_t>& region_output_ids,
    const std::map<uint32_t, uint32_t>& input_id_to_fresh_id_map,
    const std::map<uint32_t, uint32_t>& output_id_to_fresh_id_map) const {
  // Change all uses of input ids inside the region to the corresponding fresh
  // ids that will ultimately be parameters of the outlined function.
  // This is done by considering each region input id in turn.
  for (uint32_t id : region_input_ids) {
    // We then consider each use of the input id.
    ir_context->get_def_use_mgr()->ForEachUse(
        id, [ir_context, id, &input_id_to_fresh_id_map, region_blocks](
                opt::Instruction* use, uint32_t operand_index) {
          // Find the block in which this use of the input id occurs.
          opt::BasicBlock* use_block = ir_context->get_instr_block(use);
          // We want to rewrite the use id if its block occurs in the outlined
          // region.
          if (region_blocks.count(use_block) != 0) {
            // Rewrite this use of the input id.
            use->SetOperand(operand_index, {input_id_to_fresh_id_map.at(id)});
          }
        });
  }

  // Change each definition of a region output id to define the corresponding
  // fresh ids that will store intermediate value for the output ids.  Also
  // change all uses of the output id located in the outlined region.
  // This is done by considering each region output id in turn.
  for (uint32_t id : region_output_ids) {
    // First consider each use of the output id and update the relevant uses.
    ir_context->get_def_use_mgr()->ForEachUse(
        id, [ir_context, &original_region_exit_block, id,
             &output_id_to_fresh_id_map,
             region_blocks](opt::Instruction* use, uint32_t operand_index) {
          // Find the block in which this use of the output id occurs.
          auto use_block = ir_context->get_instr_block(use);
          // We want to rewrite the use id if its block occurs in the outlined
          // region, with one exception: the terminator of the exit block of
          // the region is going to remain in the original function, so if the
          // use appears in such a terminator instruction we leave it alone.
          if (
              // The block is in the region ...
              region_blocks.count(use_block) != 0 &&
              // ... and the use is not in the terminator instruction of the
              // region's exit block.
              !(use_block == &original_region_exit_block &&
                use->IsBlockTerminator())) {
            // Rewrite this use of the output id.
            use->SetOperand(operand_index, {output_id_to_fresh_id_map.at(id)});
          }
        });

    // Now change the instruction that defines the output id so that it instead
    // defines the corresponding fresh id.  We do this after changing all the
    // uses so that the definition of the original id is still registered when
    // we analyse its uses.
    ir_context->get_def_use_mgr()->GetDef(id)->SetResultId(
        output_id_to_fresh_id_map.at(id));
  }
}

void TransformationOutlineFunction::PopulateOutlinedFunction(
    const opt::BasicBlock& original_region_entry_block,
    const opt::BasicBlock& original_region_exit_block,
    const std::set<opt::BasicBlock*>& region_blocks,
    const std::vector<uint32_t>& region_output_ids,
    const std::map<uint32_t, uint32_t>& output_id_to_type_id,
    const std::map<uint32_t, uint32_t>& output_id_to_fresh_id_map,
    opt::IRContext* ir_context, opt::Function* outlined_function) const {
  // When we create the exit block for the outlined region, we use this pointer
  // to track of it so that we can manipulate it later.
  opt::BasicBlock* outlined_region_exit_block = nullptr;

  // The region entry block in the new function is identical to the entry block
  // of the region being outlined, except that it has
  // |message_.new_function_region_entry_block| as its id.
  std::unique_ptr<opt::BasicBlock> outlined_region_entry_block =
      MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpLabel, 0,
          message_.new_function_region_entry_block(),
          opt::Instruction::OperandList()));

  if (&original_region_entry_block == &original_region_exit_block) {
    outlined_region_exit_block = outlined_region_entry_block.get();
  }

  for (auto& inst : original_region_entry_block) {
    outlined_region_entry_block->AddInstruction(
        std::unique_ptr<opt::Instruction>(inst.Clone(ir_context)));
  }
  outlined_function->AddBasicBlock(std::move(outlined_region_entry_block));

  // We now go through the single-entry single-exit region defined by the entry
  // and exit blocks, adding clones of all blocks to the new function.

  // Consider every block in the enclosing function.
  auto enclosing_function = original_region_entry_block.GetParent();
  for (auto block_it = enclosing_function->begin();
       block_it != enclosing_function->end();) {
    // Skip the region's entry block - we already dealt with it above.
    if (region_blocks.count(&*block_it) == 0 ||
        &*block_it == &original_region_entry_block) {
      ++block_it;
      continue;
    }
    // Clone the block so that it can be added to the new function.
    auto cloned_block =
        std::unique_ptr<opt::BasicBlock>(block_it->Clone(ir_context));

    // If this is the region's exit block, then the cloned block is the outlined
    // region's exit block.
    if (&*block_it == &original_region_exit_block) {
      assert(outlined_region_exit_block == nullptr &&
             "We should not yet have encountered the exit block.");
      outlined_region_exit_block = cloned_block.get();
    }

    // Redirect any OpPhi operands whose predecessors are the original region
    // entry block to become the new function entry block.
    cloned_block->ForEachPhiInst([this](opt::Instruction* phi_inst) {
      for (uint32_t predecessor_index = 1;
           predecessor_index < phi_inst->NumInOperands();
           predecessor_index += 2) {
        if (phi_inst->GetSingleWordInOperand(predecessor_index) ==
            message_.entry_block()) {
          phi_inst->SetInOperand(predecessor_index,
                                 {message_.new_function_region_entry_block()});
        }
      }
    });

    outlined_function->AddBasicBlock(std::move(cloned_block));
    block_it = block_it.Erase();
  }
  assert(outlined_region_exit_block != nullptr &&
         "We should have encountered the region's exit block when iterating "
         "through the function");

  // We now need to adapt the exit block for the region - in the new function -
  // so that it ends with a return.

  // We first eliminate the merge instruction (if any) and the terminator for
  // the cloned exit block.
  for (auto inst_it = outlined_region_exit_block->begin();
       inst_it != outlined_region_exit_block->end();) {
    if (inst_it->opcode() == spv::Op::OpLoopMerge ||
        inst_it->opcode() == spv::Op::OpSelectionMerge) {
      inst_it = inst_it.Erase();
    } else if (inst_it->IsBlockTerminator()) {
      inst_it = inst_it.Erase();
    } else {
      ++inst_it;
    }
  }

  // We now add either OpReturn or OpReturnValue as the cloned exit block's
  // terminator.
  if (region_output_ids.empty()) {
    // The case where there are no region output ids is simple: we just add
    // OpReturn.
    outlined_region_exit_block->AddInstruction(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpReturn, 0, 0, opt::Instruction::OperandList()));
  } else {
    // In the case where there are output ids, we add an OpCompositeConstruct
    // instruction to pack all the non-void output values into a struct, and
    // then an OpReturnValue instruction to return this struct.
    opt::Instruction::OperandList struct_member_operands;
    for (uint32_t id : region_output_ids) {
      if (ir_context->get_def_use_mgr()
              ->GetDef(output_id_to_type_id.at(id))
              ->opcode() != spv::Op::OpTypeVoid) {
        struct_member_operands.push_back(
            {SPV_OPERAND_TYPE_ID, {output_id_to_fresh_id_map.at(id)}});
      }
    }
    outlined_region_exit_block->AddInstruction(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpCompositeConstruct,
        message_.new_function_struct_return_type_id(),
        message_.new_callee_result_id(), struct_member_operands));
    outlined_region_exit_block->AddInstruction(MakeUnique<opt::Instruction>(
        ir_context, spv::Op::OpReturnValue, 0, 0,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.new_callee_result_id()}}})));
  }

  outlined_function->SetFunctionEnd(
      MakeUnique<opt::Instruction>(ir_context, spv::Op::OpFunctionEnd, 0, 0,
                                   opt::Instruction::OperandList()));
}

void TransformationOutlineFunction::ShrinkOriginalRegion(
    opt::IRContext* ir_context, const std::set<opt::BasicBlock*>& region_blocks,
    const std::vector<uint32_t>& region_input_ids,
    const std::vector<uint32_t>& region_output_ids,
    const std::map<uint32_t, uint32_t>& output_id_to_type_id,
    uint32_t return_type_id,
    std::unique_ptr<opt::Instruction> cloned_exit_block_merge,
    std::unique_ptr<opt::Instruction> cloned_exit_block_terminator,
    opt::BasicBlock* original_region_entry_block) const {
  // Erase all blocks from the original function that are in the outlined
  // region, except for the region's entry block.
  //
  // In the process, identify all references to the exit block of the region,
  // as merge blocks, continue targets, or OpPhi predecessors, and rewrite them
  // to refer to the region entry block (the single block to which we are
  // shrinking the region).
  auto enclosing_function = original_region_entry_block->GetParent();
  for (auto block_it = enclosing_function->begin();
       block_it != enclosing_function->end();) {
    if (&*block_it == original_region_entry_block) {
      ++block_it;
    } else if (region_blocks.count(&*block_it) == 0) {
      // The block is not in the region.  Check whether it has the last block
      // of the region as an OpPhi predecessor, and if so change the
      // predecessor to be the first block of the region (i.e. the block
      // containing the call to what was outlined).
      assert(block_it->MergeBlockIdIfAny() != message_.exit_block() &&
             "Outlined region must not end with a merge block");
      assert(block_it->ContinueBlockIdIfAny() != message_.exit_block() &&
             "Outlined region must not end with a continue target");
      block_it->ForEachPhiInst([this](opt::Instruction* phi_inst) {
        for (uint32_t predecessor_index = 1;
             predecessor_index < phi_inst->NumInOperands();
             predecessor_index += 2) {
          if (phi_inst->GetSingleWordInOperand(predecessor_index) ==
              message_.exit_block()) {
            phi_inst->SetInOperand(predecessor_index, {message_.entry_block()});
          }
        }
      });
      ++block_it;
    } else {
      // The block is in the region and is not the region's entry block: kill
      // it.
      block_it = block_it.Erase();
    }
  }

  // Now erase all instructions from the region's entry block, as they have
  // been outlined.
  for (auto inst_it = original_region_entry_block->begin();
       inst_it != original_region_entry_block->end();) {
    inst_it = inst_it.Erase();
  }

  // Now we add a call to the outlined function to the region's entry block.
  opt::Instruction::OperandList function_call_operands;
  function_call_operands.push_back(
      {SPV_OPERAND_TYPE_ID, {message_.new_function_id()}});
  // The function parameters are the region input ids.
  for (auto input_id : region_input_ids) {
    function_call_operands.push_back({SPV_OPERAND_TYPE_ID, {input_id}});
  }

  original_region_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpFunctionCall, return_type_id,
      message_.new_caller_result_id(), function_call_operands));

  // If there are output ids, the function call will return a struct.  For each
  // output id, we add an extract operation to pull the appropriate struct
  // member out into an output id.  The exception is for output ids with void
  // type.  There are no struct entries for these, so we use an OpUndef of void
  // type instead.
  uint32_t struct_member_index = 0;
  for (uint32_t output_id : region_output_ids) {
    uint32_t output_type_id = output_id_to_type_id.at(output_id);
    if (ir_context->get_def_use_mgr()->GetDef(output_type_id)->opcode() ==
        spv::Op::OpTypeVoid) {
      original_region_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpUndef, output_type_id, output_id,
          opt::Instruction::OperandList()));
      // struct_member_index is not incremented since there was no struct member
      // associated with this void-typed output id.
    } else {
      original_region_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpCompositeExtract, output_type_id, output_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.new_caller_result_id()}},
               {SPV_OPERAND_TYPE_LITERAL_INTEGER, {struct_member_index}}})));
      struct_member_index++;
    }
  }

  // Finally, we terminate the block with the merge instruction (if any) that
  // used to belong to the region's exit block, and the terminator that used
  // to belong to the region's exit block.
  if (cloned_exit_block_merge != nullptr) {
    original_region_entry_block->AddInstruction(
        std::move(cloned_exit_block_merge));
  }
  original_region_entry_block->AddInstruction(
      std::move(cloned_exit_block_terminator));
}

std::unordered_set<uint32_t> TransformationOutlineFunction::GetFreshIds()
    const {
  std::unordered_set<uint32_t> result = {
      message_.new_function_struct_return_type_id(),
      message_.new_function_type_id(),
      message_.new_function_id(),
      message_.new_function_region_entry_block(),
      message_.new_caller_result_id(),
      message_.new_callee_result_id()};
  for (auto& pair : message_.input_id_to_fresh_id()) {
    result.insert(pair.second());
  }
  for (auto& pair : message_.output_id_to_fresh_id()) {
    result.insert(pair.second());
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
