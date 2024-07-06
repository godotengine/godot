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

#include "source/fuzz/transformation_inline_function.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationInlineFunction::TransformationInlineFunction(
    protobufs::TransformationInlineFunction message)
    : message_(std::move(message)) {}

TransformationInlineFunction::TransformationInlineFunction(
    uint32_t function_call_id,
    const std::map<uint32_t, uint32_t>& result_id_map) {
  message_.set_function_call_id(function_call_id);
  *message_.mutable_result_id_map() =
      fuzzerutil::MapToRepeatedUInt32Pair(result_id_map);
}

bool TransformationInlineFunction::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // The values in the |message_.result_id_map| must be all fresh and all
  // distinct.
  const auto result_id_map =
      fuzzerutil::RepeatedUInt32PairToMap(message_.result_id_map());
  std::set<uint32_t> ids_used_by_this_transformation;
  for (auto& pair : result_id_map) {
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(
            pair.second, ir_context, &ids_used_by_this_transformation)) {
      return false;
    }
  }

  // |function_call_instruction| must be suitable for inlining.
  auto* function_call_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.function_call_id());
  if (!IsSuitableForInlining(ir_context, function_call_instruction)) {
    return false;
  }

  // |function_call_instruction| must be the penultimate instruction in its
  // block and its block termination instruction must be an OpBranch. This
  // avoids the case where the penultimate instruction is an OpLoopMerge, which
  // would make the back-edge block not branch to the loop header.
  auto* function_call_instruction_block =
      ir_context->get_instr_block(function_call_instruction);
  if (function_call_instruction !=
          &*--function_call_instruction_block->tail() ||
      function_call_instruction_block->terminator()->opcode() !=
          spv::Op::OpBranch) {
    return false;
  }

  auto* called_function = fuzzerutil::FindFunction(
      ir_context, function_call_instruction->GetSingleWordInOperand(0));
  for (auto& block : *called_function) {
    // Since the entry block label will not be inlined, only the remaining
    // labels must have a corresponding value in the map.
    if (&block != &*called_function->entry() &&
        !result_id_map.count(block.id()) &&
        !transformation_context.GetOverflowIdSource()->HasOverflowIds()) {
      return false;
    }

    // |result_id_map| must have an entry for every result id in the called
    // function.
    for (auto& instruction : block) {
      // If |instruction| has result id, then it must have a mapped id in
      // |result_id_map|.
      if (instruction.HasResultId() &&
          !result_id_map.count(instruction.result_id()) &&
          !transformation_context.GetOverflowIdSource()->HasOverflowIds()) {
        return false;
      }
    }
  }

  // |result_id_map| must not contain an entry for any parameter of the function
  // that is being inlined.
  bool found_entry_for_parameter = false;
  called_function->ForEachParam(
      [&result_id_map, &found_entry_for_parameter](opt::Instruction* param) {
        if (result_id_map.count(param->result_id())) {
          found_entry_for_parameter = true;
        }
      });
  return !found_entry_for_parameter;
}

void TransformationInlineFunction::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  auto* function_call_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.function_call_id());
  auto* caller_function =
      ir_context->get_instr_block(function_call_instruction)->GetParent();
  auto* called_function = fuzzerutil::FindFunction(
      ir_context, function_call_instruction->GetSingleWordInOperand(0));
  std::map<uint32_t, uint32_t> result_id_map =
      fuzzerutil::RepeatedUInt32PairToMap(message_.result_id_map());

  // If there are gaps in the result id map, fill them using overflow ids.
  for (auto& block : *called_function) {
    if (&block != &*called_function->entry() &&
        !result_id_map.count(block.id())) {
      result_id_map.insert(
          {block.id(),
           transformation_context->GetOverflowIdSource()->GetNextOverflowId()});
    }
    for (auto& instruction : block) {
      // If |instruction| has result id, then it must have a mapped id in
      // |result_id_map|.
      if (instruction.HasResultId() &&
          !result_id_map.count(instruction.result_id())) {
        result_id_map.insert({instruction.result_id(),
                              transformation_context->GetOverflowIdSource()
                                  ->GetNextOverflowId()});
      }
    }
  }

  auto* successor_block = ir_context->cfg()->block(
      ir_context->get_instr_block(function_call_instruction)
          ->terminator()
          ->GetSingleWordInOperand(0));

  // Inline the |called_function| entry block.
  for (auto& entry_block_instruction : *called_function->entry()) {
    opt::Instruction* inlined_instruction;

    if (entry_block_instruction.opcode() == spv::Op::OpVariable) {
      // All OpVariable instructions in a function must be in the first block
      // in the function.
      inlined_instruction = caller_function->begin()->begin()->InsertBefore(
          std::unique_ptr<opt::Instruction>(
              entry_block_instruction.Clone(ir_context)));
    } else {
      inlined_instruction = function_call_instruction->InsertBefore(
          std::unique_ptr<opt::Instruction>(
              entry_block_instruction.Clone(ir_context)));
    }

    AdaptInlinedInstruction(result_id_map, ir_context, inlined_instruction);
  }

  // If the function call's successor block contains OpPhi instructions that
  // refer to the block containing the call then these will need to be rewritten
  // to instead refer to the block associated with "returning" from the inlined
  // function, as this block will be the predecessor of what used to be the
  // function call's successor block.  We look out for this block.
  uint32_t new_return_block_id = 0;

  // Inline the |called_function| non-entry blocks.
  for (auto& block : *called_function) {
    if (&block == &*called_function->entry()) {
      continue;
    }

    // Check whether this is the function's return block.  Take note if it is,
    // so that OpPhi instructions in the successor of the original function call
    // block can be re-written.
    if (block.terminator()->IsReturn()) {
      assert(new_return_block_id == 0 &&
             "There should be only one return block.");
      new_return_block_id = result_id_map.at(block.id());
    }

    auto* cloned_block = block.Clone(ir_context);
    cloned_block = caller_function->InsertBasicBlockBefore(
        std::unique_ptr<opt::BasicBlock>(cloned_block), successor_block);
    cloned_block->GetLabel()->SetResultId(result_id_map.at(cloned_block->id()));
    fuzzerutil::UpdateModuleIdBound(ir_context, cloned_block->id());

    for (auto& inlined_instruction : *cloned_block) {
      AdaptInlinedInstruction(result_id_map, ir_context, &inlined_instruction);
    }
  }

  opt::BasicBlock* block_containing_function_call =
      ir_context->get_instr_block(function_call_instruction);

  assert(((new_return_block_id == 0) ==
          called_function->entry()->terminator()->IsReturn()) &&
         "We should have found a return block unless the function being "
         "inlined returns in its first block.");
  if (new_return_block_id != 0) {
    // Rewrite any OpPhi instructions in the successor block so that they refer
    // to the new return block instead of the block that originally contained
    // the function call.
    ir_context->get_def_use_mgr()->ForEachUse(
        block_containing_function_call->id(),
        [ir_context, new_return_block_id, successor_block](
            opt::Instruction* use_instruction, uint32_t operand_index) {
          if (use_instruction->opcode() == spv::Op::OpPhi &&
              ir_context->get_instr_block(use_instruction) == successor_block) {
            use_instruction->SetOperand(operand_index, {new_return_block_id});
          }
        });
  }

  // Removes the function call instruction and its block termination instruction
  // from |caller_function|.
  ir_context->KillInst(block_containing_function_call->terminator());
  ir_context->KillInst(function_call_instruction);

  // Since the SPIR-V module has changed, no analyses must be validated.
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationInlineFunction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_inline_function() = message_;
  return result;
}

bool TransformationInlineFunction::IsSuitableForInlining(
    opt::IRContext* ir_context, opt::Instruction* function_call_instruction) {
  // |function_call_instruction| must be defined and must be an OpFunctionCall
  // instruction.
  if (!function_call_instruction ||
      function_call_instruction->opcode() != spv::Op::OpFunctionCall) {
    return false;
  }

  // If |function_call_instruction| return type is void, then
  // |function_call_instruction| must not have uses.
  if (ir_context->get_type_mgr()
          ->GetType(function_call_instruction->type_id())
          ->AsVoid() &&
      ir_context->get_def_use_mgr()->NumUses(function_call_instruction) != 0) {
    return false;
  }

  // |called_function| must not have an early return.
  auto called_function = fuzzerutil::FindFunction(
      ir_context, function_call_instruction->GetSingleWordInOperand(0));
  if (called_function->HasEarlyReturn()) {
    return false;
  }

  // |called_function| must not use OpKill or OpUnreachable.
  if (fuzzerutil::FunctionContainsOpKillOrUnreachable(*called_function)) {
    return false;
  }

  return true;
}

void TransformationInlineFunction::AdaptInlinedInstruction(
    const std::map<uint32_t, uint32_t>& result_id_map,
    opt::IRContext* ir_context,
    opt::Instruction* instruction_to_be_inlined) const {
  auto* function_call_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.function_call_id());
  auto* called_function = fuzzerutil::FindFunction(
      ir_context, function_call_instruction->GetSingleWordInOperand(0));

  const auto* function_call_block =
      ir_context->get_instr_block(function_call_instruction);
  assert(function_call_block && "OpFunctionCall must belong to some block");

  // Replaces the operand ids with their mapped result ids.
  instruction_to_be_inlined->ForEachInId(
      [called_function, function_call_instruction, &result_id_map,
       function_call_block](uint32_t* id) {
        // We are not inlining the entry block of the |called_function|.
        //
        // We must check this condition first since we can't use the fresh id
        // from |result_id_map| even if it has one. This is because that fresh
        // id will never be added to the module since entry blocks are not
        // inlined.
        if (*id == called_function->entry()->id()) {
          *id = function_call_block->id();
          return;
        }

        // If |id| is mapped, then set it to its mapped value.
        if (result_id_map.count(*id)) {
          *id = result_id_map.at(*id);
          return;
        }

        uint32_t parameter_index = 0;
        called_function->ForEachParam(
            [id, function_call_instruction,
             &parameter_index](opt::Instruction* parameter_instruction) {
              // If the id is a function parameter, then set it to the
              // parameter value passed in the function call instruction.
              if (*id == parameter_instruction->result_id()) {
                // We do + 1 because the first in-operand for OpFunctionCall is
                // the function id that is being called.
                *id = function_call_instruction->GetSingleWordInOperand(
                    parameter_index + 1);
              }
              parameter_index++;
            });
      });

  // If |instruction_to_be_inlined| has result id, then set it to its mapped
  // value.
  if (instruction_to_be_inlined->HasResultId()) {
    assert(result_id_map.count(instruction_to_be_inlined->result_id()) &&
           "Result id must be mapped to a fresh id.");
    instruction_to_be_inlined->SetResultId(
        result_id_map.at(instruction_to_be_inlined->result_id()));
    fuzzerutil::UpdateModuleIdBound(ir_context,
                                    instruction_to_be_inlined->result_id());
  }

  // The return instruction will be changed into an OpBranch to the basic
  // block that follows the block containing the function call.
  if (spvOpcodeIsReturn(instruction_to_be_inlined->opcode())) {
    uint32_t successor_block_id =
        ir_context->get_instr_block(function_call_instruction)
            ->terminator()
            ->GetSingleWordInOperand(0);
    switch (instruction_to_be_inlined->opcode()) {
      case spv::Op::OpReturn:
        instruction_to_be_inlined->AddOperand(
            {SPV_OPERAND_TYPE_ID, {successor_block_id}});
        break;
      case spv::Op::OpReturnValue: {
        instruction_to_be_inlined->InsertBefore(MakeUnique<opt::Instruction>(
            ir_context, spv::Op::OpCopyObject,
            function_call_instruction->type_id(),
            function_call_instruction->result_id(),
            opt::Instruction::OperandList(
                {{SPV_OPERAND_TYPE_ID,
                  {instruction_to_be_inlined->GetSingleWordOperand(0)}}})));
        instruction_to_be_inlined->SetInOperand(0, {successor_block_id});
        break;
      }
      default:
        break;
    }
    instruction_to_be_inlined->SetOpcode(spv::Op::OpBranch);
  }
}

std::unordered_set<uint32_t> TransformationInlineFunction::GetFreshIds() const {
  std::unordered_set<uint32_t> result;
  for (auto& pair : message_.result_id_map()) {
    result.insert(pair.second());
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
