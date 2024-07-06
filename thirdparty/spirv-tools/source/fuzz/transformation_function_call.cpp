// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_function_call.h"

#include "source/fuzz/call_graph.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationFunctionCall::TransformationFunctionCall(
    protobufs::TransformationFunctionCall message)
    : message_(std::move(message)) {}

TransformationFunctionCall::TransformationFunctionCall(
    uint32_t fresh_id, uint32_t callee_id,
    const std::vector<uint32_t>& argument_id,
    const protobufs::InstructionDescriptor& instruction_to_insert_before) {
  message_.set_fresh_id(fresh_id);
  message_.set_callee_id(callee_id);
  for (auto argument : argument_id) {
    message_.add_argument_id(argument);
  }
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
}

bool TransformationFunctionCall::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // The result id must be fresh
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }

  // The function must exist
  auto callee_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.callee_id());
  if (!callee_inst || callee_inst->opcode() != spv::Op::OpFunction) {
    return false;
  }

  // The function must not be an entry point
  if (fuzzerutil::FunctionIsEntryPoint(ir_context, message_.callee_id())) {
    return false;
  }

  auto callee_type_inst = ir_context->get_def_use_mgr()->GetDef(
      callee_inst->GetSingleWordInOperand(1));
  assert(callee_type_inst->opcode() == spv::Op::OpTypeFunction &&
         "Bad function type.");

  // The number of expected function arguments must match the number of given
  // arguments.  The number of expected arguments is one less than the function
  // type's number of input operands, as one operand is for the return type.
  if (callee_type_inst->NumInOperands() - 1 !=
      static_cast<uint32_t>(message_.argument_id().size())) {
    return false;
  }

  // The instruction descriptor must refer to a position where it is valid to
  // insert the call
  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  if (!insert_before) {
    return false;
  }
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpFunctionCall,
                                                    insert_before)) {
    return false;
  }

  auto block = ir_context->get_instr_block(insert_before);
  auto enclosing_function = block->GetParent();

  // If the block is not dead, the function must be livesafe
  bool block_is_dead =
      transformation_context.GetFactManager()->BlockIsDead(block->id());
  if (!block_is_dead &&
      !transformation_context.GetFactManager()->FunctionIsLivesafe(
          message_.callee_id())) {
    return false;
  }

  // The ids must all match and have the right types and satisfy rules on
  // pointers.  If the block is not dead, pointers must be arbitrary.
  for (uint32_t arg_index = 0;
       arg_index < static_cast<uint32_t>(message_.argument_id().size());
       arg_index++) {
    opt::Instruction* arg_inst =
        ir_context->get_def_use_mgr()->GetDef(message_.argument_id(arg_index));
    if (!arg_inst) {
      // The given argument does not correspond to an instruction.
      return false;
    }
    if (!arg_inst->type_id()) {
      // The given argument does not have a type; it is thus not suitable.
    }
    if (arg_inst->type_id() !=
        callee_type_inst->GetSingleWordInOperand(arg_index + 1)) {
      // Argument type mismatch.
      return false;
    }
    opt::Instruction* arg_type_inst =
        ir_context->get_def_use_mgr()->GetDef(arg_inst->type_id());
    if (arg_type_inst->opcode() == spv::Op::OpTypePointer) {
      switch (arg_inst->opcode()) {
        case spv::Op::OpFunctionParameter:
        case spv::Op::OpVariable:
          // These are OK
          break;
        default:
          // Other pointer ids cannot be passed as parameters
          return false;
      }
      if (!block_is_dead &&
          !transformation_context.GetFactManager()->PointeeValueIsIrrelevant(
              arg_inst->result_id())) {
        // This is not a dead block, so pointer parameters passed to the called
        // function might really have their contents modified. We thus require
        // such pointers to be to arbitrary-valued variables, which this is not.
        return false;
      }
    }

    // The argument id needs to be available (according to dominance rules) at
    // the point where the call will occur.
    if (!fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                    arg_inst->result_id())) {
      return false;
    }
  }

  // Introducing the call must not lead to recursion.
  if (message_.callee_id() == enclosing_function->result_id()) {
    // This would be direct recursion.
    return false;
  }
  // Ensure the call would not lead to indirect recursion.
  return !CallGraph(ir_context)
              .GetIndirectCallees(message_.callee_id())
              .count(block->GetParent()->result_id());
}

void TransformationFunctionCall::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Update the module's bound to reflect the fresh id for the result of the
  // function call.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());
  // Get the return type of the function being called.
  uint32_t return_type =
      ir_context->get_def_use_mgr()->GetDef(message_.callee_id())->type_id();
  // Populate the operands to the call instruction, with the function id and the
  // arguments.
  opt::Instruction::OperandList operands;
  operands.push_back({SPV_OPERAND_TYPE_ID, {message_.callee_id()}});
  for (auto arg : message_.argument_id()) {
    operands.push_back({SPV_OPERAND_TYPE_ID, {arg}});
  }
  // Insert the function call before the instruction specified in the message.
  FindInstruction(message_.instruction_to_insert_before(), ir_context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, spv::Op::OpFunctionCall, return_type, message_.fresh_id(),
          operands));
  // Invalidate all analyses since we have changed the module.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationFunctionCall::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_function_call() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationFunctionCall::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
