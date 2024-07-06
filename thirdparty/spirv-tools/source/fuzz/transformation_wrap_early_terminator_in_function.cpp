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

#include "source/fuzz/transformation_wrap_early_terminator_in_function.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

TransformationWrapEarlyTerminatorInFunction::
    TransformationWrapEarlyTerminatorInFunction(
        protobufs::TransformationWrapEarlyTerminatorInFunction message)
    : message_(std::move(message)) {}

TransformationWrapEarlyTerminatorInFunction::
    TransformationWrapEarlyTerminatorInFunction(
        uint32_t fresh_id,
        const protobufs::InstructionDescriptor& early_terminator_instruction,
        uint32_t returned_value_id) {
  message_.set_fresh_id(fresh_id);
  *message_.mutable_early_terminator_instruction() =
      early_terminator_instruction;
  message_.set_returned_value_id(returned_value_id);
}

bool TransformationWrapEarlyTerminatorInFunction::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The given id must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }

  // |message_.early_terminator_instruction| must identify an instruction, and
  // the instruction must indeed be an early terminator.
  auto early_terminator =
      FindInstruction(message_.early_terminator_instruction(), ir_context);
  if (!early_terminator) {
    return false;
  }
  switch (early_terminator->opcode()) {
    case spv::Op::OpKill:
    case spv::Op::OpUnreachable:
    case spv::Op::OpTerminateInvocation:
      break;
    default:
      return false;
  }
  // A wrapper function for the early terminator must exist.
  auto wrapper_function =
      MaybeGetWrapperFunction(ir_context, early_terminator->opcode());
  if (wrapper_function == nullptr) {
    return false;
  }
  auto enclosing_function =
      ir_context->get_instr_block(early_terminator)->GetParent();
  // The wrapper function cannot be the function containing the instruction we
  // would like to wrap.
  if (wrapper_function->result_id() == enclosing_function->result_id()) {
    return false;
  }
  if (!ir_context->get_type_mgr()
           ->GetType(enclosing_function->type_id())
           ->AsVoid()) {
    // The enclosing function has non-void return type.  We thus need to make
    // sure that |message_.returned_value_instruction| provides a suitable
    // result id to use in an OpReturnValue instruction.
    auto returned_value_instruction =
        ir_context->get_def_use_mgr()->GetDef(message_.returned_value_id());
    if (!returned_value_instruction || !returned_value_instruction->type_id() ||
        returned_value_instruction->type_id() !=
            enclosing_function->type_id()) {
      return false;
    }
    if (!fuzzerutil::IdIsAvailableBeforeInstruction(
            ir_context, early_terminator, message_.returned_value_id())) {
      return false;
    }
  }
  return true;
}

void TransformationWrapEarlyTerminatorInFunction::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());
  auto early_terminator =
      FindInstruction(message_.early_terminator_instruction(), ir_context);
  auto enclosing_block = ir_context->get_instr_block(early_terminator);
  auto enclosing_function = enclosing_block->GetParent();

  // We would like to add an OpFunctionCall before the block's terminator
  // instruction, and then change the block's terminator to OpReturn or
  // OpReturnValue.

  // We get an iterator to the instruction we would like to insert the function
  // call before.  It will be an iterator to the final instruction in the block
  // unless the block is a merge block in which case it will be to the
  // penultimate instruction (because we cannot insert an OpFunctionCall after
  // a merge instruction).
  auto iterator = enclosing_block->tail();
  if (enclosing_block->MergeBlockIdIfAny()) {
    --iterator;
  }

  auto wrapper_function =
      MaybeGetWrapperFunction(ir_context, early_terminator->opcode());

  iterator->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpFunctionCall, wrapper_function->type_id(),
      message_.fresh_id(),
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {wrapper_function->result_id()}}})));

  opt::Instruction::OperandList new_in_operands;
  if (!ir_context->get_type_mgr()
           ->GetType(enclosing_function->type_id())
           ->AsVoid()) {
    new_in_operands.push_back(
        {SPV_OPERAND_TYPE_ID, {message_.returned_value_id()}});
    early_terminator->SetOpcode(spv::Op::OpReturnValue);
  } else {
    early_terminator->SetOpcode(spv::Op::OpReturn);
  }
  early_terminator->SetInOperands(std::move(new_in_operands));

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

std::unordered_set<uint32_t>
TransformationWrapEarlyTerminatorInFunction::GetFreshIds() const {
  return std::unordered_set<uint32_t>({message_.fresh_id()});
}

protobufs::Transformation
TransformationWrapEarlyTerminatorInFunction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_wrap_early_terminator_in_function() = message_;
  return result;
}

opt::Function*
TransformationWrapEarlyTerminatorInFunction::MaybeGetWrapperFunction(
    opt::IRContext* ir_context, spv::Op early_terminator_opcode) {
  assert((early_terminator_opcode == spv::Op::OpKill ||
          early_terminator_opcode == spv::Op::OpUnreachable ||
          early_terminator_opcode == spv::Op::OpTerminateInvocation) &&
         "Invalid opcode.");
  auto void_type_id = fuzzerutil::MaybeGetVoidType(ir_context);
  if (!void_type_id) {
    return nullptr;
  }
  auto void_function_type_id =
      fuzzerutil::FindFunctionType(ir_context, {void_type_id});
  if (!void_function_type_id) {
    return nullptr;
  }
  for (auto& function : *ir_context->module()) {
    if (function.DefInst().GetSingleWordInOperand(1) != void_function_type_id) {
      continue;
    }
    if (function.begin()->begin()->opcode() == early_terminator_opcode) {
      return &function;
    }
  }
  return nullptr;
}

}  // namespace fuzz
}  // namespace spvtools
