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

#include "source/fuzz/transformation_add_early_terminator_wrapper.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

TransformationAddEarlyTerminatorWrapper::
    TransformationAddEarlyTerminatorWrapper(
        protobufs::TransformationAddEarlyTerminatorWrapper message)
    : message_(std::move(message)) {}

TransformationAddEarlyTerminatorWrapper::
    TransformationAddEarlyTerminatorWrapper(uint32_t function_fresh_id,
                                            uint32_t label_fresh_id,
                                            spv::Op opcode) {
  message_.set_function_fresh_id(function_fresh_id);
  message_.set_label_fresh_id(label_fresh_id);
  message_.set_opcode(uint32_t(opcode));
}

bool TransformationAddEarlyTerminatorWrapper::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  assert((spv::Op(message_.opcode()) == spv::Op::OpKill ||
          spv::Op(message_.opcode()) == spv::Op::OpUnreachable ||
          spv::Op(message_.opcode()) == spv::Op::OpTerminateInvocation) &&
         "Invalid opcode.");

  if (!fuzzerutil::IsFreshId(ir_context, message_.function_fresh_id())) {
    return false;
  }
  if (!fuzzerutil::IsFreshId(ir_context, message_.label_fresh_id())) {
    return false;
  }
  if (message_.function_fresh_id() == message_.label_fresh_id()) {
    return false;
  }
  uint32_t void_type_id = fuzzerutil::MaybeGetVoidType(ir_context);
  if (!void_type_id) {
    return false;
  }
  return fuzzerutil::FindFunctionType(ir_context, {void_type_id});
}

void TransformationAddEarlyTerminatorWrapper::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.function_fresh_id());
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.label_fresh_id());

  // Create a basic block of the form:
  // %label_fresh_id = OpLabel
  //                   OpKill|Unreachable|TerminateInvocation
  auto basic_block = MakeUnique<opt::BasicBlock>(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpLabel, 0, message_.label_fresh_id(),
      opt::Instruction::OperandList()));
  basic_block->AddInstruction(MakeUnique<opt::Instruction>(
      ir_context, static_cast<spv::Op>(message_.opcode()), 0, 0,
      opt::Instruction::OperandList()));

  // Create a zero-argument void function.
  auto void_type_id = fuzzerutil::MaybeGetVoidType(ir_context);
  auto function = MakeUnique<opt::Function>(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpFunction, void_type_id,
      message_.function_fresh_id(),
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_FUNCTION_CONTROL,
            {uint32_t(spv::FunctionControlMask::MaskNone)}},
           {SPV_OPERAND_TYPE_TYPE_ID,
            {fuzzerutil::FindFunctionType(ir_context, {void_type_id})}}})));

  // Add the basic block to the function as the sole block, and add the function
  // to the module.
  function->AddBasicBlock(std::move(basic_block));
  function->SetFunctionEnd(
      MakeUnique<opt::Instruction>(ir_context, spv::Op::OpFunctionEnd, 0, 0,
                                   opt::Instruction::OperandList()));
  ir_context->module()->AddFunction(std::move(function));

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

std::unordered_set<uint32_t>
TransformationAddEarlyTerminatorWrapper::GetFreshIds() const {
  return std::unordered_set<uint32_t>(
      {message_.function_fresh_id(), message_.label_fresh_id()});
}

protobufs::Transformation TransformationAddEarlyTerminatorWrapper::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_add_early_terminator_wrapper() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
