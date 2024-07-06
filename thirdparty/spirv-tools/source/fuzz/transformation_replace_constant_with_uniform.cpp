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

#include "source/fuzz/transformation_replace_constant_with_uniform.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/uniform_buffer_element_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceConstantWithUniform::
    TransformationReplaceConstantWithUniform(
        protobufs::TransformationReplaceConstantWithUniform message)
    : message_(std::move(message)) {}

TransformationReplaceConstantWithUniform::
    TransformationReplaceConstantWithUniform(
        protobufs::IdUseDescriptor id_use,
        protobufs::UniformBufferElementDescriptor uniform_descriptor,
        uint32_t fresh_id_for_access_chain, uint32_t fresh_id_for_load) {
  *message_.mutable_id_use_descriptor() = std::move(id_use);
  *message_.mutable_uniform_descriptor() = std::move(uniform_descriptor);
  message_.set_fresh_id_for_access_chain(fresh_id_for_access_chain);
  message_.set_fresh_id_for_load(fresh_id_for_load);
}

std::unique_ptr<opt::Instruction>
TransformationReplaceConstantWithUniform::MakeAccessChainInstruction(
    spvtools::opt::IRContext* ir_context, uint32_t constant_type_id) const {
  // The input operands for the access chain.
  opt::Instruction::OperandList operands_for_access_chain;

  opt::Instruction* uniform_variable =
      FindUniformVariable(message_.uniform_descriptor(), ir_context, false);

  // The first input operand is the id of the uniform variable.
  operands_for_access_chain.push_back(
      {SPV_OPERAND_TYPE_ID, {uniform_variable->result_id()}});

  // The other input operands are the ids of the constants used to index into
  // the uniform. The uniform buffer descriptor specifies a series of literals;
  // for each we find the id of the instruction that defines it, and add these
  // instruction ids as operands.
  opt::analysis::Integer int_type(32, true);
  auto registered_int_type =
      ir_context->get_type_mgr()->GetRegisteredType(&int_type)->AsInteger();
  auto int_type_id = ir_context->get_type_mgr()->GetId(&int_type);
  for (auto index : message_.uniform_descriptor().index()) {
    opt::analysis::IntConstant int_constant(registered_int_type, {index});
    auto constant_id = ir_context->get_constant_mgr()->FindDeclaredConstant(
        &int_constant, int_type_id);
    operands_for_access_chain.push_back({SPV_OPERAND_TYPE_ID, {constant_id}});
  }

  // The type id for the access chain is a uniform pointer with base type
  // matching the given constant id type.
  auto type_and_pointer_type =
      ir_context->get_type_mgr()->GetTypeAndPointerType(
          constant_type_id, spv::StorageClass::Uniform);
  assert(type_and_pointer_type.first != nullptr);
  assert(type_and_pointer_type.second != nullptr);
  auto pointer_to_uniform_constant_type_id =
      ir_context->get_type_mgr()->GetId(type_and_pointer_type.second.get());

  return MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpAccessChain, pointer_to_uniform_constant_type_id,
      message_.fresh_id_for_access_chain(), operands_for_access_chain);
}

std::unique_ptr<opt::Instruction>
TransformationReplaceConstantWithUniform::MakeLoadInstruction(
    spvtools::opt::IRContext* ir_context, uint32_t constant_type_id) const {
  opt::Instruction::OperandList operands_for_load = {
      {SPV_OPERAND_TYPE_ID, {message_.fresh_id_for_access_chain()}}};
  return MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpLoad, constant_type_id,
      message_.fresh_id_for_load(), operands_for_load);
}

opt::Instruction*
TransformationReplaceConstantWithUniform::GetInsertBeforeInstruction(
    opt::IRContext* ir_context) const {
  auto* result =
      FindInstructionContainingUse(message_.id_use_descriptor(), ir_context);
  if (!result) {
    return nullptr;
  }

  // The use might be in an OpPhi instruction.
  if (result->opcode() == spv::Op::OpPhi) {
    // OpPhi instructions must be the first instructions in a block. Thus, we
    // can't insert above the OpPhi instruction. Given the predecessor block
    // that corresponds to the id use, get the last instruction in that block
    // above which we can insert OpAccessChain and OpLoad.
    return fuzzerutil::GetLastInsertBeforeInstruction(
        ir_context,
        result->GetSingleWordInOperand(
            message_.id_use_descriptor().in_operand_index() + 1),
        spv::Op::OpLoad);
  }

  // The only operand that we could've replaced in the OpBranchConditional is
  // the condition id. But that operand has a boolean type and uniform variables
  // can't store booleans (see the spec on OpTypeBool). Thus, |result| can't be
  // an OpBranchConditional.
  assert(result->opcode() != spv::Op::OpBranchConditional &&
         "OpBranchConditional has no operands to replace");

  assert(
      fuzzerutil::CanInsertOpcodeBeforeInstruction(spv::Op::OpLoad, result) &&
      "We should be able to insert OpLoad and OpAccessChain at this point");
  return result;
}

bool TransformationReplaceConstantWithUniform::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // The following is really an invariant of the transformation rather than
  // merely a requirement of the precondition.  We check it here since we cannot
  // check it in the message_ constructor.
  assert(message_.fresh_id_for_access_chain() != message_.fresh_id_for_load() &&
         "Fresh ids for access chain and load result cannot be the same.");

  // The ids for the access chain and load instructions must both be fresh.
  if (!fuzzerutil::IsFreshId(ir_context,
                             message_.fresh_id_for_access_chain())) {
    return false;
  }
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id_for_load())) {
    return false;
  }

  // The id specified in the id use descriptor must be that of a declared scalar
  // constant.
  auto declared_constant = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.id_use_descriptor().id_of_interest());
  if (!declared_constant) {
    return false;
  }
  if (!declared_constant->AsScalarConstant()) {
    return false;
  }

  // The fact manager needs to believe that the uniform data element described
  // by the uniform buffer element descriptor will hold a scalar value.
  auto constant_id_associated_with_uniform =
      transformation_context.GetFactManager()->GetConstantFromUniformDescriptor(
          message_.uniform_descriptor());
  if (!constant_id_associated_with_uniform) {
    return false;
  }
  auto constant_associated_with_uniform =
      ir_context->get_constant_mgr()->FindDeclaredConstant(
          constant_id_associated_with_uniform);
  assert(constant_associated_with_uniform &&
         "The constant should be present in the module.");
  if (!constant_associated_with_uniform->AsScalarConstant()) {
    return false;
  }

  // The types and values of the scalar value held in the id specified by the id
  // use descriptor and in the uniform data element specified by the uniform
  // buffer element descriptor need to match on both type and value.
  if (!declared_constant->type()->IsSame(
          constant_associated_with_uniform->type())) {
    return false;
  }
  if (declared_constant->AsScalarConstant()->words() !=
      constant_associated_with_uniform->AsScalarConstant()->words()) {
    return false;
  }

  // The id use descriptor must identify some instruction with respect to the
  // module.
  auto instruction_using_constant =
      FindInstructionContainingUse(message_.id_use_descriptor(), ir_context);
  if (!instruction_using_constant) {
    return false;
  }

  // The use must not be a variable initializer; these are required to be
  // constants, so it would be illegal to replace one with a uniform access.
  if (instruction_using_constant->opcode() == spv::Op::OpVariable) {
    return false;
  }

  // The module needs to have a uniform pointer type suitable for indexing into
  // the uniform variable, i.e. matching the type of the constant we wish to
  // replace with a uniform.
  opt::analysis::Pointer pointer_to_type_of_constant(
      declared_constant->type(), spv::StorageClass::Uniform);
  if (!ir_context->get_type_mgr()->GetId(&pointer_to_type_of_constant)) {
    return false;
  }

  // In order to index into the uniform, the module has got to contain the int32
  // type, plus an OpConstant for each of the indices of interest.
  opt::analysis::Integer int_type(32, true);
  if (!ir_context->get_type_mgr()->GetId(&int_type)) {
    return false;
  }
  auto registered_int_type =
      ir_context->get_type_mgr()->GetRegisteredType(&int_type)->AsInteger();
  auto int_type_id = ir_context->get_type_mgr()->GetId(&int_type);
  for (auto index : message_.uniform_descriptor().index()) {
    opt::analysis::IntConstant int_constant(registered_int_type, {index});
    if (!ir_context->get_constant_mgr()->FindDeclaredConstant(&int_constant,
                                                              int_type_id)) {
      return false;
    }
  }

  // Once all checks are completed, we should be able to safely insert
  // OpAccessChain and OpLoad into the module.
  assert(GetInsertBeforeInstruction(ir_context) &&
         "There must exist an instruction that we can use to insert "
         "OpAccessChain and OpLoad above");

  return true;
}

void TransformationReplaceConstantWithUniform::Apply(
    spvtools::opt::IRContext* ir_context,
    TransformationContext* /*unused*/) const {
  // Get the instruction that contains the id use we wish to replace.
  auto* instruction_containing_constant_use =
      FindInstructionContainingUse(message_.id_use_descriptor(), ir_context);
  assert(instruction_containing_constant_use &&
         "Precondition requires that the id use can be found.");
  assert(instruction_containing_constant_use->GetSingleWordInOperand(
             message_.id_use_descriptor().in_operand_index()) ==
             message_.id_use_descriptor().id_of_interest() &&
         "Does not appear to be a usage of the desired id.");

  // The id of the type for the constant whose use we wish to replace.
  auto constant_type_id =
      ir_context->get_def_use_mgr()
          ->GetDef(message_.id_use_descriptor().id_of_interest())
          ->type_id();

  // Get an instruction that will be used to insert OpAccessChain and OpLoad.
  auto* insert_before_inst = GetInsertBeforeInstruction(ir_context);
  assert(insert_before_inst &&
         "There must exist an insertion point for OpAccessChain and OpLoad");
  opt::BasicBlock* enclosing_block =
      ir_context->get_instr_block(insert_before_inst);

  // Add an access chain instruction to target the uniform element.
  auto access_chain_instruction =
      MakeAccessChainInstruction(ir_context, constant_type_id);
  auto access_chain_instruction_ptr = access_chain_instruction.get();
  insert_before_inst->InsertBefore(std::move(access_chain_instruction));
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(
      access_chain_instruction_ptr);
  ir_context->set_instr_block(access_chain_instruction_ptr, enclosing_block);

  // Add a load from this access chain.
  auto load_instruction = MakeLoadInstruction(ir_context, constant_type_id);
  auto load_instruction_ptr = load_instruction.get();
  insert_before_inst->InsertBefore(std::move(load_instruction));
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(load_instruction_ptr);
  ir_context->set_instr_block(load_instruction_ptr, enclosing_block);

  // Adjust the instruction containing the usage of the constant so that this
  // usage refers instead to the result of the load.
  instruction_containing_constant_use->SetInOperand(
      message_.id_use_descriptor().in_operand_index(),
      {message_.fresh_id_for_load()});
  ir_context->get_def_use_mgr()->EraseUseRecordsOfOperandIds(
      instruction_containing_constant_use);
  ir_context->get_def_use_mgr()->AnalyzeInstUse(
      instruction_containing_constant_use);

  // Update the module id bound to reflect the new instructions.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id_for_load());
  fuzzerutil::UpdateModuleIdBound(ir_context,
                                  message_.fresh_id_for_access_chain());
}

protobufs::Transformation TransformationReplaceConstantWithUniform::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_replace_constant_with_uniform() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationReplaceConstantWithUniform::GetFreshIds() const {
  return {message_.fresh_id_for_access_chain(), message_.fresh_id_for_load()};
}

}  // namespace fuzz
}  // namespace spvtools
