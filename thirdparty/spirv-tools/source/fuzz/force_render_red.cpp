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

#include "source/fuzz/force_render_red.h"

#include "source/fuzz/fact_manager/fact_manager.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation_context.h"
#include "source/fuzz/transformation_replace_constant_with_uniform.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/opt/types.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

namespace {

// Helper method to find the fragment shader entry point, complaining if there
// is no shader or if there is no fragment entry point.
opt::Function* FindFragmentShaderEntryPoint(opt::IRContext* ir_context,
                                            MessageConsumer message_consumer) {
  // Check that this is a fragment shader
  bool found_capability_shader = false;
  for (auto& capability : ir_context->capabilities()) {
    assert(capability.opcode() == spv::Op::OpCapability);
    if (spv::Capability(capability.GetSingleWordInOperand(0)) ==
        spv::Capability::Shader) {
      found_capability_shader = true;
      break;
    }
  }
  if (!found_capability_shader) {
    message_consumer(
        SPV_MSG_ERROR, nullptr, {},
        "Forcing of red rendering requires the Shader capability.");
    return nullptr;
  }

  opt::Instruction* fragment_entry_point = nullptr;
  for (auto& entry_point : ir_context->module()->entry_points()) {
    if (spv::ExecutionModel(entry_point.GetSingleWordInOperand(0)) ==
        spv::ExecutionModel::Fragment) {
      fragment_entry_point = &entry_point;
      break;
    }
  }
  if (fragment_entry_point == nullptr) {
    message_consumer(SPV_MSG_ERROR, nullptr, {},
                     "Forcing of red rendering requires an entry point with "
                     "the Fragment execution model.");
    return nullptr;
  }

  for (auto& function : *ir_context->module()) {
    if (function.result_id() ==
        fragment_entry_point->GetSingleWordInOperand(1)) {
      return &function;
    }
  }
  assert(
      false &&
      "A valid module must have a function associate with each entry point.");
  return nullptr;
}

// Helper method to check that there is a single vec4 output variable and get a
// pointer to it.
opt::Instruction* FindVec4OutputVariable(opt::IRContext* ir_context,
                                         MessageConsumer message_consumer) {
  opt::Instruction* output_variable = nullptr;
  for (auto& inst : ir_context->types_values()) {
    if (inst.opcode() == spv::Op::OpVariable &&
        spv::StorageClass(inst.GetSingleWordInOperand(0)) ==
            spv::StorageClass::Output) {
      if (output_variable != nullptr) {
        message_consumer(SPV_MSG_ERROR, nullptr, {},
                         "Only one output variable can be handled at present; "
                         "found multiple.");
        return nullptr;
      }
      output_variable = &inst;
      // Do not break, as we want to check for multiple output variables.
    }
  }
  if (output_variable == nullptr) {
    message_consumer(SPV_MSG_ERROR, nullptr, {},
                     "No output variable to which to write red was found.");
    return nullptr;
  }

  auto output_variable_base_type = ir_context->get_type_mgr()
                                       ->GetType(output_variable->type_id())
                                       ->AsPointer()
                                       ->pointee_type()
                                       ->AsVector();
  if (!output_variable_base_type ||
      output_variable_base_type->element_count() != 4 ||
      !output_variable_base_type->element_type()->AsFloat()) {
    message_consumer(SPV_MSG_ERROR, nullptr, {},
                     "The output variable must have type vec4.");
    return nullptr;
  }

  return output_variable;
}

// Helper to get the ids of float constants 0.0 and 1.0, creating them if
// necessary.
std::pair<uint32_t, uint32_t> FindOrCreateFloatZeroAndOne(
    opt::IRContext* ir_context, opt::analysis::Float* float_type) {
  float one = 1.0;
  uint32_t one_as_uint;
  memcpy(&one_as_uint, &one, sizeof(float));
  std::vector<uint32_t> zero_bytes = {0};
  std::vector<uint32_t> one_bytes = {one_as_uint};
  auto constant_zero = ir_context->get_constant_mgr()->RegisterConstant(
      MakeUnique<opt::analysis::FloatConstant>(float_type, zero_bytes));
  auto constant_one = ir_context->get_constant_mgr()->RegisterConstant(
      MakeUnique<opt::analysis::FloatConstant>(float_type, one_bytes));
  auto constant_zero_id = ir_context->get_constant_mgr()
                              ->GetDefiningInstruction(constant_zero)
                              ->result_id();
  auto constant_one_id = ir_context->get_constant_mgr()
                             ->GetDefiningInstruction(constant_one)
                             ->result_id();
  return std::pair<uint32_t, uint32_t>(constant_zero_id, constant_one_id);
}

std::unique_ptr<TransformationReplaceConstantWithUniform>
MakeConstantUniformReplacement(opt::IRContext* ir_context,
                               const FactManager& fact_manager,
                               uint32_t constant_id,
                               uint32_t greater_than_instruction,
                               uint32_t in_operand_index) {
  return MakeUnique<TransformationReplaceConstantWithUniform>(
      MakeIdUseDescriptor(
          constant_id,
          MakeInstructionDescriptor(greater_than_instruction,
                                    spv::Op::OpFOrdGreaterThan, 0),
          in_operand_index),
      fact_manager.GetUniformDescriptorsForConstant(constant_id)[0],
      ir_context->TakeNextId(), ir_context->TakeNextId());
}

}  // namespace

bool ForceRenderRed(
    const spv_target_env& target_env, spv_validator_options validator_options,
    const std::vector<uint32_t>& binary_in,
    const spvtools::fuzz::protobufs::FactSequence& initial_facts,
    const MessageConsumer& message_consumer,
    std::vector<uint32_t>* binary_out) {
  spvtools::SpirvTools tools(target_env);
  if (!tools.IsValid()) {
    message_consumer(SPV_MSG_ERROR, nullptr, {},
                     "Failed to create SPIRV-Tools interface; stopping.");
    return false;
  }

  // Initial binary should be valid.
  if (!tools.Validate(&binary_in[0], binary_in.size(), validator_options)) {
    message_consumer(SPV_MSG_ERROR, nullptr, {},
                     "Initial binary is invalid; stopping.");
    return false;
  }

  // Build the module from the input binary.
  std::unique_ptr<opt::IRContext> ir_context = BuildModule(
      target_env, message_consumer, binary_in.data(), binary_in.size());
  assert(ir_context);

  // Set up a fact manager with any given initial facts.
  TransformationContext transformation_context(
      MakeUnique<FactManager>(ir_context.get()), validator_options);
  for (auto& fact : initial_facts.fact()) {
    transformation_context.GetFactManager()->MaybeAddFact(fact);
  }

  auto entry_point_function =
      FindFragmentShaderEntryPoint(ir_context.get(), message_consumer);
  auto output_variable =
      FindVec4OutputVariable(ir_context.get(), message_consumer);
  if (entry_point_function == nullptr || output_variable == nullptr) {
    return false;
  }

  opt::analysis::Float temp_float_type(32);
  opt::analysis::Float* float_type = ir_context->get_type_mgr()
                                         ->GetRegisteredType(&temp_float_type)
                                         ->AsFloat();
  std::pair<uint32_t, uint32_t> zero_one_float_ids =
      FindOrCreateFloatZeroAndOne(ir_context.get(), float_type);

  // Make the new exit block
  auto new_exit_block_id = ir_context->TakeNextId();
  {
    auto label = MakeUnique<opt::Instruction>(
        ir_context.get(), spv::Op::OpLabel, 0, new_exit_block_id,
        opt::Instruction::OperandList());
    auto new_exit_block = MakeUnique<opt::BasicBlock>(std::move(label));
    new_exit_block->AddInstruction(
        MakeUnique<opt::Instruction>(ir_context.get(), spv::Op::OpReturn, 0, 0,
                                     opt::Instruction::OperandList()));
    entry_point_function->AddBasicBlock(std::move(new_exit_block));
  }

  // Make the new entry block
  {
    auto label = MakeUnique<opt::Instruction>(
        ir_context.get(), spv::Op::OpLabel, 0, ir_context->TakeNextId(),
        opt::Instruction::OperandList());
    auto new_entry_block = MakeUnique<opt::BasicBlock>(std::move(label));

    // Make an instruction to construct vec4(1.0, 0.0, 0.0, 1.0), representing
    // the colour red.
    opt::Operand zero_float = {SPV_OPERAND_TYPE_ID, {zero_one_float_ids.first}};
    opt::Operand one_float = {SPV_OPERAND_TYPE_ID, {zero_one_float_ids.second}};
    opt::Instruction::OperandList op_composite_construct_operands = {
        one_float, zero_float, zero_float, one_float};
    auto temp_vec4 = opt::analysis::Vector(float_type, 4);
    auto vec4_id = ir_context->get_type_mgr()->GetId(&temp_vec4);
    auto red = MakeUnique<opt::Instruction>(
        ir_context.get(), spv::Op::OpCompositeConstruct, vec4_id,
        ir_context->TakeNextId(), op_composite_construct_operands);
    auto red_id = red->result_id();
    new_entry_block->AddInstruction(std::move(red));

    // Make an instruction to store red into the output color.
    opt::Operand variable_to_store_into = {SPV_OPERAND_TYPE_ID,
                                           {output_variable->result_id()}};
    opt::Operand value_to_be_stored = {SPV_OPERAND_TYPE_ID, {red_id}};
    opt::Instruction::OperandList op_store_operands = {variable_to_store_into,
                                                       value_to_be_stored};
    new_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
        ir_context.get(), spv::Op::OpStore, 0, 0, op_store_operands));

    // We are going to attempt to construct 'false' as an expression of the form
    // 'literal1 > literal2'. If we succeed, we will later replace each literal
    // with a uniform of the same value - we can only do that replacement once
    // we have added the entry block to the module.
    std::unique_ptr<TransformationReplaceConstantWithUniform>
        first_greater_then_operand_replacement = nullptr;
    std::unique_ptr<TransformationReplaceConstantWithUniform>
        second_greater_then_operand_replacement = nullptr;
    uint32_t id_guaranteed_to_be_false = 0;

    opt::analysis::Bool temp_bool_type;
    opt::analysis::Bool* registered_bool_type =
        ir_context->get_type_mgr()
            ->GetRegisteredType(&temp_bool_type)
            ->AsBool();

    auto float_type_id = ir_context->get_type_mgr()->GetId(float_type);
    auto types_for_which_uniforms_are_known =
        transformation_context.GetFactManager()
            ->GetTypesForWhichUniformValuesAreKnown();

    // Check whether we have any float uniforms.
    if (std::find(types_for_which_uniforms_are_known.begin(),
                  types_for_which_uniforms_are_known.end(),
                  float_type_id) != types_for_which_uniforms_are_known.end()) {
      // We have at least one float uniform; let's see whether we have at least
      // two.
      auto available_constants =
          transformation_context.GetFactManager()
              ->GetConstantsAvailableFromUniformsForType(float_type_id);
      if (available_constants.size() > 1) {
        // Grab the float constants associated with the first two known float
        // uniforms.
        auto first_constant =
            ir_context->get_constant_mgr()
                ->GetConstantFromInst(ir_context->get_def_use_mgr()->GetDef(
                    available_constants[0]))
                ->AsFloatConstant();
        auto second_constant =
            ir_context->get_constant_mgr()
                ->GetConstantFromInst(ir_context->get_def_use_mgr()->GetDef(
                    available_constants[1]))
                ->AsFloatConstant();

        // Now work out which of the two constants is larger than the other.
        uint32_t larger_constant_index = 0;
        uint32_t smaller_constant_index = 0;
        if (first_constant->GetFloat() > second_constant->GetFloat()) {
          larger_constant_index = 0;
          smaller_constant_index = 1;
        } else if (first_constant->GetFloat() < second_constant->GetFloat()) {
          larger_constant_index = 1;
          smaller_constant_index = 0;
        }

        // Only proceed with these constants if they have turned out to be
        // distinct.
        if (larger_constant_index != smaller_constant_index) {
          // We are in a position to create 'false' as 'literal1 > literal2', so
          // reserve an id for this computation; this id will end up being
          // guaranteed to be 'false'.
          id_guaranteed_to_be_false = ir_context->TakeNextId();

          auto smaller_constant = available_constants[smaller_constant_index];
          auto larger_constant = available_constants[larger_constant_index];

          opt::Instruction::OperandList greater_than_operands = {
              {SPV_OPERAND_TYPE_ID, {smaller_constant}},
              {SPV_OPERAND_TYPE_ID, {larger_constant}}};
          new_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
              ir_context.get(), spv::Op::OpFOrdGreaterThan,
              ir_context->get_type_mgr()->GetId(registered_bool_type),
              id_guaranteed_to_be_false, greater_than_operands));

          first_greater_then_operand_replacement =
              MakeConstantUniformReplacement(
                  ir_context.get(), *transformation_context.GetFactManager(),
                  smaller_constant, id_guaranteed_to_be_false, 0);
          second_greater_then_operand_replacement =
              MakeConstantUniformReplacement(
                  ir_context.get(), *transformation_context.GetFactManager(),
                  larger_constant, id_guaranteed_to_be_false, 1);
        }
      }
    }

    if (id_guaranteed_to_be_false == 0) {
      auto constant_false = ir_context->get_constant_mgr()->RegisterConstant(
          MakeUnique<opt::analysis::BoolConstant>(registered_bool_type, false));
      id_guaranteed_to_be_false = ir_context->get_constant_mgr()
                                      ->GetDefiningInstruction(constant_false)
                                      ->result_id();
    }

    opt::Operand false_condition = {SPV_OPERAND_TYPE_ID,
                                    {id_guaranteed_to_be_false}};
    opt::Operand then_block = {SPV_OPERAND_TYPE_ID,
                               {entry_point_function->entry()->id()}};
    opt::Operand else_block = {SPV_OPERAND_TYPE_ID, {new_exit_block_id}};
    opt::Instruction::OperandList op_branch_conditional_operands = {
        false_condition, then_block, else_block};
    new_entry_block->AddInstruction(MakeUnique<opt::Instruction>(
        ir_context.get(), spv::Op::OpBranchConditional, 0, 0,
        op_branch_conditional_operands));

    entry_point_function->InsertBasicBlockBefore(
        std::move(new_entry_block), entry_point_function->entry().get());

    for (auto& replacement : {first_greater_then_operand_replacement.get(),
                              second_greater_then_operand_replacement.get()}) {
      if (replacement) {
        assert(replacement->IsApplicable(ir_context.get(),
                                         transformation_context));
        replacement->Apply(ir_context.get(), &transformation_context);
      }
    }
  }

  // Write out the module as a binary.
  ir_context->module()->ToBinary(binary_out, false);
  return true;
}

}  // namespace fuzz
}  // namespace spvtools
