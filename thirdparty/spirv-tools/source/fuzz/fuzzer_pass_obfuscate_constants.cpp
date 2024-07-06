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

#include "source/fuzz/fuzzer_pass_obfuscate_constants.h"

#include <algorithm>
#include <cmath>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_replace_boolean_constant_with_constant_binary.h"
#include "source/fuzz/transformation_replace_constant_with_uniform.h"
#include "source/fuzz/uniform_buffer_element_descriptor.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

FuzzerPassObfuscateConstants::FuzzerPassObfuscateConstants(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassObfuscateConstants::ObfuscateBoolConstantViaConstantPair(
    uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
    const std::vector<spv::Op>& greater_than_opcodes,
    const std::vector<spv::Op>& less_than_opcodes, uint32_t constant_id_1,
    uint32_t constant_id_2, bool first_constant_is_larger) {
  auto bool_constant_opcode = GetIRContext()
                                  ->get_def_use_mgr()
                                  ->GetDef(bool_constant_use.id_of_interest())
                                  ->opcode();
  assert((bool_constant_opcode == spv::Op::OpConstantFalse ||
          bool_constant_opcode == spv::Op::OpConstantTrue) &&
         "Precondition: this must be a usage of a boolean constant.");

  // Pick an opcode at random.  First randomly decide whether to generate
  // a 'greater than' or 'less than' kind of opcode, and then select a
  // random opcode from the resulting subset.
  spv::Op comparison_opcode;
  if (GetFuzzerContext()->ChooseEven()) {
    comparison_opcode = greater_than_opcodes[GetFuzzerContext()->RandomIndex(
        greater_than_opcodes)];
  } else {
    comparison_opcode =
        less_than_opcodes[GetFuzzerContext()->RandomIndex(less_than_opcodes)];
  }

  // We now need to decide how to order constant_id_1 and constant_id_2 such
  // that 'constant_id_1 comparison_opcode constant_id_2' evaluates to the
  // boolean constant.
  const bool is_greater_than_opcode =
      std::find(greater_than_opcodes.begin(), greater_than_opcodes.end(),
                comparison_opcode) != greater_than_opcodes.end();
  uint32_t lhs_id;
  uint32_t rhs_id;
  if ((bool_constant_opcode == spv::Op::OpConstantTrue &&
       first_constant_is_larger == is_greater_than_opcode) ||
      (bool_constant_opcode == spv::Op::OpConstantFalse &&
       first_constant_is_larger != is_greater_than_opcode)) {
    lhs_id = constant_id_1;
    rhs_id = constant_id_2;
  } else {
    lhs_id = constant_id_2;
    rhs_id = constant_id_1;
  }

  // We can now make a transformation that will replace |bool_constant_use|
  // with an expression of the form (written using infix notation):
  // |lhs_id| |comparison_opcode| |rhs_id|
  auto transformation = TransformationReplaceBooleanConstantWithConstantBinary(
      bool_constant_use, lhs_id, rhs_id, comparison_opcode,
      GetFuzzerContext()->GetFreshId());
  // The transformation should be applicable by construction.
  assert(
      transformation.IsApplicable(GetIRContext(), *GetTransformationContext()));

  // Applying this transformation yields a pointer to the new instruction that
  // computes the result of the binary expression.
  auto binary_operator_instruction = transformation.ApplyWithResult(
      GetIRContext(), GetTransformationContext());

  // Add this transformation to the sequence of transformations that have been
  // applied.
  *GetTransformations()->add_transformation() = transformation.ToMessage();

  // Having made a binary expression, there may now be opportunities to further
  // obfuscate the constants used as the LHS and RHS of the expression (e.g. by
  // replacing them with loads from known uniforms).
  //
  // We thus consider operands 0 and 1 (LHS and RHS in turn).
  for (uint32_t index : {0u, 1u}) {
    // We randomly decide, based on the current depth of obfuscation, whether
    // to further obfuscate this operand.
    if (GetFuzzerContext()->GoDeeperInConstantObfuscation(depth)) {
      auto in_operand_use = MakeIdUseDescriptor(
          binary_operator_instruction->GetSingleWordInOperand(index),
          MakeInstructionDescriptor(binary_operator_instruction->result_id(),
                                    binary_operator_instruction->opcode(), 0),
          index);
      ObfuscateConstant(depth + 1, in_operand_use);
    }
  }
}

void FuzzerPassObfuscateConstants::ObfuscateBoolConstantViaFloatConstantPair(
    uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
    uint32_t float_constant_id_1, uint32_t float_constant_id_2) {
  auto float_constant_1 = GetIRContext()
                              ->get_constant_mgr()
                              ->FindDeclaredConstant(float_constant_id_1)
                              ->AsFloatConstant();
  auto float_constant_2 = GetIRContext()
                              ->get_constant_mgr()
                              ->FindDeclaredConstant(float_constant_id_2)
                              ->AsFloatConstant();
  assert(float_constant_1->words() != float_constant_2->words() &&
         "The constants should not be identical.");
  assert(std::isfinite(float_constant_1->GetValueAsDouble()) &&
         "The constants must be finite numbers.");
  assert(std::isfinite(float_constant_2->GetValueAsDouble()) &&
         "The constants must be finite numbers.");
  bool first_constant_is_larger;
  assert(float_constant_1->type()->AsFloat()->width() ==
             float_constant_2->type()->AsFloat()->width() &&
         "First and second floating-point constants must have the same width.");
  if (float_constant_1->type()->AsFloat()->width() == 32) {
    first_constant_is_larger =
        float_constant_1->GetFloat() > float_constant_2->GetFloat();
  } else {
    assert(float_constant_1->type()->AsFloat()->width() == 64 &&
           "Supported floating-point widths are 32 and 64.");
    first_constant_is_larger =
        float_constant_1->GetDouble() > float_constant_2->GetDouble();
  }
  std::vector<spv::Op> greater_than_opcodes{
      spv::Op::OpFOrdGreaterThan, spv::Op::OpFOrdGreaterThanEqual,
      spv::Op::OpFUnordGreaterThan, spv::Op::OpFUnordGreaterThanEqual};
  std::vector<spv::Op> less_than_opcodes{
      spv::Op::OpFOrdGreaterThan, spv::Op::OpFOrdGreaterThanEqual,
      spv::Op::OpFUnordGreaterThan, spv::Op::OpFUnordGreaterThanEqual};

  ObfuscateBoolConstantViaConstantPair(
      depth, bool_constant_use, greater_than_opcodes, less_than_opcodes,
      float_constant_id_1, float_constant_id_2, first_constant_is_larger);
}

void FuzzerPassObfuscateConstants::
    ObfuscateBoolConstantViaSignedIntConstantPair(
        uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
        uint32_t signed_int_constant_id_1, uint32_t signed_int_constant_id_2) {
  auto signed_int_constant_1 =
      GetIRContext()
          ->get_constant_mgr()
          ->FindDeclaredConstant(signed_int_constant_id_1)
          ->AsIntConstant();
  auto signed_int_constant_2 =
      GetIRContext()
          ->get_constant_mgr()
          ->FindDeclaredConstant(signed_int_constant_id_2)
          ->AsIntConstant();
  assert(signed_int_constant_1->words() != signed_int_constant_2->words() &&
         "The constants should not be identical.");
  bool first_constant_is_larger;
  assert(signed_int_constant_1->type()->AsInteger()->width() ==
             signed_int_constant_2->type()->AsInteger()->width() &&
         "First and second floating-point constants must have the same width.");
  assert(signed_int_constant_1->type()->AsInteger()->IsSigned());
  assert(signed_int_constant_2->type()->AsInteger()->IsSigned());
  if (signed_int_constant_1->type()->AsFloat()->width() == 32) {
    first_constant_is_larger =
        signed_int_constant_1->GetS32() > signed_int_constant_2->GetS32();
  } else {
    assert(signed_int_constant_1->type()->AsFloat()->width() == 64 &&
           "Supported integer widths are 32 and 64.");
    first_constant_is_larger =
        signed_int_constant_1->GetS64() > signed_int_constant_2->GetS64();
  }
  std::vector<spv::Op> greater_than_opcodes{spv::Op::OpSGreaterThan,
                                            spv::Op::OpSGreaterThanEqual};
  std::vector<spv::Op> less_than_opcodes{spv::Op::OpSLessThan,
                                         spv::Op::OpSLessThanEqual};

  ObfuscateBoolConstantViaConstantPair(
      depth, bool_constant_use, greater_than_opcodes, less_than_opcodes,
      signed_int_constant_id_1, signed_int_constant_id_2,
      first_constant_is_larger);
}

void FuzzerPassObfuscateConstants::
    ObfuscateBoolConstantViaUnsignedIntConstantPair(
        uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
        uint32_t unsigned_int_constant_id_1,
        uint32_t unsigned_int_constant_id_2) {
  auto unsigned_int_constant_1 =
      GetIRContext()
          ->get_constant_mgr()
          ->FindDeclaredConstant(unsigned_int_constant_id_1)
          ->AsIntConstant();
  auto unsigned_int_constant_2 =
      GetIRContext()
          ->get_constant_mgr()
          ->FindDeclaredConstant(unsigned_int_constant_id_2)
          ->AsIntConstant();
  assert(unsigned_int_constant_1->words() != unsigned_int_constant_2->words() &&
         "The constants should not be identical.");
  bool first_constant_is_larger;
  assert(unsigned_int_constant_1->type()->AsInteger()->width() ==
             unsigned_int_constant_2->type()->AsInteger()->width() &&
         "First and second floating-point constants must have the same width.");
  assert(!unsigned_int_constant_1->type()->AsInteger()->IsSigned());
  assert(!unsigned_int_constant_2->type()->AsInteger()->IsSigned());
  if (unsigned_int_constant_1->type()->AsFloat()->width() == 32) {
    first_constant_is_larger =
        unsigned_int_constant_1->GetU32() > unsigned_int_constant_2->GetU32();
  } else {
    assert(unsigned_int_constant_1->type()->AsFloat()->width() == 64 &&
           "Supported integer widths are 32 and 64.");
    first_constant_is_larger =
        unsigned_int_constant_1->GetU64() > unsigned_int_constant_2->GetU64();
  }
  std::vector<spv::Op> greater_than_opcodes{spv::Op::OpUGreaterThan,
                                            spv::Op::OpUGreaterThanEqual};
  std::vector<spv::Op> less_than_opcodes{spv::Op::OpULessThan,
                                         spv::Op::OpULessThanEqual};

  ObfuscateBoolConstantViaConstantPair(
      depth, bool_constant_use, greater_than_opcodes, less_than_opcodes,
      unsigned_int_constant_id_1, unsigned_int_constant_id_2,
      first_constant_is_larger);
}

std::vector<std::vector<uint32_t>>
FuzzerPassObfuscateConstants::GetConstantWordsFromUniformsForType(
    uint32_t type_id) {
  assert(type_id && "Type id can't be 0");
  std::vector<std::vector<uint32_t>> result;

  for (const auto& facts_and_types : GetTransformationContext()
                                         ->GetFactManager()
                                         ->GetConstantUniformFactsAndTypes()) {
    if (facts_and_types.second != type_id) {
      continue;
    }

    std::vector<uint32_t> words(facts_and_types.first.constant_word().begin(),
                                facts_and_types.first.constant_word().end());
    if (std::find(result.begin(), result.end(), words) == result.end()) {
      result.push_back(std::move(words));
    }
  }

  return result;
}

void FuzzerPassObfuscateConstants::ObfuscateBoolConstant(
    uint32_t depth, const protobufs::IdUseDescriptor& constant_use) {
  // We want to replace the boolean constant use with a binary expression over
  // scalar constants, but only if we can then potentially replace the constants
  // with uniforms of the same value.

  auto available_types_with_uniforms =
      GetTransformationContext()
          ->GetFactManager()
          ->GetTypesForWhichUniformValuesAreKnown();
  if (available_types_with_uniforms.empty()) {
    // Do not try to obfuscate if we do not have access to any uniform
    // elements with known values.
    return;
  }
  auto chosen_type_id =
      available_types_with_uniforms[GetFuzzerContext()->RandomIndex(
          available_types_with_uniforms)];
  auto available_constant_words =
      GetConstantWordsFromUniformsForType(chosen_type_id);
  if (available_constant_words.size() == 1) {
    // TODO(afd): for now we only obfuscate a boolean if there are at least
    //  two constants available from uniforms, so that we can do a
    //  comparison between them. It would be good to be able to do the
    //  obfuscation even if there is only one such constant, if there is
    //  also another regular constant available.
    return;
  }

  assert(!available_constant_words.empty() &&
         "There exists a fact but no constants - impossible");

  // We know we have at least two known-to-be-constant uniforms of the chosen
  // type.  Pick one of them at random.
  auto constant_index_1 =
      GetFuzzerContext()->RandomIndex(available_constant_words);
  uint32_t constant_index_2;

  // Now choose another one distinct from the first one.
  do {
    constant_index_2 =
        GetFuzzerContext()->RandomIndex(available_constant_words);
  } while (constant_index_1 == constant_index_2);

  auto constant_id_1 = FindOrCreateConstant(
      available_constant_words[constant_index_1], chosen_type_id, false);
  auto constant_id_2 = FindOrCreateConstant(
      available_constant_words[constant_index_2], chosen_type_id, false);

  assert(constant_id_1 != 0 && constant_id_2 != 0 &&
         "We should not find an available constant with an id of 0.");

  // Now perform the obfuscation, according to whether the type of the constants
  // is float, signed int, or unsigned int.
  auto chosen_type = GetIRContext()->get_type_mgr()->GetType(chosen_type_id);
  if (chosen_type->AsFloat()) {
    ObfuscateBoolConstantViaFloatConstantPair(depth, constant_use,
                                              constant_id_1, constant_id_2);
  } else {
    assert(chosen_type->AsInteger() &&
           "We should only have uniform facts about ints and floats.");
    if (chosen_type->AsInteger()->IsSigned()) {
      ObfuscateBoolConstantViaSignedIntConstantPair(
          depth, constant_use, constant_id_1, constant_id_2);
    } else {
      ObfuscateBoolConstantViaUnsignedIntConstantPair(
          depth, constant_use, constant_id_1, constant_id_2);
    }
  }
}

void FuzzerPassObfuscateConstants::ObfuscateScalarConstant(
    uint32_t /*depth*/, const protobufs::IdUseDescriptor& constant_use) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/2670): consider
  //  additional ways to obfuscate scalar constants.

  // Check whether we know that any uniforms are guaranteed to be equal to the
  // scalar constant associated with |constant_use|.
  auto uniform_descriptors =
      GetTransformationContext()
          ->GetFactManager()
          ->GetUniformDescriptorsForConstant(constant_use.id_of_interest());
  if (uniform_descriptors.empty()) {
    // No relevant uniforms, so do not obfuscate.
    return;
  }

  // Choose a random available uniform known to be equal to the constant.
  const auto& uniform_descriptor =
      uniform_descriptors[GetFuzzerContext()->RandomIndex(uniform_descriptors)];

  // Make sure the module has OpConstant instructions for each index used to
  // access a uniform.
  for (auto index : uniform_descriptor.index()) {
    FindOrCreateIntegerConstant({index}, 32, true, false);
  }

  // Make sure the module has OpTypePointer that points to the element type of
  // the uniform.
  const auto* uniform_variable_instr =
      FindUniformVariable(uniform_descriptor, GetIRContext(), true);
  assert(uniform_variable_instr &&
         "Uniform variable does not exist or not unique.");

  const auto* uniform_variable_type_intr =
      GetIRContext()->get_def_use_mgr()->GetDef(
          uniform_variable_instr->type_id());
  assert(uniform_variable_type_intr && "Uniform variable has invalid type");

  auto element_type_id = fuzzerutil::WalkCompositeTypeIndices(
      GetIRContext(), uniform_variable_type_intr->GetSingleWordInOperand(1),
      uniform_descriptor.index());
  assert(element_type_id && "Type of uniform variable is invalid");

  FindOrCreatePointerType(element_type_id, spv::StorageClass::Uniform);

  // Create, apply and record a transformation to replace the constant use with
  // the result of a load from the chosen uniform.
  ApplyTransformation(TransformationReplaceConstantWithUniform(
      constant_use, uniform_descriptor, GetFuzzerContext()->GetFreshId(),
      GetFuzzerContext()->GetFreshId()));
}

void FuzzerPassObfuscateConstants::ObfuscateConstant(
    uint32_t depth, const protobufs::IdUseDescriptor& constant_use) {
  switch (GetIRContext()
              ->get_def_use_mgr()
              ->GetDef(constant_use.id_of_interest())
              ->opcode()) {
    case spv::Op::OpConstantTrue:
    case spv::Op::OpConstantFalse:
      ObfuscateBoolConstant(depth, constant_use);
      break;
    case spv::Op::OpConstant:
      ObfuscateScalarConstant(depth, constant_use);
      break;
    default:
      assert(false && "The opcode should be one of the above.");
      break;
  }
}

void FuzzerPassObfuscateConstants::MaybeAddConstantIdUse(
    const opt::Instruction& inst, uint32_t in_operand_index,
    uint32_t base_instruction_result_id,
    const std::map<spv::Op, uint32_t>& skipped_opcode_count,
    std::vector<protobufs::IdUseDescriptor>* constant_uses) {
  if (inst.GetInOperand(in_operand_index).type != SPV_OPERAND_TYPE_ID) {
    // The operand is not an id, so it cannot be a constant id.
    return;
  }
  auto operand_id = inst.GetSingleWordInOperand(in_operand_index);
  auto operand_definition =
      GetIRContext()->get_def_use_mgr()->GetDef(operand_id);
  switch (operand_definition->opcode()) {
    case spv::Op::OpConstantFalse:
    case spv::Op::OpConstantTrue:
    case spv::Op::OpConstant: {
      // The operand is a constant id, so make an id use descriptor and record
      // it.
      protobufs::IdUseDescriptor id_use_descriptor;
      id_use_descriptor.set_id_of_interest(operand_id);
      id_use_descriptor.mutable_enclosing_instruction()
          ->set_target_instruction_opcode(uint32_t(inst.opcode()));
      id_use_descriptor.mutable_enclosing_instruction()
          ->set_base_instruction_result_id(base_instruction_result_id);
      id_use_descriptor.mutable_enclosing_instruction()
          ->set_num_opcodes_to_ignore(
              skipped_opcode_count.find(inst.opcode()) ==
                      skipped_opcode_count.end()
                  ? 0
                  : skipped_opcode_count.at(inst.opcode()));
      id_use_descriptor.set_in_operand_index(in_operand_index);
      constant_uses->push_back(id_use_descriptor);
    } break;
    default:
      break;
  }
}

void FuzzerPassObfuscateConstants::Apply() {
  // First, gather up all the constant uses available in the module, by going
  // through each block in each function.
  std::vector<protobufs::IdUseDescriptor> constant_uses;
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // For each constant use we encounter we are going to make an id use
      // descriptor. An id use is described with respect to a base instruction;
      // if there are instructions at the start of the block without result ids,
      // the base instruction will have to be the block's label.
      uint32_t base_instruction_result_id = block.id();

      // An id use descriptor also records how many instructions of a particular
      // opcode need to be skipped in order to find the instruction of interest
      // from the base instruction. We maintain a mapping that records a skip
      // count for each relevant opcode.
      std::map<spv::Op, uint32_t> skipped_opcode_count;

      // Go through each instruction in the block.
      for (auto& inst : block) {
        if (inst.HasResultId()) {
          // The instruction has a result id, so can be used as the base
          // instruction from now on, until another instruction with a result id
          // is encountered.
          base_instruction_result_id = inst.result_id();
          // Opcode skip counts were with respect to the previous base
          // instruction and are now irrelevant.
          skipped_opcode_count.clear();
        }

        // The instruction must not be an OpVariable, the only id that an
        // OpVariable uses is an initializer id, which has to remain
        // constant.
        if (inst.opcode() != spv::Op::OpVariable) {
          // Consider each operand of the instruction, and add a constant id
          // use for the operand if relevant.
          for (uint32_t in_operand_index = 0;
               in_operand_index < inst.NumInOperands(); in_operand_index++) {
            MaybeAddConstantIdUse(inst, in_operand_index,
                                  base_instruction_result_id,
                                  skipped_opcode_count, &constant_uses);
          }
        }

        if (!inst.HasResultId()) {
          // The instruction has no result id, so in order to identify future id
          // uses for instructions with this opcode from the existing base
          // instruction, we need to increase the skip count for this opcode.
          skipped_opcode_count[inst.opcode()] =
              skipped_opcode_count.find(inst.opcode()) ==
                      skipped_opcode_count.end()
                  ? 1
                  : skipped_opcode_count[inst.opcode()] + 1;
        }
      }
    }
  }

  // Go through the constant uses in a random order by repeatedly pulling out a
  // constant use at a random index.
  while (!constant_uses.empty()) {
    auto index = GetFuzzerContext()->RandomIndex(constant_uses);
    auto constant_use = std::move(constant_uses[index]);
    constant_uses.erase(constant_uses.begin() + index);
    // Decide probabilistically whether to skip or obfuscate this constant use.
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfObfuscatingConstant())) {
      continue;
    }
    ObfuscateConstant(0, constant_use);
  }
}

}  // namespace fuzz
}  // namespace spvtools
