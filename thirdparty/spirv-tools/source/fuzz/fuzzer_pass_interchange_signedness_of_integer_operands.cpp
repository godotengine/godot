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

#include "fuzzer_pass_interchange_signedness_of_integer_operands.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/id_use_descriptor.h"
#include "source/fuzz/transformation_record_synonymous_constants.h"
#include "source/fuzz/transformation_replace_id_with_synonym.h"

namespace spvtools {
namespace fuzz {

FuzzerPassInterchangeSignednessOfIntegerOperands::
    FuzzerPassInterchangeSignednessOfIntegerOperands(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations,
        bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassInterchangeSignednessOfIntegerOperands::Apply() {
  assert(!GetFuzzerContext()->IsWgslCompatible() &&
         "Cannot interchange signedness in WGSL");

  // Make vector keeping track of all the uses we want to replace.
  // This is a vector of pairs, where the first element is an id use descriptor
  // identifying the use of a constant id and the second is the id that should
  // be used to replace it.
  std::vector<std::pair<protobufs::IdUseDescriptor, uint32_t>> uses_to_replace;

  for (auto constant : GetIRContext()->GetConstants()) {
    uint32_t constant_id = constant->result_id();

    // We want to record the synonymity of an integer constant with another
    // constant with opposite signedness, and this can only be done if they are
    // not irrelevant.
    if (GetTransformationContext()->GetFactManager()->IdIsIrrelevant(
            constant_id)) {
      continue;
    }

    uint32_t toggled_id =
        FindOrCreateToggledIntegerConstant(constant->result_id());
    if (!toggled_id) {
      // Not an integer constant
      continue;
    }

    assert(!GetTransformationContext()->GetFactManager()->IdIsIrrelevant(
               toggled_id) &&
           "FindOrCreateToggledConstant can't produce an irrelevant id");

    // Record synonymous constants
    ApplyTransformation(
        TransformationRecordSynonymousConstants(constant_id, toggled_id));

    // Find all the uses of the constant and, for each, probabilistically
    // decide whether to replace it.
    GetIRContext()->get_def_use_mgr()->ForEachUse(
        constant_id,
        [this, toggled_id, &uses_to_replace](opt::Instruction* use_inst,
                                             uint32_t use_index) -> void {
          if (GetFuzzerContext()->ChoosePercentage(
                  GetFuzzerContext()
                      ->GetChanceOfInterchangingSignednessOfIntegerOperands())) {
            MaybeAddUseToReplace(use_inst, use_index, toggled_id,
                                 &uses_to_replace);
          }
        });
  }

  // Replace the ids if it is allowed.
  for (auto use_to_replace : uses_to_replace) {
    MaybeApplyTransformation(TransformationReplaceIdWithSynonym(
        use_to_replace.first, use_to_replace.second));
  }
}

uint32_t FuzzerPassInterchangeSignednessOfIntegerOperands::
    FindOrCreateToggledIntegerConstant(uint32_t id) {
  // |id| must not be a specialization constant because we do not know the value
  // of specialization constants.
  if (opt::IsSpecConstantInst(
          GetIRContext()->get_def_use_mgr()->GetDef(id)->opcode())) {
    return 0;
  }

  auto constant = GetIRContext()->get_constant_mgr()->FindDeclaredConstant(id);

  // This pass only toggles integer constants.
  if (!constant->AsIntConstant() &&
      (!constant->AsVectorConstant() ||
       !constant->AsVectorConstant()->component_type()->AsInteger())) {
    return 0;
  }

  if (auto integer = constant->AsIntConstant()) {
    auto type = integer->type()->AsInteger();

    // Find or create and return the toggled constant.
    return FindOrCreateIntegerConstant(std::vector<uint32_t>(integer->words()),
                                       type->width(), !type->IsSigned(), false);
  }

  // The constant is an integer vector.

  // Find the component type.
  auto component_type =
      constant->AsVectorConstant()->component_type()->AsInteger();

  // Find or create the toggled component type.
  uint32_t toggled_component_type = FindOrCreateIntegerType(
      component_type->width(), !component_type->IsSigned());

  // Get the information about the toggled components. We need to extract this
  // information now because the analyses might be invalidated, which would make
  // the constant and component_type variables invalid.
  std::vector<std::vector<uint32_t>> component_words;

  for (auto component : constant->AsVectorConstant()->GetComponents()) {
    component_words.push_back(component->AsIntConstant()->words());
  }
  uint32_t width = component_type->width();
  bool is_signed = !component_type->IsSigned();

  std::vector<uint32_t> toggled_components;

  // Find or create the toggled components.
  for (auto words : component_words) {
    toggled_components.push_back(
        FindOrCreateIntegerConstant(words, width, is_signed, false));
  }

  // Find or create the required toggled vector type.
  uint32_t toggled_type = FindOrCreateVectorType(
      toggled_component_type, (uint32_t)toggled_components.size());

  // Find or create and return the toggled vector constant.
  return FindOrCreateCompositeConstant(toggled_components, toggled_type, false);
}

}  // namespace fuzz
}  // namespace spvtools
