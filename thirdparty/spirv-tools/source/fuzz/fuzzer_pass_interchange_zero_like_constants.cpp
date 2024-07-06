// Copyright (c) 2020 Stefano Milizia
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

#include "source/fuzz/fuzzer_pass_interchange_zero_like_constants.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/id_use_descriptor.h"
#include "source/fuzz/transformation_record_synonymous_constants.h"
#include "source/fuzz/transformation_replace_id_with_synonym.h"

namespace spvtools {
namespace fuzz {
FuzzerPassInterchangeZeroLikeConstants::FuzzerPassInterchangeZeroLikeConstants(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

uint32_t FuzzerPassInterchangeZeroLikeConstants::FindOrCreateToggledConstant(
    opt::Instruction* declaration) {
  // |declaration| must not be a specialization constant because we do not know
  // the value of specialization constants.
  if (opt::IsSpecConstantInst(declaration->opcode())) {
    return 0;
  }

  auto constant = GetIRContext()->get_constant_mgr()->FindDeclaredConstant(
      declaration->result_id());

  // This pass only toggles zero-like constants
  if (!constant->IsZero()) {
    return 0;
  }

  if (constant->AsScalarConstant()) {
    return FindOrCreateNullConstant(declaration->type_id());
  } else if (constant->AsNullConstant()) {
    // Add declaration of equivalent scalar constant
    auto kind = constant->type()->kind();
    if (kind == opt::analysis::Type::kBool ||
        kind == opt::analysis::Type::kInteger ||
        kind == opt::analysis::Type::kFloat) {
      return FindOrCreateZeroConstant(declaration->type_id(), false);
    }
  }

  return 0;
}

void FuzzerPassInterchangeZeroLikeConstants::Apply() {
  // Make vector keeping track of all the uses we want to replace.
  // This is a vector of pairs, where the first element is an id use descriptor
  // identifying the use of a constant id and the second is the id that should
  // be used to replace it.
  std::vector<std::pair<protobufs::IdUseDescriptor, uint32_t>> uses_to_replace;

  for (auto constant : GetIRContext()->GetConstants()) {
    uint32_t constant_id = constant->result_id();
    if (GetTransformationContext()->GetFactManager()->IdIsIrrelevant(
            constant_id)) {
      continue;
    }

    uint32_t toggled_id = FindOrCreateToggledConstant(constant);
    if (!toggled_id) {
      // Not a zero-like constant
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
                      ->GetChanceOfInterchangingZeroLikeConstants())) {
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
}  // namespace fuzz
}  // namespace spvtools
