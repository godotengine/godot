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

#include "transformation_record_synonymous_constants.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationRecordSynonymousConstants::
    TransformationRecordSynonymousConstants(
        protobufs::TransformationRecordSynonymousConstants message)
    : message_(std::move(message)) {}

TransformationRecordSynonymousConstants::
    TransformationRecordSynonymousConstants(uint32_t constant1_id,
                                            uint32_t constant2_id) {
  message_.set_constant1_id(constant1_id);
  message_.set_constant2_id(constant2_id);
}

bool TransformationRecordSynonymousConstants::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // The ids must be different
  if (message_.constant1_id() == message_.constant2_id()) {
    return false;
  }

  if (transformation_context.GetFactManager()->IdIsIrrelevant(
          message_.constant1_id()) ||
      transformation_context.GetFactManager()->IdIsIrrelevant(
          message_.constant2_id())) {
    return false;
  }

  return AreEquivalentConstants(ir_context, message_.constant1_id(),
                                message_.constant2_id());
}

void TransformationRecordSynonymousConstants::Apply(
    opt::IRContext* /*unused*/,
    TransformationContext* transformation_context) const {
  // Add the fact to the fact manager
  transformation_context->GetFactManager()->AddFactDataSynonym(
      MakeDataDescriptor(message_.constant1_id(), {}),
      MakeDataDescriptor(message_.constant2_id(), {}));
}

protobufs::Transformation TransformationRecordSynonymousConstants::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_record_synonymous_constants() = message_;
  return result;
}

bool TransformationRecordSynonymousConstants::AreEquivalentConstants(
    opt::IRContext* ir_context, uint32_t constant_id1, uint32_t constant_id2) {
  const auto* def_1 = ir_context->get_def_use_mgr()->GetDef(constant_id1);
  const auto* def_2 = ir_context->get_def_use_mgr()->GetDef(constant_id2);

  // Check that the definitions exist
  if (!def_1 || !def_2) {
    // We don't use an assertion since otherwise the shrinker fails.
    return false;
  }

  auto constant1 = ir_context->get_constant_mgr()->GetConstantFromInst(def_1);
  auto constant2 = ir_context->get_constant_mgr()->GetConstantFromInst(def_2);

  // The ids must refer to constants.
  if (!constant1 || !constant2) {
    return false;
  }

  // The types must be compatible.
  if (!fuzzerutil::TypesAreEqualUpToSign(ir_context, def_1->type_id(),
                                         def_2->type_id())) {
    return false;
  }

  // If either constant is null, the other is equivalent iff it is zero-like
  if (constant1->AsNullConstant()) {
    return constant2->IsZero();
  }

  if (constant2->AsNullConstant()) {
    return constant1->IsZero();
  }

  // If the constants are scalar, they are equal iff their words are the same
  if (auto scalar1 = constant1->AsScalarConstant()) {
    // Either both or neither constant is scalar since we've already checked
    // that their types are compatible.
    assert(constant2->AsScalarConstant() && "Both constants must be scalar");
    return scalar1->words() == constant2->AsScalarConstant()->words();
  }

  // The only remaining possibility is that the constants are composite
  assert(constant1->AsCompositeConstant() && constant2->AsCompositeConstant() &&
         "Equivalence of constants can only be checked with scalar, composite "
         "or null constants.");

  // Since the types match, we already know that the number of components is
  // the same. We check that the input operands of the definitions are all
  // constants and that they are pairwise equivalent.
  for (uint32_t i = 0; i < def_1->NumInOperands(); i++) {
    if (!AreEquivalentConstants(ir_context, def_1->GetSingleWordInOperand(i),
                                def_2->GetSingleWordInOperand(i))) {
      return false;
    }
  }

  // If we get here, all the components are equivalent
  return true;
}

std::unordered_set<uint32_t>
TransformationRecordSynonymousConstants::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
