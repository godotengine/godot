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

#ifndef SOURCE_FUZZ_TRANSFORMATION_RECORD_SYNONYMOUS_CONSTANTS_H
#define SOURCE_FUZZ_TRANSFORMATION_RECORD_SYNONYMOUS_CONSTANTS_H

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

class TransformationRecordSynonymousConstants : public Transformation {
 public:
  explicit TransformationRecordSynonymousConstants(
      protobufs::TransformationRecordSynonymousConstants message);

  TransformationRecordSynonymousConstants(uint32_t constant1_id,
                                          uint32_t constant2_id);

  // - |message_.constant_id| and |message_.synonym_id| are distinct ids
  //   of constants
  // - |message_.constant_id| and |message_.synonym_id| refer to constants
  //   that are equivalent.
  // Constants are equivalent if at least one of the following holds:
  // - they are equal (i.e. they have the same type ids and equal values)
  // - both of them represent zero-like values of compatible types
  // - they are composite constants with compatible types and their
  //   components are pairwise equivalent
  // Two types are compatible if at least one of the following holds:
  // - they have the same id
  // - they are integer scalar types with the same width
  // - they are integer vectors and their components have the same width
  //   (this is always the case if the components are equivalent)
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds the fact that |message_.constant_id| and |message_.synonym_id|
  // are synonyms to the fact manager. The module is not changed.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationRecordSynonymousConstants message_;

  // Returns true if the two given constants are equivalent
  // (the description of IsApplicable specifies the conditions they must satisfy
  // to be considered equivalent)
  static bool AreEquivalentConstants(opt::IRContext* ir_context,
                                     uint32_t constant_id1,
                                     uint32_t constant_id2);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_RECORD_SYNONYMOUS_CONSTANTS
