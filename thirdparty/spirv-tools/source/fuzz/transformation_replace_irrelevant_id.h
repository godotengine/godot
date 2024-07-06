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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_IRRELEVANT_ID_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_IRRELEVANT_ID_H_

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceIrrelevantId : public Transformation {
 public:
  explicit TransformationReplaceIrrelevantId(
      protobufs::TransformationReplaceIrrelevantId message);

  TransformationReplaceIrrelevantId(
      const protobufs::IdUseDescriptor& id_use_descriptor,
      uint32_t replacement_id);

  // - The id of interest in |message_.id_use_descriptor| is irrelevant
  //   according to the fact manager.
  // - The types of the original id and of the replacement ids are the same.
  // - The replacement must not be the result id of an OpFunction instruction.
  // - |message_.replacement_id| is available to use at the enclosing
  //   instruction of |message_.id_use_descriptor|.
  // - The original id is in principle replaceable with any other id of the same
  //   type. See fuzzerutil::IdUseCanBeReplaced for details.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Replaces the use of an irrelevant id identified by
  // |message_.id_use_descriptor| with the id |message_.replacement_id|, which
  // has the same type as the id of interest.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if and only if |use_instruction| is OpVariable and
  // |replacement_for_use| is not a constant instruction - i.e., if it would be
  // illegal to replace the variable's initializer with the given instruction.
  static bool AttemptsToReplaceVariableInitializerWithNonConstant(
      const opt::Instruction& use_instruction,
      const opt::Instruction& replacement_for_use);

 private:
  protobufs::TransformationReplaceIrrelevantId message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_IRRELEVANT_ID_H_
