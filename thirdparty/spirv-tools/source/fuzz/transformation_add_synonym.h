// Copyright (c) 2020 Vasyl Teliman
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_SYNONYM_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_SYNONYM_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddSynonym : public Transformation {
 public:
  explicit TransformationAddSynonym(
      protobufs::TransformationAddSynonym message);

  TransformationAddSynonym(
      uint32_t result_id,
      protobufs::TransformationAddSynonym::SynonymType synonym_type,
      uint32_t synonym_fresh_id,
      const protobufs::InstructionDescriptor& insert_before);

  // - |result_id| must be a valid result id of some instruction in the module.
  // - |result_id| may not be an irrelevant id.
  // - |synonym_type| is a type of the synonymous instruction that will be
  //   created.
  // - |synonym_fresh_id| is a fresh id.
  // - |insert_before| must be a valid instruction descriptor and we must be
  //   able to insert a new synonymous instruction before |insert_before|.
  // - |result_id| must be available before |insert_before|.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Creates a new synonymous instruction according to the |synonym_type| with
  // result id |synonym_fresh_id|.
  // Inserts that instruction before |insert_before| and creates a fact
  // that the |synonym_fresh_id| and the |result_id| are synonymous.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if we can create a synonym of |inst| according to the
  // |synonym_type|.
  static bool IsInstructionValid(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context,
      opt::Instruction* inst,
      protobufs::TransformationAddSynonym::SynonymType synonym_type);

  // Returns true if |synonym_type| requires an additional constant instruction
  // to be present in the module.
  static bool IsAdditionalConstantRequired(
      protobufs::TransformationAddSynonym::SynonymType synonym_type);

 private:
  // Returns a new instruction which is synonymous to |message_.result_id|.
  std::unique_ptr<opt::Instruction> MakeSynonymousInstruction(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const;

  // Returns a result id of a constant instruction that is required to be
  // present in some synonym types (e.g. returns a result id of a zero constant
  // for ADD_ZERO synonym type). Returns 0 if no such instruction is present in
  // the module. This method should only be called when
  // IsAdditionalConstantRequired returns true.
  uint32_t MaybeGetConstantId(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const;

  protobufs::TransformationAddSynonym message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_SYNONYM_H_
