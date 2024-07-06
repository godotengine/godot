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

#ifndef SOURCE_FUZZ_TRANSFORMATION_COMPOSITE_INSERT_H_
#define SOURCE_FUZZ_TRANSFORMATION_COMPOSITE_INSERT_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationCompositeInsert : public Transformation {
 public:
  explicit TransformationCompositeInsert(
      protobufs::TransformationCompositeInsert message);

  TransformationCompositeInsert(
      const protobufs::InstructionDescriptor& instruction_to_insert_before,
      uint32_t fresh_id, uint32_t composite_id, uint32_t object_id,
      const std::vector<uint32_t>& index);

  // - |message_.fresh_id| must be fresh.
  // - |message_.composite_id| must refer to an existing composite value.
  // - |message_.index| must refer to a correct index in the composite.
  // - The type id of the object and the type id of the component of the
  //   composite at index |message_.index| must be the same.
  // - |message_.instruction_to_insert_before| must refer to a defined
  //   instruction.
  // - It must be possible to insert OpCompositeInsert before
  //   |instruction_to_insert_before|.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an instruction OpCompositeInsert before
  // |instruction_to_insert_before|, which creates a new composite from
  // |composite_id| by inserting |object_id| at the specified |index|.
  // Synonyms are created between those components which are identical in the
  // original and the modified composite and between the inserted object and its
  // copy in the modified composite.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Checks if |instruction| is a instruction of a composite type supported by
  // this transformation.
  static bool IsCompositeInstructionSupported(opt::IRContext* ir_context,
                                              opt::Instruction* instruction);

 private:
  // Helper method for adding data synonym facts when applying the
  // transformation to |ir_context| and |transformation_context|.
  void AddDataSynonymFacts(opt::IRContext* ir_context,
                           TransformationContext* transformation_context) const;

  protobufs::TransformationCompositeInsert message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_COMPOSITE_INSERT_H_
