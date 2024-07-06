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

#ifndef SOURCE_FUZZ_TRANSFORMATION_COMPOSITE_CONSTRUCT_H_
#define SOURCE_FUZZ_TRANSFORMATION_COMPOSITE_CONSTRUCT_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationCompositeConstruct : public Transformation {
 public:
  explicit TransformationCompositeConstruct(
      protobufs::TransformationCompositeConstruct message);

  TransformationCompositeConstruct(
      uint32_t composite_type_id, std::vector<uint32_t> component,
      const protobufs::InstructionDescriptor& instruction_to_insert_before,
      uint32_t fresh_id);

  // - |message_.fresh_id| must not be used by the module.
  // - |message_.composite_type_id| must be the id of a composite type
  // - The elements of |message_.component| must be result ids that are
  //   suitable for constructing an element of the given composite type, in
  //   order
  // - The elements of |message_.component| must not be the target of any
  //   decorations.
  // - |message_.base_instruction_id| must be the result id of an instruction
  //   'base' in some block 'blk'.
  // - 'blk' must contain an instruction 'inst' located |message_.offset|
  //   instructions after 'base' (if |message_.offset| = 0 then 'inst' =
  //   'base').
  // - It must be legal to insert an OpCompositeConstruct instruction directly
  //   before 'inst'.
  // - Each element of |message_.component| must be available directly before
  //   'inst'.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Inserts a new OpCompositeConstruct instruction, with id
  // |message_.fresh_id|, directly before the instruction identified by
  // |message_.base_instruction_id| and |message_.offset|.  The instruction
  // creates a composite of type |message_.composite_type_id| using the ids of
  // |message_.component|.
  //
  // Synonym facts are added between the elements of the resulting composite
  // and the components used to construct it, as long as the associated ids
  // support synonym creation.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  // Helper to decide whether the components of the transformation are suitable
  // for constructing an array of the given type.
  bool ComponentsForArrayConstructionAreOK(
      opt::IRContext* ir_context, const opt::analysis::Array& array_type) const;

  // Similar, but for matrices.
  bool ComponentsForMatrixConstructionAreOK(
      opt::IRContext* ir_context,
      const opt::analysis::Matrix& matrix_type) const;

  // Similar, but for structs.
  bool ComponentsForStructConstructionAreOK(
      opt::IRContext* ir_context,
      const opt::analysis::Struct& struct_type) const;

  // Similar, but for vectors.
  bool ComponentsForVectorConstructionAreOK(
      opt::IRContext* ir_context,
      const opt::analysis::Vector& vector_type) const;

  // Helper method for adding data synonym facts when applying the
  // transformation to |ir_context| and |transformation_context|.
  void AddDataSynonymFacts(opt::IRContext* ir_context,
                           TransformationContext* transformation_context) const;

  protobufs::TransformationCompositeConstruct message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_COMPOSITE_CONSTRUCT_H_
