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

#ifndef SOURCE_FUZZ_TRANSFORMATION_VECTOR_SHUFFLE_H_
#define SOURCE_FUZZ_TRANSFORMATION_VECTOR_SHUFFLE_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"
#include "source/opt/types.h"

namespace spvtools {
namespace fuzz {

class TransformationVectorShuffle : public Transformation {
 public:
  explicit TransformationVectorShuffle(
      protobufs::TransformationVectorShuffle message);

  TransformationVectorShuffle(
      const protobufs::InstructionDescriptor& instruction_to_insert_before,
      uint32_t fresh_id, uint32_t vector1, uint32_t vector2,
      const std::vector<uint32_t>& component);

  // - |message_.fresh_id| must not be in use
  // - |message_.instruction_to_insert_before| must identify an instruction
  //   before which it is legitimate to insert an OpVectorShuffle
  // - |message_.vector1| and |message_.vector2| must be instructions of vector
  //   type, and the element types of these vectors must be the same
  // - Each element of |message_.component| must either be 0xFFFFFFFF
  //   (representing an undefined component), or must be less than the combined
  //   sizes of the input vectors
  // - The module must already contain a vector type with the same element type
  //   as |message_.vector1| and |message_.vector2|, and with the size of
  //   |message_component| as its element count
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Inserts an OpVectorShuffle instruction before
  // |message_.instruction_to_insert_before|, shuffles vectors
  // |message_.vector1| and |message_.vector2| using the indices provided by
  // |message_.component|, into |message_.fresh_id|.
  //
  // If |message_.fresh_id| is irrelevant (e.g. due to being in a dead block)
  // of if one of |message_.vector1| or |message_.vector2| is irrelevant and the
  // shuffle reads components from the irrelevant vector then no synonym facts
  // are added.
  //
  // Otherwise, a fact is added recording that element of |message_.fresh_id| is
  // synonymous with the element of |message_.vector1| or |message_.vector2|
  // from which it came (with undefined components being ignored).
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  // Returns a type id that already exists in |ir_context| suitable for
  // representing the result of the shuffle, where |element_type| is known to
  // be the common element type of the vectors to which the shuffle is being
  // applied.  Returns 0 if no such id exists.
  uint32_t GetResultTypeId(opt::IRContext* ir_context,
                           const opt::analysis::Type& element_type) const;

  // Returns the type associated with |id_of_vector| in |ir_context|.
  static opt::analysis::Vector* GetVectorType(opt::IRContext* ir_context,
                                              uint32_t id_of_vector);

  // Helper method for adding data synonym facts when applying the
  // transformation to |ir_context| and |transformation_context|.
  void AddDataSynonymFacts(opt::IRContext* ir_context,
                           TransformationContext* transformation_context) const;

  protobufs::TransformationVectorShuffle message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_VECTOR_SHUFFLE_H_
