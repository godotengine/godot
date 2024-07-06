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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ACCESS_CHAIN_H_
#define SOURCE_FUZZ_TRANSFORMATION_ACCESS_CHAIN_H_

#include <utility>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAccessChain : public Transformation {
 public:
  explicit TransformationAccessChain(
      protobufs::TransformationAccessChain message);

  TransformationAccessChain(
      uint32_t fresh_id, uint32_t pointer_id,
      const std::vector<uint32_t>& index_id,
      const protobufs::InstructionDescriptor& instruction_to_insert_before,
      const std::vector<std::pair<uint32_t, uint32_t>>& fresh_ids_for_clamping =
          {});

  // - |message_.fresh_id| must be fresh.
  // - |message_.instruction_to_insert_before| must identify an instruction
  //   before which it is legitimate to insert an OpAccessChain instruction.
  // - |message_.pointer_id| must be a result id with pointer type that is
  //   available (according to dominance rules) at the insertion point.
  // - The pointer must not be OpConstantNull or OpUndef.
  // - |message_.index_id| must be a sequence of ids of 32-bit integers
  //   such that it is possible to walk the pointee type of
  //   |message_.pointer_id| using these indices.
  // - All indices used to access a struct must be OpConstant.
  // - The indices used to index non-struct composites will be clamped to be
  //   in bound. Enough fresh ids must be given in
  //   |message_.fresh_id_for_clamping| to perform clamping (2 for
  //   each index accessing a non-struct). This requires the bool type and
  //   a constant of value (bound - 1) to be declared in the module.
  // - If type t is the final type reached by walking these indices, the module
  //   must include an instruction "OpTypePointer SC %t" where SC is the storage
  //   class associated with |message_.pointer_id|.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an instruction of the form:
  //   |message_.fresh_id| = OpAccessChain %ptr |message_.index_id|
  // where %ptr is the result if of an instruction declaring a pointer to the
  // type reached by walking the pointee type of |message_.pointer_id| using
  // the indices in |message_.index_id|, and with the same storage class as
  // |message_.pointer_id|.
  //
  // For each of the indices traversing non-struct composites, two clamping
  // instructions are added using ids in |message_.fresh_id_for_clamping|.
  //
  // If the fact manager in |transformation_context| reports that
  // |message_.pointer_id| has an irrelevant pointee value, then the fact that
  // |message_.fresh_id| (the result of the access chain) also has an irrelevant
  // pointee value is also recorded.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  // Returns {false, 0} in each of the following cases:
  // - |index_id| does not correspond to a 32-bit integer constant
  // - |object_type_id| must be a struct type
  // - the constant at |index_id| is out of bounds.
  // Otherwise, returns {true, value}, where value is the value of the constant
  // at |index_id|.
  std::pair<bool, uint32_t> GetStructIndexValue(opt::IRContext* ir_context,
                                                uint32_t index_id,
                                                uint32_t object_type_id) const;

  // Returns true if |index_id| corresponds, in the given context, to a 32-bit
  // integer which can be used to index an object of the type specified by
  // |object_type_id|. Returns false otherwise.
  static bool ValidIndexToComposite(opt::IRContext* ir_context,
                                    uint32_t index_id, uint32_t object_type_id);

  protobufs::TransformationAccessChain message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ACCESS_CHAIN_H_
