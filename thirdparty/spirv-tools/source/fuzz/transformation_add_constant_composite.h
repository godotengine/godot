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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_CONSTANT_COMPOSITE_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_CONSTANT_COMPOSITE_H_

#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddConstantComposite : public Transformation {
 public:
  explicit TransformationAddConstantComposite(
      protobufs::TransformationAddConstantComposite message);

  TransformationAddConstantComposite(
      uint32_t fresh_id, uint32_t type_id,
      const std::vector<uint32_t>& constituent_ids, bool is_irrelevant);

  // - |message_.fresh_id| must be a fresh id
  // - |message_.type_id| must be the id of a composite type
  // - |message_.constituent_id| must refer to ids that match the constituent
  //   types of this composite type
  // - If |message_.type_id| is a struct type, it must not have the Block or
  //   BufferBlock decoration
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - Adds an OpConstantComposite instruction defining a constant of type
  //   |message_.type_id|, using |message_.constituent_id| as constituents, with
  //   result id |message_.fresh_id|.
  // - Creates an IdIsIrrelevant fact about |fresh_id| if |is_irrelevant| is
  //   true.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddConstantComposite message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_CONSTANT_COMPOSITE_H_
