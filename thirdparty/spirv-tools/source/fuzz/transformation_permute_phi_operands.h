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

#ifndef SOURCE_FUZZ_TRANSFORMATION_PERMUTE_PHI_OPERANDS_H_
#define SOURCE_FUZZ_TRANSFORMATION_PERMUTE_PHI_OPERANDS_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationPermutePhiOperands : public Transformation {
 public:
  explicit TransformationPermutePhiOperands(
      protobufs::TransformationPermutePhiOperands message);

  TransformationPermutePhiOperands(uint32_t result_id,
                                   const std::vector<uint32_t>& permutation);

  // - |result_id| must be a valid id of some OpPhi instruction in the module.
  // - |permutation| must contain elements in the range [0, n/2 - 1] where |n|
  //   is a number of operands to the instruction with |result_id|. All elements
  //   must be unique (i.e. |permutation.size() == n / 2|).
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Permutes operands of the OpPhi instruction with |result_id| according to
  // the elements in |permutation|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationPermutePhiOperands message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_PERMUTE_PHI_OPERANDS_H_
