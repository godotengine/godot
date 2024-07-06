// Copyright (c) 2020 André Perez Maselco
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADJUST_BRANCH_WEIGHTS_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADJUST_BRANCH_WEIGHTS_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAdjustBranchWeights : public Transformation {
 public:
  explicit TransformationAdjustBranchWeights(
      protobufs::TransformationAdjustBranchWeights message);

  TransformationAdjustBranchWeights(
      const protobufs::InstructionDescriptor& instruction_descriptor,
      const std::pair<uint32_t, uint32_t>& branch_weights);

  // - |message_.instruction_descriptor| must identify an existing
  //   branch conditional instruction
  // - At least one of |branch_weights| must be non-zero and
  //   the two weights must not overflow a 32-bit unsigned integer when added
  //   together
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adjust the branch weights of a branch conditional instruction.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAdjustBranchWeights message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADJUST_BRANCH_WEIGHTS_H_
