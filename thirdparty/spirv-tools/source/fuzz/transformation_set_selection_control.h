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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SET_SELECTION_CONTROL_H_
#define SOURCE_FUZZ_TRANSFORMATION_SET_SELECTION_CONTROL_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationSetSelectionControl : public Transformation {
 public:
  explicit TransformationSetSelectionControl(
      protobufs::TransformationSetSelectionControl message);

  TransformationSetSelectionControl(uint32_t block_id,
                                    uint32_t selection_control);

  // - |message_.block_id| must be a block containing an OpSelectionMerge
  //   instruction.
  // - |message_.selection_control| must be one of None, Flatten or
  //   DontFlatten.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - The selection control operand of the OpSelectionMergeInstruction in
  //   |message_.block_id| is overwritten with |message_.selection_control|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationSetSelectionControl message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_SET_SELECTION_CONTROL_H_
