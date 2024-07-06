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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SET_LOOP_CONTROL_H_
#define SOURCE_FUZZ_TRANSFORMATION_SET_LOOP_CONTROL_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationSetLoopControl : public Transformation {
 public:
  const static uint32_t kLoopControlMaskInOperandIndex = 2;
  const static uint32_t kLoopControlFirstLiteralInOperandIndex = 3;

  explicit TransformationSetLoopControl(
      protobufs::TransformationSetLoopControl message);

  TransformationSetLoopControl(uint32_t block_id, uint32_t loop_control,
                               uint32_t peel_count, uint32_t partial_count);

  // - |message_.block_id| must be a block containing an OpLoopMerge
  //   instruction.
  // - |message_.loop_control| must be a legal loop control mask that
  //   only uses controls available in the SPIR-V version associated with
  //   |ir_context|, and must not add loop controls that are only valid in the
  //   presence of guarantees about what the loop does (e.g. MinIterations).
  // - |message_.peel_count| (respectively |message_.partial_count|) must be
  //   zero PeelCount (respectively PartialCount) is set in
  //   |message_.loop_control|.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - The loop control operand of the OpLoopMergeInstruction in
  //   |message_.block_id| is overwritten with |message_.loop_control|.
  // - The literals associated with the loop control are updated to reflect any
  //   controls with associated literals that have been removed (e.g.
  //   MinIterations), and any that have been added (PeelCount and/or
  //   PartialCount).
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Does the version of SPIR-V being used support the PartialCount loop
  // control?
  static bool PartialCountIsSupported(opt::IRContext* ir_context);

  // Does the version of SPIR-V being used support the PeelCount loop control?
  static bool PeelCountIsSupported(opt::IRContext* ir_context);

 private:
  // Returns true if and only if |loop_single_bit_mask| is *not* set in
  // |existing_loop_control| but *is* set in |message_.loop_control|.
  bool LoopControlBitIsAddedByTransformation(
      spv::LoopControlMask loop_control_single_bit_mask,
      uint32_t existing_loop_control_mask) const;

  protobufs::TransformationSetLoopControl message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_SET_LOOP_CONTROL_H_
