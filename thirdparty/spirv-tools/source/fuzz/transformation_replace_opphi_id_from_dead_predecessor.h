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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_OPPHI_ID_FROM_DEAD_PREDECESSOR_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_OPPHI_ID_FROM_DEAD_PREDECESSOR_H_

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceOpPhiIdFromDeadPredecessor : public Transformation {
 public:
  explicit TransformationReplaceOpPhiIdFromDeadPredecessor(
      protobufs::TransformationReplaceOpPhiIdFromDeadPredecessor message);

  TransformationReplaceOpPhiIdFromDeadPredecessor(uint32_t opphi_id,
                                                  uint32_t pred_label_id,
                                                  uint32_t replacement_id);

  // - |message_.opphi_id| is the id of an OpPhi instruction.
  // - |message_.pred_label_id| is the label id of one of the predecessors of
  //   the block containing the OpPhi instruction.
  // - The predecessor has been recorded as dead.
  // - |message_.replacement_id| is the id of an instruction with the same type
  //   as the OpPhi instruction, available at the end of the predecessor.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Replaces the id corresponding to predecessor |message_.pred_label_id|, in
  // the OpPhi instruction |message_.opphi_id|, with |message_.replacement_id|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationReplaceOpPhiIdFromDeadPredecessor message_;
};
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_OPPHI_ID_FROM_DEAD_PREDECESSOR_H_
