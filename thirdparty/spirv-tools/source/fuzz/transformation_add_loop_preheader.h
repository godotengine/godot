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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_LOOP_PREHEADER_H
#define SOURCE_FUZZ_TRANSFORMATION_ADD_LOOP_PREHEADER_H

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

class TransformationAddLoopPreheader : public Transformation {
 public:
  explicit TransformationAddLoopPreheader(
      protobufs::TransformationAddLoopPreheader message);

  TransformationAddLoopPreheader(uint32_t loop_header_block, uint32_t fresh_id,
                                 std::vector<uint32_t> phi_id);

  // - |message_.loop_header_block| must be the id of a loop header block in
  //   the given module.
  // - |message_.fresh_id| must be an available id.
  // - |message_.phi_ids| must be a list of available ids.
  //   It can be empty if the loop header only has one predecessor outside of
  //   the loop. Otherwise, it must contain at least as many ids as OpPhi
  //   instructions in the loop header block.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds a preheader block as the unique out-of-loop predecessor of the given
  // loop header block. All of the existing out-of-loop predecessors of the
  // header are changed so that they branch to the preheader instead.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddLoopPreheader message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_LOOP_PREHEADER_H
