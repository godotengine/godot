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

#ifndef SOURCE_FUZZ_TRANSFORMATION_INVERT_COMPARISON_OPERATOR_H_
#define SOURCE_FUZZ_TRANSFORMATION_INVERT_COMPARISON_OPERATOR_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationInvertComparisonOperator : public Transformation {
 public:
  explicit TransformationInvertComparisonOperator(
      protobufs::TransformationInvertComparisonOperator message);

  TransformationInvertComparisonOperator(uint32_t operator_id,
                                         uint32_t fresh_id);

  // - |operator_id| should be a result id of some instruction for which
  //   IsInversionSupported returns true.
  // - |fresh_id| must be a fresh id.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Inverts the opcode of the instruction with result id |operator_id| (e.g >=
  // becomes <) and inserts OpLogicalNot instruction after |operator_id|. Also,
  // changes the result id of OpLogicalNot to |operator_id| and the result id of
  // the inverted operator to |fresh_id|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if |opcode| is supported by this transformation.
  static bool IsInversionSupported(spv::Op opcode);

 private:
  // Returns an inverted |opcode| (e.g. < becomes >=, == becomes != etc.)
  static spv::Op InvertOpcode(spv::Op opcode);

  protobufs::TransformationInvertComparisonOperator message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_INVERT_COMPARISON_OPERATOR_H_
