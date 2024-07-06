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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_ADD_SUB_MUL_WITH_CARRYING_EXTENDED_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_ADD_SUB_MUL_WITH_CARRYING_EXTENDED_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceAddSubMulWithCarryingExtended
    : public Transformation {
 public:
  explicit TransformationReplaceAddSubMulWithCarryingExtended(
      protobufs::TransformationReplaceAddSubMulWithCarryingExtended message);

  explicit TransformationReplaceAddSubMulWithCarryingExtended(
      uint32_t struct_fresh_id, uint32_t result_id);

  // - |message_.struct_fresh_id| must be fresh.
  // - |message_.result_id| must refer to an OpIAdd or OpISub or OpIMul
  //   instruction. In this instruction the result type id and the type ids of
  //   the operands must be the same.
  // - The type of struct holding the intermediate result must exists in the
  //   module.
  // - For OpIAdd, OpISub both operands must be unsigned.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // A transformation that replaces instructions OpIAdd, OpISub, OpIMul with
  // pairs of instructions. The first one (OpIAddCarry, OpISubBorrow,
  // OpUMulExtended, OpSMulExtended) computes the result into a struct. The
  // second one extracts the appropriate component from the struct to yield the
  // original result.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Checks if an OpIAdd, OpISub or OpIMul instruction can be used by the
  // transformation.
  bool static IsInstructionSuitable(opt::IRContext* ir_context,
                                    const opt::Instruction& instruction);

 private:
  protobufs::TransformationReplaceAddSubMulWithCarryingExtended message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_ADD_SUB_MUL_WITH_CARRYING_EXTENDED_H_
