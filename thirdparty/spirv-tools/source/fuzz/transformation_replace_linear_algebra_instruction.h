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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_LINEAR_ALGEBRA_INSTRUCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_LINEAR_ALGEBRA_INSTRUCTION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceLinearAlgebraInstruction : public Transformation {
 public:
  explicit TransformationReplaceLinearAlgebraInstruction(
      protobufs::TransformationReplaceLinearAlgebraInstruction message);

  TransformationReplaceLinearAlgebraInstruction(
      const std::vector<uint32_t>& fresh_ids,
      const protobufs::InstructionDescriptor& instruction_descriptor);

  // - |message_.fresh_ids| must be fresh ids needed to apply the
  // transformation.
  // - |message_.instruction_descriptor| must be a linear algebra instruction
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Replaces a linear algebra instruction.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns the number of ids needed to apply the transformation.
  static uint32_t GetRequiredFreshIdCount(opt::IRContext* ir_context,
                                          opt::Instruction* instruction);

 private:
  protobufs::TransformationReplaceLinearAlgebraInstruction message_;

  // Replaces an OpTranspose instruction.
  void ReplaceOpTranspose(opt::IRContext* ir_context,
                          opt::Instruction* instruction) const;

  // Replaces an OpVectorTimesScalar instruction.
  void ReplaceOpVectorTimesScalar(opt::IRContext* ir_context,
                                  opt::Instruction* instruction) const;

  // Replaces an OpMatrixTimesScalar instruction.
  void ReplaceOpMatrixTimesScalar(opt::IRContext* ir_context,
                                  opt::Instruction* instruction) const;

  // Replaces an OpVectorTimesMatrix instruction.
  void ReplaceOpVectorTimesMatrix(opt::IRContext* ir_context,
                                  opt::Instruction* instruction) const;

  // Replaces an OpMatrixTimesVector instruction.
  void ReplaceOpMatrixTimesVector(opt::IRContext* ir_context,
                                  opt::Instruction* instruction) const;

  // Replaces an OpMatrixTimesMatrix instruction.
  void ReplaceOpMatrixTimesMatrix(opt::IRContext* ir_context,
                                  opt::Instruction* instruction) const;

  // Replaces an OpOuterProduct instruction.
  void ReplaceOpOuterProduct(opt::IRContext* ir_context,
                             opt::Instruction* instruction) const;

  // Replaces an OpDot instruction.
  void ReplaceOpDot(opt::IRContext* ir_context,
                    opt::Instruction* instruction) const;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_LINEAR_ALGEBRA_INSTRUCTION_H_
