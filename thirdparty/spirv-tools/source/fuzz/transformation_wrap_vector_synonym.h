// Copyright (c) 2021 Shiyu Liu
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_WRAP_VECTOR_SYNONYM_H_
#define SOURCE_FUZZ_TRANSFORMATION_WRAP_VECTOR_SYNONYM_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationWrapVectorSynonym : public Transformation {
 public:
  explicit TransformationWrapVectorSynonym(
      protobufs::TransformationWrapVectorSynonym message);

  TransformationWrapVectorSynonym(uint32_t instruction_id,
                                  uint32_t vector_operand1,
                                  uint32_t vector_operand2, uint32_t fresh_id,
                                  uint32_t pos);
  // - |instruction_id| must be the id of a supported arithmetic operation
  //   and must be relevant.
  // - |vector_operand1| and |vector_operand2| represents the result ids of the
  //   two vector operands.
  // - |fresh_id| is an unused id that will be used as a result id of the
  //   created instruction.
  // - |vector_operand1| and |vector_operand2| must have compatible vector types
  //   that are supported by this transformation.
  // - |pos| is an index of the operands of |instruction_id| in the
  //   |vector_operand1| and |vector_operand2|. It must be less than the size
  //   of those vector operands.
  // - A vector type with the same width as the types of the vector operands,
  //   and element type matching the type of |instruction_id|, must exist in the
  //   module.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds a new instruction before the |instruction_id| with |fresh_id|
  // result id and |instruction_id|'s opcode. The added instruction has
  // two operands: |vector_operand1| and |vector_operand2| and its type
  // id is equal to the type ids of those operands. A new fact is added
  // to the fact manager specifying that |fresh_id[pos]| is synonymous
  // to |instruction_id|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;
  protobufs::Transformation ToMessage() const override;

  // Checks whether the instruction given is supported by the transformation.
  // A valid instruction must:
  // - has both result id and type id.
  // - is a supported scalar operation instruction.
  // - has a supported type that is either int or float.
  static bool IsInstructionSupported(opt::IRContext* ir_context,
                                     const opt::Instruction& instruction);

 private:
  protobufs::TransformationWrapVectorSynonym message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_WRAP_VECTOR_SYNONYM_H_
