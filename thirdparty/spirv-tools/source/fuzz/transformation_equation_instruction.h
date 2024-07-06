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

#ifndef SOURCE_FUZZ_TRANSFORMATION_EQUATION_INSTRUCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_EQUATION_INSTRUCTION_H_

#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationEquationInstruction : public Transformation {
 public:
  explicit TransformationEquationInstruction(
      protobufs::TransformationEquationInstruction message);

  TransformationEquationInstruction(
      uint32_t fresh_id, spv::Op opcode,
      const std::vector<uint32_t>& in_operand_id,
      const protobufs::InstructionDescriptor& instruction_to_insert_before);

  // - |message_.fresh_id| must be fresh.
  // - |message_.instruction_to_insert_before| must identify an instruction
  //   before which an equation instruction can legitimately be inserted.
  // - Each id in |message_.in_operand_id| must exist, not be an OpUndef, and
  //   be available before |message_.instruction_to_insert_before|.
  // - |message_.opcode| must be an opcode for which we know how to handle
  //   equations, the types of the ids in |message_.in_operand_id| must be
  //   suitable for use with this opcode, and the module must contain an
  //   appropriate result type id.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an instruction to the module, right before
  // |message_.instruction_to_insert_before|, of the form:
  //
  // |message_.fresh_id| = |message_.opcode| %type |message_.in_operand_ids|
  //
  // where %type is a type id that already exists in the module and that is
  // compatible with the opcode and input operands.
  //
  // The fact manager is also updated to inform it of this equation fact.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  // Returns type id for the equation instruction. Returns 0 if result type does
  // not exist.
  uint32_t MaybeGetResultTypeId(opt::IRContext* ir_context) const;

  protobufs::TransformationEquationInstruction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_EQUATION_INSTRUCTION_H_
