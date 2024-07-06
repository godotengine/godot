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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_SPEC_CONSTANT_OP_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_SPEC_CONSTANT_OP_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddSpecConstantOp : public Transformation {
 public:
  explicit TransformationAddSpecConstantOp(
      protobufs::TransformationAddSpecConstantOp message);

  TransformationAddSpecConstantOp(
      uint32_t fresh_id, uint32_t type_id, spv::Op opcode,
      const opt::Instruction::OperandList& operands);

  // - |fresh_id| is a fresh result id in the module.
  // - |type_id| is a valid result id of some OpType* instruction in the
  // module. It is also a valid type id with respect to |opcode|.
  // - |opcode| is one of the opcodes supported by OpSpecConstantOp.
  // - |operands| are valid with respect to |opcode|
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // |%fresh_id = OpSpecConstantOp %type_id opcode operands...| is added to the
  // module.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  void ApplyImpl(opt::IRContext* ir_context) const;

  protobufs::TransformationAddSpecConstantOp message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_SPEC_CONSTANT_OP_H_
