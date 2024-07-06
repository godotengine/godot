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

#ifndef SOURCE_FUZZ_TRANSFORMATION_WRAP_EARLY_TERMINATOR_IN_FUNCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_WRAP_EARLY_TERMINATOR_IN_FUNCTION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationWrapEarlyTerminatorInFunction : public Transformation {
 public:
  explicit TransformationWrapEarlyTerminatorInFunction(
      protobufs::TransformationWrapEarlyTerminatorInFunction message);

  TransformationWrapEarlyTerminatorInFunction(
      uint32_t fresh_id,
      const protobufs::InstructionDescriptor& early_terminator_instruction,
      uint32_t returned_value_id);

  // - |message_.fresh_id| must be fresh.
  // - |message_.early_terminator_instruction| must identify an early terminator
  //   instruction, i.e. an instruction with opcode OpKill, OpUnreachable or
  //   OpTerminateInvocation.
  // - A suitable wrapper function for the early terminator must exist, and it
  //   must be distinct from the function containing
  //   |message_.early_terminator_instruction|.
  // - If the enclosing function has non-void return type then
  //   |message_.returned_value_instruction| must be the id of an instruction of
  //   the return type that is available at the point of the early terminator
  //   instruction.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // An OpFunctionCall instruction to an appropriate wrapper function is
  // inserted before |message_.early_terminator_instruction|, and
  // |message_.early_terminator_instruction| is replaced with either OpReturn
  // or OpReturnValue |message_.returned_value_instruction| depending on whether
  // the enclosing function's return type is void.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  static opt::Function* MaybeGetWrapperFunction(
      opt::IRContext* ir_context, spv::Op early_terminator_opcode);

 private:
  protobufs::TransformationWrapEarlyTerminatorInFunction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_WRAP_EARLY_TERMINATOR_IN_FUNCTION_H_
