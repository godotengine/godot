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

#ifndef SOURCE_FUZZ_TRANSFORMATION_INLINE_FUNCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_INLINE_FUNCTION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationInlineFunction : public Transformation {
 public:
  explicit TransformationInlineFunction(
      protobufs::TransformationInlineFunction message);

  TransformationInlineFunction(
      uint32_t function_call_id,
      const std::map<uint32_t, uint32_t>& result_id_map);

  // - |message_.result_id_map| must map the instructions of the called function
  //   to fresh ids, unless overflow ids are available.
  // - |message_.function_call_id| must be an OpFunctionCall instruction.
  //   It must not have an early return and must not use OpUnreachable or
  //   OpKill. This is to guard against making the module invalid when the
  //   caller is inside a continue construct.
  //   TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3735):
  //     Allow functions that use OpKill or OpUnreachable to be inlined if the
  //     function call is not part of a continue construct.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Replaces the OpFunctionCall instruction, identified by
  // |message_.function_call_id|, with a copy of the function's body.
  // |message_.result_id_map| is used to provide fresh ids for duplicate
  // instructions.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if |function_call_instruction| is defined, is an
  // OpFunctionCall instruction, has no uses if its return type is void, has no
  // early returns and has no uses of OpKill or OpUnreachable.
  static bool IsSuitableForInlining(
      opt::IRContext* ir_context, opt::Instruction* function_call_instruction);

 private:
  protobufs::TransformationInlineFunction message_;

  // Inline |instruction_to_be_inlined| by setting its ids to the corresponding
  // ids in |result_id_map|.
  void AdaptInlinedInstruction(
      const std::map<uint32_t, uint32_t>& result_id_map,
      opt::IRContext* ir_context, opt::Instruction* instruction) const;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_INLINE_FUNCTION_H_
