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

#ifndef SOURCE_FUZZ_TRANSFORMATION_FUNCTION_CALL_H_
#define SOURCE_FUZZ_TRANSFORMATION_FUNCTION_CALL_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationFunctionCall : public Transformation {
 public:
  explicit TransformationFunctionCall(
      protobufs::TransformationFunctionCall message);

  TransformationFunctionCall(
      uint32_t fresh_id, uint32_t callee_id,
      const std::vector<uint32_t>& argument_id,
      const protobufs::InstructionDescriptor& instruction_to_insert_before);

  // - |message_.fresh_id| must be fresh
  // - |message_.instruction_to_insert_before| must identify an instruction
  //   before which an OpFunctionCall can be legitimately inserted
  // - |message_.function_id| must be the id of a function, and calling the
  //   function before the identified instruction must not introduce recursion
  // - |message_.arg_id| must provide suitable arguments for the function call
  //   (they must have the right types and be available according to dominance
  //   rules)
  // - If the insertion point is not in a dead block then |message_function_id|
  //   must refer to a livesafe function, and every pointer argument in
  //   |message_.arg_id| must refer to an arbitrary-valued variable
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an instruction of the form:
  //   |fresh_id| = OpFunctionCall %type |callee_id| |arg_id...|
  // before |instruction_to_insert_before|, where %type is the return type of
  // |callee_id|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationFunctionCall message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_FUNCTION_CALL_H_
