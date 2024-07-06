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

#ifndef SOURCE_FUZZ_TRANSFORMATION_add_early_terminator_wrapper_H_
#define SOURCE_FUZZ_TRANSFORMATION_add_early_terminator_wrapper_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddEarlyTerminatorWrapper : public Transformation {
 public:
  explicit TransformationAddEarlyTerminatorWrapper(
      protobufs::TransformationAddEarlyTerminatorWrapper message);

  TransformationAddEarlyTerminatorWrapper(uint32_t function_fresh_id,
                                          uint32_t label_fresh_id,
                                          spv::Op opcode);

  // - |message_.function_fresh_id| and |message_.label_fresh_id| must be fresh
  //   and distinct.
  // - OpTypeVoid must be declared in the module.
  // - The module must contain a type for a zero-argument void function.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds a function to the module of the form:
  //
  // |message_.function_fresh_id| = OpFunction %void None %zero_args_return_void
  //    |message_.label_fresh_id| = OpLabel
  //                                |message_.opcode|
  //                                OpFunctionEnd
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddEarlyTerminatorWrapper message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_add_early_terminator_wrapper_H_
