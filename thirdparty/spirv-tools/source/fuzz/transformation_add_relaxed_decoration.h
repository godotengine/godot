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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_RELAXED_DECORATION_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_RELAXED_DECORATION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddRelaxedDecoration : public Transformation {
 public:
  explicit TransformationAddRelaxedDecoration(
      protobufs::TransformationAddRelaxedDecoration message);

  explicit TransformationAddRelaxedDecoration(uint32_t fresh_id);

  // - |message_.result_id| must be the result id of an instruction, which is
  //   located in a dead block and Relaxed decoration can be applied.
  // - It does not matter whether this instruction is already annotated with the
  //   Relaxed decoration.
  bool IsApplicable(

      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds a decoration of the form:
  //   'OpDecoration |message_.result_id| RelaxedPrecision'
  // to the module.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if and only if |opcode| is the opcode of an instruction
  // that operates on 32-bit integers and 32-bit floats
  // as defined by the SPIR-V specification.
  static bool IsNumeric(spv::Op opcode);

 private:
  protobufs::TransformationAddRelaxedDecoration message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_RELAXED_DECORATION_H_
