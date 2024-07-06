// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_NO_CONTRACTION_DECORATION_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_NO_CONTRACTION_DECORATION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddNoContractionDecoration : public Transformation {
 public:
  explicit TransformationAddNoContractionDecoration(
      protobufs::TransformationAddNoContractionDecoration message);

  explicit TransformationAddNoContractionDecoration(uint32_t fresh_id);

  // - |message_.result_id| must be the result id of an arithmetic instruction,
  //   as defined by the SPIR-V specification.
  // - It does not matter whether this instruction is already annotated with the
  //   NoContraction decoration.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds a decoration of the form:
  //   'OpDecoration |message_.result_id| NoContraction'
  // to the module.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if and only if |opcode| is the opcode of an arithmetic
  // instruction, as defined by the SPIR-V specification.
  static bool IsArithmetic(spv::Op opcode);

 private:
  protobufs::TransformationAddNoContractionDecoration message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_NO_CONTRACTION_DECORATION_H_
