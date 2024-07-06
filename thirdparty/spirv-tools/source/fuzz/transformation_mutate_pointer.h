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

#ifndef SOURCE_FUZZ_TRANSFORMATION_MUTATE_POINTER_H_
#define SOURCE_FUZZ_TRANSFORMATION_MUTATE_POINTER_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationMutatePointer : public Transformation {
 public:
  explicit TransformationMutatePointer(
      protobufs::TransformationMutatePointer message);

  explicit TransformationMutatePointer(
      uint32_t pointer_id, uint32_t fresh_id,
      const protobufs::InstructionDescriptor& insert_before);

  // - |fresh_id| must be fresh.
  // - |insert_before| must be a valid instruction descriptor of some
  //   instruction in the module.
  // - It should be possible to insert OpLoad and OpStore before
  //   |insert_before|.
  // - |pointer_id| must be a result id of some instruction in the module.
  // - Instruction with result id |pointer_id| must be valid (see
  //   IsValidPointerInstruction method).
  // - There must exist an irrelevant constant in the module. Type of the
  //   constant must be equal to the type of the |pointer_id|'s pointee.
  // - |pointer_id| must be available (according to the dominance rules) before
  //   |insert_before|.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Inserts the following instructions before |insert_before|:
  //   %fresh_id = OpLoad %pointee_type_id %pointer_id
  //               OpStore %pointer_id %constant_id
  //               OpStore %pointer_id %fresh_id
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if |inst| valid pointer according to the following:
  // - |inst| has result id and type id.
  // - |inst| is neither OpUndef nor OpConstantNull.
  // - |inst| has a pointer type.
  // - |inst|'s storage class is either Private, Function or Workgroup.
  // - |inst|'s pointee type and all its constituents are either scalar or
  //   composite.
  static bool IsValidPointerInstruction(opt::IRContext* ir_context,
                                        const opt::Instruction& inst);

 private:
  protobufs::TransformationMutatePointer message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_MUTATE_POINTER_H_
