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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_COPY_MEMORY_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_COPY_MEMORY_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddCopyMemory : public Transformation {
 public:
  explicit TransformationAddCopyMemory(
      protobufs::TransformationAddCopyMemory message);

  TransformationAddCopyMemory(
      const protobufs::InstructionDescriptor& instruction_descriptor,
      uint32_t fresh_id, uint32_t source_id, spv::StorageClass storage_class,
      uint32_t initializer_id);

  // - |instruction_descriptor| must point to a valid instruction in the module.
  // - it should be possible to insert OpCopyMemory before
  //   |instruction_descriptor| (i.e. the module remains valid after the
  //   insertion).
  // - |source_id| must be a result id for some valid instruction in the module.
  // - |fresh_id| must be a fresh id to copy memory into.
  // - type of |source_id| must be OpTypePointer where pointee can be used with
  //   OpCopyMemory.
  // - If the pointee type of |source_id| is a struct type, it must not have the
  //   Block or BufferBlock decoration.
  // - |storage_class| must be either Private or Function.
  // - type ids of instructions with result ids |source_id| and |initialize_id|
  //   must be the same.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // A global or local variable with id |target_id| and |storage_class| class is
  // created. An 'OpCopyMemory %fresh_id %source_id' instruction is inserted
  // before the |instruction_descriptor|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if we can copy memory from |instruction| using OpCopyMemory.
  static bool IsInstructionSupported(opt::IRContext* ir_context,
                                     opt::Instruction* inst);

 private:
  // Returns whether the type, pointed to by some OpTypePointer, can be used
  // with OpCopyMemory instruction.
  static bool CanUsePointeeWithCopyMemory(const opt::analysis::Type& type);

  protobufs::TransformationAddCopyMemory message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_COPY_MEMORY_H_
