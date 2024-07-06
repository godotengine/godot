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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_COPY_MEMORY_WITH_LOAD_STORE_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_COPY_MEMORY_WITH_LOAD_STORE_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceCopyMemoryWithLoadStore : public Transformation {
 public:
  explicit TransformationReplaceCopyMemoryWithLoadStore(
      protobufs::TransformationReplaceCopyMemoryWithLoadStore message);

  TransformationReplaceCopyMemoryWithLoadStore(
      uint32_t fresh_id, const protobufs::InstructionDescriptor&
                             copy_memory_instruction_descriptor);

  // - |message_.fresh_id| must be fresh.
  // - |message_.copy_memory_instruction_descriptor| must refer to an
  //   OpCopyMemory instruction.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Replaces instruction OpCopyMemory with loading the source variable to an
  // intermediate value and storing this value into the target variable of the
  // original OpCopyMemory instruction.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationReplaceCopyMemoryWithLoadStore message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_COPY_MEMORY_WITH_LOAD_STORE_H_
