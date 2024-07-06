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

#ifndef SOURCE_FUZZ_TRANSFORMATION_PUSH_ID_THROUGH_VARIABLE_H_
#define SOURCE_FUZZ_TRANSFORMATION_PUSH_ID_THROUGH_VARIABLE_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationPushIdThroughVariable : public Transformation {
 public:
  explicit TransformationPushIdThroughVariable(
      protobufs::TransformationPushIdThroughVariable message);

  TransformationPushIdThroughVariable(
      uint32_t value_id, uint32_t value_synonym_fresh_id,
      uint32_t variable_fresh_id, uint32_t variable_storage_class,
      uint32_t initializer_id,
      const protobufs::InstructionDescriptor& instruction_descriptor);

  // - |message_.value_id| must be an instruction result id that has the same
  //   type as the pointee type of |message_.pointer_id|
  // - |message_.value_synonym_id| must be fresh
  // - |message_.variable_id| must be fresh
  // - |message_.variable_storage_class| must be either StorageClassPrivate or
  //   StorageClassFunction
  // - |message_.initializer_id| must be a result id of some constant in the
  //   module. Its type must be equal to the pointee type of the variable that
  //   will be created.
  // - |message_.instruction_descriptor| must identify an instruction
  //   which it is valid to insert the OpStore and OpLoad instructions before it
  //   and must be belongs to a reachable block.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Stores |value_id| to |variable_id|, loads |variable_id| to
  // |value_synonym_id|. Adds the fact that |value_synonym_id| and |value_id|
  // are synonymous if |value_id| and |value_synonym_id| are not irrelevant.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationPushIdThroughVariable message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_PUSH_ID_THROUGH_VARIABLE_H_
