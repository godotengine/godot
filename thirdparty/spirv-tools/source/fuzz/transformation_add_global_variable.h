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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_GLOBAL_VARIABLE_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_GLOBAL_VARIABLE_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddGlobalVariable : public Transformation {
 public:
  explicit TransformationAddGlobalVariable(
      protobufs::TransformationAddGlobalVariable message);

  TransformationAddGlobalVariable(uint32_t fresh_id, uint32_t type_id,
                                  spv::StorageClass storage_class,
                                  uint32_t initializer_id,
                                  bool value_is_irrelevant);

  // - |message_.fresh_id| must be fresh
  // - |message_.type_id| must be the id of a pointer type with the same storage
  //   class as |message_.storage_class|
  // - |message_.storage_class| must be Private or Workgroup
  // - |message_.initializer_id| must be 0 if |message_.storage_class| is
  //   Workgroup, and otherwise may either be 0 or the id of a constant whose
  //   type is the pointee type of |message_.type_id|
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds a global variable with storage class |message_.storage_class| to the
  // module, with type |message_.type_id| and either no initializer or
  // |message_.initializer_id| as an initializer, depending on whether
  // |message_.initializer_id| is 0.  The global variable has result id
  // |message_.fresh_id|.
  //
  // If |message_.value_is_irrelevant| holds, adds a corresponding fact to the
  // fact manager in |transformation_context|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddGlobalVariable message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_GLOBAL_VARIABLE_H_
