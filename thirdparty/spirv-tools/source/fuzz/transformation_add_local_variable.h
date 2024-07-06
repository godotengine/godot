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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_LOCAL_VARIABLE_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_LOCAL_VARIABLE_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddLocalVariable : public Transformation {
 public:
  explicit TransformationAddLocalVariable(
      protobufs::TransformationAddLocalVariable message);

  TransformationAddLocalVariable(uint32_t fresh_id, uint32_t type_id,
                                 uint32_t function_id, uint32_t initializer_id,
                                 bool value_is_irrelevant);

  // - |message_.fresh_id| must not be used by the module
  // - |message_.type_id| must be the id of a pointer type with Function
  //   storage class
  // - |message_.initializer_id| must be the id of a constant with the same
  //   type as the pointer's pointee type
  // - |message_.function_id| must be the id of a function
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an instruction to the start of |message_.function_id|, of the form:
  //   |message_.fresh_id| = OpVariable |message_.type_id| Function
  //                         |message_.initializer_id|
  // If |message_.value_is_irrelevant| holds, adds a corresponding fact to the
  // fact manager in |transformation_context|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddLocalVariable message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_LOCAL_VARIABLE_H_
