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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_FUNCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_FUNCTION_H_

#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddTypeFunction : public Transformation {
 public:
  explicit TransformationAddTypeFunction(
      protobufs::TransformationAddTypeFunction message);

  TransformationAddTypeFunction(uint32_t fresh_id, uint32_t return_type_id,
                                const std::vector<uint32_t>& argument_type_ids);

  // - |message_.fresh_id| must not be used by the module
  // - |message_.return_type_id| and each element of |message_.argument_type_id|
  //   must be the ids of non-function types
  // - The module must not contain an OpTypeFunction instruction defining a
  //   function type with the signature provided by the given return and
  //   argument types
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an OpTypeFunction instruction to the module, with signature given by
  // |message_.return_type_id| and |message_.argument_type_id|.  The result id
  // for the instruction is |message_.fresh_id|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddTypeFunction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_FUNCTION_H_
