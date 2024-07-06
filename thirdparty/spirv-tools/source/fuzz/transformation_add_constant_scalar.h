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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_CONSTANT_SCALAR_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_CONSTANT_SCALAR_H_

#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddConstantScalar : public Transformation {
 public:
  explicit TransformationAddConstantScalar(
      protobufs::TransformationAddConstantScalar message);

  TransformationAddConstantScalar(uint32_t fresh_id, uint32_t type_id,
                                  const std::vector<uint32_t>& words,
                                  bool is_irrelevant);

  // - |message_.fresh_id| must not be used by the module
  // - |message_.type_id| must be the id of a floating-point or integer type
  // - The size of |message_.word| must be compatible with the width of this
  //   type
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds a new OpConstant instruction with the given type and words.
  // Creates an IdIsIrrelevant fact about |fresh_id| if |is_irrelevant| is true.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddConstantScalar message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_CONSTANT_SCALAR_H_
