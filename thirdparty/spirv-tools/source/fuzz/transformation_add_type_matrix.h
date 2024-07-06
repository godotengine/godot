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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_MATRIX_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_MATRIX_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddTypeMatrix : public Transformation {
 public:
  explicit TransformationAddTypeMatrix(
      protobufs::TransformationAddTypeMatrix message);

  TransformationAddTypeMatrix(uint32_t fresh_id, uint32_t column_type_id,
                              uint32_t column_count);

  // - |message_.fresh_id| must be a fresh id
  // - |message_.column_type_id| must be the id of a floating-point vector type
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an OpTypeMatrix instruction to the module, with column type
  // |message_.column_type_id| and |message_.column_count| columns, with result
  // id |message_.fresh_id|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddTypeMatrix message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_MATRIX_H_
