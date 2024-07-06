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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_STRUCT_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_STRUCT_H_

#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddTypeStruct : public Transformation {
 public:
  explicit TransformationAddTypeStruct(
      protobufs::TransformationAddTypeStruct message);

  TransformationAddTypeStruct(uint32_t fresh_id,
                              const std::vector<uint32_t>& component_type_ids);

  // - |message_.fresh_id| must be a fresh id
  // - |message_.member_type_id| must be a sequence of non-function type ids
  // - |message_.member_type_id| may not contain a result id of an OpTypeStruct
  //   instruction with BuiltIn members (i.e. members of the struct are
  //   decorated via OpMemberDecorate with BuiltIn decoration)
  // - |message_.member_type_id| may not contain a result id of an OpTypeStruct
  //   instruction that has the Block or BufferBlock decoration
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an OpTypeStruct instruction whose field types are given by
  // |message_.member_type_id|, with result id |message_.fresh_id|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddTypeStruct message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_STRUCT_H_
