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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_GLOBAL_UNDEF_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_GLOBAL_UNDEF_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddGlobalUndef : public Transformation {
 public:
  explicit TransformationAddGlobalUndef(
      protobufs::TransformationAddGlobalUndef message);

  TransformationAddGlobalUndef(uint32_t fresh_id, uint32_t type_id);

  // - |message_.fresh_id| must be fresh
  // - |message_.type_id| must be the id of a non-function, non-pointer type
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an OpUndef instruction to the module, with |message_.type_id| as its
  // type.  The instruction has result id |message_.fresh_id|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddGlobalUndef message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_GLOBAL_UNDEF_H_
