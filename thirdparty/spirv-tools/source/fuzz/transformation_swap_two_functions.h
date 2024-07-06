// Copyright (c) 2021 Shiyu Liu
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SWAP_TWO_FUNCTIONS_H_
#define SOURCE_FUZZ_TRANSFORMATION_SWAP_TWO_FUNCTIONS_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationSwapTwoFunctions : public Transformation {
 public:
  explicit TransformationSwapTwoFunctions(
      protobufs::TransformationSwapTwoFunctions message);

  TransformationSwapTwoFunctions(uint32_t function_id1, uint32_t function_id2);

  // |function_id1| and  |function_id1| should all be existing ids.
  //  Swap function operation is only permitted if:
  //  - both ids must be ids of functions.
  //  - both ids can be found in the module.
  //  - function_id1 and function_id2 are not the same.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // OpFunction with |function_id1| and |function_id1| are swapped.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;
  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationSwapTwoFunctions message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_SWAP_TWO_FUNCTIONS_H_
