// Copyright (c) 2021 Mostafa Ashraf
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SWAP_FUNCTION_VARIABLES_H_
#define SOURCE_FUZZ_TRANSFORMATION_SWAP_FUNCTION_VARIABLES_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// A transformation that swaps two variable declaration instructions that appear
// in the same function.
class TransformationSwapFunctionVariables : public Transformation {
 public:
  explicit TransformationSwapFunctionVariables(
      protobufs::TransformationSwapFunctionVariables message);

  TransformationSwapFunctionVariables(uint32_t result_id1, uint32_t result_id2);

  // - |message_.result_id1| and |message_.result_id2| must be the ids of
  //   distinct OpVariable instructions appearing in the same function.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Swaps two OpVariable instructions with result ids |message_.result_id1|
  // and |message_.result_id2|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

 private:
  protobufs::TransformationSwapFunctionVariables message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_SWAP_FUNCTION_VARIABLES_H_
