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

#ifndef SOURCE_FUZZ_FUZZER_PASS_ADD_FUNCTION_CALLS_H_
#define SOURCE_FUZZ_FUZZER_PASS_ADD_FUNCTION_CALLS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Fuzzer pass that adds calls at random to (a) livesafe functions, from
// anywhere, and (b) any functions, from dead blocks.
class FuzzerPassAddFunctionCalls : public FuzzerPass {
 public:
  FuzzerPassAddFunctionCalls(opt::IRContext* ir_context,
                             TransformationContext* transformation_context,
                             FuzzerContext* fuzzer_context,
                             protobufs::TransformationSequence* transformations,
                             bool ignore_inapplicable_transformations);

  void Apply() override;

 private:
  // Randomly chooses suitable arguments to invoke |callee| right before
  // instruction |caller_inst_it| of block |caller_block| in |caller_function|,
  // based on both existing available instructions and the addition of new
  // instructions to the module.
  std::vector<uint32_t> ChooseFunctionCallArguments(
      const opt::Function& callee, opt::Function* caller_function,
      opt::BasicBlock* caller_block,
      const opt::BasicBlock::iterator& caller_inst_it);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_ADD_FUNCTION_CALLS_H_
