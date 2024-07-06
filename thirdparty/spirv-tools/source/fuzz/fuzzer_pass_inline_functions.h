// Copyright (c) 2020 André Perez Maselco
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

#ifndef SOURCE_FUZZ_FUZZER_PASS_INLINE_FUNCTIONS_H_
#define SOURCE_FUZZ_FUZZER_PASS_INLINE_FUNCTIONS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Looks for OpFunctionCall instructions and randomly decides which ones to
// inline. If the instructions of the called function are going to be inlined,
// then a mapping, between their result ids and suitable ids, is done.
class FuzzerPassInlineFunctions : public FuzzerPass {
 public:
  FuzzerPassInlineFunctions(opt::IRContext* ir_context,
                            TransformationContext* transformation_context,
                            FuzzerContext* fuzzer_context,
                            protobufs::TransformationSequence* transformations,
                            bool ignore_inapplicable_transformations);

  void Apply() override;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_INLINE_FUNCTIONS_H_
