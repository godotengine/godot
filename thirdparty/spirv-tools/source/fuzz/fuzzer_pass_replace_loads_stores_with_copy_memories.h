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

#ifndef SOURCE_FUZZ_FUZZER_PASS_REPLACE_LOADS_STORES_WITH_COPY_MEMORIES_H_
#define SOURCE_FUZZ_FUZZER_PASS_REPLACE_LOADS_STORES_WITH_COPY_MEMORIES_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass that takes pairs of instruction descriptors to OpLoad and
// OpStore that have the same intermediate value and in each pair replaces the
// OpStore with an equivalent OpCopyMemory.
class FuzzerPassReplaceLoadsStoresWithCopyMemories : public FuzzerPass {
 public:
  FuzzerPassReplaceLoadsStoresWithCopyMemories(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_REPLACE_LOADS_STORES_WITH_COPY_MEMORIES_H_
