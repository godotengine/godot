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

#include "source/fuzz/fuzzer_pass_add_loop_preheaders.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_add_loop_preheader.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddLoopPreheaders::FuzzerPassAddLoopPreheaders(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddLoopPreheaders::Apply() {
  for (auto& function : *GetIRContext()->module()) {
    // Keep track of all the loop headers we want to add a preheader to.
    std::vector<uint32_t> loop_header_ids_to_consider;
    for (auto& block : function) {
      // We only care about loop headers.
      if (!block.IsLoopHeader()) {
        continue;
      }

      // Randomly decide whether to consider this header.
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfAddingLoopPreheader())) {
        continue;
      }

      // We exclude loop headers with just one predecessor (the back-edge block)
      // because they are unreachable.
      if (GetIRContext()->cfg()->preds(block.id()).size() < 2) {
        continue;
      }

      loop_header_ids_to_consider.push_back(block.id());
    }

    for (uint32_t header_id : loop_header_ids_to_consider) {
      // If not already present, add a preheader which is not also a loop
      // header.
      GetOrCreateSimpleLoopPreheader(header_id);
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
