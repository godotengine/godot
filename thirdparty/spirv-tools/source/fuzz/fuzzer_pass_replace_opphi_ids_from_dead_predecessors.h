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

#ifndef SOURCE_FUZZ_FUZZER_PASS_REPLACE_OPPHI_IDS_FROM_DEAD_PREDECESSORS_H_
#define SOURCE_FUZZ_FUZZER_PASS_REPLACE_OPPHI_IDS_FROM_DEAD_PREDECESSORS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Replaces id operands in OpPhi instructions with other available ids of the
// right type, where the corresponding predecessor is dead.
class FuzzerPassReplaceOpPhiIdsFromDeadPredecessors : public FuzzerPass {
 public:
  FuzzerPassReplaceOpPhiIdsFromDeadPredecessors(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;
};
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_REPLACE_OPPHI_IDS_FROM_DEAD_PREDECESSORS_H_
