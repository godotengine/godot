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

#ifndef SOURCE_FUZZ_FUZZER_PASS_ADD_COMPOSITE_INSERTS_H_
#define SOURCE_FUZZ_FUZZER_PASS_ADD_COMPOSITE_INSERTS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Fuzzer pass that randomly adds new OpCompositeInsert instructions to
// available values that have the composite type.
class FuzzerPassAddCompositeInserts : public FuzzerPass {
 public:
  FuzzerPassAddCompositeInserts(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;

  // Checks if any component of a composite is a pointer.
  static bool ContainsPointer(const opt::analysis::Type& type);

  // Checks if any component of a composite has type OpTypeRuntimeArray.
  static bool ContainsRuntimeArray(const opt::analysis::Type& type);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_ADD_COMPOSITE_INSERTS_H_
