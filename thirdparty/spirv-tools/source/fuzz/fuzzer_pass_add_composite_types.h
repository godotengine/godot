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

#ifndef SOURCE_FUZZ_FUZZER_PASS_ADD_COMPOSITE_TYPES_H_
#define SOURCE_FUZZ_FUZZER_PASS_ADD_COMPOSITE_TYPES_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Fuzzer pass that randomly adds missing vector and matrix types, and new
// array and struct types, to the module.
class FuzzerPassAddCompositeTypes : public FuzzerPass {
 public:
  FuzzerPassAddCompositeTypes(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;

 private:
  // Creates an array of a random size with a random existing base type and adds
  // it to the module.
  void AddNewArrayType();

  // Creates a struct with fields of random existing types and adds it to the
  // module.
  void AddNewStructType();

  // For each vector type not already present in the module, randomly decides
  // whether to add it to the module.
  void MaybeAddMissingVectorTypes();

  // For each matrix type not already present in the module, randomly decides
  // whether to add it to the module.
  void MaybeAddMissingMatrixTypes();

  // Returns the id of a scalar or composite type declared in the module,
  // chosen randomly.
  uint32_t ChooseScalarOrCompositeType();
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_ADD_COMPOSITE_TYPES_H_
