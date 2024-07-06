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

#ifndef SOURCE_FUZZ_REPEATED_PASS_MANAGER_SIMPLE_H_
#define SOURCE_FUZZ_REPEATED_PASS_MANAGER_SIMPLE_H_

#include "source/fuzz/pass_management/repeated_pass_manager.h"

namespace spvtools {
namespace fuzz {

// Selects the next pass to run uniformly at random from the enabled repeated
// passes.  Recommendations are not used.
class RepeatedPassManagerSimple : public RepeatedPassManager {
 public:
  RepeatedPassManagerSimple(FuzzerContext* fuzzer_context,
                            RepeatedPassInstances* pass_instances);

  ~RepeatedPassManagerSimple() override;

  FuzzerPass* ChoosePass(const protobufs::TransformationSequence&
                             applied_transformations) override;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_REPEATED_PASS_MANAGER_SIMPLE_H_
