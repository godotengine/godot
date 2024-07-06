// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_FUZZ_FUZZER_PASS_APPLY_ID_SYNONYMS_H_
#define SOURCE_FUZZ_FUZZER_PASS_APPLY_ID_SYNONYMS_H_

#include "source/fuzz/fuzzer_pass.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// A pass that replaces ids with other ids, or accesses into structures, that
// are known to hold the same values.
class FuzzerPassApplyIdSynonyms : public FuzzerPass {
 public:
  FuzzerPassApplyIdSynonyms(opt::IRContext* ir_context,
                            TransformationContext* transformation_context,
                            FuzzerContext* fuzzer_context,
                            protobufs::TransformationSequence* transformations,
                            bool ignore_inapplicable_transformations);

  void Apply() override;

 private:
  // Returns true if uses of |dd1| can be replaced with |dd2| and vice-versa
  // with respect to the type. Concretely, returns true if |dd1| and |dd2| have
  // the same type or both |dd1| and |dd2| are either a numerical or a vector
  // type of integral components with possibly different signedness.
  bool DataDescriptorsHaveCompatibleTypes(spv::Op opcode,
                                          uint32_t use_in_operand_index,
                                          const protobufs::DataDescriptor& dd1,
                                          const protobufs::DataDescriptor& dd2);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_APPLY_ID_SYNONYMS_H_
