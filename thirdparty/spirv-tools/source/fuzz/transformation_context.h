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

#ifndef SOURCE_FUZZ_TRANSFORMATION_CONTEXT_H_
#define SOURCE_FUZZ_TRANSFORMATION_CONTEXT_H_

#include <memory>

#include "source/fuzz/fact_manager/fact_manager.h"
#include "source/fuzz/overflow_id_source.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace fuzz {

// Encapsulates all information that is required to inform how to apply a
// transformation to a module.
class TransformationContext {
 public:
  // Constructs a transformation context with a given fact manager and validator
  // options.  Overflow ids are not available from a transformation context
  // constructed in this way.
  TransformationContext(std::unique_ptr<FactManager>,
                        spv_validator_options validator_options);

  // Constructs a transformation context with a given fact manager, validator
  // options and overflow id source.
  TransformationContext(std::unique_ptr<FactManager>,
                        spv_validator_options validator_options,
                        std::unique_ptr<OverflowIdSource> overflow_id_source);

  ~TransformationContext();

  FactManager* GetFactManager() { return fact_manager_.get(); }

  const FactManager* GetFactManager() const { return fact_manager_.get(); }

  OverflowIdSource* GetOverflowIdSource() { return overflow_id_source_.get(); }

  const OverflowIdSource* GetOverflowIdSource() const {
    return overflow_id_source_.get();
  }

  spv_validator_options GetValidatorOptions() const {
    return validator_options_;
  }

 private:
  // Manages facts that inform whether transformations can be applied, and that
  // are produced by applying transformations.
  std::unique_ptr<FactManager> fact_manager_;

  // Options to control validation when deciding whether transformations can be
  // applied.
  spv_validator_options validator_options_;

  std::unique_ptr<OverflowIdSource> overflow_id_source_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_CONTEXT_H_
