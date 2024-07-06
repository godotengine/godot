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

#include "source/fuzz/transformation_context.h"

#include <cassert>

#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {
namespace {

// An overflow id source that should never be used: its methods assert false.
// This is the right id source for use during fuzzing, when overflow ids should
// never be required.
class NullOverflowIdSource : public OverflowIdSource {
  bool HasOverflowIds() const override {
    assert(false && "Bad attempt to query whether overflow ids are available.");
    return false;
  }

  uint32_t GetNextOverflowId() override {
    assert(false && "Bad attempt to request an overflow id.");
    return 0;
  }

  const std::unordered_set<uint32_t>& GetIssuedOverflowIds() const override {
    assert(false && "Operation not supported.");
    return placeholder_;
  }

 private:
  std::unordered_set<uint32_t> placeholder_;
};

}  // namespace

TransformationContext::TransformationContext(
    std::unique_ptr<FactManager> fact_manager,
    spv_validator_options validator_options)
    : fact_manager_(std::move(fact_manager)),
      validator_options_(validator_options),
      overflow_id_source_(MakeUnique<NullOverflowIdSource>()) {}

TransformationContext::TransformationContext(
    std::unique_ptr<FactManager> fact_manager,
    spv_validator_options validator_options,
    std::unique_ptr<OverflowIdSource> overflow_id_source)
    : fact_manager_(std::move(fact_manager)),
      validator_options_(validator_options),
      overflow_id_source_(std::move(overflow_id_source)) {}

TransformationContext::~TransformationContext() = default;

}  // namespace fuzz
}  // namespace spvtools
