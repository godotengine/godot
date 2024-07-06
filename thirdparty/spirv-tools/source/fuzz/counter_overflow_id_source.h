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

#ifndef SOURCE_FUZZ_COUNTER_OVERFLOW_ID_SOURCE_H_
#define SOURCE_FUZZ_COUNTER_OVERFLOW_ID_SOURCE_H_

#include "source/fuzz/overflow_id_source.h"

namespace spvtools {
namespace fuzz {

// A source of overflow ids that uses a counter to provide successive ids from
// a given starting value.
class CounterOverflowIdSource : public OverflowIdSource {
 public:
  // |first_available_id| is the starting value for the counter.
  explicit CounterOverflowIdSource(uint32_t first_available_id);

  // Always returns true.
  bool HasOverflowIds() const override;

  // Returns the current counter value and increments the counter.
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/2541) We should
  //  account for the case where the maximum allowed id is reached.
  uint32_t GetNextOverflowId() override;

  const std::unordered_set<uint32_t>& GetIssuedOverflowIds() const override;

 private:
  uint32_t next_available_id_;

  std::unordered_set<uint32_t> issued_ids_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_OVERFLOW_ID_SOURCE_COUNTER_H_
