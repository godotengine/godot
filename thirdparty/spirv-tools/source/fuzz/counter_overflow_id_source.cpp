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

#include "source/fuzz/counter_overflow_id_source.h"

namespace spvtools {
namespace fuzz {

CounterOverflowIdSource::CounterOverflowIdSource(uint32_t first_available_id)
    : next_available_id_(first_available_id), issued_ids_() {}

bool CounterOverflowIdSource::HasOverflowIds() const { return true; }

uint32_t CounterOverflowIdSource::GetNextOverflowId() {
  issued_ids_.insert(next_available_id_);
  return next_available_id_++;
}

const std::unordered_set<uint32_t>&
CounterOverflowIdSource::GetIssuedOverflowIds() const {
  return issued_ids_;
}

}  // namespace fuzz
}  // namespace spvtools
