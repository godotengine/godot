// Copyright (c) 2017 Google Inc.
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

#ifndef SOURCE_ID_DESCRIPTOR_H_
#define SOURCE_ID_DESCRIPTOR_H_

#include <unordered_map>
#include <vector>

#include "spirv-tools/libspirv.hpp"

namespace spvtools {

using CustomHashFunc = std::function<uint32_t(const std::vector<uint32_t>&)>;

// Computes and stores id descriptors.
//
// Descriptors are computed as hash of all words in the instruction where ids
// were substituted with previously computed descriptors.
class IdDescriptorCollection {
 public:
  explicit IdDescriptorCollection(
      CustomHashFunc custom_hash_func = CustomHashFunc())
      : custom_hash_func_(custom_hash_func) {
    words_.reserve(16);
  }

  // Computes descriptor for the result id of the given instruction and
  // registers it in id_to_descriptor_. Returns the computed descriptor.
  // This function needs to be sequentially called for every instruction in the
  // module.
  uint32_t ProcessInstruction(const spv_parsed_instruction_t& inst);

  // Returns a previously computed descriptor id.
  uint32_t GetDescriptor(uint32_t id) const {
    const auto it = id_to_descriptor_.find(id);
    if (it == id_to_descriptor_.end()) return 0;
    return it->second;
  }

 private:
  std::unordered_map<uint32_t, uint32_t> id_to_descriptor_;

  std::function<uint32_t(const std::vector<uint32_t>&)> custom_hash_func_;

  // Scratch buffer used for hashing. Class member to optimize on allocation.
  std::vector<uint32_t> words_;
};

}  // namespace spvtools

#endif  // SOURCE_ID_DESCRIPTOR_H_
