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

#include "source/fuzz/data_descriptor.h"

#include <algorithm>

namespace spvtools {
namespace fuzz {

protobufs::DataDescriptor MakeDataDescriptor(
    uint32_t object, const std::vector<uint32_t>& indices) {
  protobufs::DataDescriptor result;
  result.set_object(object);
  for (auto index : indices) {
    result.add_index(index);
  }
  return result;
}

size_t DataDescriptorHash::operator()(
    const protobufs::DataDescriptor* data_descriptor) const {
  std::u32string hash;
  hash.push_back(data_descriptor->object());
  for (auto an_index : data_descriptor->index()) {
    hash.push_back(an_index);
  }
  return std::hash<std::u32string>()(hash);
}

bool DataDescriptorEquals::operator()(
    const protobufs::DataDescriptor* first,
    const protobufs::DataDescriptor* second) const {
  return first->object() == second->object() &&
         first->index().size() == second->index().size() &&
         std::equal(first->index().begin(), first->index().end(),
                    second->index().begin());
}

std::ostream& operator<<(std::ostream& out,
                         const protobufs::DataDescriptor& data_descriptor) {
  out << data_descriptor.object();
  out << "[";
  bool first = true;
  for (auto index : data_descriptor.index()) {
    if (first) {
      first = false;
    } else {
      out << ", ";
    }
    out << index;
  }
  out << "]";
  return out;
}

}  // namespace fuzz
}  // namespace spvtools
