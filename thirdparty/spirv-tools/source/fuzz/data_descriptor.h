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

#ifndef SOURCE_FUZZ_DATA_DESCRIPTOR_H_
#define SOURCE_FUZZ_DATA_DESCRIPTOR_H_

#include <ostream>
#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"

namespace spvtools {
namespace fuzz {

// Factory method to create a data descriptor message from an object id and a
// list of indices.
protobufs::DataDescriptor MakeDataDescriptor(
    uint32_t object, const std::vector<uint32_t>& indices);

// Hash function for data descriptors.
struct DataDescriptorHash {
  size_t operator()(const protobufs::DataDescriptor* data_descriptor) const;
};

// Equality function for data descriptors.
struct DataDescriptorEquals {
  bool operator()(const protobufs::DataDescriptor* first,
                  const protobufs::DataDescriptor* second) const;
};

std::ostream& operator<<(std::ostream& out,
                         const protobufs::DataDescriptor& data_descriptor);

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_DATA_DESCRIPTOR_H_
