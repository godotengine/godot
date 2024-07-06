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

#ifndef SOURCE_FUZZ_UNIFORM_BUFFER_ELEMENT_DESCRIPTOR_H_
#define SOURCE_FUZZ_UNIFORM_BUFFER_ELEMENT_DESCRIPTOR_H_

#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// Factory method to create a uniform buffer element descriptor message from
// descriptor set and binding ids and a list of indices.
protobufs::UniformBufferElementDescriptor MakeUniformBufferElementDescriptor(
    uint32_t descriptor_set, uint32_t binding, std::vector<uint32_t>&& indices);

// Equality function for uniform buffer element descriptors.
struct UniformBufferElementDescriptorEquals {
  bool operator()(
      const protobufs::UniformBufferElementDescriptor* first,
      const protobufs::UniformBufferElementDescriptor* second) const;
};

// Returns a pointer to an OpVariable in |context| that is decorated with the
// descriptor set and binding associated with |uniform_buffer_element|.  Returns
// nullptr if no such variable exists.  If multiple such variables exist, a
// pointer to an arbitrary one of the associated instructions is returned if
// |check_unique| is false, and nullptr is returned if |check_unique| is true.
opt::Instruction* FindUniformVariable(
    const protobufs::UniformBufferElementDescriptor&
        uniform_buffer_element_descriptor,
    opt::IRContext* context, bool check_unique);

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_UNIFORM_BUFFER_ELEMENT_DESCRIPTOR_H_
