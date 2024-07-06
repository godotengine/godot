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

#include "source/fuzz/uniform_buffer_element_descriptor.h"

#include <algorithm>

namespace spvtools {
namespace fuzz {

protobufs::UniformBufferElementDescriptor MakeUniformBufferElementDescriptor(
    uint32_t descriptor_set, uint32_t binding,
    std::vector<uint32_t>&& indices) {
  protobufs::UniformBufferElementDescriptor result;
  result.set_descriptor_set(descriptor_set);
  result.set_binding(binding);
  for (auto index : indices) {
    result.add_index(index);
  }
  return result;
}

bool UniformBufferElementDescriptorEquals::operator()(
    const protobufs::UniformBufferElementDescriptor* first,
    const protobufs::UniformBufferElementDescriptor* second) const {
  return first->descriptor_set() == second->descriptor_set() &&
         first->binding() == second->binding() &&
         first->index().size() == second->index().size() &&
         std::equal(first->index().begin(), first->index().end(),
                    second->index().begin());
}

opt::Instruction* FindUniformVariable(
    const protobufs::UniformBufferElementDescriptor&
        uniform_buffer_element_descriptor,
    opt::IRContext* context, bool check_unique) {
  opt::Instruction* result = nullptr;

  for (auto& inst : context->types_values()) {
    // Consider all global variables with uniform storage class.
    if (inst.opcode() != spv::Op::OpVariable) {
      continue;
    }
    if (spv::StorageClass(inst.GetSingleWordInOperand(0)) !=
        spv::StorageClass::Uniform) {
      continue;
    }

    // Determine whether the variable is decorated with a descriptor set
    // matching that in |uniform_buffer_element|.
    bool descriptor_set_matches = false;
    context->get_decoration_mgr()->ForEachDecoration(
        inst.result_id(), uint32_t(spv::Decoration::DescriptorSet),
        [&descriptor_set_matches, &uniform_buffer_element_descriptor](
            const opt::Instruction& decoration_inst) {
          const uint32_t kDescriptorSetOperandIndex = 2;
          if (decoration_inst.GetSingleWordInOperand(
                  kDescriptorSetOperandIndex) ==
              uniform_buffer_element_descriptor.descriptor_set()) {
            descriptor_set_matches = true;
          }
        });
    if (!descriptor_set_matches) {
      // Descriptor set does not match.
      continue;
    }

    // Determine whether the variable is decorated with a binding matching that
    // in |uniform_buffer_element|.
    bool binding_matches = false;
    context->get_decoration_mgr()->ForEachDecoration(
        inst.result_id(), uint32_t(spv::Decoration::Binding),
        [&binding_matches, &uniform_buffer_element_descriptor](
            const opt::Instruction& decoration_inst) {
          const uint32_t kBindingOperandIndex = 2;
          if (decoration_inst.GetSingleWordInOperand(kBindingOperandIndex) ==
              uniform_buffer_element_descriptor.binding()) {
            binding_matches = true;
          }
        });
    if (!binding_matches) {
      // Binding does not match.
      continue;
    }

    // This instruction is a uniform variable with the right descriptor set and
    // binding.
    if (!check_unique) {
      // If we aren't checking uniqueness, return it.
      return &inst;
    }

    if (result) {
      // More than one uniform variable is decorated with the given descriptor
      // set and binding. This means the fact is ambiguous.
      return nullptr;
    }
    result = &inst;
  }

  // We get here either if no match was found, or if |check_unique| holds and
  // exactly one match was found.
  assert(result == nullptr || check_unique);
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
