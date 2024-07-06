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

#ifndef SOURCE_FORCE_RENDER_RED_H_
#define SOURCE_FORCE_RENDER_RED_H_

#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace fuzz {

// Requires |binary_in| to be a valid SPIR-V module with Shader capability,
// containing an entry point with the Fragment execution model, and a single
// output variable of type vec4.
//
// Turns the body of this entry point into effectively:
//
// output_variable = vec4(1.0, 0.0, 0.0, 1.0);
// if (false) {
//    original_body
// }
//
// If suitable facts about values of uniforms are available, the 'false' will
// instead become: 'u > v', where 'u' and 'v' are pieces of uniform data for
// which it is known that 'u < v' holds.
bool ForceRenderRed(
    const spv_target_env& target_env, spv_validator_options validator_options,
    const std::vector<uint32_t>& binary_in,
    const spvtools::fuzz::protobufs::FactSequence& initial_facts,
    const MessageConsumer& message_consumer, std::vector<uint32_t>* binary_out);

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FORCE_RENDER_RED_H_
