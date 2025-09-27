// Copyright (c) 2018 Google LLC.
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

// Validates correctness of scopes for SPIR-V instructions.

#include "source/opcode.h"
#include "source/val/validate.h"

namespace spvtools {
namespace val {

spv_result_t ValidateScope(ValidationState_t& _, const Instruction* inst,
                           uint32_t scope);

spv_result_t ValidateExecutionScope(ValidationState_t& _,
                                    const Instruction* inst, uint32_t scope);

spv_result_t ValidateMemoryScope(ValidationState_t& _, const Instruction* inst,
                                 uint32_t scope);

}  // namespace val
}  // namespace spvtools
