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

#ifndef SOURCE_FUZZ_ID_USE_DESCRIPTOR_H_
#define SOURCE_FUZZ_ID_USE_DESCRIPTOR_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// Looks for an instruction in |context| that contains a use
// identified by |id_use_descriptor|.
// Returns |nullptr| if no such instruction can be found.
opt::Instruction* FindInstructionContainingUse(
    const protobufs::IdUseDescriptor& id_use_descriptor,
    opt::IRContext* context);

// Creates an IdUseDescriptor protobuf message from the given components.
// See the protobuf definition for details of what these components mean.
protobufs::IdUseDescriptor MakeIdUseDescriptor(
    uint32_t id_of_interest,
    const protobufs::InstructionDescriptor& enclosing_instruction,
    uint32_t in_operand_index);

// Given an id use, represented by the instruction |inst| that uses the id, and
// the input operand index |in_operand_index| associated with the usage, returns
// an IdUseDescriptor that represents the use.
protobufs::IdUseDescriptor MakeIdUseDescriptorFromUse(
    opt::IRContext* context, opt::Instruction* inst, uint32_t in_operand_index);

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_ID_USE_DESCRIPTOR_H_
