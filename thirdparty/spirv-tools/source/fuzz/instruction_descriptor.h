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

#ifndef SOURCE_FUZZ_INSTRUCTION_DESCRIPTOR_H_
#define SOURCE_FUZZ_INSTRUCTION_DESCRIPTOR_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/basic_block.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// Looks for an instruction in |context| corresponding to |descriptor|.
// Returns |nullptr| if no such instruction can be found.
opt::Instruction* FindInstruction(
    const protobufs::InstructionDescriptor& instruction_descriptor,
    opt::IRContext* context);

// Creates an InstructionDescriptor protobuf message from the given
// components.  See the protobuf definition for details of what these
// components mean.
protobufs::InstructionDescriptor MakeInstructionDescriptor(
    uint32_t base_instruction_result_id, spv::Op target_instruction_opcode,
    uint32_t num_opcodes_to_ignore);

// Returns an instruction descriptor that describing the instruction at
// |inst_it|, which must be inside |block|.  The descriptor will be with
// respect to the first instruction at or before |inst_it| that has a result
// id.
protobufs::InstructionDescriptor MakeInstructionDescriptor(
    const opt::BasicBlock& block,
    const opt::BasicBlock::const_iterator& inst_it);

// Returns an InstructionDescriptor that describes the given instruction |inst|.
protobufs::InstructionDescriptor MakeInstructionDescriptor(
    opt::IRContext* context, opt::Instruction* inst);

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_INSTRUCTION_DESCRIPTOR_H_
