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

#ifndef SOURCE_FUZZ_INSTRUCTION_MESSAGE_H_
#define SOURCE_FUZZ_INSTRUCTION_MESSAGE_H_

#include <memory>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// Creates an Instruction protobuf message from its component parts.
protobufs::Instruction MakeInstructionMessage(
    spv::Op opcode, uint32_t result_type_id, uint32_t result_id,
    const opt::Instruction::OperandList& input_operands);

// Creates an Instruction protobuf message from a parsed instruction.
protobufs::Instruction MakeInstructionMessage(
    const opt::Instruction* instruction);

// Creates and returns an opt::Instruction from protobuf message
// |instruction_message|, relative to |ir_context|.  In the process, the module
// id bound associated with |ir_context| is updated to be at least as large as
// the result id (if any) associated with the new instruction.
std::unique_ptr<opt::Instruction> InstructionFromMessage(
    opt::IRContext* ir_context,
    const protobufs::Instruction& instruction_message);

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_INSTRUCTION_MESSAGE_H_
