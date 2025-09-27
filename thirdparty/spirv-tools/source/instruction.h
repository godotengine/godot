// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#ifndef SOURCE_INSTRUCTION_H_
#define SOURCE_INSTRUCTION_H_

#include <cstdint>
#include <vector>

#include "source/latest_version_spirv_header.h"
#include "spirv-tools/libspirv.h"

// Describes an instruction.
struct spv_instruction_t {
  // Normally, both opcode and extInstType contain valid data.
  // However, when the assembler parses !<number> as the first word in
  // an instruction and opcode and extInstType are invalid.
  spv::Op opcode;
  spv_ext_inst_type_t extInstType;

  // The Id of the result type, if this instruction has one.  Zero otherwise.
  uint32_t resultTypeId;

  // The instruction, as a sequence of 32-bit words.
  // For a regular instruction the opcode and word count are combined
  // in words[0], as described in the SPIR-V spec.
  // Otherwise, the first token was !<number>, and that number appears
  // in words[0].  Subsequent elements are the result of parsing
  // tokens in the alternate parsing mode as described in syntax.md.
  std::vector<uint32_t> words;
};

// Appends a word to an instruction, without checking for overflow.
inline void spvInstructionAddWord(spv_instruction_t* inst, uint32_t value) {
  inst->words.push_back(value);
}

#endif  // SOURCE_INSTRUCTION_H_
