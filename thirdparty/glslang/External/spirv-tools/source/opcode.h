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

#ifndef SOURCE_OPCODE_H_
#define SOURCE_OPCODE_H_

#include "source/instruction.h"
#include "source/latest_version_spirv_header.h"
#include "source/table.h"
#include "spirv-tools/libspirv.h"

// Returns the name of a registered SPIR-V generator as a null-terminated
// string. If the generator is not known, then returns the string "Unknown".
// The generator parameter should be most significant 16-bits of the generator
// word in the SPIR-V module header.
//
// See the registry at https://www.khronos.org/registry/spir-v/api/spir-v.xml.
const char* spvGeneratorStr(uint32_t generator);

// Combines word_count and opcode enumerant in single word.
uint32_t spvOpcodeMake(uint16_t word_count, spv::Op opcode);

// Splits word into into two constituent parts: word_count and opcode.
void spvOpcodeSplit(const uint32_t word, uint16_t* word_count,
                    uint16_t* opcode);

// Finds the named opcode in the given opcode table. On success, returns
// SPV_SUCCESS and writes a handle of the table entry into *entry.
spv_result_t spvOpcodeTableNameLookup(spv_target_env,
                                      const spv_opcode_table table,
                                      const char* name, spv_opcode_desc* entry);

// Finds the opcode by enumerant in the given opcode table. On success, returns
// SPV_SUCCESS and writes a handle of the table entry into *entry.
spv_result_t spvOpcodeTableValueLookup(spv_target_env,
                                       const spv_opcode_table table,
                                       const spv::Op opcode,
                                       spv_opcode_desc* entry);

// Copies an instruction's word and fixes the endianness to host native. The
// source instruction's stream/opcode/endianness is in the words/opcode/endian
// parameter. The word_count parameter specifies the number of words to copy.
// Writes copied instruction into *inst.
void spvInstructionCopy(const uint32_t* words, const spv::Op opcode,
                        const uint16_t word_count,
                        const spv_endianness_t endian, spv_instruction_t* inst);

// Determine if the given opcode is a scalar type. Returns zero if false,
// non-zero otherwise.
int32_t spvOpcodeIsScalarType(const spv::Op opcode);

// Determines if the given opcode is a specialization constant. Returns zero if
// false, non-zero otherwise.
int32_t spvOpcodeIsSpecConstant(const spv::Op opcode);

// Determines if the given opcode is a constant. Returns zero if false, non-zero
// otherwise.
int32_t spvOpcodeIsConstant(const spv::Op opcode);

// Returns true if the given opcode is a constant or undef.
bool spvOpcodeIsConstantOrUndef(const spv::Op opcode);

// Returns true if the given opcode is a scalar specialization constant.
bool spvOpcodeIsScalarSpecConstant(const spv::Op opcode);

// Determines if the given opcode is a composite type. Returns zero if false,
// non-zero otherwise.
int32_t spvOpcodeIsComposite(const spv::Op opcode);

// Determines if the given opcode results in a pointer when using the logical
// addressing model. Returns zero if false, non-zero otherwise.
int32_t spvOpcodeReturnsLogicalPointer(const spv::Op opcode);

// Returns whether the given opcode could result in a pointer or a variable
// pointer when using the logical addressing model.
bool spvOpcodeReturnsLogicalVariablePointer(const spv::Op opcode);

// Determines if the given opcode generates a type. Returns zero if false,
// non-zero otherwise.
int32_t spvOpcodeGeneratesType(spv::Op opcode);

// Returns true if the opcode adds a decoration to an id.
bool spvOpcodeIsDecoration(const spv::Op opcode);

// Returns true if the opcode is a load from memory into a result id.  This
// function only considers core instructions.
bool spvOpcodeIsLoad(const spv::Op opcode);

// Returns true if the opcode is an atomic operation that uses the original
// value.
bool spvOpcodeIsAtomicWithLoad(const spv::Op opcode);

// Returns true if the opcode is an atomic operation.
bool spvOpcodeIsAtomicOp(const spv::Op opcode);

// Returns true if the given opcode is a branch instruction.
bool spvOpcodeIsBranch(spv::Op opcode);

// Returns true if the given opcode is a return instruction.
bool spvOpcodeIsReturn(spv::Op opcode);

// Returns true if the given opcode aborts execution.  To abort means that after
// executing that instruction, no other instructions will be executed regardless
// of the context in which the instruction appears.  Note that `OpUnreachable`
// is considered an abort even if its behaviour is undefined.
bool spvOpcodeIsAbort(spv::Op opcode);

// Returns true if the given opcode is a return instruction or it aborts
// execution.
bool spvOpcodeIsReturnOrAbort(spv::Op opcode);

// Returns true if the given opcode is a basic block terminator.
bool spvOpcodeIsBlockTerminator(spv::Op opcode);

// Returns true if the given opcode always defines an opaque type.
bool spvOpcodeIsBaseOpaqueType(spv::Op opcode);

// Returns true if the given opcode is a non-uniform group operation.
bool spvOpcodeIsNonUniformGroupOperation(spv::Op opcode);

// Returns true if the opcode with vector inputs could be divided into a series
// of independent scalar operations that would give the same result.
bool spvOpcodeIsScalarizable(spv::Op opcode);

// Returns true if the given opcode is a debug instruction.
bool spvOpcodeIsDebug(spv::Op opcode);

// Returns true for opcodes that are binary operators,
// where the order of the operands is irrelevant.
bool spvOpcodeIsCommutativeBinaryOperator(spv::Op opcode);

// Returns true for opcodes that represent linear algebra instructions.
bool spvOpcodeIsLinearAlgebra(spv::Op opcode);

// Returns true for opcodes that represent image sample instructions.
bool spvOpcodeIsImageSample(spv::Op opcode);

// Returns a vector containing the indices of the memory semantics <id>
// operands for |opcode|.
std::vector<uint32_t> spvOpcodeMemorySemanticsOperandIndices(spv::Op opcode);

// Returns true for opcodes that represent access chain instructions.
bool spvOpcodeIsAccessChain(spv::Op opcode);

// Returns true for opcodes that represent bit instructions.
bool spvOpcodeIsBit(spv::Op opcode);

// Gets the name of an instruction, without the "Op" prefix.
const char* spvOpcodeString(const spv::Op opcode);

#endif  // SOURCE_OPCODE_H_
