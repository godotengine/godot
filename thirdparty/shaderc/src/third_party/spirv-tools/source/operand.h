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

#ifndef SOURCE_OPERAND_H_
#define SOURCE_OPERAND_H_

#include <functional>
#include <vector>

#include "source/table.h"
#include "spirv-tools/libspirv.h"

// A sequence of operand types.
//
// A SPIR-V parser uses an operand pattern to describe what is expected
// next on the input.
//
// As we parse an instruction in text or binary form from left to right,
// we pop and push at the end of the pattern vector. Symbols later in the
// pattern vector are matched against the input before symbols earlier in the
// pattern vector are matched.

// Using a vector in this way reduces memory traffic, which is good for
// performance.
using spv_operand_pattern_t = std::vector<spv_operand_type_t>;

// Finds the named operand in the table. The type parameter specifies the
// operand's group. A handle of the operand table entry for this operand will
// be written into *entry.
spv_result_t spvOperandTableNameLookup(spv_target_env,
                                       const spv_operand_table table,
                                       const spv_operand_type_t type,
                                       const char* name,
                                       const size_t name_length,
                                       spv_operand_desc* entry);

// Finds the operand with value in the table. The type parameter specifies the
// operand's group. A handle of the operand table entry for this operand will
// be written into *entry.
spv_result_t spvOperandTableValueLookup(spv_target_env,
                                        const spv_operand_table table,
                                        const spv_operand_type_t type,
                                        const uint32_t value,
                                        spv_operand_desc* entry);

// Gets the name string of the non-variable operand type.
const char* spvOperandTypeStr(spv_operand_type_t type);

// Returns true if the given type is concrete.
bool spvOperandIsConcrete(spv_operand_type_t type);

// Returns true if the given type is concrete and also a mask.
bool spvOperandIsConcreteMask(spv_operand_type_t type);

// Returns true if an operand of the given type is optional.
bool spvOperandIsOptional(spv_operand_type_t type);

// Returns true if an operand type represents zero or more logical operands.
//
// Note that a single logical operand may still be a variable number of words.
// For example, a literal string may be many words, but is just one logical
// operand.
bool spvOperandIsVariable(spv_operand_type_t type);

// Append a list of operand types to the end of the pattern vector.
// The types parameter specifies the source array of types, ending with
// SPV_OPERAND_TYPE_NONE.
void spvPushOperandTypes(const spv_operand_type_t* types,
                         spv_operand_pattern_t* pattern);

// Appends the operands expected after the given typed mask onto the
// end of the given pattern.
//
// Each set bit in the mask represents zero or more operand types that should
// be appended onto the pattern.  Operands for a less significant bit always
// appear after operands for a more significant bit.
//
// If a set bit is unknown, then we assume it has no operands.
void spvPushOperandTypesForMask(spv_target_env,
                                const spv_operand_table operand_table,
                                const spv_operand_type_t mask_type,
                                const uint32_t mask,
                                spv_operand_pattern_t* pattern);

// Expands an operand type representing zero or more logical operands,
// exactly once.
//
// If the given type represents potentially several logical operands,
// then prepend the given pattern with the first expansion of the logical
// operands, followed by original type.  Otherwise, don't modify the pattern.
//
// For example, the SPV_OPERAND_TYPE_VARIABLE_ID represents zero or more
// IDs.  In that case we would prepend the pattern with SPV_OPERAND_TYPE_ID
// followed by SPV_OPERAND_TYPE_VARIABLE_ID again.
//
// This also applies to zero or more tuples of logical operands.  In that case
// we prepend pattern with for the members of the tuple, followed by the
// original type argument.  The pattern must encode the fact that if any part
// of the tuple is present, then all tuple members should be.  So the first
// member of the tuple must be optional, and the remaining members
// non-optional.
//
// Returns true if we modified the pattern.
bool spvExpandOperandSequenceOnce(spv_operand_type_t type,
                                  spv_operand_pattern_t* pattern);

// Expands the first element in the pattern until it is a matchable operand
// type, then pops it off the front and returns it.  The pattern must not be
// empty.
//
// A matchable operand type is anything other than a zero-or-more-items
// operand type.
spv_operand_type_t spvTakeFirstMatchableOperand(spv_operand_pattern_t* pattern);

// Calculates the corresponding post-immediate alternate pattern, which allows
// a limited set of operand types.
spv_operand_pattern_t spvAlternatePatternFollowingImmediate(
    const spv_operand_pattern_t& pattern);

// Is the operand an ID?
bool spvIsIdType(spv_operand_type_t type);

// Is the operand an input ID?
bool spvIsInIdType(spv_operand_type_t type);

// Takes the opcode of an instruction and returns
// a function object that will return true if the index
// of the operand can be forward declared. This function will
// used in the SSA validation stage of the pipeline
std::function<bool(unsigned)> spvOperandCanBeForwardDeclaredFunction(
    SpvOp opcode);

#endif  // SOURCE_OPERAND_H_
