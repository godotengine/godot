// Copyright (c) 2015-2020 The Khronos Group Inc.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights
// reserved.
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

#ifndef INCLUDE_SPIRV_TOOLS_LIBSPIRV_H_
#define INCLUDE_SPIRV_TOOLS_LIBSPIRV_H_

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

#include <stddef.h>
#include <stdint.h>

#if defined(SPIRV_TOOLS_SHAREDLIB)
#if defined(_WIN32)
#if defined(SPIRV_TOOLS_IMPLEMENTATION)
#define SPIRV_TOOLS_EXPORT __declspec(dllexport)
#else
#define SPIRV_TOOLS_EXPORT __declspec(dllimport)
#endif
#define SPIRV_TOOLS_LOCAL
#else
#if defined(SPIRV_TOOLS_IMPLEMENTATION)
#define SPIRV_TOOLS_EXPORT __attribute__((visibility("default")))
#define SPIRV_TOOLS_LOCAL __attribute__((visibility("hidden")))
#else
#define SPIRV_TOOLS_EXPORT
#define SPIRV_TOOLS_LOCAL
#endif
#endif
#else
#define SPIRV_TOOLS_EXPORT
#define SPIRV_TOOLS_LOCAL
#endif

// Helpers

#define SPV_BIT(shift) (1 << (shift))

#define SPV_FORCE_16_BIT_ENUM(name) SPV_FORCE_16BIT_##name = 0x7fff
#define SPV_FORCE_32_BIT_ENUM(name) SPV_FORCE_32BIT_##name = 0x7fffffff

// Enumerations

typedef enum spv_result_t {
  SPV_SUCCESS = 0,
  SPV_UNSUPPORTED = 1,
  SPV_END_OF_STREAM = 2,
  SPV_WARNING = 3,
  SPV_FAILED_MATCH = 4,
  SPV_REQUESTED_TERMINATION = 5,  // Success, but signals early termination.
  SPV_ERROR_INTERNAL = -1,
  SPV_ERROR_OUT_OF_MEMORY = -2,
  SPV_ERROR_INVALID_POINTER = -3,
  SPV_ERROR_INVALID_BINARY = -4,
  SPV_ERROR_INVALID_TEXT = -5,
  SPV_ERROR_INVALID_TABLE = -6,
  SPV_ERROR_INVALID_VALUE = -7,
  SPV_ERROR_INVALID_DIAGNOSTIC = -8,
  SPV_ERROR_INVALID_LOOKUP = -9,
  SPV_ERROR_INVALID_ID = -10,
  SPV_ERROR_INVALID_CFG = -11,
  SPV_ERROR_INVALID_LAYOUT = -12,
  SPV_ERROR_INVALID_CAPABILITY = -13,
  SPV_ERROR_INVALID_DATA = -14,  // Indicates data rules validation failure.
  SPV_ERROR_MISSING_EXTENSION = -15,
  SPV_ERROR_WRONG_VERSION = -16,  // Indicates wrong SPIR-V version
  SPV_FORCE_32_BIT_ENUM(spv_result_t)
} spv_result_t;

// Severity levels of messages communicated to the consumer.
typedef enum spv_message_level_t {
  SPV_MSG_FATAL,           // Unrecoverable error due to environment.
                           // Will exit the program immediately. E.g.,
                           // out of memory.
  SPV_MSG_INTERNAL_ERROR,  // Unrecoverable error due to SPIRV-Tools
                           // internals.
                           // Will exit the program immediately. E.g.,
                           // unimplemented feature.
  SPV_MSG_ERROR,           // Normal error due to user input.
  SPV_MSG_WARNING,         // Warning information.
  SPV_MSG_INFO,            // General information.
  SPV_MSG_DEBUG,           // Debug information.
} spv_message_level_t;

typedef enum spv_endianness_t {
  SPV_ENDIANNESS_LITTLE,
  SPV_ENDIANNESS_BIG,
  SPV_FORCE_32_BIT_ENUM(spv_endianness_t)
} spv_endianness_t;

// The kinds of operands that an instruction may have.
//
// Some operand types are "concrete".  The binary parser uses a concrete
// operand type to describe an operand of a parsed instruction.
//
// The assembler uses all operand types.  In addition to determining what
// kind of value an operand may be, non-concrete operand types capture the
// fact that an operand might be optional (may be absent, or present exactly
// once), or might occur zero or more times.
//
// Sometimes we also need to be able to express the fact that an operand
// is a member of an optional tuple of values.  In that case the first member
// would be optional, and the subsequent members would be required.
//
// NOTE: Although we don't promise binary compatibility, as a courtesy, please
// add new enum values at the end.
typedef enum spv_operand_type_t {
  // A sentinel value.
  SPV_OPERAND_TYPE_NONE = 0,

  // Set 1:  Operands that are IDs.
  SPV_OPERAND_TYPE_ID,
  SPV_OPERAND_TYPE_TYPE_ID,
  SPV_OPERAND_TYPE_RESULT_ID,
  SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID,  // SPIR-V Sec 3.25
  SPV_OPERAND_TYPE_SCOPE_ID,             // SPIR-V Sec 3.27

  // Set 2:  Operands that are literal numbers.
  SPV_OPERAND_TYPE_LITERAL_INTEGER,  // Always unsigned 32-bits.
  // The Instruction argument to OpExtInst. It's an unsigned 32-bit literal
  // number indicating which instruction to use from an extended instruction
  // set.
  SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
  // The Opcode argument to OpSpecConstantOp. It determines the operation
  // to be performed on constant operands to compute a specialization constant
  // result.
  SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER,
  // A literal number whose format and size are determined by a previous operand
  // in the same instruction.  It's a signed integer, an unsigned integer, or a
  // floating point number.  It also has a specified bit width.  The width
  // may be larger than 32, which would require such a typed literal value to
  // occupy multiple SPIR-V words.
  SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER,
  SPV_OPERAND_TYPE_LITERAL_FLOAT,  // Always 32-bit float.

  // Set 3:  The literal string operand type.
  SPV_OPERAND_TYPE_LITERAL_STRING,

  // Set 4:  Operands that are a single word enumerated value.
  SPV_OPERAND_TYPE_SOURCE_LANGUAGE,               // SPIR-V Sec 3.2
  SPV_OPERAND_TYPE_EXECUTION_MODEL,               // SPIR-V Sec 3.3
  SPV_OPERAND_TYPE_ADDRESSING_MODEL,              // SPIR-V Sec 3.4
  SPV_OPERAND_TYPE_MEMORY_MODEL,                  // SPIR-V Sec 3.5
  SPV_OPERAND_TYPE_EXECUTION_MODE,                // SPIR-V Sec 3.6
  SPV_OPERAND_TYPE_STORAGE_CLASS,                 // SPIR-V Sec 3.7
  SPV_OPERAND_TYPE_DIMENSIONALITY,                // SPIR-V Sec 3.8
  SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE,       // SPIR-V Sec 3.9
  SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE,           // SPIR-V Sec 3.10
  SPV_OPERAND_TYPE_SAMPLER_IMAGE_FORMAT,          // SPIR-V Sec 3.11
  SPV_OPERAND_TYPE_IMAGE_CHANNEL_ORDER,           // SPIR-V Sec 3.12
  SPV_OPERAND_TYPE_IMAGE_CHANNEL_DATA_TYPE,       // SPIR-V Sec 3.13
  SPV_OPERAND_TYPE_FP_ROUNDING_MODE,              // SPIR-V Sec 3.16
  SPV_OPERAND_TYPE_LINKAGE_TYPE,                  // SPIR-V Sec 3.17
  SPV_OPERAND_TYPE_ACCESS_QUALIFIER,              // SPIR-V Sec 3.18
  SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE,  // SPIR-V Sec 3.19
  SPV_OPERAND_TYPE_DECORATION,                    // SPIR-V Sec 3.20
  SPV_OPERAND_TYPE_BUILT_IN,                      // SPIR-V Sec 3.21
  SPV_OPERAND_TYPE_GROUP_OPERATION,               // SPIR-V Sec 3.28
  SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS,              // SPIR-V Sec 3.29
  SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO,         // SPIR-V Sec 3.30
  SPV_OPERAND_TYPE_CAPABILITY,                    // SPIR-V Sec 3.31
  SPV_OPERAND_TYPE_FPENCODING,                    // SPIR-V Sec 3.51

  // NOTE: New concrete enum values should be added at the end.

  // Set 5:  Operands that are a single word bitmask.
  // Sometimes a set bit indicates the instruction requires still more operands.
  SPV_OPERAND_TYPE_IMAGE,                  // SPIR-V Sec 3.14
  SPV_OPERAND_TYPE_FP_FAST_MATH_MODE,      // SPIR-V Sec 3.15
  SPV_OPERAND_TYPE_SELECTION_CONTROL,      // SPIR-V Sec 3.22
  SPV_OPERAND_TYPE_LOOP_CONTROL,           // SPIR-V Sec 3.23
  SPV_OPERAND_TYPE_FUNCTION_CONTROL,       // SPIR-V Sec 3.24
  SPV_OPERAND_TYPE_MEMORY_ACCESS,          // SPIR-V Sec 3.26
  SPV_OPERAND_TYPE_FRAGMENT_SHADING_RATE,  // SPIR-V Sec 3.FSR

// NOTE: New concrete enum values should be added at the end.

// The "optional" and "variable"  operand types are only used internally by
// the assembler and the binary parser.
// There are two categories:
//    Optional : expands to 0 or 1 operand, like ? in regular expressions.
//    Variable : expands to 0, 1 or many operands or pairs of operands.
//               This is similar to * in regular expressions.

// NOTE: These FIRST_* and LAST_* enum values are DEPRECATED.
// The concept of "optional" and "variable" operand types are only intended
// for use as an implementation detail of parsing SPIR-V, either in text or
// binary form.  Instead of using enum ranges, use characteristic function
// spvOperandIsConcrete.
// The use of enum value ranges in a public API makes it difficult to insert
// new values into a range without also breaking binary compatibility.
//
// Macros for defining bounds on optional and variable operand types.
// Any variable operand type is also optional.
// TODO(dneto): Remove SPV_OPERAND_TYPE_FIRST_* and SPV_OPERAND_TYPE_LAST_*
#define FIRST_OPTIONAL(ENUM) ENUM, SPV_OPERAND_TYPE_FIRST_OPTIONAL_TYPE = ENUM
#define FIRST_VARIABLE(ENUM) ENUM, SPV_OPERAND_TYPE_FIRST_VARIABLE_TYPE = ENUM
#define LAST_VARIABLE(ENUM)                         \
  ENUM, SPV_OPERAND_TYPE_LAST_VARIABLE_TYPE = ENUM, \
        SPV_OPERAND_TYPE_LAST_OPTIONAL_TYPE = ENUM

  // An optional operand represents zero or one logical operands.
  // In an instruction definition, this may only appear at the end of the
  // operand types.
  FIRST_OPTIONAL(SPV_OPERAND_TYPE_OPTIONAL_ID),
  // An optional image operand type.
  SPV_OPERAND_TYPE_OPTIONAL_IMAGE,
  // An optional memory access type.
  SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS,
  // An optional literal integer.
  SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER,
  // An optional literal number, which may be either integer or floating point.
  SPV_OPERAND_TYPE_OPTIONAL_LITERAL_NUMBER,
  // Like SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER, but optional, and integral.
  SPV_OPERAND_TYPE_OPTIONAL_TYPED_LITERAL_INTEGER,
  // An optional literal string.
  SPV_OPERAND_TYPE_OPTIONAL_LITERAL_STRING,
  // An optional access qualifier
  SPV_OPERAND_TYPE_OPTIONAL_ACCESS_QUALIFIER,
  // An optional context-independent value, or CIV.  CIVs are tokens that we can
  // assemble regardless of where they occur -- literals, IDs, immediate
  // integers, etc.
  SPV_OPERAND_TYPE_OPTIONAL_CIV,
  // An optional floating point encoding enum
  SPV_OPERAND_TYPE_OPTIONAL_FPENCODING,

  // A variable operand represents zero or more logical operands.
  // In an instruction definition, this may only appear at the end of the
  // operand types.
  FIRST_VARIABLE(SPV_OPERAND_TYPE_VARIABLE_ID),
  SPV_OPERAND_TYPE_VARIABLE_LITERAL_INTEGER,
  // A sequence of zero or more pairs of (typed literal integer, Id).
  // Expands to zero or more:
  //  (SPV_OPERAND_TYPE_TYPED_LITERAL_INTEGER, SPV_OPERAND_TYPE_ID)
  // where the literal number must always be an integer of some sort.
  SPV_OPERAND_TYPE_VARIABLE_LITERAL_INTEGER_ID,
  // A sequence of zero or more pairs of (Id, Literal integer)
  LAST_VARIABLE(SPV_OPERAND_TYPE_VARIABLE_ID_LITERAL_INTEGER),

  // The following are concrete enum types from the DebugInfo extended
  // instruction set.
  SPV_OPERAND_TYPE_DEBUG_INFO_FLAGS,  // DebugInfo Sec 3.2.  A mask.
  SPV_OPERAND_TYPE_DEBUG_BASE_TYPE_ATTRIBUTE_ENCODING,  // DebugInfo Sec 3.3
  SPV_OPERAND_TYPE_DEBUG_COMPOSITE_TYPE,                // DebugInfo Sec 3.4
  SPV_OPERAND_TYPE_DEBUG_TYPE_QUALIFIER,                // DebugInfo Sec 3.5
  SPV_OPERAND_TYPE_DEBUG_OPERATION,                     // DebugInfo Sec 3.6

  // The following are concrete enum types from the OpenCL.DebugInfo.100
  // extended instruction set.
  SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_INFO_FLAGS,  // Sec 3.2. A Mask
  SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_BASE_TYPE_ATTRIBUTE_ENCODING,  // Sec 3.3
  SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_COMPOSITE_TYPE,                // Sec 3.4
  SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_TYPE_QUALIFIER,                // Sec 3.5
  SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_OPERATION,                     // Sec 3.6
  SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_IMPORTED_ENTITY,               // Sec 3.7

  // The following are concrete enum types from SPV_INTEL_float_controls2
  // https://github.com/intel/llvm/blob/39fa9b0cbfbae88327118990a05c5b387b56d2ef/sycl/doc/extensions/SPIRV/SPV_INTEL_float_controls2.asciidoc
  SPV_OPERAND_TYPE_FPDENORM_MODE,     // Sec 3.17 FP Denorm Mode
  SPV_OPERAND_TYPE_FPOPERATION_MODE,  // Sec 3.18 FP Operation Mode
  // A value enum from https://github.com/KhronosGroup/SPIRV-Headers/pull/177
  SPV_OPERAND_TYPE_QUANTIZATION_MODES,
  // A value enum from https://github.com/KhronosGroup/SPIRV-Headers/pull/177
  SPV_OPERAND_TYPE_OVERFLOW_MODES,

  // Concrete operand types for the provisional Vulkan ray tracing feature.
  SPV_OPERAND_TYPE_RAY_FLAGS,               // SPIR-V Sec 3.RF
  SPV_OPERAND_TYPE_RAY_QUERY_INTERSECTION,  // SPIR-V Sec 3.RQIntersection
  SPV_OPERAND_TYPE_RAY_QUERY_COMMITTED_INTERSECTION_TYPE,  // SPIR-V Sec
                                                           // 3.RQCommitted
  SPV_OPERAND_TYPE_RAY_QUERY_CANDIDATE_INTERSECTION_TYPE,  // SPIR-V Sec
                                                           // 3.RQCandidate

  // Concrete operand types for integer dot product.
  // Packed vector format
  SPV_OPERAND_TYPE_PACKED_VECTOR_FORMAT,  // SPIR-V Sec 3.x
  // An optional packed vector format
  SPV_OPERAND_TYPE_OPTIONAL_PACKED_VECTOR_FORMAT,

  // Concrete operand types for cooperative matrix.
  SPV_OPERAND_TYPE_COOPERATIVE_MATRIX_OPERANDS,
  // An optional cooperative matrix operands
  SPV_OPERAND_TYPE_OPTIONAL_COOPERATIVE_MATRIX_OPERANDS,
  SPV_OPERAND_TYPE_COOPERATIVE_MATRIX_LAYOUT,
  SPV_OPERAND_TYPE_COOPERATIVE_MATRIX_USE,

  // Enum type from SPV_INTEL_global_variable_fpga_decorations
  SPV_OPERAND_TYPE_INITIALIZATION_MODE_QUALIFIER,
  // Enum type from SPV_INTEL_global_variable_host_access
  SPV_OPERAND_TYPE_HOST_ACCESS_QUALIFIER,
  // Enum type from SPV_INTEL_cache_controls
  SPV_OPERAND_TYPE_LOAD_CACHE_CONTROL,
  // Enum type from SPV_INTEL_cache_controls
  SPV_OPERAND_TYPE_STORE_CACHE_CONTROL,
  // Enum type from SPV_INTEL_maximum_registers
  SPV_OPERAND_TYPE_NAMED_MAXIMUM_NUMBER_OF_REGISTERS,
  // Enum type from SPV_NV_raw_access_chains
  SPV_OPERAND_TYPE_RAW_ACCESS_CHAIN_OPERANDS,
  // Optional enum type from SPV_NV_raw_access_chains
  SPV_OPERAND_TYPE_OPTIONAL_RAW_ACCESS_CHAIN_OPERANDS,

  // This is a sentinel value, and does not represent an operand type.
  // It should come last.
  SPV_OPERAND_TYPE_NUM_OPERAND_TYPES,

  SPV_FORCE_32_BIT_ENUM(spv_operand_type_t)
} spv_operand_type_t;

// Returns true if the given type is concrete.
bool spvOperandIsConcrete(spv_operand_type_t type);

// Returns true if the given type is concrete and also a mask.
bool spvOperandIsConcreteMask(spv_operand_type_t type);

typedef enum spv_ext_inst_type_t {
  SPV_EXT_INST_TYPE_NONE = 0,
  SPV_EXT_INST_TYPE_GLSL_STD_450,
  SPV_EXT_INST_TYPE_OPENCL_STD,
  SPV_EXT_INST_TYPE_SPV_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER,
  SPV_EXT_INST_TYPE_SPV_AMD_SHADER_TRINARY_MINMAX,
  SPV_EXT_INST_TYPE_SPV_AMD_GCN_SHADER,
  SPV_EXT_INST_TYPE_SPV_AMD_SHADER_BALLOT,
  SPV_EXT_INST_TYPE_DEBUGINFO,
  SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100,
  SPV_EXT_INST_TYPE_NONSEMANTIC_CLSPVREFLECTION,
  SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100,
  SPV_EXT_INST_TYPE_NONSEMANTIC_VKSPREFLECTION,

  // Multiple distinct extended instruction set types could return this
  // value, if they are prefixed with NonSemantic. and are otherwise
  // unrecognised
  SPV_EXT_INST_TYPE_NONSEMANTIC_UNKNOWN,

  SPV_FORCE_32_BIT_ENUM(spv_ext_inst_type_t)
} spv_ext_inst_type_t;

// This determines at a high level the kind of a binary-encoded literal
// number, but not the bit width.
// In principle, these could probably be folded into new entries in
// spv_operand_type_t.  But then we'd have some special case differences
// between the assembler and disassembler.
typedef enum spv_number_kind_t {
  SPV_NUMBER_NONE = 0,  // The default for value initialization.
  SPV_NUMBER_UNSIGNED_INT,
  SPV_NUMBER_SIGNED_INT,
  SPV_NUMBER_FLOATING,
} spv_number_kind_t;

typedef enum spv_text_to_binary_options_t {
  SPV_TEXT_TO_BINARY_OPTION_NONE = SPV_BIT(0),
  // Numeric IDs in the binary will have the same values as in the source.
  // Non-numeric IDs are allocated by filling in the gaps, starting with 1
  // and going up.
  SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS = SPV_BIT(1),
  SPV_FORCE_32_BIT_ENUM(spv_text_to_binary_options_t)
} spv_text_to_binary_options_t;

typedef enum spv_binary_to_text_options_t {
  SPV_BINARY_TO_TEXT_OPTION_NONE = SPV_BIT(0),
  SPV_BINARY_TO_TEXT_OPTION_PRINT = SPV_BIT(1),
  SPV_BINARY_TO_TEXT_OPTION_COLOR = SPV_BIT(2),
  SPV_BINARY_TO_TEXT_OPTION_INDENT = SPV_BIT(3),
  SPV_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET = SPV_BIT(4),
  // Do not output the module header as leading comments in the assembly.
  SPV_BINARY_TO_TEXT_OPTION_NO_HEADER = SPV_BIT(5),
  // Use friendly names where possible.  The heuristic may expand over
  // time, but will use common names for scalar types, and debug names from
  // OpName instructions.
  SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES = SPV_BIT(6),
  // Add some comments to the generated assembly
  SPV_BINARY_TO_TEXT_OPTION_COMMENT = SPV_BIT(7),
  // Use nested indentation for more readable SPIR-V
  SPV_BINARY_TO_TEXT_OPTION_NESTED_INDENT = SPV_BIT(8),
  // Reorder blocks to match the structured control flow of SPIR-V to increase
  // readability.
  SPV_BINARY_TO_TEXT_OPTION_REORDER_BLOCKS = SPV_BIT(9),
  SPV_FORCE_32_BIT_ENUM(spv_binary_to_text_options_t)
} spv_binary_to_text_options_t;

// Constants

// The default id bound is to the minimum value for the id limit
// in the spir-v specification under the section "Universal Limits".
const uint32_t kDefaultMaxIdBound = 0x3FFFFF;

// Structures

// Information about an operand parsed from a binary SPIR-V module.
// Note that the values are not included.  You still need access to the binary
// to extract the values.
typedef struct spv_parsed_operand_t {
  // Location of the operand, in words from the start of the instruction.
  uint16_t offset;
  // Number of words occupied by this operand.
  uint16_t num_words;
  // The "concrete" operand type.  See the definition of spv_operand_type_t
  // for details.
  spv_operand_type_t type;
  // If type is a literal number type, then number_kind says whether it's
  // a signed integer, an unsigned integer, or a floating point number.
  spv_number_kind_t number_kind;
  // The number of bits for a literal number type.
  uint32_t number_bit_width;
} spv_parsed_operand_t;

// An instruction parsed from a binary SPIR-V module.
typedef struct spv_parsed_instruction_t {
  // An array of words for this instruction, in native endianness.
  const uint32_t* words;
  // The number of words in this instruction.
  uint16_t num_words;
  uint16_t opcode;
  // The extended instruction type, if opcode is OpExtInst.  Otherwise
  // this is the "none" value.
  spv_ext_inst_type_t ext_inst_type;
  // The type id, or 0 if this instruction doesn't have one.
  uint32_t type_id;
  // The result id, or 0 if this instruction doesn't have one.
  uint32_t result_id;
  // The array of parsed operands.
  const spv_parsed_operand_t* operands;
  uint16_t num_operands;
} spv_parsed_instruction_t;

typedef struct spv_parsed_header_t {
  // The magic number of the SPIR-V module.
  uint32_t magic;
  // Version number.
  uint32_t version;
  // Generator's magic number.
  uint32_t generator;
  // IDs bound for this module (0 < id < bound).
  uint32_t bound;
  // reserved.
  uint32_t reserved;
} spv_parsed_header_t;

typedef struct spv_const_binary_t {
  const uint32_t* code;
  const size_t wordCount;
} spv_const_binary_t;

typedef struct spv_binary_t {
  uint32_t* code;
  size_t wordCount;
} spv_binary_t;

typedef struct spv_text_t {
  const char* str;
  size_t length;
} spv_text_t;

typedef struct spv_position_t {
  size_t line;
  size_t column;
  size_t index;
} spv_position_t;

typedef struct spv_diagnostic_t {
  spv_position_t position;
  char* error;
  bool isTextSource;
} spv_diagnostic_t;

// Opaque struct containing the context used to operate on a SPIR-V module.
// Its object is used by various translation API functions.
typedef struct spv_context_t spv_context_t;

typedef struct spv_validator_options_t spv_validator_options_t;

typedef struct spv_optimizer_options_t spv_optimizer_options_t;

typedef struct spv_reducer_options_t spv_reducer_options_t;

typedef struct spv_fuzzer_options_t spv_fuzzer_options_t;

typedef struct spv_optimizer_t spv_optimizer_t;

// Type Definitions

typedef spv_const_binary_t* spv_const_binary;
typedef spv_binary_t* spv_binary;
typedef spv_text_t* spv_text;
typedef spv_position_t* spv_position;
typedef spv_diagnostic_t* spv_diagnostic;
typedef const spv_context_t* spv_const_context;
typedef spv_context_t* spv_context;
typedef spv_validator_options_t* spv_validator_options;
typedef const spv_validator_options_t* spv_const_validator_options;
typedef spv_optimizer_options_t* spv_optimizer_options;
typedef const spv_optimizer_options_t* spv_const_optimizer_options;
typedef spv_reducer_options_t* spv_reducer_options;
typedef const spv_reducer_options_t* spv_const_reducer_options;
typedef spv_fuzzer_options_t* spv_fuzzer_options;
typedef const spv_fuzzer_options_t* spv_const_fuzzer_options;

// Platform API

// Returns the SPIRV-Tools software version as a null-terminated string.
// The contents of the underlying storage is valid for the remainder of
// the process.
SPIRV_TOOLS_EXPORT const char* spvSoftwareVersionString(void);
// Returns a null-terminated string containing the name of the project,
// the software version string, and commit details.
// The contents of the underlying storage is valid for the remainder of
// the process.
SPIRV_TOOLS_EXPORT const char* spvSoftwareVersionDetailsString(void);

// Certain target environments impose additional restrictions on SPIR-V, so it's
// often necessary to specify which one applies.  SPV_ENV_UNIVERSAL_* implies an
// environment-agnostic SPIR-V.
//
// When an API method needs to derive a SPIR-V version from a target environment
// (from the spv_context object), the method will choose the highest version of
// SPIR-V supported by the target environment.  Examples:
//    SPV_ENV_VULKAN_1_0           ->  SPIR-V 1.0
//    SPV_ENV_VULKAN_1_1           ->  SPIR-V 1.3
//    SPV_ENV_VULKAN_1_1_SPIRV_1_4 ->  SPIR-V 1.4
//    SPV_ENV_VULKAN_1_2           ->  SPIR-V 1.5
//    SPV_ENV_VULKAN_1_3           ->  SPIR-V 1.6
// Consult the description of API entry points for specific rules.
typedef enum {
  SPV_ENV_UNIVERSAL_1_0,  // SPIR-V 1.0 latest revision, no other restrictions.
  SPV_ENV_VULKAN_1_0,     // Vulkan 1.0 latest revision.
  SPV_ENV_UNIVERSAL_1_1,  // SPIR-V 1.1 latest revision, no other restrictions.
  SPV_ENV_OPENCL_2_1,     // OpenCL Full Profile 2.1 latest revision.
  SPV_ENV_OPENCL_2_2,     // OpenCL Full Profile 2.2 latest revision.
  SPV_ENV_OPENGL_4_0,     // OpenGL 4.0 plus GL_ARB_gl_spirv, latest revisions.
  SPV_ENV_OPENGL_4_1,     // OpenGL 4.1 plus GL_ARB_gl_spirv, latest revisions.
  SPV_ENV_OPENGL_4_2,     // OpenGL 4.2 plus GL_ARB_gl_spirv, latest revisions.
  SPV_ENV_OPENGL_4_3,     // OpenGL 4.3 plus GL_ARB_gl_spirv, latest revisions.
  // There is no variant for OpenGL 4.4.
  SPV_ENV_OPENGL_4_5,     // OpenGL 4.5 plus GL_ARB_gl_spirv, latest revisions.
  SPV_ENV_UNIVERSAL_1_2,  // SPIR-V 1.2, latest revision, no other restrictions.
  SPV_ENV_OPENCL_1_2,     // OpenCL Full Profile 1.2 plus cl_khr_il_program,
                          // latest revision.
  SPV_ENV_OPENCL_EMBEDDED_1_2,  // OpenCL Embedded Profile 1.2 plus
                                // cl_khr_il_program, latest revision.
  SPV_ENV_OPENCL_2_0,  // OpenCL Full Profile 2.0 plus cl_khr_il_program,
                       // latest revision.
  SPV_ENV_OPENCL_EMBEDDED_2_0,  // OpenCL Embedded Profile 2.0 plus
                                // cl_khr_il_program, latest revision.
  SPV_ENV_OPENCL_EMBEDDED_2_1,  // OpenCL Embedded Profile 2.1 latest revision.
  SPV_ENV_OPENCL_EMBEDDED_2_2,  // OpenCL Embedded Profile 2.2 latest revision.
  SPV_ENV_UNIVERSAL_1_3,  // SPIR-V 1.3 latest revision, no other restrictions.
  SPV_ENV_VULKAN_1_1,     // Vulkan 1.1 latest revision.
  SPV_ENV_WEBGPU_0,       // DEPRECATED, may be removed in the future.
  SPV_ENV_UNIVERSAL_1_4,  // SPIR-V 1.4 latest revision, no other restrictions.

  // Vulkan 1.1 with VK_KHR_spirv_1_4, i.e. SPIR-V 1.4 binary.
  SPV_ENV_VULKAN_1_1_SPIRV_1_4,

  SPV_ENV_UNIVERSAL_1_5,  // SPIR-V 1.5 latest revision, no other restrictions.
  SPV_ENV_VULKAN_1_2,     // Vulkan 1.2 latest revision.

  SPV_ENV_UNIVERSAL_1_6,  // SPIR-V 1.6 latest revision, no other restrictions.
  SPV_ENV_VULKAN_1_3,     // Vulkan 1.3 latest revision.

  SPV_ENV_MAX  // Keep this as the last enum value.
} spv_target_env;

// SPIR-V Validator can be parameterized with the following Universal Limits.
typedef enum {
  spv_validator_limit_max_struct_members,
  spv_validator_limit_max_struct_depth,
  spv_validator_limit_max_local_variables,
  spv_validator_limit_max_global_variables,
  spv_validator_limit_max_switch_branches,
  spv_validator_limit_max_function_args,
  spv_validator_limit_max_control_flow_nesting_depth,
  spv_validator_limit_max_access_chain_indexes,
  spv_validator_limit_max_id_bound,
} spv_validator_limit;

// Returns a string describing the given SPIR-V target environment.
SPIRV_TOOLS_EXPORT const char* spvTargetEnvDescription(spv_target_env env);

// Parses s into *env and returns true if successful.  If unparsable, returns
// false and sets *env to SPV_ENV_UNIVERSAL_1_0.
SPIRV_TOOLS_EXPORT bool spvParseTargetEnv(const char* s, spv_target_env* env);

// Determines the target env value with the least features but which enables
// the given Vulkan and SPIR-V versions. If such a target is supported, returns
// true and writes the value to |env|, otherwise returns false.
//
// The Vulkan version is given as an unsigned 32-bit number as specified in
// Vulkan section "29.2.1 Version Numbers": the major version number appears
// in bits 22 to 21, and the minor version is in bits 12 to 21.  The SPIR-V
// version is given in the SPIR-V version header word: major version in bits
// 16 to 23, and minor version in bits 8 to 15.
SPIRV_TOOLS_EXPORT bool spvParseVulkanEnv(uint32_t vulkan_ver,
                                          uint32_t spirv_ver,
                                          spv_target_env* env);

// Creates a context object for most of the SPIRV-Tools API.
// Returns null if env is invalid.
//
// See specific API calls for how the target environment is interpreted
// (particularly assembly and validation).
SPIRV_TOOLS_EXPORT spv_context spvContextCreate(spv_target_env env);

// Destroys the given context object.
SPIRV_TOOLS_EXPORT void spvContextDestroy(spv_context context);

// Creates a Validator options object with default options. Returns a valid
// options object. The object remains valid until it is passed into
// spvValidatorOptionsDestroy.
SPIRV_TOOLS_EXPORT spv_validator_options spvValidatorOptionsCreate(void);

// Destroys the given Validator options object.
SPIRV_TOOLS_EXPORT void spvValidatorOptionsDestroy(
    spv_validator_options options);

// Records the maximum Universal Limit that is considered valid in the given
// Validator options object. <options> argument must be a valid options object.
SPIRV_TOOLS_EXPORT void spvValidatorOptionsSetUniversalLimit(
    spv_validator_options options, spv_validator_limit limit_type,
    uint32_t limit);

// Record whether or not the validator should relax the rules on types for
// stores to structs.  When relaxed, it will allow a type mismatch as long as
// the types are structs with the same layout.  Two structs have the same layout
// if
//
// 1) the members of the structs are either the same type or are structs with
// same layout, and
//
// 2) the decorations that affect the memory layout are identical for both
// types.  Other decorations are not relevant.
SPIRV_TOOLS_EXPORT void spvValidatorOptionsSetRelaxStoreStruct(
    spv_validator_options options, bool val);

// Records whether or not the validator should relax the rules on pointer usage
// in logical addressing mode.
//
// When relaxed, it will allow the following usage cases of pointers:
// 1) OpVariable allocating an object whose type is a pointer type
// 2) OpReturnValue returning a pointer value
SPIRV_TOOLS_EXPORT void spvValidatorOptionsSetRelaxLogicalPointer(
    spv_validator_options options, bool val);

// Records whether or not the validator should relax the rules because it is
// expected that the optimizations will make the code legal.
//
// When relaxed, it will allow the following:
// 1) It will allow relaxed logical pointers.  Setting this option will also
//    set that option.
// 2) Pointers that are pass as parameters to function calls do not have to
//    match the storage class of the formal parameter.
// 3) Pointers that are actual parameters on function calls do not have to point
//    to the same type pointed as the formal parameter.  The types just need to
//    logically match.
// 4) GLSLstd450 Interpolate* instructions can have a load of an interpolant
//    for a first argument.
SPIRV_TOOLS_EXPORT void spvValidatorOptionsSetBeforeHlslLegalization(
    spv_validator_options options, bool val);

// Records whether the validator should use "relaxed" block layout rules.
// Relaxed layout rules are described by Vulkan extension
// VK_KHR_relaxed_block_layout, and they affect uniform blocks, storage blocks,
// and push constants.
//
// This is enabled by default when targeting Vulkan 1.1 or later.
// Relaxed layout is more permissive than the default rules in Vulkan 1.0.
SPIRV_TOOLS_EXPORT void spvValidatorOptionsSetRelaxBlockLayout(
    spv_validator_options options, bool val);

// Records whether the validator should use standard block layout rules for
// uniform blocks.
SPIRV_TOOLS_EXPORT void spvValidatorOptionsSetUniformBufferStandardLayout(
    spv_validator_options options, bool val);

// Records whether the validator should use "scalar" block layout rules.
// Scalar layout rules are more permissive than relaxed block layout.
//
// See Vulkan extension VK_EXT_scalar_block_layout.  The scalar alignment is
// defined as follows:
// - scalar alignment of a scalar is the scalar size
// - scalar alignment of a vector is the scalar alignment of its component
// - scalar alignment of a matrix is the scalar alignment of its component
// - scalar alignment of an array is the scalar alignment of its element
// - scalar alignment of a struct is the max scalar alignment among its
//   members
//
// For a struct in Uniform, StorageClass, or PushConstant:
// - a member Offset must be a multiple of the member's scalar alignment
// - ArrayStride or MatrixStride must be a multiple of the array or matrix
//   scalar alignment
SPIRV_TOOLS_EXPORT void spvValidatorOptionsSetScalarBlockLayout(
    spv_validator_options options, bool val);

// Records whether the validator should use "scalar" block layout
// rules (as defined above) for Workgroup blocks.  See Vulkan
// extension VK_KHR_workgroup_memory_explicit_layout.
SPIRV_TOOLS_EXPORT void spvValidatorOptionsSetWorkgroupScalarBlockLayout(
    spv_validator_options options, bool val);

// Records whether or not the validator should skip validating standard
// uniform/storage block layout.
SPIRV_TOOLS_EXPORT void spvValidatorOptionsSetSkipBlockLayout(
    spv_validator_options options, bool val);

// Records whether or not the validator should allow the LocalSizeId
// decoration where the environment otherwise would not allow it.
SPIRV_TOOLS_EXPORT void spvValidatorOptionsSetAllowLocalSizeId(
    spv_validator_options options, bool val);

// Whether friendly names should be used in validation error messages.
SPIRV_TOOLS_EXPORT void spvValidatorOptionsSetFriendlyNames(
    spv_validator_options options, bool val);

// Creates an optimizer options object with default options. Returns a valid
// options object. The object remains valid until it is passed into
// |spvOptimizerOptionsDestroy|.
SPIRV_TOOLS_EXPORT spv_optimizer_options spvOptimizerOptionsCreate(void);

// Destroys the given optimizer options object.
SPIRV_TOOLS_EXPORT void spvOptimizerOptionsDestroy(
    spv_optimizer_options options);

// Records whether or not the optimizer should run the validator before
// optimizing.  If |val| is true, the validator will be run.
SPIRV_TOOLS_EXPORT void spvOptimizerOptionsSetRunValidator(
    spv_optimizer_options options, bool val);

// Records the validator options that should be passed to the validator if it is
// run.
SPIRV_TOOLS_EXPORT void spvOptimizerOptionsSetValidatorOptions(
    spv_optimizer_options options, spv_validator_options val);

// Records the maximum possible value for the id bound.
SPIRV_TOOLS_EXPORT void spvOptimizerOptionsSetMaxIdBound(
    spv_optimizer_options options, uint32_t val);

// Records whether all bindings within the module should be preserved.
SPIRV_TOOLS_EXPORT void spvOptimizerOptionsSetPreserveBindings(
    spv_optimizer_options options, bool val);

// Records whether all specialization constants within the module
// should be preserved.
SPIRV_TOOLS_EXPORT void spvOptimizerOptionsSetPreserveSpecConstants(
    spv_optimizer_options options, bool val);

// Creates a reducer options object with default options. Returns a valid
// options object. The object remains valid until it is passed into
// |spvReducerOptionsDestroy|.
SPIRV_TOOLS_EXPORT spv_reducer_options spvReducerOptionsCreate(void);

// Destroys the given reducer options object.
SPIRV_TOOLS_EXPORT void spvReducerOptionsDestroy(spv_reducer_options options);

// Sets the maximum number of reduction steps that should run before the reducer
// gives up.
SPIRV_TOOLS_EXPORT void spvReducerOptionsSetStepLimit(
    spv_reducer_options options, uint32_t step_limit);

// Sets the fail-on-validation-error option; if true, the reducer will return
// kStateInvalid if a reduction step yields a state that fails SPIR-V
// validation. Otherwise, an invalid state is treated as uninteresting and the
// reduction backtracks and continues.
SPIRV_TOOLS_EXPORT void spvReducerOptionsSetFailOnValidationError(
    spv_reducer_options options, bool fail_on_validation_error);

// Sets the function that the reducer should target.  If set to zero the reducer
// will target all functions as well as parts of the module that lie outside
// functions.  Otherwise the reducer will restrict reduction to the function
// with result id |target_function|, which is required to exist.
SPIRV_TOOLS_EXPORT void spvReducerOptionsSetTargetFunction(
    spv_reducer_options options, uint32_t target_function);

// Creates a fuzzer options object with default options. Returns a valid
// options object. The object remains valid until it is passed into
// |spvFuzzerOptionsDestroy|.
SPIRV_TOOLS_EXPORT spv_fuzzer_options spvFuzzerOptionsCreate(void);

// Destroys the given fuzzer options object.
SPIRV_TOOLS_EXPORT void spvFuzzerOptionsDestroy(spv_fuzzer_options options);

// Enables running the validator after every transformation is applied during
// a replay.
SPIRV_TOOLS_EXPORT void spvFuzzerOptionsEnableReplayValidation(
    spv_fuzzer_options options);

// Sets the seed with which the random number generator used by the fuzzer
// should be initialized.
SPIRV_TOOLS_EXPORT void spvFuzzerOptionsSetRandomSeed(
    spv_fuzzer_options options, uint32_t seed);

// Sets the range of transformations that should be applied during replay: 0
// means all transformations, +N means the first N transformations, -N means all
// except the final N transformations.
SPIRV_TOOLS_EXPORT void spvFuzzerOptionsSetReplayRange(
    spv_fuzzer_options options, int32_t replay_range);

// Sets the maximum number of steps that the shrinker should take before giving
// up.
SPIRV_TOOLS_EXPORT void spvFuzzerOptionsSetShrinkerStepLimit(
    spv_fuzzer_options options, uint32_t shrinker_step_limit);

// Enables running the validator after every pass is applied during a fuzzing
// run.
SPIRV_TOOLS_EXPORT void spvFuzzerOptionsEnableFuzzerPassValidation(
    spv_fuzzer_options options);

// Enables all fuzzer passes during a fuzzing run (instead of a random subset
// of passes).
SPIRV_TOOLS_EXPORT void spvFuzzerOptionsEnableAllPasses(
    spv_fuzzer_options options);

// Encodes the given SPIR-V assembly text to its binary representation. The
// length parameter specifies the number of bytes for text. Encoded binary will
// be stored into *binary. Any error will be written into *diagnostic if
// diagnostic is non-null, otherwise the context's message consumer will be
// used. The generated binary is independent of the context and may outlive it.
// The SPIR-V binary version is set to the highest version of SPIR-V supported
// by the context's target environment.
SPIRV_TOOLS_EXPORT spv_result_t spvTextToBinary(const spv_const_context context,
                                                const char* text,
                                                const size_t length,
                                                spv_binary* binary,
                                                spv_diagnostic* diagnostic);

// Encodes the given SPIR-V assembly text to its binary representation. Same as
// spvTextToBinary but with options. The options parameter is a bit field of
// spv_text_to_binary_options_t.
SPIRV_TOOLS_EXPORT spv_result_t spvTextToBinaryWithOptions(
    const spv_const_context context, const char* text, const size_t length,
    const uint32_t options, spv_binary* binary, spv_diagnostic* diagnostic);

// Frees an allocated text stream. This is a no-op if the text parameter
// is a null pointer.
SPIRV_TOOLS_EXPORT void spvTextDestroy(spv_text text);

// Decodes the given SPIR-V binary representation to its assembly text. The
// word_count parameter specifies the number of words for binary. The options
// parameter is a bit field of spv_binary_to_text_options_t. Decoded text will
// be stored into *text. Any error will be written into *diagnostic if
// diagnostic is non-null, otherwise the context's message consumer will be
// used.
SPIRV_TOOLS_EXPORT spv_result_t spvBinaryToText(const spv_const_context context,
                                                const uint32_t* binary,
                                                const size_t word_count,
                                                const uint32_t options,
                                                spv_text* text,
                                                spv_diagnostic* diagnostic);

// Frees a binary stream from memory. This is a no-op if binary is a null
// pointer.
SPIRV_TOOLS_EXPORT void spvBinaryDestroy(spv_binary binary);

// Validates a SPIR-V binary for correctness. Any errors will be written into
// *diagnostic if diagnostic is non-null, otherwise the context's message
// consumer will be used.
//
// Validate for SPIR-V spec rules for the SPIR-V version named in the
// binary's header (at word offset 1).  Additionally, if the context target
// environment is a client API (such as Vulkan 1.1), then validate for that
// client API version, to the extent that it is verifiable from data in the
// binary itself.
SPIRV_TOOLS_EXPORT spv_result_t spvValidate(const spv_const_context context,
                                            const spv_const_binary binary,
                                            spv_diagnostic* diagnostic);

// Validates a SPIR-V binary for correctness. Uses the provided Validator
// options. Any errors will be written into *diagnostic if diagnostic is
// non-null, otherwise the context's message consumer will be used.
//
// Validate for SPIR-V spec rules for the SPIR-V version named in the
// binary's header (at word offset 1).  Additionally, if the context target
// environment is a client API (such as Vulkan 1.1), then validate for that
// client API version, to the extent that it is verifiable from data in the
// binary itself, or in the validator options.
SPIRV_TOOLS_EXPORT spv_result_t spvValidateWithOptions(
    const spv_const_context context, const spv_const_validator_options options,
    const spv_const_binary binary, spv_diagnostic* diagnostic);

// Validates a raw SPIR-V binary for correctness. Any errors will be written
// into *diagnostic if diagnostic is non-null, otherwise the context's message
// consumer will be used.
SPIRV_TOOLS_EXPORT spv_result_t
spvValidateBinary(const spv_const_context context, const uint32_t* words,
                  const size_t num_words, spv_diagnostic* diagnostic);

// Creates a diagnostic object. The position parameter specifies the location in
// the text/binary stream. The message parameter, copied into the diagnostic
// object, contains the error message to display.
SPIRV_TOOLS_EXPORT spv_diagnostic
spvDiagnosticCreate(const spv_position position, const char* message);

// Destroys a diagnostic object.  This is a no-op if diagnostic is a null
// pointer.
SPIRV_TOOLS_EXPORT void spvDiagnosticDestroy(spv_diagnostic diagnostic);

// Prints the diagnostic to stderr.
SPIRV_TOOLS_EXPORT spv_result_t
spvDiagnosticPrint(const spv_diagnostic diagnostic);

// Gets the name of an instruction, without the "Op" prefix.
SPIRV_TOOLS_EXPORT const char* spvOpcodeString(const uint32_t opcode);

// The binary parser interface.

// A pointer to a function that accepts a parsed SPIR-V header.
// The integer arguments are the 32-bit words from the header, as specified
// in SPIR-V 1.0 Section 2.3 Table 1.
// The function should return SPV_SUCCESS if parsing should continue.
typedef spv_result_t (*spv_parsed_header_fn_t)(
    void* user_data, spv_endianness_t endian, uint32_t magic, uint32_t version,
    uint32_t generator, uint32_t id_bound, uint32_t reserved);

// A pointer to a function that accepts a parsed SPIR-V instruction.
// The parsed_instruction value is transient: it may be overwritten
// or released immediately after the function has returned.  That also
// applies to the words array member of the parsed instruction.  The
// function should return SPV_SUCCESS if and only if parsing should
// continue.
typedef spv_result_t (*spv_parsed_instruction_fn_t)(
    void* user_data, const spv_parsed_instruction_t* parsed_instruction);

// Parses a SPIR-V binary, specified as counted sequence of 32-bit words.
// Parsing feedback is provided via two callbacks provided as function
// pointers.  Each callback function pointer can be a null pointer, in
// which case it is never called.  Otherwise, in a valid parse the
// parsed-header callback is called once, and then the parsed-instruction
// callback once for each instruction in the stream.  The user_data parameter
// is supplied as context to the callbacks.  Returns SPV_SUCCESS on successful
// parse where the callbacks always return SPV_SUCCESS.  For an invalid parse,
// returns a status code other than SPV_SUCCESS, and if diagnostic is non-null
// also emits a diagnostic. If diagnostic is null the context's message consumer
// will be used to emit any errors. If a callback returns anything other than
// SPV_SUCCESS, then that status code is returned, no further callbacks are
// issued, and no additional diagnostics are emitted.
SPIRV_TOOLS_EXPORT spv_result_t spvBinaryParse(
    const spv_const_context context, void* user_data, const uint32_t* words,
    const size_t num_words, spv_parsed_header_fn_t parse_header,
    spv_parsed_instruction_fn_t parse_instruction, spv_diagnostic* diagnostic);

// The optimizer interface.

// A pointer to a function that accepts a log message from an optimizer.
typedef void (*spv_message_consumer)(
    spv_message_level_t, const char*, const spv_position_t*, const char*);

// Creates and returns an optimizer object.  This object must be passed to
// optimizer APIs below and is valid until passed to spvOptimizerDestroy.
SPIRV_TOOLS_EXPORT spv_optimizer_t* spvOptimizerCreate(spv_target_env env);

// Destroys the given optimizer object.
SPIRV_TOOLS_EXPORT void spvOptimizerDestroy(spv_optimizer_t* optimizer);

// Sets an spv_message_consumer on an optimizer object.
SPIRV_TOOLS_EXPORT void spvOptimizerSetMessageConsumer(
    spv_optimizer_t* optimizer, spv_message_consumer consumer);

// Registers passes that attempt to legalize the generated code.
SPIRV_TOOLS_EXPORT void spvOptimizerRegisterLegalizationPasses(
    spv_optimizer_t* optimizer);

// Registers passes that attempt to improve performance of generated code.
SPIRV_TOOLS_EXPORT void spvOptimizerRegisterPerformancePasses(
    spv_optimizer_t* optimizer);

// Registers passes that attempt to improve the size of generated code.
SPIRV_TOOLS_EXPORT void spvOptimizerRegisterSizePasses(
    spv_optimizer_t* optimizer);

// Registers a pass specified by a flag in an optimizer object.
SPIRV_TOOLS_EXPORT bool spvOptimizerRegisterPassFromFlag(
    spv_optimizer_t* optimizer, const char* flag);

// Registers passes specified by length number of flags in an optimizer object.
// Passes may remove interface variables that are unused.
SPIRV_TOOLS_EXPORT bool spvOptimizerRegisterPassesFromFlags(
    spv_optimizer_t* optimizer, const char** flags, const size_t flag_count);

// Registers passes specified by length number of flags in an optimizer object.
// Passes will not remove interface variables.
SPIRV_TOOLS_EXPORT bool
spvOptimizerRegisterPassesFromFlagsWhilePreservingTheInterface(
    spv_optimizer_t* optimizer, const char** flags, const size_t flag_count);

// Optimizes the SPIR-V code of size |word_count| pointed to by |binary| and
// returns an optimized spv_binary in |optimized_binary|.
//
// Returns SPV_SUCCESS on successful optimization, whether or not the module is
// modified.  Returns an SPV_ERROR_* if the module fails to validate or if
// errors occur when processing using any of the registered passes.  In that
// case, no further passes are executed and the |optimized_binary| contents may
// be invalid.
//
// By default, the binary is validated before any transforms are performed,
// and optionally after each transform.  Validation uses SPIR-V spec rules
// for the SPIR-V version named in the binary's header (at word offset 1).
// Additionally, if the target environment is a client API (such as
// Vulkan 1.1), then validate for that client API version, to the extent
// that it is verifiable from data in the binary itself, or from the
// validator options set on the optimizer options.
SPIRV_TOOLS_EXPORT spv_result_t spvOptimizerRun(
    spv_optimizer_t* optimizer, const uint32_t* binary, const size_t word_count,
    spv_binary* optimized_binary, const spv_optimizer_options options);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_SPIRV_TOOLS_LIBSPIRV_H_
