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

#include "source/text.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "source/assembly_grammar.h"
#include "source/binary.h"
#include "source/diagnostic.h"
#include "source/ext_inst.h"
#include "source/instruction.h"
#include "source/opcode.h"
#include "source/operand.h"
#include "source/spirv_constant.h"
#include "source/spirv_target_env.h"
#include "source/table.h"
#include "source/text_handler.h"
#include "source/util/bitutils.h"
#include "source/util/parse_number.h"
#include "spirv-tools/libspirv.h"

bool spvIsValidIDCharacter(const char value) {
  return value == '_' || 0 != ::isalnum(value);
}

// Returns true if the given string represents a valid ID name.
bool spvIsValidID(const char* textValue) {
  const char* c = textValue;
  for (; *c != '\0'; ++c) {
    if (!spvIsValidIDCharacter(*c)) {
      return false;
    }
  }
  // If the string was empty, then the ID also is not valid.
  return c != textValue;
}

// Text API

spv_result_t spvTextToLiteral(const char* textValue, spv_literal_t* pLiteral) {
  bool isSigned = false;
  int numPeriods = 0;
  bool isString = false;

  const size_t len = strlen(textValue);
  if (len == 0) return SPV_FAILED_MATCH;

  for (uint64_t index = 0; index < len; ++index) {
    switch (textValue[index]) {
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        break;
      case '.':
        numPeriods++;
        break;
      case '-':
        if (index == 0) {
          isSigned = true;
        } else {
          isString = true;
        }
        break;
      default:
        isString = true;
        index = len;  // break out of the loop too.
        break;
    }
  }

  pLiteral->type = spv_literal_type_t(99);

  if (isString || numPeriods > 1 || (isSigned && len == 1)) {
    if (len < 2 || textValue[0] != '"' || textValue[len - 1] != '"')
      return SPV_FAILED_MATCH;
    bool escaping = false;
    for (const char* val = textValue + 1; val != textValue + len - 1; ++val) {
      if ((*val == '\\') && (!escaping)) {
        escaping = true;
      } else {
        // Have to save space for the null-terminator
        if (pLiteral->str.size() >= SPV_LIMIT_LITERAL_STRING_BYTES_MAX)
          return SPV_ERROR_OUT_OF_MEMORY;
        pLiteral->str.push_back(*val);
        escaping = false;
      }
    }

    pLiteral->type = SPV_LITERAL_TYPE_STRING;
  } else if (numPeriods == 1) {
    double d = std::strtod(textValue, nullptr);
    float f = (float)d;
    if (d == (double)f) {
      pLiteral->type = SPV_LITERAL_TYPE_FLOAT_32;
      pLiteral->value.f = f;
    } else {
      pLiteral->type = SPV_LITERAL_TYPE_FLOAT_64;
      pLiteral->value.d = d;
    }
  } else if (isSigned) {
    int64_t i64 = strtoll(textValue, nullptr, 10);
    int32_t i32 = (int32_t)i64;
    if (i64 == (int64_t)i32) {
      pLiteral->type = SPV_LITERAL_TYPE_INT_32;
      pLiteral->value.i32 = i32;
    } else {
      pLiteral->type = SPV_LITERAL_TYPE_INT_64;
      pLiteral->value.i64 = i64;
    }
  } else {
    uint64_t u64 = strtoull(textValue, nullptr, 10);
    uint32_t u32 = (uint32_t)u64;
    if (u64 == (uint64_t)u32) {
      pLiteral->type = SPV_LITERAL_TYPE_UINT_32;
      pLiteral->value.u32 = u32;
    } else {
      pLiteral->type = SPV_LITERAL_TYPE_UINT_64;
      pLiteral->value.u64 = u64;
    }
  }

  return SPV_SUCCESS;
}

namespace {

/// Parses an immediate integer from text, guarding against overflow.  If
/// successful, adds the parsed value to pInst, advances the context past it,
/// and returns SPV_SUCCESS.  Otherwise, leaves pInst alone, emits diagnostics,
/// and returns SPV_ERROR_INVALID_TEXT.
spv_result_t encodeImmediate(spvtools::AssemblyContext* context,
                             const char* text, spv_instruction_t* pInst) {
  assert(*text == '!');
  uint32_t parse_result;
  if (!spvtools::utils::ParseNumber(text + 1, &parse_result)) {
    return context->diagnostic(SPV_ERROR_INVALID_TEXT)
           << "Invalid immediate integer: !" << text + 1;
  }
  context->binaryEncodeU32(parse_result, pInst);
  context->seekForward(static_cast<uint32_t>(strlen(text)));
  return SPV_SUCCESS;
}

}  // anonymous namespace

/// @brief Translate an Opcode operand to binary form
///
/// @param[in] grammar the grammar to use for compilation
/// @param[in, out] context the dynamic compilation info
/// @param[in] type of the operand
/// @param[in] textValue word of text to be parsed
/// @param[out] pInst return binary Opcode
/// @param[in,out] pExpectedOperands the operand types expected
///
/// @return result code
spv_result_t spvTextEncodeOperand(const spvtools::AssemblyGrammar& grammar,
                                  spvtools::AssemblyContext* context,
                                  const spv_operand_type_t type,
                                  const char* textValue,
                                  spv_instruction_t* pInst,
                                  spv_operand_pattern_t* pExpectedOperands) {
  // NOTE: Handle immediate int in the stream
  if ('!' == textValue[0]) {
    if (auto error = encodeImmediate(context, textValue, pInst)) {
      return error;
    }
    *pExpectedOperands =
        spvAlternatePatternFollowingImmediate(*pExpectedOperands);
    return SPV_SUCCESS;
  }

  // Optional literal operands can fail to parse. In that case use
  // SPV_FAILED_MATCH to avoid emitting a diagostic.  Use the following
  // for those situations.
  spv_result_t error_code_for_literals =
      spvOperandIsOptional(type) ? SPV_FAILED_MATCH : SPV_ERROR_INVALID_TEXT;

  switch (type) {
    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_TYPE_ID:
    case SPV_OPERAND_TYPE_RESULT_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
    case SPV_OPERAND_TYPE_SCOPE_ID:
    case SPV_OPERAND_TYPE_OPTIONAL_ID: {
      if ('%' == textValue[0]) {
        textValue++;
      } else {
        return context->diagnostic() << "Expected id to start with %.";
      }
      if (!spvIsValidID(textValue)) {
        return context->diagnostic() << "Invalid ID " << textValue;
      }
      const uint32_t id = context->spvNamedIdAssignOrGet(textValue);
      if (type == SPV_OPERAND_TYPE_TYPE_ID) pInst->resultTypeId = id;
      spvInstructionAddWord(pInst, id);

      // Set the extended instruction type.
      // The import set id is the 3rd operand of OpExtInst.
      if (pInst->opcode == SpvOpExtInst && pInst->words.size() == 4) {
        auto ext_inst_type = context->getExtInstTypeForId(pInst->words[3]);
        if (ext_inst_type == SPV_EXT_INST_TYPE_NONE) {
          return context->diagnostic()
                 << "Invalid extended instruction import Id "
                 << pInst->words[2];
        }
        pInst->extInstType = ext_inst_type;
      }
    } break;

    case SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER: {
      // The assembler accepts the symbolic name for an extended instruction,
      // and emits its corresponding number.
      spv_ext_inst_desc extInst;
      if (grammar.lookupExtInst(pInst->extInstType, textValue, &extInst)) {
        return context->diagnostic()
               << "Invalid extended instruction name '" << textValue << "'.";
      }
      spvInstructionAddWord(pInst, extInst->ext_inst);

      // Prepare to parse the operands for the extended instructions.
      spvPushOperandTypes(extInst->operandTypes, pExpectedOperands);
    } break;

    case SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER: {
      // The assembler accepts the symbolic name for the opcode, but without
      // the "Op" prefix.  For example, "IAdd" is accepted.  The number
      // of the opcode is emitted.
      SpvOp opcode;
      if (grammar.lookupSpecConstantOpcode(textValue, &opcode)) {
        return context->diagnostic() << "Invalid " << spvOperandTypeStr(type)
                                     << " '" << textValue << "'.";
      }
      spv_opcode_desc opcodeEntry = nullptr;
      if (grammar.lookupOpcode(opcode, &opcodeEntry)) {
        return context->diagnostic(SPV_ERROR_INTERNAL)
               << "OpSpecConstant opcode table out of sync";
      }
      spvInstructionAddWord(pInst, uint32_t(opcodeEntry->opcode));

      // Prepare to parse the operands for the opcode.  Except skip the
      // type Id and result Id, since they've already been processed.
      assert(opcodeEntry->hasType);
      assert(opcodeEntry->hasResult);
      assert(opcodeEntry->numTypes >= 2);
      spvPushOperandTypes(opcodeEntry->operandTypes + 2, pExpectedOperands);
    } break;

    case SPV_OPERAND_TYPE_LITERAL_INTEGER:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER: {
      // The current operand is an *unsigned* 32-bit integer.
      // That's just how the grammar works.
      spvtools::IdType expected_type = {
          32, false, spvtools::IdTypeClass::kScalarIntegerType};
      if (auto error = context->binaryEncodeNumericLiteral(
              textValue, error_code_for_literals, expected_type, pInst)) {
        return error;
      }
    } break;

    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_NUMBER:
      // This is a context-independent literal number which can be a 32-bit
      // number of floating point value.
      if (auto error = context->binaryEncodeNumericLiteral(
              textValue, error_code_for_literals, spvtools::kUnknownType,
              pInst)) {
        return error;
      }
      break;

    case SPV_OPERAND_TYPE_OPTIONAL_TYPED_LITERAL_INTEGER:
    case SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER: {
      spvtools::IdType expected_type = spvtools::kUnknownType;
      // The encoding for OpConstant, OpSpecConstant and OpSwitch all
      // depend on either their own result-id or the result-id of
      // one of their parameters.
      if (SpvOpConstant == pInst->opcode ||
          SpvOpSpecConstant == pInst->opcode) {
        // The type of the literal is determined by the type Id of the
        // instruction.
        expected_type =
            context->getTypeOfTypeGeneratingValue(pInst->resultTypeId);
        if (!spvtools::isScalarFloating(expected_type) &&
            !spvtools::isScalarIntegral(expected_type)) {
          spv_opcode_desc d;
          const char* opcode_name = "opcode";
          if (SPV_SUCCESS == grammar.lookupOpcode(pInst->opcode, &d)) {
            opcode_name = d->name;
          }
          return context->diagnostic()
                 << "Type for " << opcode_name
                 << " must be a scalar floating point or integer type";
        }
      } else if (pInst->opcode == SpvOpSwitch) {
        // The type of the literal is the same as the type of the selector.
        expected_type = context->getTypeOfValueInstruction(pInst->words[1]);
        if (!spvtools::isScalarIntegral(expected_type)) {
          return context->diagnostic()
                 << "The selector operand for OpSwitch must be the result"
                    " of an instruction that generates an integer scalar";
        }
      }
      if (auto error = context->binaryEncodeNumericLiteral(
              textValue, error_code_for_literals, expected_type, pInst)) {
        return error;
      }
    } break;

    case SPV_OPERAND_TYPE_LITERAL_STRING:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_STRING: {
      spv_literal_t literal = {};
      spv_result_t error = spvTextToLiteral(textValue, &literal);
      if (error != SPV_SUCCESS) {
        if (error == SPV_ERROR_OUT_OF_MEMORY) return error;
        return context->diagnostic(error_code_for_literals)
               << "Invalid literal string '" << textValue << "'.";
      }
      if (literal.type != SPV_LITERAL_TYPE_STRING) {
        return context->diagnostic()
               << "Expected literal string, found literal number '" << textValue
               << "'.";
      }

      // NOTE: Special case for extended instruction library import
      if (SpvOpExtInstImport == pInst->opcode) {
        const spv_ext_inst_type_t ext_inst_type =
            spvExtInstImportTypeGet(literal.str.c_str());
        if (SPV_EXT_INST_TYPE_NONE == ext_inst_type) {
          return context->diagnostic()
                 << "Invalid extended instruction import '" << literal.str
                 << "'";
        }
        if ((error = context->recordIdAsExtInstImport(pInst->words[1],
                                                      ext_inst_type)))
          return error;
      }

      if (context->binaryEncodeString(literal.str.c_str(), pInst))
        return SPV_ERROR_INVALID_TEXT;
    } break;

    // Masks.
    case SPV_OPERAND_TYPE_FP_FAST_MATH_MODE:
    case SPV_OPERAND_TYPE_FUNCTION_CONTROL:
    case SPV_OPERAND_TYPE_LOOP_CONTROL:
    case SPV_OPERAND_TYPE_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS:
    case SPV_OPERAND_TYPE_SELECTION_CONTROL:
    case SPV_OPERAND_TYPE_DEBUG_INFO_FLAGS: {
      uint32_t value;
      if (grammar.parseMaskOperand(type, textValue, &value)) {
        return context->diagnostic() << "Invalid " << spvOperandTypeStr(type)
                                     << " operand '" << textValue << "'.";
      }
      if (auto error = context->binaryEncodeU32(value, pInst)) return error;
      // Prepare to parse the operands for this logical operand.
      grammar.pushOperandTypesForMask(type, value, pExpectedOperands);
    } break;
    case SPV_OPERAND_TYPE_OPTIONAL_CIV: {
      auto error = spvTextEncodeOperand(
          grammar, context, SPV_OPERAND_TYPE_OPTIONAL_LITERAL_NUMBER, textValue,
          pInst, pExpectedOperands);
      if (error == SPV_FAILED_MATCH) {
        // It's not a literal number -- is it a literal string?
        error = spvTextEncodeOperand(grammar, context,
                                     SPV_OPERAND_TYPE_OPTIONAL_LITERAL_STRING,
                                     textValue, pInst, pExpectedOperands);
      }
      if (error == SPV_FAILED_MATCH) {
        // It's not a literal -- is it an ID?
        error =
            spvTextEncodeOperand(grammar, context, SPV_OPERAND_TYPE_OPTIONAL_ID,
                                 textValue, pInst, pExpectedOperands);
      }
      if (error) {
        return context->diagnostic(error)
               << "Invalid word following !<integer>: " << textValue;
      }
      if (pExpectedOperands->empty()) {
        pExpectedOperands->push_back(SPV_OPERAND_TYPE_OPTIONAL_CIV);
      }
    } break;
    default: {
      // NOTE: All non literal operands are handled here using the operand
      // table.
      spv_operand_desc entry;
      if (grammar.lookupOperand(type, textValue, strlen(textValue), &entry)) {
        return context->diagnostic() << "Invalid " << spvOperandTypeStr(type)
                                     << " '" << textValue << "'.";
      }
      if (context->binaryEncodeU32(entry->value, pInst)) {
        return context->diagnostic() << "Invalid " << spvOperandTypeStr(type)
                                     << " '" << textValue << "'.";
      }

      // Prepare to parse the operands for this logical operand.
      spvPushOperandTypes(entry->operandTypes, pExpectedOperands);
    } break;
  }
  return SPV_SUCCESS;
}

namespace {

/// Encodes an instruction started by !<integer> at the given position in text.
///
/// Puts the encoded words into *pInst.  If successful, moves position past the
/// instruction and returns SPV_SUCCESS.  Otherwise, returns an error code and
/// leaves position pointing to the error in text.
spv_result_t encodeInstructionStartingWithImmediate(
    const spvtools::AssemblyGrammar& grammar,
    spvtools::AssemblyContext* context, spv_instruction_t* pInst) {
  std::string firstWord;
  spv_position_t nextPosition = {};
  auto error = context->getWord(&firstWord, &nextPosition);
  if (error) return context->diagnostic(error) << "Internal Error";

  if ((error = encodeImmediate(context, firstWord.c_str(), pInst))) {
    return error;
  }
  while (context->advance() != SPV_END_OF_STREAM) {
    // A beginning of a new instruction means we're done.
    if (context->isStartOfNewInst()) return SPV_SUCCESS;

    // Otherwise, there must be an operand that's either a literal, an ID, or
    // an immediate.
    std::string operandValue;
    if ((error = context->getWord(&operandValue, &nextPosition)))
      return context->diagnostic(error) << "Internal Error";

    if (operandValue == "=")
      return context->diagnostic() << firstWord << " not allowed before =.";

    // Needed to pass to spvTextEncodeOpcode(), but it shouldn't ever be
    // expanded.
    spv_operand_pattern_t dummyExpectedOperands;
    error = spvTextEncodeOperand(
        grammar, context, SPV_OPERAND_TYPE_OPTIONAL_CIV, operandValue.c_str(),
        pInst, &dummyExpectedOperands);
    if (error) return error;
    context->setPosition(nextPosition);
  }
  return SPV_SUCCESS;
}

/// @brief Translate single Opcode and operands to binary form
///
/// @param[in] grammar the grammar to use for compilation
/// @param[in, out] context the dynamic compilation info
/// @param[in] text stream to translate
/// @param[out] pInst returned binary Opcode
/// @param[in,out] pPosition in the text stream
///
/// @return result code
spv_result_t spvTextEncodeOpcode(const spvtools::AssemblyGrammar& grammar,
                                 spvtools::AssemblyContext* context,
                                 spv_instruction_t* pInst) {
  // Check for !<integer> first.
  if ('!' == context->peek()) {
    return encodeInstructionStartingWithImmediate(grammar, context, pInst);
  }

  std::string firstWord;
  spv_position_t nextPosition = {};
  spv_result_t error = context->getWord(&firstWord, &nextPosition);
  if (error) return context->diagnostic() << "Internal Error";

  std::string opcodeName;
  std::string result_id;
  spv_position_t result_id_position = {};
  if (context->startsWithOp()) {
    opcodeName = firstWord;
  } else {
    result_id = firstWord;
    if ('%' != result_id.front()) {
      return context->diagnostic()
             << "Expected <opcode> or <result-id> at the beginning "
                "of an instruction, found '"
             << result_id << "'.";
    }
    result_id_position = context->position();

    // The '=' sign.
    context->setPosition(nextPosition);
    if (context->advance())
      return context->diagnostic() << "Expected '=', found end of stream.";
    std::string equal_sign;
    error = context->getWord(&equal_sign, &nextPosition);
    if ("=" != equal_sign)
      return context->diagnostic() << "'=' expected after result id.";

    // The <opcode> after the '=' sign.
    context->setPosition(nextPosition);
    if (context->advance())
      return context->diagnostic() << "Expected opcode, found end of stream.";
    error = context->getWord(&opcodeName, &nextPosition);
    if (error) return context->diagnostic(error) << "Internal Error";
    if (!context->startsWithOp()) {
      return context->diagnostic()
             << "Invalid Opcode prefix '" << opcodeName << "'.";
    }
  }

  // NOTE: The table contains Opcode names without the "Op" prefix.
  const char* pInstName = opcodeName.data() + 2;

  spv_opcode_desc opcodeEntry;
  error = grammar.lookupOpcode(pInstName, &opcodeEntry);
  if (error) {
    return context->diagnostic(error)
           << "Invalid Opcode name '" << opcodeName << "'";
  }
  if (opcodeEntry->hasResult && result_id.empty()) {
    return context->diagnostic()
           << "Expected <result-id> at the beginning of an instruction, found '"
           << firstWord << "'.";
  }
  pInst->opcode = opcodeEntry->opcode;
  context->setPosition(nextPosition);
  // Reserve the first word for the instruction.
  spvInstructionAddWord(pInst, 0);

  // Maintains the ordered list of expected operand types.
  // For many instructions we only need the {numTypes, operandTypes}
  // entries in opcodeEntry.  However, sometimes we need to modify
  // the list as we parse the operands. This occurs when an operand
  // has its own logical operands (such as the LocalSize operand for
  // ExecutionMode), or for extended instructions that may have their
  // own operands depending on the selected extended instruction.
  spv_operand_pattern_t expectedOperands;
  expectedOperands.reserve(opcodeEntry->numTypes);
  for (auto i = 0; i < opcodeEntry->numTypes; i++)
    expectedOperands.push_back(
        opcodeEntry->operandTypes[opcodeEntry->numTypes - i - 1]);

  while (!expectedOperands.empty()) {
    const spv_operand_type_t type = expectedOperands.back();
    expectedOperands.pop_back();

    // Expand optional tuples lazily.
    if (spvExpandOperandSequenceOnce(type, &expectedOperands)) continue;

    if (type == SPV_OPERAND_TYPE_RESULT_ID && !result_id.empty()) {
      // Handle the <result-id> for value generating instructions.
      // We've already consumed it from the text stream.  Here
      // we inject its words into the instruction.
      spv_position_t temp_pos = context->position();
      error = spvTextEncodeOperand(grammar, context, SPV_OPERAND_TYPE_RESULT_ID,
                                   result_id.c_str(), pInst, nullptr);
      result_id_position = context->position();
      // Because we are injecting we have to reset the position afterwards.
      context->setPosition(temp_pos);
      if (error) return error;
    } else {
      // Find the next word.
      error = context->advance();
      if (error == SPV_END_OF_STREAM) {
        if (spvOperandIsOptional(type)) {
          // This would have been the last potential operand for the
          // instruction,
          // and we didn't find one.  We're finished parsing this instruction.
          break;
        } else {
          return context->diagnostic()
                 << "Expected operand, found end of stream.";
        }
      }
      assert(error == SPV_SUCCESS && "Somebody added another way to fail");

      if (context->isStartOfNewInst()) {
        if (spvOperandIsOptional(type)) {
          break;
        } else {
          return context->diagnostic()
                 << "Expected operand, found next instruction instead.";
        }
      }

      std::string operandValue;
      error = context->getWord(&operandValue, &nextPosition);
      if (error) return context->diagnostic(error) << "Internal Error";

      error = spvTextEncodeOperand(grammar, context, type, operandValue.c_str(),
                                   pInst, &expectedOperands);

      if (error == SPV_FAILED_MATCH && spvOperandIsOptional(type))
        return SPV_SUCCESS;

      if (error) return error;

      context->setPosition(nextPosition);
    }
  }

  if (spvOpcodeGeneratesType(pInst->opcode)) {
    if (context->recordTypeDefinition(pInst) != SPV_SUCCESS) {
      return SPV_ERROR_INVALID_TEXT;
    }
  } else if (opcodeEntry->hasType) {
    // SPIR-V dictates that if an instruction has both a return value and a
    // type ID then the type id is first, and the return value is second.
    assert(opcodeEntry->hasResult &&
           "Unknown opcode: has a type but no result.");
    context->recordTypeIdForValue(pInst->words[2], pInst->words[1]);
  }

  if (pInst->words.size() > SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX) {
    return context->diagnostic()
           << "Instruction too long: " << pInst->words.size()
           << " words, but the limit is "
           << SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX;
  }

  pInst->words[0] =
      spvOpcodeMake(uint16_t(pInst->words.size()), opcodeEntry->opcode);

  return SPV_SUCCESS;
}

enum { kAssemblerVersion = 0 };

// Populates a binary stream's |header|. The target environment is specified via
// |env| and Id bound is via |bound|.
spv_result_t SetHeader(spv_target_env env, const uint32_t bound,
                       uint32_t* header) {
  if (!header) return SPV_ERROR_INVALID_BINARY;

  header[SPV_INDEX_MAGIC_NUMBER] = SpvMagicNumber;
  header[SPV_INDEX_VERSION_NUMBER] = spvVersionForTargetEnv(env);
  header[SPV_INDEX_GENERATOR_NUMBER] =
      SPV_GENERATOR_WORD(SPV_GENERATOR_KHRONOS_ASSEMBLER, kAssemblerVersion);
  header[SPV_INDEX_BOUND] = bound;
  header[SPV_INDEX_SCHEMA] = 0;  // NOTE: Reserved

  return SPV_SUCCESS;
}

// Collects all numeric ids in the module source into |numeric_ids|.
// This function is essentially a dry-run of spvTextToBinary.
spv_result_t GetNumericIds(const spvtools::AssemblyGrammar& grammar,
                           const spvtools::MessageConsumer& consumer,
                           const spv_text text,
                           std::set<uint32_t>* numeric_ids) {
  spvtools::AssemblyContext context(text, consumer);

  if (!text->str) return context.diagnostic() << "Missing assembly text.";

  if (!grammar.isValid()) {
    return SPV_ERROR_INVALID_TABLE;
  }

  // Skip past whitespace and comments.
  context.advance();

  while (context.hasText()) {
    spv_instruction_t inst;

    if (spvTextEncodeOpcode(grammar, &context, &inst)) {
      return SPV_ERROR_INVALID_TEXT;
    }

    if (context.advance()) break;
  }

  *numeric_ids = context.GetNumericIds();
  return SPV_SUCCESS;
}

// Translates a given assembly language module into binary form.
// If a diagnostic is generated, it is not yet marked as being
// for a text-based input.
spv_result_t spvTextToBinaryInternal(const spvtools::AssemblyGrammar& grammar,
                                     const spvtools::MessageConsumer& consumer,
                                     const spv_text text,
                                     const uint32_t options,
                                     spv_binary* pBinary) {
  // The ids in this set will have the same values both in source and binary.
  // All other ids will be generated by filling in the gaps.
  std::set<uint32_t> ids_to_preserve;

  if (options & SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS) {
    // Collect all numeric ids from the source into ids_to_preserve.
    const spv_result_t result =
        GetNumericIds(grammar, consumer, text, &ids_to_preserve);
    if (result != SPV_SUCCESS) return result;
  }

  spvtools::AssemblyContext context(text, consumer, std::move(ids_to_preserve));

  if (!text->str) return context.diagnostic() << "Missing assembly text.";

  if (!grammar.isValid()) {
    return SPV_ERROR_INVALID_TABLE;
  }
  if (!pBinary) return SPV_ERROR_INVALID_POINTER;

  std::vector<spv_instruction_t> instructions;

  // Skip past whitespace and comments.
  context.advance();

  while (context.hasText()) {
    instructions.push_back({});
    spv_instruction_t& inst = instructions.back();

    if (spvTextEncodeOpcode(grammar, &context, &inst)) {
      return SPV_ERROR_INVALID_TEXT;
    }

    if (context.advance()) break;
  }

  size_t totalSize = SPV_INDEX_INSTRUCTION;
  for (auto& inst : instructions) {
    totalSize += inst.words.size();
  }

  uint32_t* data = new uint32_t[totalSize];
  if (!data) return SPV_ERROR_OUT_OF_MEMORY;
  uint64_t currentIndex = SPV_INDEX_INSTRUCTION;
  for (auto& inst : instructions) {
    memcpy(data + currentIndex, inst.words.data(),
           sizeof(uint32_t) * inst.words.size());
    currentIndex += inst.words.size();
  }

  if (auto error = SetHeader(grammar.target_env(), context.getBound(), data))
    return error;

  spv_binary binary = new spv_binary_t();
  if (!binary) {
    delete[] data;
    return SPV_ERROR_OUT_OF_MEMORY;
  }
  binary->code = data;
  binary->wordCount = totalSize;

  *pBinary = binary;

  return SPV_SUCCESS;
}

}  // anonymous namespace

spv_result_t spvTextToBinary(const spv_const_context context,
                             const char* input_text,
                             const size_t input_text_size, spv_binary* pBinary,
                             spv_diagnostic* pDiagnostic) {
  return spvTextToBinaryWithOptions(context, input_text, input_text_size,
                                    SPV_TEXT_TO_BINARY_OPTION_NONE, pBinary,
                                    pDiagnostic);
}

spv_result_t spvTextToBinaryWithOptions(const spv_const_context context,
                                        const char* input_text,
                                        const size_t input_text_size,
                                        const uint32_t options,
                                        spv_binary* pBinary,
                                        spv_diagnostic* pDiagnostic) {
  spv_context_t hijack_context = *context;
  if (pDiagnostic) {
    *pDiagnostic = nullptr;
    spvtools::UseDiagnosticAsMessageConsumer(&hijack_context, pDiagnostic);
  }

  spv_text_t text = {input_text, input_text_size};
  spvtools::AssemblyGrammar grammar(&hijack_context);

  spv_result_t result = spvTextToBinaryInternal(
      grammar, hijack_context.consumer, &text, options, pBinary);
  if (pDiagnostic && *pDiagnostic) (*pDiagnostic)->isTextSource = true;

  return result;
}

void spvTextDestroy(spv_text text) {
  if (!text) return;
  delete[] text->str;
  delete text;
}
