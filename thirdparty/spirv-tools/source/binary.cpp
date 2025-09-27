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

#include "source/binary.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "source/assembly_grammar.h"
#include "source/diagnostic.h"
#include "source/ext_inst.h"
#include "source/latest_version_spirv_header.h"
#include "source/opcode.h"
#include "source/operand.h"
#include "source/spirv_constant.h"
#include "source/spirv_endian.h"
#include "source/util/string_utils.h"

spv_result_t spvBinaryHeaderGet(const spv_const_binary binary,
                                const spv_endianness_t endian,
                                spv_header_t* pHeader) {
  if (!binary->code) return SPV_ERROR_INVALID_BINARY;
  if (binary->wordCount < SPV_INDEX_INSTRUCTION)
    return SPV_ERROR_INVALID_BINARY;
  if (!pHeader) return SPV_ERROR_INVALID_POINTER;

  // TODO: Validation checking?
  pHeader->magic = spvFixWord(binary->code[SPV_INDEX_MAGIC_NUMBER], endian);
  pHeader->version = spvFixWord(binary->code[SPV_INDEX_VERSION_NUMBER], endian);
  // Per 2.3.1 version's high and low bytes are 0
  if ((pHeader->version & 0x000000ff) || pHeader->version & 0xff000000)
    return SPV_ERROR_INVALID_BINARY;
  // Minimum version was 1.0 and max version is defined by SPV_VERSION.
  if (pHeader->version < SPV_SPIRV_VERSION_WORD(1, 0) ||
      pHeader->version > SPV_VERSION)
    return SPV_ERROR_INVALID_BINARY;

  pHeader->generator =
      spvFixWord(binary->code[SPV_INDEX_GENERATOR_NUMBER], endian);
  pHeader->bound = spvFixWord(binary->code[SPV_INDEX_BOUND], endian);
  pHeader->schema = spvFixWord(binary->code[SPV_INDEX_SCHEMA], endian);
  pHeader->instructions = &binary->code[SPV_INDEX_INSTRUCTION];

  return SPV_SUCCESS;
}

std::string spvDecodeLiteralStringOperand(const spv_parsed_instruction_t& inst,
                                          const uint16_t operand_index) {
  assert(operand_index < inst.num_operands);
  const spv_parsed_operand_t& operand = inst.operands[operand_index];

  return spvtools::utils::MakeString(inst.words + operand.offset,
                                     operand.num_words);
}

namespace {

// A SPIR-V binary parser.  A parser instance communicates detailed parse
// results via callbacks.
class Parser {
 public:
  // The user_data value is provided to the callbacks as context.
  Parser(const spv_const_context context, void* user_data,
         spv_parsed_header_fn_t parsed_header_fn,
         spv_parsed_instruction_fn_t parsed_instruction_fn)
      : grammar_(context),
        consumer_(context->consumer),
        user_data_(user_data),
        parsed_header_fn_(parsed_header_fn),
        parsed_instruction_fn_(parsed_instruction_fn) {}

  // Parses the specified binary SPIR-V module, issuing callbacks on a parsed
  // header and for each parsed instruction.  Returns SPV_SUCCESS on success.
  // Otherwise returns an error code and issues a diagnostic.
  spv_result_t parse(const uint32_t* words, size_t num_words,
                     spv_diagnostic* diagnostic);

 private:
  // All remaining methods work on the current module parse state.

  // Like the parse method, but works on the current module parse state.
  spv_result_t parseModule();

  // Parses an instruction at the current position of the binary.  Assumes
  // the header has been parsed, the endian has been set, and the word index is
  // still in range.  Advances the parsing position past the instruction, and
  // updates other parsing state for the current module.
  // On success, returns SPV_SUCCESS and issues the parsed-instruction callback.
  // On failure, returns an error code and issues a diagnostic.
  spv_result_t parseInstruction();

  // Parses an instruction operand with the given type, for an instruction
  // starting at inst_offset words into the SPIR-V binary.
  // If the SPIR-V binary is the same endianness as the host, then the
  // endian_converted_inst_words parameter is ignored.  Otherwise, this method
  // appends the words for this operand, converted to host native endianness,
  // to the end of endian_converted_inst_words.  This method also updates the
  // expected_operands parameter, and the scalar members of the inst parameter.
  // On success, returns SPV_SUCCESS, advances past the operand, and pushes a
  // new entry on to the operands vector.  Otherwise returns an error code and
  // issues a diagnostic.
  spv_result_t parseOperand(size_t inst_offset, spv_parsed_instruction_t* inst,
                            const spv_operand_type_t type,
                            std::vector<uint32_t>* endian_converted_inst_words,
                            std::vector<spv_parsed_operand_t>* operands,
                            spv_operand_pattern_t* expected_operands);

  // Records the numeric type for an operand according to the type information
  // associated with the given non-zero type Id.  This can fail if the type Id
  // is not a type Id, or if the type Id does not reference a scalar numeric
  // type.  On success, return SPV_SUCCESS and populates the num_words,
  // number_kind, and number_bit_width fields of parsed_operand.
  spv_result_t setNumericTypeInfoForType(spv_parsed_operand_t* parsed_operand,
                                         uint32_t type_id);

  // Records the number type for an instruction at the given offset, if that
  // instruction generates a type.  For types that aren't scalar numbers,
  // record something with number kind SPV_NUMBER_NONE.
  void recordNumberType(size_t inst_offset,
                        const spv_parsed_instruction_t* inst);

  // Returns a diagnostic stream object initialized with current position in
  // the input stream, and for the given error code. Any data written to the
  // returned object will be propagated to the current parse's diagnostic
  // object.
  spvtools::DiagnosticStream diagnostic(spv_result_t error) {
    return spvtools::DiagnosticStream({0, 0, _.instruction_count}, consumer_,
                                      "", error);
  }

  // Returns a diagnostic stream object with the default parse error code.
  spvtools::DiagnosticStream diagnostic() {
    // The default failure for parsing is invalid binary.
    return diagnostic(SPV_ERROR_INVALID_BINARY);
  }

  // Issues a diagnostic describing an exhaustion of input condition when
  // trying to decode an instruction operand, and returns
  // SPV_ERROR_INVALID_BINARY.
  spv_result_t exhaustedInputDiagnostic(size_t inst_offset, spv::Op opcode,
                                        spv_operand_type_t type) {
    return diagnostic() << "End of input reached while decoding Op"
                        << spvOpcodeString(opcode) << " starting at word "
                        << inst_offset
                        << ((_.word_index < _.num_words) ? ": truncated "
                                                         : ": missing ")
                        << spvOperandTypeStr(type) << " operand at word offset "
                        << _.word_index - inst_offset << ".";
  }

  // Returns the endian-corrected word at the current position.
  uint32_t peek() const { return peekAt(_.word_index); }

  // Returns the endian-corrected word at the given position.
  uint32_t peekAt(size_t index) const {
    assert(index < _.num_words);
    return spvFixWord(_.words[index], _.endian);
  }

  // Data members

  const spvtools::AssemblyGrammar grammar_;        // SPIR-V syntax utility.
  const spvtools::MessageConsumer& consumer_;      // Message consumer callback.
  void* const user_data_;                          // Context for the callbacks
  const spv_parsed_header_fn_t parsed_header_fn_;  // Parsed header callback
  const spv_parsed_instruction_fn_t
      parsed_instruction_fn_;  // Parsed instruction callback

  // Describes the format of a typed literal number.
  struct NumberType {
    spv_number_kind_t type;
    uint32_t bit_width;
  };

  // The state used to parse a single SPIR-V binary module.
  struct State {
    State(const uint32_t* words_arg, size_t num_words_arg,
          spv_diagnostic* diagnostic_arg)
        : words(words_arg),
          num_words(num_words_arg),
          diagnostic(diagnostic_arg),
          word_index(0),
          instruction_count(0),
          endian(),
          requires_endian_conversion(false) {
      // Temporary storage for parser state within a single instruction.
      // Most instructions require fewer than 25 words or operands.
      operands.reserve(25);
      endian_converted_words.reserve(25);
      expected_operands.reserve(25);
    }
    State() : State(0, 0, nullptr) {}
    const uint32_t* words;       // Words in the binary SPIR-V module.
    size_t num_words;            // Number of words in the module.
    spv_diagnostic* diagnostic;  // Where diagnostics go.
    size_t word_index;           // The current position in words.
    size_t instruction_count;    // The count of processed instructions
    spv_endianness_t endian;     // The endianness of the binary.
    // Is the SPIR-V binary in a different endianness from the host native
    // endianness?
    bool requires_endian_conversion;

    // Maps a result ID to its type ID.  By convention:
    //  - a result ID that is a type definition maps to itself.
    //  - a result ID without a type maps to 0.  (E.g. for OpLabel)
    std::unordered_map<uint32_t, uint32_t> id_to_type_id;
    // Maps a type ID to its number type description.
    std::unordered_map<uint32_t, NumberType> type_id_to_number_type_info;
    // Maps an ExtInstImport id to the extended instruction type.
    std::unordered_map<uint32_t, spv_ext_inst_type_t>
        import_id_to_ext_inst_type;

    // Used by parseOperand
    std::vector<spv_parsed_operand_t> operands;
    std::vector<uint32_t> endian_converted_words;
    spv_operand_pattern_t expected_operands;
  } _;
};

spv_result_t Parser::parse(const uint32_t* words, size_t num_words,
                           spv_diagnostic* diagnostic_arg) {
  _ = State(words, num_words, diagnostic_arg);

  const spv_result_t result = parseModule();

  // Clear the module state.  The tables might be big.
  _ = State();

  return result;
}

spv_result_t Parser::parseModule() {
  if (!_.words) return diagnostic() << "Missing module.";

  if (_.num_words < SPV_INDEX_INSTRUCTION)
    return diagnostic() << "Module has incomplete header: only " << _.num_words
                        << " words instead of " << SPV_INDEX_INSTRUCTION;

  // Check the magic number and detect the module's endianness.
  spv_const_binary_t binary{_.words, _.num_words};
  if (spvBinaryEndianness(&binary, &_.endian)) {
    return diagnostic() << "Invalid SPIR-V magic number '" << std::hex
                        << _.words[0] << "'.";
  }
  _.requires_endian_conversion = !spvIsHostEndian(_.endian);

  // Process the header.
  spv_header_t header;
  if (spvBinaryHeaderGet(&binary, _.endian, &header)) {
    // It turns out there is no way to trigger this error since the only
    // failure cases are already handled above, with better messages.
    return diagnostic(SPV_ERROR_INTERNAL)
           << "Internal error: unhandled header parse failure";
  }
  if (parsed_header_fn_) {
    if (auto error = parsed_header_fn_(user_data_, _.endian, header.magic,
                                       header.version, header.generator,
                                       header.bound, header.schema)) {
      return error;
    }
  }

  // Process the instructions.
  _.word_index = SPV_INDEX_INSTRUCTION;
  while (_.word_index < _.num_words)
    if (auto error = parseInstruction()) return error;

  // Running off the end should already have been reported earlier.
  assert(_.word_index == _.num_words);

  return SPV_SUCCESS;
}

spv_result_t Parser::parseInstruction() {
  _.instruction_count++;

  // The zero values for all members except for opcode are the
  // correct initial values.
  spv_parsed_instruction_t inst = {};

  const uint32_t first_word = peek();

  // If the module's endianness is different from the host native endianness,
  // then converted_words contains the endian-translated words in the
  // instruction.
  _.endian_converted_words.clear();
  _.endian_converted_words.push_back(first_word);

  // After a successful parse of the instruction, the inst.operands member
  // will point to this vector's storage.
  _.operands.clear();

  assert(_.word_index < _.num_words);
  // Decompose and check the first word.
  uint16_t inst_word_count = 0;
  spvOpcodeSplit(first_word, &inst_word_count, &inst.opcode);
  if (inst_word_count < 1) {
    return diagnostic() << "Invalid instruction word count: "
                        << inst_word_count;
  }
  spv_opcode_desc opcode_desc;
  if (grammar_.lookupOpcode(static_cast<spv::Op>(inst.opcode), &opcode_desc))
    return diagnostic() << "Invalid opcode: " << inst.opcode;

  // Advance past the opcode word.  But remember the of the start
  // of the instruction.
  const size_t inst_offset = _.word_index;
  _.word_index++;

  // Maintains the ordered list of expected operand types.
  // For many instructions we only need the {numTypes, operandTypes}
  // entries in opcode_desc.  However, sometimes we need to modify
  // the list as we parse the operands. This occurs when an operand
  // has its own logical operands (such as the LocalSize operand for
  // ExecutionMode), or for extended instructions that may have their
  // own operands depending on the selected extended instruction.
  _.expected_operands.clear();
  for (auto i = 0; i < opcode_desc->numTypes; i++)
    _.expected_operands.push_back(
        opcode_desc->operandTypes[opcode_desc->numTypes - i - 1]);

  while (_.word_index < inst_offset + inst_word_count) {
    const uint16_t inst_word_index = uint16_t(_.word_index - inst_offset);
    if (_.expected_operands.empty()) {
      return diagnostic() << "Invalid instruction Op" << opcode_desc->name
                          << " starting at word " << inst_offset
                          << ": expected no more operands after "
                          << inst_word_index
                          << " words, but stated word count is "
                          << inst_word_count << ".";
    }

    spv_operand_type_t type =
        spvTakeFirstMatchableOperand(&_.expected_operands);

    if (auto error =
            parseOperand(inst_offset, &inst, type, &_.endian_converted_words,
                         &_.operands, &_.expected_operands)) {
      return error;
    }
  }

  if (!_.expected_operands.empty() &&
      !spvOperandIsOptional(_.expected_operands.back())) {
    return diagnostic() << "End of input reached while decoding Op"
                        << opcode_desc->name << " starting at word "
                        << inst_offset << ": expected more operands after "
                        << inst_word_count << " words.";
  }

  if ((inst_offset + inst_word_count) != _.word_index) {
    return diagnostic() << "Invalid word count: Op" << opcode_desc->name
                        << " starting at word " << inst_offset
                        << " says it has " << inst_word_count
                        << " words, but found " << _.word_index - inst_offset
                        << " words instead.";
  }

  // Check the computed length of the endian-converted words vector against
  // the declared number of words in the instruction.  If endian conversion
  // is required, then they should match.  If no endian conversion was
  // performed, then the vector only contains the initial opcode/word-count
  // word.
  assert(!_.requires_endian_conversion ||
         (inst_word_count == _.endian_converted_words.size()));
  assert(_.requires_endian_conversion ||
         (_.endian_converted_words.size() == 1));

  recordNumberType(inst_offset, &inst);

  if (_.requires_endian_conversion) {
    // We must wait until here to set this pointer, because the vector might
    // have been be resized while we accumulated its elements.
    inst.words = _.endian_converted_words.data();
  } else {
    // If no conversion is required, then just point to the underlying binary.
    // This saves time and space.
    inst.words = _.words + inst_offset;
  }
  inst.num_words = inst_word_count;

  // We must wait until here to set this pointer, because the vector might
  // have been be resized while we accumulated its elements.
  inst.operands = _.operands.data();
  inst.num_operands = uint16_t(_.operands.size());

  // Issue the callback.  The callee should know that all the storage in inst
  // is transient, and will disappear immediately afterward.
  if (parsed_instruction_fn_) {
    if (auto error = parsed_instruction_fn_(user_data_, &inst)) return error;
  }

  return SPV_SUCCESS;
}

spv_result_t Parser::parseOperand(size_t inst_offset,
                                  spv_parsed_instruction_t* inst,
                                  const spv_operand_type_t type,
                                  std::vector<uint32_t>* words,
                                  std::vector<spv_parsed_operand_t>* operands,
                                  spv_operand_pattern_t* expected_operands) {
  const spv::Op opcode = static_cast<spv::Op>(inst->opcode);
  // We'll fill in this result as we go along.
  spv_parsed_operand_t parsed_operand;
  parsed_operand.offset = uint16_t(_.word_index - inst_offset);
  // Most operands occupy one word.  This might be be adjusted later.
  parsed_operand.num_words = 1;
  // The type argument is the one used by the grammar to parse the instruction.
  // But it can exposes internal parser details such as whether an operand is
  // optional or actually represents a variable-length sequence of operands.
  // The resulting type should be adjusted to avoid those internal details.
  // In most cases, the resulting operand type is the same as the grammar type.
  parsed_operand.type = type;

  // Assume non-numeric values.  This will be updated for literal numbers.
  parsed_operand.number_kind = SPV_NUMBER_NONE;
  parsed_operand.number_bit_width = 0;

  if (_.word_index >= _.num_words)
    return exhaustedInputDiagnostic(inst_offset, opcode, type);

  const uint32_t word = peek();

  // Do the words in this operand have to be converted to native endianness?
  // True for all but literal strings.
  bool convert_operand_endianness = true;

  switch (type) {
    case SPV_OPERAND_TYPE_TYPE_ID:
      if (!word)
        return diagnostic(SPV_ERROR_INVALID_ID) << "Error: Type Id is 0";
      inst->type_id = word;
      break;

    case SPV_OPERAND_TYPE_RESULT_ID:
      if (!word)
        return diagnostic(SPV_ERROR_INVALID_ID) << "Error: Result Id is 0";
      inst->result_id = word;
      // Save the result ID to type ID mapping.
      // In the grammar, type ID always appears before result ID.
      if (_.id_to_type_id.find(inst->result_id) != _.id_to_type_id.end())
        return diagnostic(SPV_ERROR_INVALID_ID)
               << "Id " << inst->result_id << " is defined more than once";
      // Record it.
      // A regular value maps to its type.  Some instructions (e.g. OpLabel)
      // have no type Id, and will map to 0.  The result Id for a
      // type-generating instruction (e.g. OpTypeInt) maps to itself.
      _.id_to_type_id[inst->result_id] =
          spvOpcodeGeneratesType(opcode) ? inst->result_id : inst->type_id;
      break;

    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_OPTIONAL_ID:
      if (!word) return diagnostic(SPV_ERROR_INVALID_ID) << "Id is 0";
      parsed_operand.type = SPV_OPERAND_TYPE_ID;

      if (opcode == spv::Op::OpExtInst && parsed_operand.offset == 3) {
        // The current word is the extended instruction set Id.
        // Set the extended instruction set type for the current instruction.
        auto ext_inst_type_iter = _.import_id_to_ext_inst_type.find(word);
        if (ext_inst_type_iter == _.import_id_to_ext_inst_type.end()) {
          return diagnostic(SPV_ERROR_INVALID_ID)
                 << "OpExtInst set Id " << word
                 << " does not reference an OpExtInstImport result Id";
        }
        inst->ext_inst_type = ext_inst_type_iter->second;
      }
      break;

    case SPV_OPERAND_TYPE_SCOPE_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
      // Check for trivially invalid values.  The operand descriptions already
      // have the word "ID" in them.
      if (!word) return diagnostic() << spvOperandTypeStr(type) << " is 0";
      break;

    case SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER: {
      assert(spv::Op::OpExtInst == opcode);
      assert(inst->ext_inst_type != SPV_EXT_INST_TYPE_NONE);
      spv_ext_inst_desc ext_inst;
      if (grammar_.lookupExtInst(inst->ext_inst_type, word, &ext_inst) ==
          SPV_SUCCESS) {
        // if we know about this ext inst, push the expected operands
        spvPushOperandTypes(ext_inst->operandTypes, expected_operands);
      } else {
        // if we don't know this extended instruction and the set isn't
        // non-semantic, we cannot process further
        if (!spvExtInstIsNonSemantic(inst->ext_inst_type)) {
          return diagnostic()
                 << "Invalid extended instruction number: " << word;
        } else {
          // for non-semantic instruction sets, we know the form of all such
          // extended instructions contains a series of IDs as parameters
          expected_operands->push_back(SPV_OPERAND_TYPE_VARIABLE_ID);
        }
      }
    } break;

    case SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER: {
      assert(spv::Op::OpSpecConstantOp == opcode);
      if (word > static_cast<uint32_t>(spv::Op::Max) ||
          grammar_.lookupSpecConstantOpcode(spv::Op(word))) {
        return diagnostic()
               << "Invalid " << spvOperandTypeStr(type) << ": " << word;
      }
      spv_opcode_desc opcode_entry = nullptr;
      if (grammar_.lookupOpcode(spv::Op(word), &opcode_entry)) {
        return diagnostic(SPV_ERROR_INTERNAL)
               << "OpSpecConstant opcode table out of sync";
      }
      // OpSpecConstant opcodes must have a type and result. We've already
      // processed them, so skip them when preparing to parse the other
      // operants for the opcode.
      assert(opcode_entry->hasType);
      assert(opcode_entry->hasResult);
      assert(opcode_entry->numTypes >= 2);
      spvPushOperandTypes(opcode_entry->operandTypes + 2, expected_operands);
    } break;

    case SPV_OPERAND_TYPE_LITERAL_INTEGER:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER:
      // These are regular single-word literal integer operands.
      // Post-parsing validation should check the range of the parsed value.
      parsed_operand.type = SPV_OPERAND_TYPE_LITERAL_INTEGER;
      // It turns out they are always unsigned integers!
      parsed_operand.number_kind = SPV_NUMBER_UNSIGNED_INT;
      parsed_operand.number_bit_width = 32;
      break;

    case SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER:
    case SPV_OPERAND_TYPE_OPTIONAL_TYPED_LITERAL_INTEGER:
      parsed_operand.type = SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER;
      if (opcode == spv::Op::OpSwitch) {
        // The literal operands have the same type as the value
        // referenced by the selector Id.
        const uint32_t selector_id = peekAt(inst_offset + 1);
        const auto type_id_iter = _.id_to_type_id.find(selector_id);
        if (type_id_iter == _.id_to_type_id.end() ||
            type_id_iter->second == 0) {
          return diagnostic() << "Invalid OpSwitch: selector id " << selector_id
                              << " has no type";
        }
        uint32_t type_id = type_id_iter->second;

        if (selector_id == type_id) {
          // Recall that by convention, a result ID that is a type definition
          // maps to itself.
          return diagnostic() << "Invalid OpSwitch: selector id " << selector_id
                              << " is a type, not a value";
        }
        if (auto error = setNumericTypeInfoForType(&parsed_operand, type_id))
          return error;
        if (parsed_operand.number_kind != SPV_NUMBER_UNSIGNED_INT &&
            parsed_operand.number_kind != SPV_NUMBER_SIGNED_INT) {
          return diagnostic() << "Invalid OpSwitch: selector id " << selector_id
                              << " is not a scalar integer";
        }
      } else {
        assert(opcode == spv::Op::OpConstant ||
               opcode == spv::Op::OpSpecConstant);
        // The literal number type is determined by the type Id for the
        // constant.
        assert(inst->type_id);
        if (auto error =
                setNumericTypeInfoForType(&parsed_operand, inst->type_id))
          return error;
      }
      break;

    case SPV_OPERAND_TYPE_LITERAL_STRING:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_STRING: {
      const size_t max_words = _.num_words - _.word_index;
      std::string string =
          spvtools::utils::MakeString(_.words + _.word_index, max_words, false);

      if (string.length() == max_words * 4)
        return exhaustedInputDiagnostic(inst_offset, opcode, type);

      // Make sure we can record the word count without overflow.
      //
      // This error can't currently be triggered because of validity
      // checks elsewhere.
      const size_t string_num_words = string.length() / 4 + 1;
      if (string_num_words > std::numeric_limits<uint16_t>::max()) {
        return diagnostic() << "Literal string is longer than "
                            << std::numeric_limits<uint16_t>::max()
                            << " words: " << string_num_words << " words long";
      }
      parsed_operand.num_words = uint16_t(string_num_words);
      parsed_operand.type = SPV_OPERAND_TYPE_LITERAL_STRING;

      if (spv::Op::OpExtInstImport == opcode) {
        // Record the extended instruction type for the ID for this import.
        // There is only one string literal argument to OpExtInstImport,
        // so it's sufficient to guard this just on the opcode.
        const spv_ext_inst_type_t ext_inst_type =
            spvExtInstImportTypeGet(string.c_str());
        if (SPV_EXT_INST_TYPE_NONE == ext_inst_type) {
          return diagnostic()
                 << "Invalid extended instruction import '" << string << "'";
        }
        // We must have parsed a valid result ID.  It's a condition
        // of the grammar, and we only accept non-zero result Ids.
        assert(inst->result_id);
        _.import_id_to_ext_inst_type[inst->result_id] = ext_inst_type;
      }
    } break;

    case SPV_OPERAND_TYPE_CAPABILITY:
    case SPV_OPERAND_TYPE_SOURCE_LANGUAGE:
    case SPV_OPERAND_TYPE_EXECUTION_MODEL:
    case SPV_OPERAND_TYPE_ADDRESSING_MODEL:
    case SPV_OPERAND_TYPE_MEMORY_MODEL:
    case SPV_OPERAND_TYPE_EXECUTION_MODE:
    case SPV_OPERAND_TYPE_STORAGE_CLASS:
    case SPV_OPERAND_TYPE_DIMENSIONALITY:
    case SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE:
    case SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE:
    case SPV_OPERAND_TYPE_SAMPLER_IMAGE_FORMAT:
    case SPV_OPERAND_TYPE_FP_ROUNDING_MODE:
    case SPV_OPERAND_TYPE_LINKAGE_TYPE:
    case SPV_OPERAND_TYPE_ACCESS_QUALIFIER:
    case SPV_OPERAND_TYPE_OPTIONAL_ACCESS_QUALIFIER:
    case SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE:
    case SPV_OPERAND_TYPE_DECORATION:
    case SPV_OPERAND_TYPE_BUILT_IN:
    case SPV_OPERAND_TYPE_GROUP_OPERATION:
    case SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS:
    case SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO:
    case SPV_OPERAND_TYPE_RAY_FLAGS:
    case SPV_OPERAND_TYPE_RAY_QUERY_INTERSECTION:
    case SPV_OPERAND_TYPE_RAY_QUERY_COMMITTED_INTERSECTION_TYPE:
    case SPV_OPERAND_TYPE_RAY_QUERY_CANDIDATE_INTERSECTION_TYPE:
    case SPV_OPERAND_TYPE_DEBUG_BASE_TYPE_ATTRIBUTE_ENCODING:
    case SPV_OPERAND_TYPE_DEBUG_COMPOSITE_TYPE:
    case SPV_OPERAND_TYPE_DEBUG_TYPE_QUALIFIER:
    case SPV_OPERAND_TYPE_DEBUG_OPERATION:
    case SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_BASE_TYPE_ATTRIBUTE_ENCODING:
    case SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_COMPOSITE_TYPE:
    case SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_TYPE_QUALIFIER:
    case SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_OPERATION:
    case SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_IMPORTED_ENTITY:
    case SPV_OPERAND_TYPE_FPDENORM_MODE:
    case SPV_OPERAND_TYPE_FPOPERATION_MODE:
    case SPV_OPERAND_TYPE_QUANTIZATION_MODES:
    case SPV_OPERAND_TYPE_OVERFLOW_MODES:
    case SPV_OPERAND_TYPE_PACKED_VECTOR_FORMAT:
    case SPV_OPERAND_TYPE_OPTIONAL_PACKED_VECTOR_FORMAT: {
      // A single word that is a plain enum value.

      // Map an optional operand type to its corresponding concrete type.
      if (type == SPV_OPERAND_TYPE_OPTIONAL_ACCESS_QUALIFIER)
        parsed_operand.type = SPV_OPERAND_TYPE_ACCESS_QUALIFIER;
      if (type == SPV_OPERAND_TYPE_OPTIONAL_PACKED_VECTOR_FORMAT)
        parsed_operand.type = SPV_OPERAND_TYPE_PACKED_VECTOR_FORMAT;

      spv_operand_desc entry;
      if (grammar_.lookupOperand(type, word, &entry)) {
        return diagnostic()
               << "Invalid " << spvOperandTypeStr(parsed_operand.type)
               << " operand: " << word;
      }
      // Prepare to accept operands to this operand, if needed.
      spvPushOperandTypes(entry->operandTypes, expected_operands);
    } break;

    case SPV_OPERAND_TYPE_FP_FAST_MATH_MODE:
    case SPV_OPERAND_TYPE_FUNCTION_CONTROL:
    case SPV_OPERAND_TYPE_LOOP_CONTROL:
    case SPV_OPERAND_TYPE_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS:
    case SPV_OPERAND_TYPE_SELECTION_CONTROL:
    case SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_INFO_FLAGS:
    case SPV_OPERAND_TYPE_DEBUG_INFO_FLAGS: {
      // This operand is a mask.

      // Map an optional operand type to its corresponding concrete type.
      if (type == SPV_OPERAND_TYPE_OPTIONAL_IMAGE)
        parsed_operand.type = SPV_OPERAND_TYPE_IMAGE;
      else if (type == SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS)
        parsed_operand.type = SPV_OPERAND_TYPE_MEMORY_ACCESS;

      // Check validity of set mask bits. Also prepare for operands for those
      // masks if they have any.  To get operand order correct, scan from
      // MSB to LSB since we can only prepend operands to a pattern.
      // The only case in the grammar where you have more than one mask bit
      // having an operand is for image operands.  See SPIR-V 3.14 Image
      // Operands.
      uint32_t remaining_word = word;
      for (uint32_t mask = (1u << 31); remaining_word; mask >>= 1) {
        if (remaining_word & mask) {
          spv_operand_desc entry;
          if (grammar_.lookupOperand(type, mask, &entry)) {
            return diagnostic()
                   << "Invalid " << spvOperandTypeStr(parsed_operand.type)
                   << " operand: " << word << " has invalid mask component "
                   << mask;
          }
          remaining_word ^= mask;
          spvPushOperandTypes(entry->operandTypes, expected_operands);
        }
      }
      if (word == 0) {
        // An all-zeroes mask *might* also be valid.
        spv_operand_desc entry;
        if (SPV_SUCCESS == grammar_.lookupOperand(type, 0, &entry)) {
          // Prepare for its operands, if any.
          spvPushOperandTypes(entry->operandTypes, expected_operands);
        }
      }
    } break;
    default:
      return diagnostic() << "Internal error: Unhandled operand type: " << type;
  }

  assert(spvOperandIsConcrete(parsed_operand.type));

  operands->push_back(parsed_operand);

  const size_t index_after_operand = _.word_index + parsed_operand.num_words;

  // Avoid buffer overrun for the cases where the operand has more than one
  // word, and where it isn't a string.  (Those other cases have already been
  // handled earlier.)  For example, this error can occur for a multi-word
  // argument to OpConstant, or a multi-word case literal operand for OpSwitch.
  if (_.num_words < index_after_operand)
    return exhaustedInputDiagnostic(inst_offset, opcode, type);

  if (_.requires_endian_conversion) {
    // Copy instruction words.  Translate to native endianness as needed.
    if (convert_operand_endianness) {
      const spv_endianness_t endianness = _.endian;
      std::transform(_.words + _.word_index, _.words + index_after_operand,
                     std::back_inserter(*words),
                     [endianness](const uint32_t raw_word) {
                       return spvFixWord(raw_word, endianness);
                     });
    } else {
      words->insert(words->end(), _.words + _.word_index,
                    _.words + index_after_operand);
    }
  }

  // Advance past the operand.
  _.word_index = index_after_operand;

  return SPV_SUCCESS;
}

spv_result_t Parser::setNumericTypeInfoForType(
    spv_parsed_operand_t* parsed_operand, uint32_t type_id) {
  assert(type_id != 0);
  auto type_info_iter = _.type_id_to_number_type_info.find(type_id);
  if (type_info_iter == _.type_id_to_number_type_info.end()) {
    return diagnostic() << "Type Id " << type_id << " is not a type";
  }
  const NumberType& info = type_info_iter->second;
  if (info.type == SPV_NUMBER_NONE) {
    // This is a valid type, but for something other than a scalar number.
    return diagnostic() << "Type Id " << type_id
                        << " is not a scalar numeric type";
  }

  parsed_operand->number_kind = info.type;
  parsed_operand->number_bit_width = info.bit_width;
  // Round up the word count.
  parsed_operand->num_words = static_cast<uint16_t>((info.bit_width + 31) / 32);
  return SPV_SUCCESS;
}

void Parser::recordNumberType(size_t inst_offset,
                              const spv_parsed_instruction_t* inst) {
  const spv::Op opcode = static_cast<spv::Op>(inst->opcode);
  if (spvOpcodeGeneratesType(opcode)) {
    NumberType info = {SPV_NUMBER_NONE, 0};
    if (spv::Op::OpTypeInt == opcode) {
      const bool is_signed = peekAt(inst_offset + 3) != 0;
      info.type = is_signed ? SPV_NUMBER_SIGNED_INT : SPV_NUMBER_UNSIGNED_INT;
      info.bit_width = peekAt(inst_offset + 2);
    } else if (spv::Op::OpTypeFloat == opcode) {
      info.type = SPV_NUMBER_FLOATING;
      info.bit_width = peekAt(inst_offset + 2);
    }
    // The *result* Id of a type generating instruction is the type Id.
    _.type_id_to_number_type_info[inst->result_id] = info;
  }
}

}  // anonymous namespace

spv_result_t spvBinaryParse(const spv_const_context context, void* user_data,
                            const uint32_t* code, const size_t num_words,
                            spv_parsed_header_fn_t parsed_header,
                            spv_parsed_instruction_fn_t parsed_instruction,
                            spv_diagnostic* diagnostic) {
  spv_context_t hijack_context = *context;
  if (diagnostic) {
    *diagnostic = nullptr;
    spvtools::UseDiagnosticAsMessageConsumer(&hijack_context, diagnostic);
  }
  Parser parser(&hijack_context, user_data, parsed_header, parsed_instruction);
  return parser.parse(code, num_words, diagnostic);
}

// TODO(dneto): This probably belongs in text.cpp since that's the only place
// that a spv_binary_t value is created.
void spvBinaryDestroy(spv_binary binary) {
  if (binary) {
    if (binary->code) delete[] binary->code;
    delete binary;
  }
}

size_t spv_strnlen_s(const char* str, size_t strsz) {
  if (!str) return 0;
  for (size_t i = 0; i < strsz; i++) {
    if (!str[i]) return i;
  }
  return strsz;
}
