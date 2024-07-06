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

#include "source/text_handler.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <tuple>

#include "source/assembly_grammar.h"
#include "source/binary.h"
#include "source/ext_inst.h"
#include "source/instruction.h"
#include "source/opcode.h"
#include "source/text.h"
#include "source/util/bitutils.h"
#include "source/util/hex_float.h"
#include "source/util/parse_number.h"
#include "source/util/string_utils.h"

namespace spvtools {
namespace {

// Advances |text| to the start of the next line and writes the new position to
// |position|.
spv_result_t advanceLine(spv_text text, spv_position position) {
  while (true) {
    if (position->index >= text->length) return SPV_END_OF_STREAM;
    switch (text->str[position->index]) {
      case '\0':
        return SPV_END_OF_STREAM;
      case '\n':
        position->column = 0;
        position->line++;
        position->index++;
        return SPV_SUCCESS;
      default:
        position->column++;
        position->index++;
        break;
    }
  }
}

// Advances |text| to first non white space character and writes the new
// position to |position|.
// If a null terminator is found during the text advance, SPV_END_OF_STREAM is
// returned, SPV_SUCCESS otherwise. No error checking is performed on the
// parameters, its the users responsibility to ensure these are non null.
spv_result_t advance(spv_text text, spv_position position) {
  // NOTE: Consume white space, otherwise don't advance.
  while (true) {
    if (position->index >= text->length) return SPV_END_OF_STREAM;
    switch (text->str[position->index]) {
      case '\0':
        return SPV_END_OF_STREAM;
      case ';':
        if (spv_result_t error = advanceLine(text, position)) return error;
        continue;
      case ' ':
      case '\t':
      case '\r':
        position->column++;
        position->index++;
        continue;
      case '\n':
        position->column = 0;
        position->line++;
        position->index++;
        continue;
      default:
        return SPV_SUCCESS;
    }
  }
}

// Fetches the next word from the given text stream starting from the given
// *position. On success, writes the decoded word into *word and updates
// *position to the location past the returned word.
//
// A word ends at the next comment or whitespace.  However, double-quoted
// strings remain intact, and a backslash always escapes the next character.
spv_result_t getWord(spv_text text, spv_position position, std::string* word) {
  if (!text->str || !text->length) return SPV_ERROR_INVALID_TEXT;
  if (!position) return SPV_ERROR_INVALID_POINTER;

  const size_t start_index = position->index;

  bool quoting = false;
  bool escaping = false;

  // NOTE: Assumes first character is not white space!
  while (true) {
    if (position->index >= text->length) {
      word->assign(text->str + start_index, text->str + position->index);
      return SPV_SUCCESS;
    }
    const char ch = text->str[position->index];
    if (ch == '\\') {
      escaping = !escaping;
    } else {
      switch (ch) {
        case '"':
          if (!escaping) quoting = !quoting;
          break;
        case ' ':
        case ';':
        case '\t':
        case '\n':
        case '\r':
          if (escaping || quoting) break;
          word->assign(text->str + start_index, text->str + position->index);
          return SPV_SUCCESS;
        case '\0': {  // NOTE: End of word found!
          word->assign(text->str + start_index, text->str + position->index);
          return SPV_SUCCESS;
        }
        default:
          break;
      }
      escaping = false;
    }

    position->column++;
    position->index++;
  }
}

// Returns true if the characters in the text as position represent
// the start of an Opcode.
bool startsWithOp(spv_text text, spv_position position) {
  if (text->length < position->index + 3) return false;
  char ch0 = text->str[position->index];
  char ch1 = text->str[position->index + 1];
  char ch2 = text->str[position->index + 2];
  return ('O' == ch0 && 'p' == ch1 && ('A' <= ch2 && ch2 <= 'Z'));
}

}  // namespace

const IdType kUnknownType = {0, false, IdTypeClass::kBottom};

// TODO(dneto): Reorder AssemblyContext definitions to match declaration order.

// This represents all of the data that is only valid for the duration of
// a single compilation.
uint32_t AssemblyContext::spvNamedIdAssignOrGet(const char* textValue) {
  if (!ids_to_preserve_.empty()) {
    uint32_t id = 0;
    if (spvtools::utils::ParseNumber(textValue, &id)) {
      if (ids_to_preserve_.find(id) != ids_to_preserve_.end()) {
        bound_ = std::max(bound_, id + 1);
        return id;
      }
    }
  }

  const auto it = named_ids_.find(textValue);
  if (it == named_ids_.end()) {
    uint32_t id = next_id_++;
    if (!ids_to_preserve_.empty()) {
      while (ids_to_preserve_.find(id) != ids_to_preserve_.end()) {
        id = next_id_++;
      }
    }

    named_ids_.emplace(textValue, id);
    bound_ = std::max(bound_, id + 1);
    return id;
  }

  return it->second;
}

uint32_t AssemblyContext::getBound() const { return bound_; }

spv_result_t AssemblyContext::advance() {
  return spvtools::advance(text_, &current_position_);
}

spv_result_t AssemblyContext::getWord(std::string* word,
                                      spv_position next_position) {
  *next_position = current_position_;
  return spvtools::getWord(text_, next_position, word);
}

bool AssemblyContext::startsWithOp() {
  return spvtools::startsWithOp(text_, &current_position_);
}

bool AssemblyContext::isStartOfNewInst() {
  spv_position_t pos = current_position_;
  if (spvtools::advance(text_, &pos)) return false;
  if (spvtools::startsWithOp(text_, &pos)) return true;

  std::string word;
  pos = current_position_;
  if (spvtools::getWord(text_, &pos, &word)) return false;
  if ('%' != word.front()) return false;

  if (spvtools::advance(text_, &pos)) return false;
  if (spvtools::getWord(text_, &pos, &word)) return false;
  if ("=" != word) return false;

  if (spvtools::advance(text_, &pos)) return false;
  if (spvtools::startsWithOp(text_, &pos)) return true;
  return false;
}

char AssemblyContext::peek() const {
  return text_->str[current_position_.index];
}

bool AssemblyContext::hasText() const {
  return text_->length > current_position_.index;
}

void AssemblyContext::seekForward(uint32_t size) {
  current_position_.index += size;
  current_position_.column += size;
}

spv_result_t AssemblyContext::binaryEncodeU32(const uint32_t value,
                                              spv_instruction_t* pInst) {
  pInst->words.insert(pInst->words.end(), value);
  return SPV_SUCCESS;
}

spv_result_t AssemblyContext::binaryEncodeNumericLiteral(
    const char* val, spv_result_t error_code, const IdType& type,
    spv_instruction_t* pInst) {
  using spvtools::utils::EncodeNumberStatus;
  // Populate the NumberType from the IdType for parsing.
  spvtools::utils::NumberType number_type;
  switch (type.type_class) {
    case IdTypeClass::kOtherType:
      return diagnostic(SPV_ERROR_INTERNAL)
             << "Unexpected numeric literal type";
    case IdTypeClass::kScalarIntegerType:
      if (type.isSigned) {
        number_type = {type.bitwidth, SPV_NUMBER_SIGNED_INT};
      } else {
        number_type = {type.bitwidth, SPV_NUMBER_UNSIGNED_INT};
      }
      break;
    case IdTypeClass::kScalarFloatType:
      number_type = {type.bitwidth, SPV_NUMBER_FLOATING};
      break;
    case IdTypeClass::kBottom:
      // kBottom means the type is unknown and we need to infer the type before
      // parsing the number. The rule is: If there is a decimal point, treat
      // the value as a floating point value, otherwise a integer value, then
      // if the first char of the integer text is '-', treat the integer as a
      // signed integer, otherwise an unsigned integer.
      uint32_t bitwidth = static_cast<uint32_t>(assumedBitWidth(type));
      if (strchr(val, '.')) {
        number_type = {bitwidth, SPV_NUMBER_FLOATING};
      } else if (type.isSigned || val[0] == '-') {
        number_type = {bitwidth, SPV_NUMBER_SIGNED_INT};
      } else {
        number_type = {bitwidth, SPV_NUMBER_UNSIGNED_INT};
      }
      break;
  }

  std::string error_msg;
  EncodeNumberStatus parse_status = ParseAndEncodeNumber(
      val, number_type,
      [this, pInst](uint32_t d) { this->binaryEncodeU32(d, pInst); },
      &error_msg);
  switch (parse_status) {
    case EncodeNumberStatus::kSuccess:
      return SPV_SUCCESS;
    case EncodeNumberStatus::kInvalidText:
      return diagnostic(error_code) << error_msg;
    case EncodeNumberStatus::kUnsupported:
      return diagnostic(SPV_ERROR_INTERNAL) << error_msg;
    case EncodeNumberStatus::kInvalidUsage:
      return diagnostic(SPV_ERROR_INVALID_TEXT) << error_msg;
  }
  // This line is not reachable, only added to satisfy the compiler.
  return diagnostic(SPV_ERROR_INTERNAL)
         << "Unexpected result code from ParseAndEncodeNumber()";
}

spv_result_t AssemblyContext::binaryEncodeString(const char* value,
                                                 spv_instruction_t* pInst) {
  const size_t length = strlen(value);
  const size_t wordCount = (length / 4) + 1;
  const size_t oldWordCount = pInst->words.size();
  const size_t newWordCount = oldWordCount + wordCount;

  // TODO(dneto): We can just defer this check until later.
  if (newWordCount > SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX) {
    return diagnostic() << "Instruction too long: more than "
                        << SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX << " words.";
  }

  pInst->words.reserve(newWordCount);
  spvtools::utils::AppendToVector(value, &pInst->words);

  return SPV_SUCCESS;
}

spv_result_t AssemblyContext::recordTypeDefinition(
    const spv_instruction_t* pInst) {
  uint32_t value = pInst->words[1];
  if (types_.find(value) != types_.end()) {
    return diagnostic() << "Value " << value
                        << " has already been used to generate a type";
  }

  if (pInst->opcode == spv::Op::OpTypeInt) {
    if (pInst->words.size() != 4)
      return diagnostic() << "Invalid OpTypeInt instruction";
    types_[value] = {pInst->words[2], pInst->words[3] != 0,
                     IdTypeClass::kScalarIntegerType};
  } else if (pInst->opcode == spv::Op::OpTypeFloat) {
    if ((pInst->words.size() != 3) && (pInst->words.size() != 4))
      return diagnostic() << "Invalid OpTypeFloat instruction";
    // TODO(kpet) Do we need to record the FP Encoding here?
    types_[value] = {pInst->words[2], false, IdTypeClass::kScalarFloatType};
  } else {
    types_[value] = {0, false, IdTypeClass::kOtherType};
  }
  return SPV_SUCCESS;
}

IdType AssemblyContext::getTypeOfTypeGeneratingValue(uint32_t value) const {
  auto type = types_.find(value);
  if (type == types_.end()) {
    return kUnknownType;
  }
  return std::get<1>(*type);
}

IdType AssemblyContext::getTypeOfValueInstruction(uint32_t value) const {
  auto type_value = value_types_.find(value);
  if (type_value == value_types_.end()) {
    return {0, false, IdTypeClass::kBottom};
  }
  return getTypeOfTypeGeneratingValue(std::get<1>(*type_value));
}

spv_result_t AssemblyContext::recordTypeIdForValue(uint32_t value,
                                                   uint32_t type) {
  bool successfully_inserted = false;
  std::tie(std::ignore, successfully_inserted) =
      value_types_.insert(std::make_pair(value, type));
  if (!successfully_inserted)
    return diagnostic() << "Value is being defined a second time";
  return SPV_SUCCESS;
}

spv_result_t AssemblyContext::recordIdAsExtInstImport(
    uint32_t id, spv_ext_inst_type_t type) {
  bool successfully_inserted = false;
  std::tie(std::ignore, successfully_inserted) =
      import_id_to_ext_inst_type_.insert(std::make_pair(id, type));
  if (!successfully_inserted)
    return diagnostic() << "Import Id is being defined a second time";
  return SPV_SUCCESS;
}

spv_ext_inst_type_t AssemblyContext::getExtInstTypeForId(uint32_t id) const {
  auto type = import_id_to_ext_inst_type_.find(id);
  if (type == import_id_to_ext_inst_type_.end()) {
    return SPV_EXT_INST_TYPE_NONE;
  }
  return std::get<1>(*type);
}

std::set<uint32_t> AssemblyContext::GetNumericIds() const {
  std::set<uint32_t> ids;
  for (const auto& kv : named_ids_) {
    uint32_t id;
    if (spvtools::utils::ParseNumber(kv.first.c_str(), &id)) ids.insert(id);
  }
  return ids;
}

}  // namespace spvtools
