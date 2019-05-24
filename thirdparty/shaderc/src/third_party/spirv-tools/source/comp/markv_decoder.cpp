// Copyright (c) 2018 Google LLC
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

#include "source/comp/markv_decoder.h"

#include <cstring>
#include <iterator>
#include <numeric>

#include "source/ext_inst.h"
#include "source/opcode.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace comp {

spv_result_t MarkvDecoder::DecodeNonIdWord(uint32_t* word) {
  auto* codec = model_->GetNonIdWordHuffmanCodec(inst_.opcode, operand_index_);

  if (codec) {
    uint64_t decoded_value = 0;
    if (!codec->DecodeFromStream(GetReadBitCallback(), &decoded_value))
      return Diag(SPV_ERROR_INVALID_BINARY)
             << "Failed to decode non-id word with Huffman";

    if (decoded_value != MarkvModel::GetMarkvNoneOfTheAbove()) {
      // The word decoded successfully.
      *word = uint32_t(decoded_value);
      assert(*word == decoded_value);
      return SPV_SUCCESS;
    }

    // Received kMarkvNoneOfTheAbove signal, use fallback decoding.
  }

  const size_t chunk_length =
      model_->GetOperandVariableWidthChunkLength(operand_.type);
  if (chunk_length) {
    if (!reader_.ReadVariableWidthU32(word, chunk_length))
      return Diag(SPV_ERROR_INVALID_BINARY)
             << "Failed to decode non-id word with varint";
  } else {
    if (!reader_.ReadUnencoded(word))
      return Diag(SPV_ERROR_INVALID_BINARY)
             << "Failed to read unencoded non-id word";
  }
  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeOpcodeAndNumberOfOperands(
    uint32_t* opcode, uint32_t* num_operands) {
  // First try to use the Markov chain codec.
  auto* codec =
      model_->GetOpcodeAndNumOperandsMarkovHuffmanCodec(GetPrevOpcode());
  if (codec) {
    uint64_t decoded_value = 0;
    if (!codec->DecodeFromStream(GetReadBitCallback(), &decoded_value))
      return Diag(SPV_ERROR_INTERNAL)
             << "Failed to decode opcode_and_num_operands, previous opcode is "
             << spvOpcodeString(GetPrevOpcode());

    if (decoded_value != MarkvModel::GetMarkvNoneOfTheAbove()) {
      // The word was successfully decoded.
      *opcode = uint32_t(decoded_value & 0xFFFF);
      *num_operands = uint32_t(decoded_value >> 16);
      return SPV_SUCCESS;
    }

    // Received kMarkvNoneOfTheAbove signal, use fallback decoding.
  }

  // Fallback to base-rate codec.
  codec = model_->GetOpcodeAndNumOperandsMarkovHuffmanCodec(SpvOpNop);
  assert(codec);
  uint64_t decoded_value = 0;
  if (!codec->DecodeFromStream(GetReadBitCallback(), &decoded_value))
    return Diag(SPV_ERROR_INTERNAL)
           << "Failed to decode opcode_and_num_operands with global codec";

  if (decoded_value == MarkvModel::GetMarkvNoneOfTheAbove()) {
    // Received kMarkvNoneOfTheAbove signal, fallback further.
    return SPV_UNSUPPORTED;
  }

  *opcode = uint32_t(decoded_value & 0xFFFF);
  *num_operands = uint32_t(decoded_value >> 16);
  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeMtfRankHuffman(uint64_t mtf,
                                                uint32_t fallback_method,
                                                uint32_t* rank) {
  const auto* codec = GetMtfHuffmanCodec(mtf);
  if (!codec) {
    assert(fallback_method != kMtfNone);
    codec = GetMtfHuffmanCodec(fallback_method);
  }

  if (!codec) return Diag(SPV_ERROR_INTERNAL) << "No codec to decode MTF rank";

  uint32_t decoded_value = 0;
  if (!codec->DecodeFromStream(GetReadBitCallback(), &decoded_value))
    return Diag(SPV_ERROR_INTERNAL) << "Failed to decode MTF rank with Huffman";

  if (decoded_value == kMtfRankEncodedByValueSignal) {
    // Decode by value.
    if (!reader_.ReadVariableWidthU32(rank, model_->mtf_rank_chunk_length()))
      return Diag(SPV_ERROR_INTERNAL)
             << "Failed to decode MTF rank with varint";
    *rank += MarkvCodec::kMtfSmallestRankEncodedByValue;
  } else {
    // Decode using Huffman coding.
    assert(decoded_value < MarkvCodec::kMtfSmallestRankEncodedByValue);
    *rank = decoded_value;
  }
  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeIdWithDescriptor(uint32_t* id) {
  auto* codec =
      model_->GetIdDescriptorHuffmanCodec(inst_.opcode, operand_index_);

  uint64_t mtf = kMtfNone;
  if (codec) {
    uint64_t decoded_value = 0;
    if (!codec->DecodeFromStream(GetReadBitCallback(), &decoded_value))
      return Diag(SPV_ERROR_INTERNAL)
             << "Failed to decode descriptor with Huffman";

    if (decoded_value != MarkvModel::GetMarkvNoneOfTheAbove()) {
      const uint32_t long_descriptor = uint32_t(decoded_value);
      mtf = GetMtfLongIdDescriptor(long_descriptor);
    }
  }

  if (mtf == kMtfNone) {
    if (model_->id_fallback_strategy() !=
        MarkvModel::IdFallbackStrategy::kShortDescriptor) {
      return SPV_UNSUPPORTED;
    }

    uint64_t decoded_value = 0;
    if (!reader_.ReadBits(&decoded_value, MarkvCodec::kShortDescriptorNumBits))
      return Diag(SPV_ERROR_INTERNAL) << "Failed to read short descriptor";
    const uint32_t short_descriptor = uint32_t(decoded_value);
    if (short_descriptor == 0) {
      // Forward declared id.
      return SPV_UNSUPPORTED;
    }
    mtf = GetMtfShortIdDescriptor(short_descriptor);
  }

  return DecodeExistingId(mtf, id);
}

spv_result_t MarkvDecoder::DecodeExistingId(uint64_t mtf, uint32_t* id) {
  assert(multi_mtf_.GetSize(mtf) > 0);
  *id = 0;

  uint32_t rank = 0;

  if (multi_mtf_.GetSize(mtf) == 1) {
    rank = 1;
  } else {
    const spv_result_t result =
        DecodeMtfRankHuffman(mtf, kMtfGenericNonZeroRank, &rank);
    if (result != SPV_SUCCESS) return result;
  }

  assert(rank);
  if (!multi_mtf_.ValueFromRank(mtf, rank, id))
    return Diag(SPV_ERROR_INTERNAL) << "MTF rank is out of bounds";

  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeRefId(uint32_t* id) {
  {
    const spv_result_t result = DecodeIdWithDescriptor(id);
    if (result != SPV_UNSUPPORTED) return result;
  }

  const bool can_forward_declare = spvOperandCanBeForwardDeclaredFunction(
      SpvOp(inst_.opcode))(operand_index_);
  uint32_t rank = 0;
  *id = 0;

  if (model_->id_fallback_strategy() ==
      MarkvModel::IdFallbackStrategy::kRuleBased) {
    uint64_t mtf = GetRuleBasedMtf();
    if (mtf != kMtfNone && !can_forward_declare) {
      return DecodeExistingId(mtf, id);
    }

    if (mtf == kMtfNone) mtf = kMtfAll;
    {
      const spv_result_t result = DecodeMtfRankHuffman(mtf, kMtfAll, &rank);
      if (result != SPV_SUCCESS) return result;
    }

    if (rank == 0) {
      // This is the first occurrence of a forward declared id.
      *id = GetIdBound();
      SetIdBound(*id + 1);
      multi_mtf_.Insert(kMtfAll, *id);
      multi_mtf_.Insert(kMtfForwardDeclared, *id);
      if (mtf != kMtfAll) multi_mtf_.Insert(mtf, *id);
    } else {
      if (!multi_mtf_.ValueFromRank(mtf, rank, id))
        return Diag(SPV_ERROR_INTERNAL) << "MTF rank out of bounds";
    }
  } else {
    assert(can_forward_declare);

    if (!reader_.ReadVariableWidthU32(&rank, model_->mtf_rank_chunk_length()))
      return Diag(SPV_ERROR_INTERNAL)
             << "Failed to decode MTF rank with varint";

    if (rank == 0) {
      // This is the first occurrence of a forward declared id.
      *id = GetIdBound();
      SetIdBound(*id + 1);
      multi_mtf_.Insert(kMtfForwardDeclared, *id);
    } else {
      if (!multi_mtf_.ValueFromRank(kMtfForwardDeclared, rank, id))
        return Diag(SPV_ERROR_INTERNAL) << "MTF rank out of bounds";
    }
  }
  assert(*id);
  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeTypeId() {
  if (inst_.opcode == SpvOpFunctionParameter) {
    assert(!remaining_function_parameter_types_.empty());
    inst_.type_id = remaining_function_parameter_types_.front();
    remaining_function_parameter_types_.pop_front();
    return SPV_SUCCESS;
  }

  {
    const spv_result_t result = DecodeIdWithDescriptor(&inst_.type_id);
    if (result != SPV_UNSUPPORTED) return result;
  }

  assert(model_->id_fallback_strategy() ==
         MarkvModel::IdFallbackStrategy::kRuleBased);

  uint64_t mtf = GetRuleBasedMtf();
  assert(!spvOperandCanBeForwardDeclaredFunction(SpvOp(inst_.opcode))(
      operand_index_));

  if (mtf == kMtfNone) {
    mtf = kMtfTypeNonFunction;
    // Function types should have been handled by GetRuleBasedMtf.
    assert(inst_.opcode != SpvOpFunction);
  }

  return DecodeExistingId(mtf, &inst_.type_id);
}

spv_result_t MarkvDecoder::DecodeResultId() {
  uint32_t rank = 0;

  const uint64_t num_still_forward_declared =
      multi_mtf_.GetSize(kMtfForwardDeclared);

  if (num_still_forward_declared) {
    // Some ids were forward declared. Check if this id is one of them.
    uint64_t id_was_forward_declared;
    if (!reader_.ReadBits(&id_was_forward_declared, 1))
      return Diag(SPV_ERROR_INVALID_BINARY)
             << "Failed to read id_was_forward_declared flag";

    if (id_was_forward_declared) {
      if (!reader_.ReadVariableWidthU32(&rank, model_->mtf_rank_chunk_length()))
        return Diag(SPV_ERROR_INVALID_BINARY)
               << "Failed to read MTF rank of forward declared id";

      if (rank) {
        // The id was forward declared, recover it from kMtfForwardDeclared.
        if (!multi_mtf_.ValueFromRank(kMtfForwardDeclared, rank,
                                      &inst_.result_id))
          return Diag(SPV_ERROR_INTERNAL)
                 << "Forward declared MTF rank is out of bounds";

        // We can now remove the id from kMtfForwardDeclared.
        if (!multi_mtf_.Remove(kMtfForwardDeclared, inst_.result_id))
          return Diag(SPV_ERROR_INTERNAL)
                 << "Failed to remove id from kMtfForwardDeclared";
      }
    }
  }

  if (inst_.result_id == 0) {
    // The id was not forward declared, issue a new id.
    inst_.result_id = GetIdBound();
    SetIdBound(inst_.result_id + 1);
  }

  if (model_->id_fallback_strategy() ==
      MarkvModel::IdFallbackStrategy::kRuleBased) {
    if (!rank) {
      multi_mtf_.Insert(kMtfAll, inst_.result_id);
    }
  }

  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeLiteralNumber(
    const spv_parsed_operand_t& operand) {
  if (operand.number_bit_width <= 32) {
    uint32_t word = 0;
    const spv_result_t result = DecodeNonIdWord(&word);
    if (result != SPV_SUCCESS) return result;
    inst_words_.push_back(word);
  } else {
    assert(operand.number_bit_width <= 64);
    uint64_t word = 0;
    if (operand.number_kind == SPV_NUMBER_UNSIGNED_INT) {
      if (!reader_.ReadVariableWidthU64(&word, model_->u64_chunk_length()))
        return Diag(SPV_ERROR_INVALID_BINARY) << "Failed to read literal U64";
    } else if (operand.number_kind == SPV_NUMBER_SIGNED_INT) {
      int64_t val = 0;
      if (!reader_.ReadVariableWidthS64(&val, model_->s64_chunk_length(),
                                        model_->s64_block_exponent()))
        return Diag(SPV_ERROR_INVALID_BINARY) << "Failed to read literal S64";
      std::memcpy(&word, &val, 8);
    } else if (operand.number_kind == SPV_NUMBER_FLOATING) {
      if (!reader_.ReadUnencoded(&word))
        return Diag(SPV_ERROR_INVALID_BINARY) << "Failed to read literal F64";
    } else {
      return Diag(SPV_ERROR_INTERNAL) << "Unsupported bit length";
    }
    inst_words_.push_back(static_cast<uint32_t>(word));
    inst_words_.push_back(static_cast<uint32_t>(word >> 32));
  }
  return SPV_SUCCESS;
}

bool MarkvDecoder::ReadToByteBreak(size_t byte_break_if_less_than) {
  const size_t num_bits_to_next_byte =
      GetNumBitsToNextByte(reader_.GetNumReadBits());
  if (num_bits_to_next_byte == 0 ||
      num_bits_to_next_byte > byte_break_if_less_than)
    return true;

  uint64_t bits = 0;
  if (!reader_.ReadBits(&bits, num_bits_to_next_byte)) return false;

  assert(bits == 0);
  if (bits != 0) return false;

  return true;
}

spv_result_t MarkvDecoder::DecodeModule(std::vector<uint32_t>* spirv_binary) {
  const bool header_read_success =
      reader_.ReadUnencoded(&header_.magic_number) &&
      reader_.ReadUnencoded(&header_.markv_version) &&
      reader_.ReadUnencoded(&header_.markv_model) &&
      reader_.ReadUnencoded(&header_.markv_length_in_bits) &&
      reader_.ReadUnencoded(&header_.spirv_version) &&
      reader_.ReadUnencoded(&header_.spirv_generator);

  if (!header_read_success)
    return Diag(SPV_ERROR_INVALID_BINARY) << "Unable to read MARK-V header";

  if (header_.markv_length_in_bits == 0)
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "Header markv_length_in_bits field is zero";

  if (header_.magic_number != MarkvCodec::kMarkvMagicNumber)
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "MARK-V binary has incorrect magic number";

  // TODO(atgoo@github.com): Print version strings.
  if (header_.markv_version != MarkvCodec::GetMarkvVersion())
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "MARK-V binary and the codec have different versions";

  const uint32_t model_type = header_.markv_model >> 16;
  const uint32_t model_version = header_.markv_model & 0xFFFF;
  if (model_type != model_->model_type())
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "MARK-V binary and the codec use different MARK-V models";

  if (model_version != model_->model_version())
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "MARK-V binary and the codec use different versions if the same "
           << "MARK-V model";

  spirv_.reserve(header_.markv_length_in_bits / 2);  // Heuristic.
  spirv_.resize(5, 0);
  spirv_[0] = SpvMagicNumber;
  spirv_[1] = header_.spirv_version;
  spirv_[2] = header_.spirv_generator;

  if (logger_) {
    reader_.SetCallback(
        [this](const std::string& str) { logger_->AppendBitSequence(str); });
  }

  while (reader_.GetNumReadBits() < header_.markv_length_in_bits) {
    inst_ = {};
    const spv_result_t decode_result = DecodeInstruction();
    if (decode_result != SPV_SUCCESS) return decode_result;
  }

  if (validator_options_) {
    spv_const_binary_t validation_binary = {spirv_.data(), spirv_.size()};
    const spv_result_t result = spvValidateWithOptions(
        context_, validator_options_, &validation_binary, nullptr);
    if (result != SPV_SUCCESS) return result;
  }

  // Validate the decode binary
  if (reader_.GetNumReadBits() != header_.markv_length_in_bits ||
      !reader_.OnlyZeroesLeft()) {
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "MARK-V binary has wrong stated bit length "
           << reader_.GetNumReadBits() << " " << header_.markv_length_in_bits;
  }

  // Decoding of the module is finished, validation state should have correct
  // id bound.
  spirv_[3] = GetIdBound();

  *spirv_binary = std::move(spirv_);
  return SPV_SUCCESS;
}

// TODO(atgoo@github.com): The implementation borrows heavily from
// Parser::parseOperand.
// Consider coupling them together in some way once MARK-V codec is more mature.
// For now it's better to keep the code independent for experimentation
// purposes.
spv_result_t MarkvDecoder::DecodeOperand(
    size_t operand_offset, const spv_operand_type_t type,
    spv_operand_pattern_t* expected_operands) {
  const SpvOp opcode = static_cast<SpvOp>(inst_.opcode);

  memset(&operand_, 0, sizeof(operand_));

  assert((operand_offset >> 16) == 0);
  operand_.offset = static_cast<uint16_t>(operand_offset);
  operand_.type = type;

  // Set default values, may be updated later.
  operand_.number_kind = SPV_NUMBER_NONE;
  operand_.number_bit_width = 0;

  const size_t first_word_index = inst_words_.size();

  switch (type) {
    case SPV_OPERAND_TYPE_RESULT_ID: {
      const spv_result_t result = DecodeResultId();
      if (result != SPV_SUCCESS) return result;

      inst_words_.push_back(inst_.result_id);
      SetIdBound(std::max(GetIdBound(), inst_.result_id + 1));
      PromoteIfNeeded(inst_.result_id);
      break;
    }

    case SPV_OPERAND_TYPE_TYPE_ID: {
      const spv_result_t result = DecodeTypeId();
      if (result != SPV_SUCCESS) return result;

      inst_words_.push_back(inst_.type_id);
      SetIdBound(std::max(GetIdBound(), inst_.type_id + 1));
      PromoteIfNeeded(inst_.type_id);
      break;
    }

    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_OPTIONAL_ID:
    case SPV_OPERAND_TYPE_SCOPE_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID: {
      uint32_t id = 0;
      const spv_result_t result = DecodeRefId(&id);
      if (result != SPV_SUCCESS) return result;

      if (id == 0) return Diag(SPV_ERROR_INVALID_BINARY) << "Decoded id is 0";

      if (type == SPV_OPERAND_TYPE_ID || type == SPV_OPERAND_TYPE_OPTIONAL_ID) {
        operand_.type = SPV_OPERAND_TYPE_ID;

        if (opcode == SpvOpExtInst && operand_.offset == 3) {
          // The current word is the extended instruction set id.
          // Set the extended instruction set type for the current
          // instruction.
          auto ext_inst_type_iter = import_id_to_ext_inst_type_.find(id);
          if (ext_inst_type_iter == import_id_to_ext_inst_type_.end()) {
            return Diag(SPV_ERROR_INVALID_ID)
                   << "OpExtInst set id " << id
                   << " does not reference an OpExtInstImport result Id";
          }
          inst_.ext_inst_type = ext_inst_type_iter->second;
        }
      }

      inst_words_.push_back(id);
      SetIdBound(std::max(GetIdBound(), id + 1));
      PromoteIfNeeded(id);
      break;
    }

    case SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER: {
      uint32_t word = 0;
      const spv_result_t result = DecodeNonIdWord(&word);
      if (result != SPV_SUCCESS) return result;

      inst_words_.push_back(word);

      assert(SpvOpExtInst == opcode);
      assert(inst_.ext_inst_type != SPV_EXT_INST_TYPE_NONE);
      spv_ext_inst_desc ext_inst;
      if (grammar_.lookupExtInst(inst_.ext_inst_type, word, &ext_inst))
        return Diag(SPV_ERROR_INVALID_BINARY)
               << "Invalid extended instruction number: " << word;
      spvPushOperandTypes(ext_inst->operandTypes, expected_operands);
      break;
    }

    case SPV_OPERAND_TYPE_LITERAL_INTEGER:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER: {
      // These are regular single-word literal integer operands.
      // Post-parsing validation should check the range of the parsed value.
      operand_.type = SPV_OPERAND_TYPE_LITERAL_INTEGER;
      // It turns out they are always unsigned integers!
      operand_.number_kind = SPV_NUMBER_UNSIGNED_INT;
      operand_.number_bit_width = 32;

      uint32_t word = 0;
      const spv_result_t result = DecodeNonIdWord(&word);
      if (result != SPV_SUCCESS) return result;

      inst_words_.push_back(word);
      break;
    }

    case SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER:
    case SPV_OPERAND_TYPE_OPTIONAL_TYPED_LITERAL_INTEGER: {
      operand_.type = SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER;
      if (opcode == SpvOpSwitch) {
        // The literal operands have the same type as the value
        // referenced by the selector Id.
        const uint32_t selector_id = inst_words_.at(1);
        const auto type_id_iter = id_to_type_id_.find(selector_id);
        if (type_id_iter == id_to_type_id_.end() || type_id_iter->second == 0) {
          return Diag(SPV_ERROR_INVALID_BINARY)
                 << "Invalid OpSwitch: selector id " << selector_id
                 << " has no type";
        }
        uint32_t type_id = type_id_iter->second;

        if (selector_id == type_id) {
          // Recall that by convention, a result ID that is a type definition
          // maps to itself.
          return Diag(SPV_ERROR_INVALID_BINARY)
                 << "Invalid OpSwitch: selector id " << selector_id
                 << " is a type, not a value";
        }
        if (auto error = SetNumericTypeInfoForType(&operand_, type_id))
          return error;
        if (operand_.number_kind != SPV_NUMBER_UNSIGNED_INT &&
            operand_.number_kind != SPV_NUMBER_SIGNED_INT) {
          return Diag(SPV_ERROR_INVALID_BINARY)
                 << "Invalid OpSwitch: selector id " << selector_id
                 << " is not a scalar integer";
        }
      } else {
        assert(opcode == SpvOpConstant || opcode == SpvOpSpecConstant);
        // The literal number type is determined by the type Id for the
        // constant.
        assert(inst_.type_id);
        if (auto error = SetNumericTypeInfoForType(&operand_, inst_.type_id))
          return error;
      }

      if (auto error = DecodeLiteralNumber(operand_)) return error;

      break;
    }

    case SPV_OPERAND_TYPE_LITERAL_STRING:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_STRING: {
      operand_.type = SPV_OPERAND_TYPE_LITERAL_STRING;
      std::vector<char> str;
      auto* codec = model_->GetLiteralStringHuffmanCodec(inst_.opcode);

      if (codec) {
        std::string decoded_string;
        const bool huffman_result =
            codec->DecodeFromStream(GetReadBitCallback(), &decoded_string);
        assert(huffman_result);
        if (!huffman_result)
          return Diag(SPV_ERROR_INVALID_BINARY)
                 << "Failed to read literal string";

        if (decoded_string != "kMarkvNoneOfTheAbove") {
          std::copy(decoded_string.begin(), decoded_string.end(),
                    std::back_inserter(str));
          str.push_back('\0');
        }
      }

      // The loop is expected to terminate once we encounter '\0' or exhaust
      // the bit stream.
      if (str.empty()) {
        while (true) {
          char ch = 0;
          if (!reader_.ReadUnencoded(&ch))
            return Diag(SPV_ERROR_INVALID_BINARY)
                   << "Failed to read literal string";

          str.push_back(ch);

          if (ch == '\0') break;
        }
      }

      while (str.size() % 4 != 0) str.push_back('\0');

      inst_words_.resize(inst_words_.size() + str.size() / 4);
      std::memcpy(&inst_words_[first_word_index], str.data(), str.size());

      if (SpvOpExtInstImport == opcode) {
        // Record the extended instruction type for the ID for this import.
        // There is only one string literal argument to OpExtInstImport,
        // so it's sufficient to guard this just on the opcode.
        const spv_ext_inst_type_t ext_inst_type =
            spvExtInstImportTypeGet(str.data());
        if (SPV_EXT_INST_TYPE_NONE == ext_inst_type) {
          return Diag(SPV_ERROR_INVALID_BINARY)
                 << "Invalid extended instruction import '" << str.data()
                 << "'";
        }
        // We must have parsed a valid result ID.  It's a condition
        // of the grammar, and we only accept non-zero result Ids.
        assert(inst_.result_id);
        const bool inserted =
            import_id_to_ext_inst_type_.emplace(inst_.result_id, ext_inst_type)
                .second;
        (void)inserted;
        assert(inserted);
      }
      break;
    }

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
    case SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO: {
      // A single word that is a plain enum value.
      uint32_t word = 0;
      const spv_result_t result = DecodeNonIdWord(&word);
      if (result != SPV_SUCCESS) return result;

      inst_words_.push_back(word);

      // Map an optional operand type to its corresponding concrete type.
      if (type == SPV_OPERAND_TYPE_OPTIONAL_ACCESS_QUALIFIER)
        operand_.type = SPV_OPERAND_TYPE_ACCESS_QUALIFIER;

      spv_operand_desc entry;
      if (grammar_.lookupOperand(type, word, &entry)) {
        return Diag(SPV_ERROR_INVALID_BINARY)
               << "Invalid " << spvOperandTypeStr(operand_.type)
               << " operand: " << word;
      }

      // Prepare to accept operands to this operand, if needed.
      spvPushOperandTypes(entry->operandTypes, expected_operands);
      break;
    }

    case SPV_OPERAND_TYPE_FP_FAST_MATH_MODE:
    case SPV_OPERAND_TYPE_FUNCTION_CONTROL:
    case SPV_OPERAND_TYPE_LOOP_CONTROL:
    case SPV_OPERAND_TYPE_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS:
    case SPV_OPERAND_TYPE_SELECTION_CONTROL: {
      // This operand is a mask.
      uint32_t word = 0;
      const spv_result_t result = DecodeNonIdWord(&word);
      if (result != SPV_SUCCESS) return result;

      inst_words_.push_back(word);

      // Map an optional operand type to its corresponding concrete type.
      if (type == SPV_OPERAND_TYPE_OPTIONAL_IMAGE)
        operand_.type = SPV_OPERAND_TYPE_IMAGE;
      else if (type == SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS)
        operand_.type = SPV_OPERAND_TYPE_MEMORY_ACCESS;

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
            return Diag(SPV_ERROR_INVALID_BINARY)
                   << "Invalid " << spvOperandTypeStr(operand_.type)
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
      break;
    }
    default:
      return Diag(SPV_ERROR_INVALID_BINARY)
             << "Internal error: Unhandled operand type: " << type;
  }

  operand_.num_words = uint16_t(inst_words_.size() - first_word_index);

  assert(spvOperandIsConcrete(operand_.type));

  parsed_operands_.push_back(operand_);

  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeInstruction() {
  parsed_operands_.clear();
  inst_words_.clear();

  // Opcode/num_words placeholder, the word will be filled in later.
  inst_words_.push_back(0);

  bool num_operands_still_unknown = true;
  {
    uint32_t opcode = 0;
    uint32_t num_operands = 0;

    const spv_result_t opcode_decoding_result =
        DecodeOpcodeAndNumberOfOperands(&opcode, &num_operands);
    if (opcode_decoding_result < 0) return opcode_decoding_result;

    if (opcode_decoding_result == SPV_SUCCESS) {
      inst_.num_operands = static_cast<uint16_t>(num_operands);
      num_operands_still_unknown = false;
    } else {
      if (!reader_.ReadVariableWidthU32(&opcode,
                                        model_->opcode_chunk_length())) {
        return Diag(SPV_ERROR_INVALID_BINARY)
               << "Failed to read opcode of instruction";
      }
    }

    inst_.opcode = static_cast<uint16_t>(opcode);
  }

  const SpvOp opcode = static_cast<SpvOp>(inst_.opcode);

  spv_opcode_desc opcode_desc;
  if (grammar_.lookupOpcode(opcode, &opcode_desc) != SPV_SUCCESS) {
    return Diag(SPV_ERROR_INVALID_BINARY) << "Invalid opcode";
  }

  spv_operand_pattern_t expected_operands;
  expected_operands.reserve(opcode_desc->numTypes);
  for (auto i = 0; i < opcode_desc->numTypes; i++) {
    expected_operands.push_back(
        opcode_desc->operandTypes[opcode_desc->numTypes - i - 1]);
  }

  if (num_operands_still_unknown) {
    if (!OpcodeHasFixedNumberOfOperands(opcode)) {
      if (!reader_.ReadVariableWidthU16(&inst_.num_operands,
                                        model_->num_operands_chunk_length()))
        return Diag(SPV_ERROR_INVALID_BINARY)
               << "Failed to read num_operands of instruction";
    } else {
      inst_.num_operands = static_cast<uint16_t>(expected_operands.size());
    }
  }

  for (operand_index_ = 0;
       operand_index_ < static_cast<size_t>(inst_.num_operands);
       ++operand_index_) {
    assert(!expected_operands.empty());
    const spv_operand_type_t type =
        spvTakeFirstMatchableOperand(&expected_operands);

    const size_t operand_offset = inst_words_.size();

    const spv_result_t decode_result =
        DecodeOperand(operand_offset, type, &expected_operands);

    if (decode_result != SPV_SUCCESS) return decode_result;
  }

  assert(inst_.num_operands == parsed_operands_.size());

  // Only valid while inst_words_ and parsed_operands_ remain unchanged (until
  // next DecodeInstruction call).
  inst_.words = inst_words_.data();
  inst_.operands = parsed_operands_.empty() ? nullptr : parsed_operands_.data();
  inst_.num_words = static_cast<uint16_t>(inst_words_.size());
  inst_words_[0] = spvOpcodeMake(inst_.num_words, SpvOp(inst_.opcode));

  std::copy(inst_words_.begin(), inst_words_.end(), std::back_inserter(spirv_));

  assert(inst_.num_words ==
             std::accumulate(
                 parsed_operands_.begin(), parsed_operands_.end(), 1,
                 [](int num_words, const spv_parsed_operand_t& operand) {
                   return num_words += operand.num_words;
                 }) &&
         "num_words in instruction doesn't correspond to the sum of num_words"
         "in the operands");

  RecordNumberType();
  ProcessCurInstruction();

  if (!ReadToByteBreak(MarkvCodec::kByteBreakAfterInstIfLessThanUntilNextByte))
    return Diag(SPV_ERROR_INVALID_BINARY) << "Failed to read to byte break";

  if (logger_) {
    logger_->NewLine();
    std::stringstream ss;
    ss << spvOpcodeString(opcode) << " ";
    for (size_t index = 1; index < inst_words_.size(); ++index)
      ss << inst_words_[index] << " ";
    logger_->AppendText(ss.str());
    logger_->NewLine();
    logger_->NewLine();
    if (!logger_->DebugInstruction(inst_)) return SPV_REQUESTED_TERMINATION;
  }

  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::SetNumericTypeInfoForType(
    spv_parsed_operand_t* parsed_operand, uint32_t type_id) {
  assert(type_id != 0);
  auto type_info_iter = type_id_to_number_type_info_.find(type_id);
  if (type_info_iter == type_id_to_number_type_info_.end()) {
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "Type Id " << type_id << " is not a type";
  }

  const NumberType& info = type_info_iter->second;
  if (info.type == SPV_NUMBER_NONE) {
    // This is a valid type, but for something other than a scalar number.
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "Type Id " << type_id << " is not a scalar numeric type";
  }

  parsed_operand->number_kind = info.type;
  parsed_operand->number_bit_width = info.bit_width;
  // Round up the word count.
  parsed_operand->num_words = static_cast<uint16_t>((info.bit_width + 31) / 32);
  return SPV_SUCCESS;
}

void MarkvDecoder::RecordNumberType() {
  const SpvOp opcode = static_cast<SpvOp>(inst_.opcode);
  if (spvOpcodeGeneratesType(opcode)) {
    NumberType info = {SPV_NUMBER_NONE, 0};
    if (SpvOpTypeInt == opcode) {
      info.bit_width = inst_.words[inst_.operands[1].offset];
      info.type = inst_.words[inst_.operands[2].offset]
                      ? SPV_NUMBER_SIGNED_INT
                      : SPV_NUMBER_UNSIGNED_INT;
    } else if (SpvOpTypeFloat == opcode) {
      info.bit_width = inst_.words[inst_.operands[1].offset];
      info.type = SPV_NUMBER_FLOATING;
    }
    // The *result* Id of a type generating instruction is the type Id.
    type_id_to_number_type_info_[inst_.result_id] = info;
  }
}

}  // namespace comp
}  // namespace spvtools
