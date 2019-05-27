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

#include "source/comp/markv_encoder.h"

#include "source/binary.h"
#include "source/opcode.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace comp {
namespace {

const size_t kCommentNumWhitespaces = 2;

}  // namespace

spv_result_t MarkvEncoder::EncodeNonIdWord(uint32_t word) {
  auto* codec = model_->GetNonIdWordHuffmanCodec(inst_.opcode, operand_index_);

  if (codec) {
    uint64_t bits = 0;
    size_t num_bits = 0;
    if (codec->Encode(word, &bits, &num_bits)) {
      // Encoding successful.
      writer_.WriteBits(bits, num_bits);
      return SPV_SUCCESS;
    } else {
      // Encoding failed, write kMarkvNoneOfTheAbove flag.
      if (!codec->Encode(MarkvModel::GetMarkvNoneOfTheAbove(), &bits,
                         &num_bits))
        return Diag(SPV_ERROR_INTERNAL)
               << "Non-id word Huffman table for "
               << spvOpcodeString(SpvOp(inst_.opcode)) << " operand index "
               << operand_index_ << " is missing kMarkvNoneOfTheAbove";
      writer_.WriteBits(bits, num_bits);
    }
  }

  // Fallback encoding.
  const size_t chunk_length =
      model_->GetOperandVariableWidthChunkLength(operand_.type);
  if (chunk_length) {
    writer_.WriteVariableWidthU32(word, chunk_length);
  } else {
    writer_.WriteUnencoded(word);
  }
  return SPV_SUCCESS;
}

spv_result_t MarkvEncoder::EncodeOpcodeAndNumOperands(uint32_t opcode,
                                                      uint32_t num_operands) {
  uint64_t bits = 0;
  size_t num_bits = 0;

  const uint32_t word = opcode | (num_operands << 16);

  // First try to use the Markov chain codec.
  auto* codec =
      model_->GetOpcodeAndNumOperandsMarkovHuffmanCodec(GetPrevOpcode());
  if (codec) {
    if (codec->Encode(word, &bits, &num_bits)) {
      // The word was successfully encoded into bits/num_bits.
      writer_.WriteBits(bits, num_bits);
      return SPV_SUCCESS;
    } else {
      // The word is not in the Huffman table. Write kMarkvNoneOfTheAbove
      // and use fallback encoding.
      if (!codec->Encode(MarkvModel::GetMarkvNoneOfTheAbove(), &bits,
                         &num_bits))
        return Diag(SPV_ERROR_INTERNAL)
               << "opcode_and_num_operands Huffman table for "
               << spvOpcodeString(GetPrevOpcode())
               << "is missing kMarkvNoneOfTheAbove";
      writer_.WriteBits(bits, num_bits);
    }
  }

  // Fallback to base-rate codec.
  codec = model_->GetOpcodeAndNumOperandsMarkovHuffmanCodec(SpvOpNop);
  assert(codec);
  if (codec->Encode(word, &bits, &num_bits)) {
    // The word was successfully encoded into bits/num_bits.
    writer_.WriteBits(bits, num_bits);
    return SPV_SUCCESS;
  } else {
    // The word is not in the Huffman table. Write kMarkvNoneOfTheAbove
    // and return false.
    if (!codec->Encode(MarkvModel::GetMarkvNoneOfTheAbove(), &bits, &num_bits))
      return Diag(SPV_ERROR_INTERNAL)
             << "Global opcode_and_num_operands Huffman table is missing "
             << "kMarkvNoneOfTheAbove";
    writer_.WriteBits(bits, num_bits);
    return SPV_UNSUPPORTED;
  }
}

spv_result_t MarkvEncoder::EncodeMtfRankHuffman(uint32_t rank, uint64_t mtf,
                                                uint64_t fallback_method) {
  const auto* codec = GetMtfHuffmanCodec(mtf);
  if (!codec) {
    assert(fallback_method != kMtfNone);
    codec = GetMtfHuffmanCodec(fallback_method);
  }

  if (!codec) return Diag(SPV_ERROR_INTERNAL) << "No codec to encode MTF rank";

  uint64_t bits = 0;
  size_t num_bits = 0;
  if (rank < MarkvCodec::kMtfSmallestRankEncodedByValue) {
    // Encode using Huffman coding.
    if (!codec->Encode(rank, &bits, &num_bits))
      return Diag(SPV_ERROR_INTERNAL)
             << "Failed to encode MTF rank with Huffman";

    writer_.WriteBits(bits, num_bits);
  } else {
    // Encode by value.
    if (!codec->Encode(MarkvCodec::kMtfRankEncodedByValueSignal, &bits,
                       &num_bits))
      return Diag(SPV_ERROR_INTERNAL)
             << "Failed to encode kMtfRankEncodedByValueSignal";

    writer_.WriteBits(bits, num_bits);
    writer_.WriteVariableWidthU32(
        rank - MarkvCodec::kMtfSmallestRankEncodedByValue,
        model_->mtf_rank_chunk_length());
  }
  return SPV_SUCCESS;
}

spv_result_t MarkvEncoder::EncodeIdWithDescriptor(uint32_t id) {
  // Get the descriptor for id.
  const uint32_t long_descriptor = long_id_descriptors_.GetDescriptor(id);
  auto* codec =
      model_->GetIdDescriptorHuffmanCodec(inst_.opcode, operand_index_);
  uint64_t bits = 0;
  size_t num_bits = 0;
  uint64_t mtf = kMtfNone;
  if (long_descriptor && codec &&
      codec->Encode(long_descriptor, &bits, &num_bits)) {
    // If the descriptor exists and is in the table, write the descriptor and
    // proceed to encoding the rank.
    writer_.WriteBits(bits, num_bits);
    mtf = GetMtfLongIdDescriptor(long_descriptor);
  } else {
    if (codec) {
      // The descriptor doesn't exist or we have no coding for it. Write
      // kMarkvNoneOfTheAbove and go to fallback method.
      if (!codec->Encode(MarkvModel::GetMarkvNoneOfTheAbove(), &bits,
                         &num_bits))
        return Diag(SPV_ERROR_INTERNAL)
               << "Descriptor Huffman table for "
               << spvOpcodeString(SpvOp(inst_.opcode)) << " operand index "
               << operand_index_ << " is missing kMarkvNoneOfTheAbove";

      writer_.WriteBits(bits, num_bits);
    }

    if (model_->id_fallback_strategy() !=
        MarkvModel::IdFallbackStrategy::kShortDescriptor) {
      return SPV_UNSUPPORTED;
    }

    const uint32_t short_descriptor = short_id_descriptors_.GetDescriptor(id);
    writer_.WriteBits(short_descriptor, MarkvCodec::kShortDescriptorNumBits);

    if (short_descriptor == 0) {
      // Forward declared id.
      return SPV_UNSUPPORTED;
    }

    mtf = GetMtfShortIdDescriptor(short_descriptor);
  }

  // Descriptor has been encoded. Now encode the rank of the id in the
  // associated mtf sequence.
  return EncodeExistingId(mtf, id);
}

spv_result_t MarkvEncoder::EncodeExistingId(uint64_t mtf, uint32_t id) {
  assert(multi_mtf_.GetSize(mtf) > 0);
  if (multi_mtf_.GetSize(mtf) == 1) {
    // If the sequence has only one element no need to write rank, the decoder
    // would make the same decision.
    return SPV_SUCCESS;
  }

  uint32_t rank = 0;
  if (!multi_mtf_.RankFromValue(mtf, id, &rank))
    return Diag(SPV_ERROR_INTERNAL) << "Id is not in the MTF sequence";

  return EncodeMtfRankHuffman(rank, mtf, kMtfGenericNonZeroRank);
}

spv_result_t MarkvEncoder::EncodeRefId(uint32_t id) {
  {
    // Try to encode using id descriptor mtfs.
    const spv_result_t result = EncodeIdWithDescriptor(id);
    if (result != SPV_UNSUPPORTED) return result;
    // If can't be done continue with other methods.
  }

  const bool can_forward_declare = spvOperandCanBeForwardDeclaredFunction(
      SpvOp(inst_.opcode))(operand_index_);
  uint32_t rank = 0;

  if (model_->id_fallback_strategy() ==
      MarkvModel::IdFallbackStrategy::kRuleBased) {
    // Encode using rule-based mtf.
    uint64_t mtf = GetRuleBasedMtf();

    if (mtf != kMtfNone && !can_forward_declare) {
      assert(multi_mtf_.HasValue(kMtfAll, id));
      return EncodeExistingId(mtf, id);
    }

    if (mtf == kMtfNone) mtf = kMtfAll;

    if (!multi_mtf_.RankFromValue(mtf, id, &rank)) {
      // This is the first occurrence of a forward declared id.
      multi_mtf_.Insert(kMtfAll, id);
      multi_mtf_.Insert(kMtfForwardDeclared, id);
      if (mtf != kMtfAll) multi_mtf_.Insert(mtf, id);
      rank = 0;
    }

    return EncodeMtfRankHuffman(rank, mtf, kMtfAll);
  } else {
    assert(can_forward_declare);

    if (!multi_mtf_.RankFromValue(kMtfForwardDeclared, id, &rank)) {
      // This is the first occurrence of a forward declared id.
      multi_mtf_.Insert(kMtfForwardDeclared, id);
      rank = 0;
    }

    writer_.WriteVariableWidthU32(rank, model_->mtf_rank_chunk_length());
    return SPV_SUCCESS;
  }
}

spv_result_t MarkvEncoder::EncodeTypeId() {
  if (inst_.opcode == SpvOpFunctionParameter) {
    assert(!remaining_function_parameter_types_.empty());
    assert(inst_.type_id == remaining_function_parameter_types_.front());
    remaining_function_parameter_types_.pop_front();
    return SPV_SUCCESS;
  }

  {
    // Try to encode using id descriptor mtfs.
    const spv_result_t result = EncodeIdWithDescriptor(inst_.type_id);
    if (result != SPV_UNSUPPORTED) return result;
    // If can't be done continue with other methods.
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

  return EncodeExistingId(mtf, inst_.type_id);
}

spv_result_t MarkvEncoder::EncodeResultId() {
  uint32_t rank = 0;

  const uint64_t num_still_forward_declared =
      multi_mtf_.GetSize(kMtfForwardDeclared);

  if (num_still_forward_declared) {
    // We write the rank only if kMtfForwardDeclared is not empty. If it is
    // empty the decoder knows that there are no forward declared ids to expect.
    if (multi_mtf_.RankFromValue(kMtfForwardDeclared, inst_.result_id, &rank)) {
      // This is a definition of a forward declared id. We can remove the id
      // from kMtfForwardDeclared.
      if (!multi_mtf_.Remove(kMtfForwardDeclared, inst_.result_id))
        return Diag(SPV_ERROR_INTERNAL)
               << "Failed to remove id from kMtfForwardDeclared";
      writer_.WriteBits(1, 1);
      writer_.WriteVariableWidthU32(rank, model_->mtf_rank_chunk_length());
    } else {
      rank = 0;
      writer_.WriteBits(0, 1);
    }
  }

  if (model_->id_fallback_strategy() ==
      MarkvModel::IdFallbackStrategy::kRuleBased) {
    if (!rank) {
      multi_mtf_.Insert(kMtfAll, inst_.result_id);
    }
  }

  return SPV_SUCCESS;
}

spv_result_t MarkvEncoder::EncodeLiteralNumber(
    const spv_parsed_operand_t& operand) {
  if (operand.number_bit_width <= 32) {
    const uint32_t word = inst_.words[operand.offset];
    return EncodeNonIdWord(word);
  } else {
    assert(operand.number_bit_width <= 64);
    const uint64_t word = uint64_t(inst_.words[operand.offset]) |
                          (uint64_t(inst_.words[operand.offset + 1]) << 32);
    if (operand.number_kind == SPV_NUMBER_UNSIGNED_INT) {
      writer_.WriteVariableWidthU64(word, model_->u64_chunk_length());
    } else if (operand.number_kind == SPV_NUMBER_SIGNED_INT) {
      int64_t val = 0;
      std::memcpy(&val, &word, 8);
      writer_.WriteVariableWidthS64(val, model_->s64_chunk_length(),
                                    model_->s64_block_exponent());
    } else if (operand.number_kind == SPV_NUMBER_FLOATING) {
      writer_.WriteUnencoded(word);
    } else {
      return Diag(SPV_ERROR_INTERNAL) << "Unsupported bit length";
    }
  }
  return SPV_SUCCESS;
}

void MarkvEncoder::AddByteBreak(size_t byte_break_if_less_than) {
  const size_t num_bits_to_next_byte =
      GetNumBitsToNextByte(writer_.GetNumBits());
  if (num_bits_to_next_byte == 0 ||
      num_bits_to_next_byte > byte_break_if_less_than)
    return;

  if (logger_) {
    logger_->AppendWhitespaces(kCommentNumWhitespaces);
    logger_->AppendText("<byte break>");
  }

  writer_.WriteBits(0, num_bits_to_next_byte);
}

spv_result_t MarkvEncoder::EncodeInstruction(
    const spv_parsed_instruction_t& inst) {
  SpvOp opcode = SpvOp(inst.opcode);
  inst_ = inst;

  LogDisassemblyInstruction();

  const spv_result_t opcode_encodig_result =
      EncodeOpcodeAndNumOperands(opcode, inst.num_operands);
  if (opcode_encodig_result < 0) return opcode_encodig_result;

  if (opcode_encodig_result != SPV_SUCCESS) {
    // Fallback encoding for opcode and num_operands.
    writer_.WriteVariableWidthU32(opcode, model_->opcode_chunk_length());

    if (!OpcodeHasFixedNumberOfOperands(opcode)) {
      // If the opcode has a variable number of operands, encode the number of
      // operands with the instruction.

      if (logger_) logger_->AppendWhitespaces(kCommentNumWhitespaces);

      writer_.WriteVariableWidthU16(inst.num_operands,
                                    model_->num_operands_chunk_length());
    }
  }

  // Write operands.
  const uint32_t num_operands = inst_.num_operands;
  for (operand_index_ = 0; operand_index_ < num_operands; ++operand_index_) {
    operand_ = inst_.operands[operand_index_];

    if (logger_) {
      logger_->AppendWhitespaces(kCommentNumWhitespaces);
      logger_->AppendText("<");
      logger_->AppendText(spvOperandTypeStr(operand_.type));
      logger_->AppendText(">");
    }

    switch (operand_.type) {
      case SPV_OPERAND_TYPE_RESULT_ID:
      case SPV_OPERAND_TYPE_TYPE_ID:
      case SPV_OPERAND_TYPE_ID:
      case SPV_OPERAND_TYPE_OPTIONAL_ID:
      case SPV_OPERAND_TYPE_SCOPE_ID:
      case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID: {
        const uint32_t id = inst_.words[operand_.offset];
        if (operand_.type == SPV_OPERAND_TYPE_TYPE_ID) {
          const spv_result_t result = EncodeTypeId();
          if (result != SPV_SUCCESS) return result;
        } else if (operand_.type == SPV_OPERAND_TYPE_RESULT_ID) {
          const spv_result_t result = EncodeResultId();
          if (result != SPV_SUCCESS) return result;
        } else {
          const spv_result_t result = EncodeRefId(id);
          if (result != SPV_SUCCESS) return result;
        }

        PromoteIfNeeded(id);
        break;
      }

      case SPV_OPERAND_TYPE_LITERAL_INTEGER: {
        const spv_result_t result =
            EncodeNonIdWord(inst_.words[operand_.offset]);
        if (result != SPV_SUCCESS) return result;
        break;
      }

      case SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER: {
        const spv_result_t result = EncodeLiteralNumber(operand_);
        if (result != SPV_SUCCESS) return result;
        break;
      }

      case SPV_OPERAND_TYPE_LITERAL_STRING: {
        const char* src =
            reinterpret_cast<const char*>(&inst_.words[operand_.offset]);

        auto* codec = model_->GetLiteralStringHuffmanCodec(opcode);
        if (codec) {
          uint64_t bits = 0;
          size_t num_bits = 0;
          const std::string str = src;
          if (codec->Encode(str, &bits, &num_bits)) {
            writer_.WriteBits(bits, num_bits);
            break;
          } else {
            bool result =
                codec->Encode("kMarkvNoneOfTheAbove", &bits, &num_bits);
            (void)result;
            assert(result);
            writer_.WriteBits(bits, num_bits);
          }
        }

        const size_t length = spv_strnlen_s(src, operand_.num_words * 4);
        if (length == operand_.num_words * 4)
          return Diag(SPV_ERROR_INVALID_BINARY)
                 << "Failed to find terminal character of literal string";
        for (size_t i = 0; i < length + 1; ++i) writer_.WriteUnencoded(src[i]);
        break;
      }

      default: {
        for (int i = 0; i < operand_.num_words; ++i) {
          const uint32_t word = inst_.words[operand_.offset + i];
          const spv_result_t result = EncodeNonIdWord(word);
          if (result != SPV_SUCCESS) return result;
        }
        break;
      }
    }
  }

  AddByteBreak(MarkvCodec::kByteBreakAfterInstIfLessThanUntilNextByte);

  if (logger_) {
    logger_->NewLine();
    logger_->NewLine();
    if (!logger_->DebugInstruction(inst_)) return SPV_REQUESTED_TERMINATION;
  }

  ProcessCurInstruction();

  return SPV_SUCCESS;
}

}  // namespace comp
}  // namespace spvtools
