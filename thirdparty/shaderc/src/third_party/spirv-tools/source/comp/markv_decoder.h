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

#include "source/comp/bit_stream.h"
#include "source/comp/markv.h"
#include "source/comp/markv_codec.h"
#include "source/comp/markv_logger.h"
#include "source/util/make_unique.h"

#ifndef SOURCE_COMP_MARKV_DECODER_H_
#define SOURCE_COMP_MARKV_DECODER_H_

namespace spvtools {
namespace comp {

class MarkvLogger;

// Decodes MARK-V buffers written by MarkvEncoder.
class MarkvDecoder : public MarkvCodec {
 public:
  // |model| is owned by the caller, must be not null and valid during the
  // lifetime of MarkvEncoder.
  MarkvDecoder(spv_const_context context, const std::vector<uint8_t>& markv,
               const MarkvCodecOptions& options, const MarkvModel* model)
      : MarkvCodec(context, GetValidatorOptions(options), model),
        options_(options),
        reader_(markv) {
    SetIdBound(1);
    parsed_operands_.reserve(25);
    inst_words_.reserve(25);
  }
  ~MarkvDecoder() = default;

  // Creates an internal logger which writes comments on the decoding process.
  void CreateLogger(MarkvLogConsumer log_consumer,
                    MarkvDebugConsumer debug_consumer) {
    logger_ = MakeUnique<MarkvLogger>(log_consumer, debug_consumer);
  }

  // Decodes SPIR-V from MARK-V and stores the words in |spirv_binary|.
  // Can be called only once. Fails if data of wrong format or ends prematurely,
  // of if validation fails.
  spv_result_t DecodeModule(std::vector<uint32_t>* spirv_binary);

  // Creates and returns validator options. Returned value owned by the caller.
  static spv_validator_options GetValidatorOptions(
      const MarkvCodecOptions& options) {
    return options.validate_spirv_binary ? spvValidatorOptionsCreate()
                                         : nullptr;
  }

 private:
  // Describes the format of a typed literal number.
  struct NumberType {
    spv_number_kind_t type;
    uint32_t bit_width;
  };

  // Reads a single bit from reader_. The read bit is stored in |bit|.
  // Returns false iff reader_ fails.
  bool ReadBit(bool* bit) {
    uint64_t bits = 0;
    const bool result = reader_.ReadBits(&bits, 1);
    if (result) *bit = bits ? true : false;
    return result;
  };

  // Returns ReadBit bound to the class object.
  std::function<bool(bool*)> GetReadBitCallback() {
    return std::bind(&MarkvDecoder::ReadBit, this, std::placeholders::_1);
  }

  // Reads a single non-id word from bit stream. operand_.type determines if
  // the word needs to be decoded and how.
  spv_result_t DecodeNonIdWord(uint32_t* word);

  // Reads and decodes both opcode and num_operands as a single code.
  // Returns SPV_UNSUPPORTED iff no suitable codec was found.
  spv_result_t DecodeOpcodeAndNumberOfOperands(uint32_t* opcode,
                                               uint32_t* num_operands);

  // Reads mtf rank from bit stream. |mtf| is used to determine the codec
  // scheme. |fallback_method| is used if no codec defined for |mtf|.
  spv_result_t DecodeMtfRankHuffman(uint64_t mtf, uint32_t fallback_method,
                                    uint32_t* rank);

  // Reads id using coding based on mtf associated with the id descriptor.
  // Returns SPV_UNSUPPORTED iff fallback method needs to be used.
  spv_result_t DecodeIdWithDescriptor(uint32_t* id);

  // Reads id using coding based on the given |mtf|, which is expected to
  // contain the needed |id|.
  spv_result_t DecodeExistingId(uint64_t mtf, uint32_t* id);

  // Reads type id of the current instruction if can't be inferred.
  spv_result_t DecodeTypeId();

  // Reads result id of the current instruction if can't be inferred.
  spv_result_t DecodeResultId();

  // Reads id which is neither type nor result id.
  spv_result_t DecodeRefId(uint32_t* id);

  // Reads and discards bits until the beginning of the next byte if the
  // number of bits until the next byte is less than |byte_break_if_less_than|.
  bool ReadToByteBreak(size_t byte_break_if_less_than);

  // Returns instruction words decoded up to this point.
  const uint32_t* GetInstWords() const override { return inst_words_.data(); }

  // Reads a literal number as it is described in |operand| from the bit stream,
  // decodes and writes it to spirv_.
  spv_result_t DecodeLiteralNumber(const spv_parsed_operand_t& operand);

  // Reads instruction from bit stream, decodes and validates it.
  // Decoded instruction is valid until the next call of DecodeInstruction().
  spv_result_t DecodeInstruction();

  // Read operand from the stream decodes and validates it.
  spv_result_t DecodeOperand(size_t operand_offset,
                             const spv_operand_type_t type,
                             spv_operand_pattern_t* expected_operands);

  // Records the numeric type for an operand according to the type information
  // associated with the given non-zero type Id.  This can fail if the type Id
  // is not a type Id, or if the type Id does not reference a scalar numeric
  // type.  On success, return SPV_SUCCESS and populates the num_words,
  // number_kind, and number_bit_width fields of parsed_operand.
  spv_result_t SetNumericTypeInfoForType(spv_parsed_operand_t* parsed_operand,
                                         uint32_t type_id);

  // Records the number type for the current instruction, if it generates a
  // type. For types that aren't scalar numbers, record something with number
  // kind SPV_NUMBER_NONE.
  void RecordNumberType();

  MarkvCodecOptions options_;

  // Temporary sink where decoded SPIR-V words are written. Once it contains the
  // entire module, the container is moved and returned.
  std::vector<uint32_t> spirv_;

  // Bit stream containing encoded data.
  BitReaderWord64 reader_;

  // Temporary storage for operands of the currently parsed instruction.
  // Valid until next DecodeInstruction call.
  std::vector<spv_parsed_operand_t> parsed_operands_;

  // Temporary storage for current instruction words.
  // Valid until next DecodeInstruction call.
  std::vector<uint32_t> inst_words_;

  // Maps a type ID to its number type description.
  std::unordered_map<uint32_t, NumberType> type_id_to_number_type_info_;

  // Maps an ExtInstImport id to the extended instruction type.
  std::unordered_map<uint32_t, spv_ext_inst_type_t> import_id_to_ext_inst_type_;
};

}  // namespace comp
}  // namespace spvtools

#endif  // SOURCE_COMP_MARKV_DECODER_H_
