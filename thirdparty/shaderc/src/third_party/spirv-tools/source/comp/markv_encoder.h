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

#ifndef SOURCE_COMP_MARKV_ENCODER_H_
#define SOURCE_COMP_MARKV_ENCODER_H_

#include <cstring>

namespace spvtools {
namespace comp {

// SPIR-V to MARK-V encoder. Exposes functions EncodeHeader and
// EncodeInstruction which can be used as callback by spvBinaryParse.
// Encoded binary is written to an internally maintained bitstream.
// After the last instruction is encoded, the resulting MARK-V binary can be
// acquired by calling GetMarkvBinary().
//
// The encoder uses SPIR-V validator to keep internal state, therefore
// SPIR-V binary needs to be able to pass validator checks.
// CreateCommentsLogger() can be used to enable the encoder to write comments
// on how encoding was done, which can later be accessed with GetComments().
class MarkvEncoder : public MarkvCodec {
 public:
  // |model| is owned by the caller, must be not null and valid during the
  // lifetime of MarkvEncoder.
  MarkvEncoder(spv_const_context context, const MarkvCodecOptions& options,
               const MarkvModel* model)
      : MarkvCodec(context, GetValidatorOptions(options), model),
        options_(options) {}
  ~MarkvEncoder() override = default;

  // Writes data from SPIR-V header to MARK-V header.
  spv_result_t EncodeHeader(spv_endianness_t /* endian */, uint32_t /* magic */,
                            uint32_t version, uint32_t generator,
                            uint32_t id_bound, uint32_t /* schema */) {
    SetIdBound(id_bound);
    header_.spirv_version = version;
    header_.spirv_generator = generator;
    return SPV_SUCCESS;
  }

  // Creates an internal logger which writes comments on the encoding process.
  void CreateLogger(MarkvLogConsumer log_consumer,
                    MarkvDebugConsumer debug_consumer) {
    logger_ = MakeUnique<MarkvLogger>(log_consumer, debug_consumer);
    writer_.SetCallback(
        [this](const std::string& str) { logger_->AppendBitSequence(str); });
  }

  // Encodes SPIR-V instruction to MARK-V and writes to bit stream.
  // Operation can fail if the instruction fails to pass the validator or if
  // the encoder stubmles on something unexpected.
  spv_result_t EncodeInstruction(const spv_parsed_instruction_t& inst);

  // Concatenates MARK-V header and the bit stream with encoded instructions
  // into a single buffer and returns it as spv_markv_binary. The returned
  // value is owned by the caller and needs to be destroyed with
  // spvMarkvBinaryDestroy().
  std::vector<uint8_t> GetMarkvBinary() {
    header_.markv_length_in_bits =
        static_cast<uint32_t>(sizeof(header_) * 8 + writer_.GetNumBits());
    header_.markv_model =
        (model_->model_type() << 16) | model_->model_version();

    const size_t num_bytes = sizeof(header_) + writer_.GetDataSizeBytes();
    std::vector<uint8_t> markv(num_bytes);

    assert(writer_.GetData());
    std::memcpy(markv.data(), &header_, sizeof(header_));
    std::memcpy(markv.data() + sizeof(header_), writer_.GetData(),
                writer_.GetDataSizeBytes());
    return markv;
  }

  // Optionally adds disassembly to the comments.
  // Disassembly should contain all instructions in the module separated by
  // \n, and no header.
  void SetDisassembly(std::string&& disassembly) {
    disassembly_ = MakeUnique<std::stringstream>(std::move(disassembly));
  }

  // Extracts the next instruction line from the disassembly and logs it.
  void LogDisassemblyInstruction() {
    if (logger_ && disassembly_) {
      std::string line;
      std::getline(*disassembly_, line, '\n');
      logger_->AppendTextNewLine(line);
    }
  }

 private:
  // Creates and returns validator options. Returned value owned by the caller.
  static spv_validator_options GetValidatorOptions(
      const MarkvCodecOptions& options) {
    return options.validate_spirv_binary ? spvValidatorOptionsCreate()
                                         : nullptr;
  }

  // Writes a single word to bit stream. operand_.type determines if the word is
  // encoded and how.
  spv_result_t EncodeNonIdWord(uint32_t word);

  // Writes both opcode and num_operands as a single code.
  // Returns SPV_UNSUPPORTED iff no suitable codec was found.
  spv_result_t EncodeOpcodeAndNumOperands(uint32_t opcode,
                                          uint32_t num_operands);

  // Writes mtf rank to bit stream. |mtf| is used to determine the codec
  // scheme. |fallback_method| is used if no codec defined for |mtf|.
  spv_result_t EncodeMtfRankHuffman(uint32_t rank, uint64_t mtf,
                                    uint64_t fallback_method);

  // Writes id using coding based on mtf associated with the id descriptor.
  // Returns SPV_UNSUPPORTED iff fallback method needs to be used.
  spv_result_t EncodeIdWithDescriptor(uint32_t id);

  // Writes id using coding based on the given |mtf|, which is expected to
  // contain the given |id|.
  spv_result_t EncodeExistingId(uint64_t mtf, uint32_t id);

  // Writes type id of the current instruction if can't be inferred.
  spv_result_t EncodeTypeId();

  // Writes result id of the current instruction if can't be inferred.
  spv_result_t EncodeResultId();

  // Writes ids which are neither type nor result ids.
  spv_result_t EncodeRefId(uint32_t id);

  // Writes bits to the stream until the beginning of the next byte if the
  // number of bits until the next byte is less than |byte_break_if_less_than|.
  void AddByteBreak(size_t byte_break_if_less_than);

  // Encodes a literal number operand and writes it to the bit stream.
  spv_result_t EncodeLiteralNumber(const spv_parsed_operand_t& operand);

  MarkvCodecOptions options_;

  // Bit stream where encoded instructions are written.
  BitWriterWord64 writer_;

  // If not nullptr, disassembled instruction lines will be written to comments.
  // Format: \n separated instruction lines, no header.
  std::unique_ptr<std::stringstream> disassembly_;
};

}  // namespace comp
}  // namespace spvtools

#endif  // SOURCE_COMP_MARKV_ENCODER_H_
