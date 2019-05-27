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

#include "source/comp/markv.h"

#include "source/comp/markv_decoder.h"
#include "source/comp/markv_encoder.h"

namespace spvtools {
namespace comp {
namespace {

spv_result_t EncodeHeader(void* user_data, spv_endianness_t endian,
                          uint32_t magic, uint32_t version, uint32_t generator,
                          uint32_t id_bound, uint32_t schema) {
  MarkvEncoder* encoder = reinterpret_cast<MarkvEncoder*>(user_data);
  return encoder->EncodeHeader(endian, magic, version, generator, id_bound,
                               schema);
}

spv_result_t EncodeInstruction(void* user_data,
                               const spv_parsed_instruction_t* inst) {
  MarkvEncoder* encoder = reinterpret_cast<MarkvEncoder*>(user_data);
  return encoder->EncodeInstruction(*inst);
}

}  // namespace

spv_result_t SpirvToMarkv(
    spv_const_context context, const std::vector<uint32_t>& spirv,
    const MarkvCodecOptions& options, const MarkvModel& markv_model,
    MessageConsumer message_consumer, MarkvLogConsumer log_consumer,
    MarkvDebugConsumer debug_consumer, std::vector<uint8_t>* markv) {
  spv_context_t hijack_context = *context;
  SetContextMessageConsumer(&hijack_context, message_consumer);

  spv_validator_options validator_options =
      MarkvDecoder::GetValidatorOptions(options);
  if (validator_options) {
    spv_const_binary_t spirv_binary = {spirv.data(), spirv.size()};
    const spv_result_t result = spvValidateWithOptions(
        &hijack_context, validator_options, &spirv_binary, nullptr);
    if (result != SPV_SUCCESS) return result;
  }

  MarkvEncoder encoder(&hijack_context, options, &markv_model);

  spv_position_t position = {};
  if (log_consumer || debug_consumer) {
    encoder.CreateLogger(log_consumer, debug_consumer);

    spv_text text = nullptr;
    if (spvBinaryToText(&hijack_context, spirv.data(), spirv.size(),
                        SPV_BINARY_TO_TEXT_OPTION_NO_HEADER, &text,
                        nullptr) != SPV_SUCCESS) {
      return DiagnosticStream(position, hijack_context.consumer, "",
                              SPV_ERROR_INVALID_BINARY)
             << "Failed to disassemble SPIR-V binary.";
    }
    assert(text);
    encoder.SetDisassembly(std::string(text->str, text->length));
    spvTextDestroy(text);
  }

  if (spvBinaryParse(&hijack_context, &encoder, spirv.data(), spirv.size(),
                     EncodeHeader, EncodeInstruction, nullptr) != SPV_SUCCESS) {
    return DiagnosticStream(position, hijack_context.consumer, "",
                            SPV_ERROR_INVALID_BINARY)
           << "Unable to encode to MARK-V.";
  }

  *markv = encoder.GetMarkvBinary();
  return SPV_SUCCESS;
}

spv_result_t MarkvToSpirv(
    spv_const_context context, const std::vector<uint8_t>& markv,
    const MarkvCodecOptions& options, const MarkvModel& markv_model,
    MessageConsumer message_consumer, MarkvLogConsumer log_consumer,
    MarkvDebugConsumer debug_consumer, std::vector<uint32_t>* spirv) {
  spv_position_t position = {};
  spv_context_t hijack_context = *context;
  SetContextMessageConsumer(&hijack_context, message_consumer);

  MarkvDecoder decoder(&hijack_context, markv, options, &markv_model);

  if (log_consumer || debug_consumer)
    decoder.CreateLogger(log_consumer, debug_consumer);

  if (decoder.DecodeModule(spirv) != SPV_SUCCESS) {
    return DiagnosticStream(position, hijack_context.consumer, "",
                            SPV_ERROR_INVALID_BINARY)
           << "Unable to decode MARK-V.";
  }

  assert(!spirv->empty());
  return SPV_SUCCESS;
}

}  // namespace comp
}  // namespace spvtools
