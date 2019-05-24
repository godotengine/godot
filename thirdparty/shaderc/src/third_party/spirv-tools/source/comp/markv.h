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

// MARK-V is a compression format for SPIR-V binaries. It strips away
// non-essential information (such as result ids which can be regenerated) and
// uses various bit reduction techiniques to reduce the size of the binary and
// make it more similar to other compressed SPIR-V files to further improve
// compression of the dataset.

#ifndef SOURCE_COMP_MARKV_H_
#define SOURCE_COMP_MARKV_H_

#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace comp {

class MarkvModel;

struct MarkvCodecOptions {
  bool validate_spirv_binary = false;
};

// Debug callback. Called once per instruction.
// |words| is instruction SPIR-V words.
// |bits| is a textual representation of the MARK-V bit sequence used to encode
// the instruction (char '0' for 0, char '1' for 1).
// |comment| contains all logs generated while processing the instruction.
using MarkvDebugConsumer =
    std::function<bool(const std::vector<uint32_t>& words,
                       const std::string& bits, const std::string& comment)>;

// Logging callback. Called often (if decoder reads a single bit, the log
// consumer will receive 1 character string with that bit).
// This callback is more suitable for continous output than MarkvDebugConsumer,
// for example if the codec crashes it would allow to pinpoint on which operand
// or bit the crash happened.
// |snippet| could be any atomic fragment of text logged by the codec. It can
// contain a paragraph of text with newlines, or can be just one character.
using MarkvLogConsumer = std::function<void(const std::string& snippet)>;

// Encodes the given SPIR-V binary to MARK-V binary.
// |log_consumer| is optional (pass MarkvLogConsumer() to disable).
// |debug_consumer| is optional (pass MarkvDebugConsumer() to disable).
spv_result_t SpirvToMarkv(
    spv_const_context context, const std::vector<uint32_t>& spirv,
    const MarkvCodecOptions& options, const MarkvModel& markv_model,
    MessageConsumer message_consumer, MarkvLogConsumer log_consumer,
    MarkvDebugConsumer debug_consumer, std::vector<uint8_t>* markv);

// Decodes a SPIR-V binary from the given MARK-V binary.
// |log_consumer| is optional (pass MarkvLogConsumer() to disable).
// |debug_consumer| is optional (pass MarkvDebugConsumer() to disable).
spv_result_t MarkvToSpirv(
    spv_const_context context, const std::vector<uint8_t>& markv,
    const MarkvCodecOptions& options, const MarkvModel& markv_model,
    MessageConsumer message_consumer, MarkvLogConsumer log_consumer,
    MarkvDebugConsumer debug_consumer, std::vector<uint32_t>* spirv);

}  // namespace comp
}  // namespace spvtools

#endif  // SOURCE_COMP_MARKV_H_
