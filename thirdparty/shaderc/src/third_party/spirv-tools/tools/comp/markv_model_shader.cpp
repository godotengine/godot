// Copyright (c) 2017 Google Inc.
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

#include "tools/comp/markv_model_shader.h"

#include <algorithm>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/util/make_unique.h"

namespace spvtools {
namespace comp {
namespace {

// Signals that the value is not in the coding scheme and a fallback method
// needs to be used.
const uint64_t kMarkvNoneOfTheAbove = MarkvModel::GetMarkvNoneOfTheAbove();

inline uint32_t CombineOpcodeAndNumOperands(uint32_t opcode,
                                            uint32_t num_operands) {
  return opcode | (num_operands << 16);
}

#include "tools/comp/markv_model_shader_default_autogen.inc"

}  // namespace

MarkvModelShaderLite::MarkvModelShaderLite() {
  const uint16_t kVersionNumber = 1;
  SetModelVersion(kVersionNumber);

  opcode_and_num_operands_huffman_codec_ =
      MakeUnique<HuffmanCodec<uint64_t>>(GetOpcodeAndNumOperandsHist());

  id_fallback_strategy_ = IdFallbackStrategy::kShortDescriptor;
}

MarkvModelShaderMid::MarkvModelShaderMid() {
  const uint16_t kVersionNumber = 1;
  SetModelVersion(kVersionNumber);

  opcode_and_num_operands_huffman_codec_ =
      MakeUnique<HuffmanCodec<uint64_t>>(GetOpcodeAndNumOperandsHist());
  non_id_word_huffman_codecs_ = GetNonIdWordHuffmanCodecs();
  id_descriptor_huffman_codecs_ = GetIdDescriptorHuffmanCodecs();
  descriptors_with_coding_scheme_ = GetDescriptorsWithCodingScheme();
  literal_string_huffman_codecs_ = GetLiteralStringHuffmanCodecs();

  id_fallback_strategy_ = IdFallbackStrategy::kShortDescriptor;
}

MarkvModelShaderMax::MarkvModelShaderMax() {
  const uint16_t kVersionNumber = 1;
  SetModelVersion(kVersionNumber);

  opcode_and_num_operands_huffman_codec_ =
      MakeUnique<HuffmanCodec<uint64_t>>(GetOpcodeAndNumOperandsHist());
  opcode_and_num_operands_markov_huffman_codecs_ =
      GetOpcodeAndNumOperandsMarkovHuffmanCodecs();
  non_id_word_huffman_codecs_ = GetNonIdWordHuffmanCodecs();
  id_descriptor_huffman_codecs_ = GetIdDescriptorHuffmanCodecs();
  descriptors_with_coding_scheme_ = GetDescriptorsWithCodingScheme();
  literal_string_huffman_codecs_ = GetLiteralStringHuffmanCodecs();

  id_fallback_strategy_ = IdFallbackStrategy::kRuleBased;
}

}  // namespace comp
}  // namespace spvtools
