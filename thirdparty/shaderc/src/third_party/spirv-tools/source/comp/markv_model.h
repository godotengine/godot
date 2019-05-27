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

#ifndef SOURCE_COMP_MARKV_MODEL_H_
#define SOURCE_COMP_MARKV_MODEL_H_

#include <unordered_set>

#include "source/comp/huffman_codec.h"
#include "source/latest_version_spirv_header.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace comp {

// Base class for MARK-V models.
// The class contains encoding/decoding model with various constants and
// codecs used by the compression algorithm.
class MarkvModel {
 public:
  MarkvModel()
      : operand_chunk_lengths_(
            static_cast<size_t>(SPV_OPERAND_TYPE_NUM_OPERAND_TYPES), 0) {
    // Set default values.
    operand_chunk_lengths_[SPV_OPERAND_TYPE_TYPE_ID] = 4;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_RESULT_ID] = 8;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_ID] = 8;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_SCOPE_ID] = 8;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID] = 8;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_LITERAL_INTEGER] = 6;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER] = 6;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_CAPABILITY] = 6;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_SOURCE_LANGUAGE] = 3;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_EXECUTION_MODEL] = 3;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_ADDRESSING_MODEL] = 2;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_MEMORY_MODEL] = 2;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_EXECUTION_MODE] = 6;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_STORAGE_CLASS] = 4;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_DIMENSIONALITY] = 3;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE] = 3;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE] = 2;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_SAMPLER_IMAGE_FORMAT] = 6;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_FP_ROUNDING_MODE] = 2;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_LINKAGE_TYPE] = 2;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_ACCESS_QUALIFIER] = 2;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_OPTIONAL_ACCESS_QUALIFIER] = 2;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE] = 3;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_DECORATION] = 6;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_BUILT_IN] = 6;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_GROUP_OPERATION] = 2;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS] = 2;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO] = 2;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_FP_FAST_MATH_MODE] = 4;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_FUNCTION_CONTROL] = 4;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_LOOP_CONTROL] = 4;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_IMAGE] = 4;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_OPTIONAL_IMAGE] = 4;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS] = 4;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_SELECTION_CONTROL] = 4;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER] = 6;
    operand_chunk_lengths_[SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER] = 6;
  }

  uint32_t model_type() const { return model_type_; }
  uint32_t model_version() const { return model_version_; }

  uint32_t opcode_chunk_length() const { return opcode_chunk_length_; }
  uint32_t num_operands_chunk_length() const {
    return num_operands_chunk_length_;
  }
  uint32_t mtf_rank_chunk_length() const { return mtf_rank_chunk_length_; }

  uint32_t u64_chunk_length() const { return u64_chunk_length_; }
  uint32_t s64_chunk_length() const { return s64_chunk_length_; }
  uint32_t s64_block_exponent() const { return s64_block_exponent_; }

  enum class IdFallbackStrategy {
    kRuleBased = 0,
    kShortDescriptor,
  };

  IdFallbackStrategy id_fallback_strategy() const {
    return id_fallback_strategy_;
  }

  // Returns a codec for common opcode_and_num_operands words for the given
  // previous opcode. May return nullptr if the codec doesn't exist.
  const HuffmanCodec<uint64_t>* GetOpcodeAndNumOperandsMarkovHuffmanCodec(
      uint32_t prev_opcode) const {
    if (prev_opcode == SpvOpNop)
      return opcode_and_num_operands_huffman_codec_.get();

    const auto it =
        opcode_and_num_operands_markov_huffman_codecs_.find(prev_opcode);
    if (it == opcode_and_num_operands_markov_huffman_codecs_.end())
      return nullptr;
    return it->second.get();
  }

  // Returns a codec for common non-id words used for given operand slot.
  // Operand slot is defined by the opcode and the operand index.
  // May return nullptr if the codec doesn't exist.
  const HuffmanCodec<uint64_t>* GetNonIdWordHuffmanCodec(
      uint32_t opcode, uint32_t operand_index) const {
    const auto it = non_id_word_huffman_codecs_.find(
        std::pair<uint32_t, uint32_t>(opcode, operand_index));
    if (it == non_id_word_huffman_codecs_.end()) return nullptr;
    return it->second.get();
  }

  // Returns a codec for common id descriptos used for given operand slot.
  // Operand slot is defined by the opcode and the operand index.
  // May return nullptr if the codec doesn't exist.
  const HuffmanCodec<uint64_t>* GetIdDescriptorHuffmanCodec(
      uint32_t opcode, uint32_t operand_index) const {
    const auto it = id_descriptor_huffman_codecs_.find(
        std::pair<uint32_t, uint32_t>(opcode, operand_index));
    if (it == id_descriptor_huffman_codecs_.end()) return nullptr;
    return it->second.get();
  }

  // Returns a codec for common strings used by the given opcode.
  // Operand slot is defined by the opcode and the operand index.
  // May return nullptr if the codec doesn't exist.
  const HuffmanCodec<std::string>* GetLiteralStringHuffmanCodec(
      uint32_t opcode) const {
    const auto it = literal_string_huffman_codecs_.find(opcode);
    if (it == literal_string_huffman_codecs_.end()) return nullptr;
    return it->second.get();
  }

  // Checks if |descriptor| has a coding scheme in any of
  // id_descriptor_huffman_codecs_.
  bool DescriptorHasCodingScheme(uint32_t descriptor) const {
    return descriptors_with_coding_scheme_.count(descriptor);
  }

  // Checks if any descriptor has a coding scheme.
  bool AnyDescriptorHasCodingScheme() const {
    return !descriptors_with_coding_scheme_.empty();
  }

  // Returns chunk length used for variable length encoding of spirv operand
  // words.
  uint32_t GetOperandVariableWidthChunkLength(spv_operand_type_t type) const {
    return operand_chunk_lengths_.at(static_cast<size_t>(type));
  }

  // Sets model type.
  void SetModelType(uint32_t in_model_type) { model_type_ = in_model_type; }

  // Sets model version.
  void SetModelVersion(uint32_t in_model_version) {
    model_version_ = in_model_version;
  }

  // Returns value used by Huffman codecs as a signal that a value is not in the
  // coding table.
  static uint64_t GetMarkvNoneOfTheAbove() {
    // Magic number.
    return 1111111111111111111;
  }

  MarkvModel(const MarkvModel&) = delete;
  const MarkvModel& operator=(const MarkvModel&) = delete;

 protected:
  // Huffman codec for base-rate of opcode_and_num_operands.
  std::unique_ptr<HuffmanCodec<uint64_t>>
      opcode_and_num_operands_huffman_codec_;

  // Huffman codecs for opcode_and_num_operands. The map key is previous opcode.
  std::map<uint32_t, std::unique_ptr<HuffmanCodec<uint64_t>>>
      opcode_and_num_operands_markov_huffman_codecs_;

  // Huffman codecs for non-id single-word operand values.
  // The map key is pair <opcode, operand_index>.
  std::map<std::pair<uint32_t, uint32_t>,
           std::unique_ptr<HuffmanCodec<uint64_t>>>
      non_id_word_huffman_codecs_;

  // Huffman codecs for id descriptors. The map key is pair
  // <opcode, operand_index>.
  std::map<std::pair<uint32_t, uint32_t>,
           std::unique_ptr<HuffmanCodec<uint64_t>>>
      id_descriptor_huffman_codecs_;

  // Set of all descriptors which have a coding scheme in any of
  // id_descriptor_huffman_codecs_.
  std::unordered_set<uint32_t> descriptors_with_coding_scheme_;

  // Huffman codecs for literal strings. The map key is the opcode of the
  // current instruction. This assumes, that there is no more than one literal
  // string operand per instruction, but would still work even if this is not
  // the case. Names and debug information strings are not collected.
  std::map<uint32_t, std::unique_ptr<HuffmanCodec<std::string>>>
      literal_string_huffman_codecs_;

  // Chunk lengths used for variable width encoding of operands (index is
  // spv_operand_type of the operand).
  std::vector<uint32_t> operand_chunk_lengths_;

  uint32_t opcode_chunk_length_ = 7;
  uint32_t num_operands_chunk_length_ = 3;
  uint32_t mtf_rank_chunk_length_ = 5;

  uint32_t u64_chunk_length_ = 8;
  uint32_t s64_chunk_length_ = 8;
  uint32_t s64_block_exponent_ = 10;

  IdFallbackStrategy id_fallback_strategy_ =
      IdFallbackStrategy::kShortDescriptor;

  uint32_t model_type_ = 0;
  uint32_t model_version_ = 0;
};

}  // namespace comp
}  // namespace spvtools

#endif  // SOURCE_COMP_MARKV_MODEL_H_
