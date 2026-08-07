// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_ANS_H_
#define LIB_JXL_ENC_ANS_H_

// Library to encode the ANS population counts to the bit-stream and encode
// symbols based on the respective distributions.

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_bit_writer.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;

#define USE_MULT_BY_RECIPROCAL

// precision must be equal to:  #bits(state_) + #bits(freq)
#define RECIPROCAL_PRECISION (32 + ANS_LOG_TAB_SIZE)

// Data structure representing one element of the encoding table built
// from a distribution.
// TODO(veluca): split this up, or use an union.
struct ANSEncSymbolInfo {
  // ANS
  uint16_t freq_;
  std::vector<uint16_t> reverse_map_;
#ifdef USE_MULT_BY_RECIPROCAL
  uint64_t ifreq_;
#endif
  // Prefix coding.
  uint8_t depth;
  uint16_t bits;
};

class ANSCoder {
 public:
  ANSCoder() : state_(ANS_SIGNATURE << 16) {}

  uint32_t PutSymbol(const ANSEncSymbolInfo& t, uint8_t* nbits) {
    uint32_t bits = 0;
    *nbits = 0;
    if ((state_ >> (32 - ANS_LOG_TAB_SIZE)) >= t.freq_) {
      bits = state_ & 0xffff;
      state_ >>= 16;
      *nbits = 16;
    }
#ifdef USE_MULT_BY_RECIPROCAL
    // We use mult-by-reciprocal trick, but that requires 64b calc.
    const uint32_t v = (state_ * t.ifreq_) >> RECIPROCAL_PRECISION;
    const uint32_t offset = t.reverse_map_[state_ - v * t.freq_];
    state_ = (v << ANS_LOG_TAB_SIZE) + offset;
#else
    state_ = ((state_ / t.freq_) << ANS_LOG_TAB_SIZE) +
             t.reverse_map_[state_ % t.freq_];
#endif
    return bits;
  }

  uint32_t GetState() const { return state_; }

 private:
  uint32_t state_;
};

static const int kNumFixedHistograms = 1;

struct EntropyEncodingData {
  std::vector<std::vector<ANSEncSymbolInfo>> encoding_info;
  bool use_prefix_code;
  std::vector<HybridUintConfig> uint_config;
  LZ77Params lz77;
  std::vector<BitWriter> encoded_histograms;
};

// Integer to be encoded by an entropy coder, either ANS or Huffman.
struct Token {
  Token() = default;
  Token(uint32_t c, uint32_t value)
      : is_lz77_length(false), context(c), value(value) {}
  uint32_t is_lz77_length : 1;
  uint32_t context : 31;
  uint32_t value;
};

// Returns an estimate of the number of bits required to encode the given
// histogram (header bits plus data bits).
StatusOr<float> ANSPopulationCost(const ANSHistBin* data, size_t alphabet_size);

// Writes the context map to the bitstream and concatenates the individual
// histogram bitstreams in codes.encoded_histograms. Used in streaming mode.
Status EncodeHistograms(const std::vector<uint8_t>& context_map,
                        const EntropyEncodingData& codes, BitWriter* writer,
                        LayerType layer, AuxOut* aux_out);

// Apply context clustering, compute histograms and encode them. Returns an
// estimate of the total bits used for encoding the stream. If `writer` ==
// nullptr, the bit estimate will not take into account the context map (which
// does not get written if `num_contexts` == 1).
// Returns cost
StatusOr<size_t> BuildAndEncodeHistograms(
    JxlMemoryManager* memory_manager, const HistogramParams& params,
    size_t num_contexts, std::vector<std::vector<Token>>& tokens,
    EntropyEncodingData* codes, std::vector<uint8_t>* context_map,
    BitWriter* writer, LayerType layer, AuxOut* aux_out);

// Write the tokens to a string.
Status WriteTokens(const std::vector<Token>& tokens,
                   const EntropyEncodingData& codes,
                   const std::vector<uint8_t>& context_map,
                   size_t context_offset, BitWriter* writer, LayerType layer,
                   AuxOut* aux_out);

// Same as above, but assumes allotment created by caller.
size_t WriteTokens(const std::vector<Token>& tokens,
                   const EntropyEncodingData& codes,
                   const std::vector<uint8_t>& context_map,
                   size_t context_offset, BitWriter* writer);

// Exposed for tests; to be used with Writer=BitWriter only.
template <typename Writer>
void EncodeUintConfigs(const std::vector<HybridUintConfig>& uint_config,
                       Writer* writer, size_t log_alpha_size);
extern template void EncodeUintConfigs(const std::vector<HybridUintConfig>&,
                                       BitWriter*, size_t);

// Globally set the option to create fuzzer-friendly ANS streams. Negatively
// impacts compression. Not thread-safe.
void SetANSFuzzerFriendly(bool ans_fuzzer_friendly);
}  // namespace jxl

#endif  // LIB_JXL_ENC_ANS_H_
