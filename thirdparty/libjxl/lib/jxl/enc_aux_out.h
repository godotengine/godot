// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_AUX_OUT_H_
#define LIB_JXL_AUX_OUT_H_

// Optional output information for debugging and analyzing size usage.

#include <array>
#include <cstddef>
#include <cstdint>

namespace jxl {

struct ColorEncoding;

// For LayerName and AuxOut::layers[] index. Order does not matter.
enum class LayerType : uint8_t {
  Header = 0,
  Toc,
  Dictionary,
  Splines,
  Noise,
  Quant,
  ModularTree,
  ModularGlobal,
  Dc,
  ModularDcGroup,
  ControlFields,
  Order,
  Ac,
  AcTokens,
  ModularAcGroup,
};

constexpr uint8_t kNumImageLayers =
    static_cast<uint8_t>(LayerType::ModularAcGroup) + 1;

const char* LayerName(LayerType layer);

// Statistics gathered during compression or decompression.
struct AuxOut {
 private:
  struct LayerTotals {
    void Assimilate(const LayerTotals& victim) {
      num_clustered_histograms += victim.num_clustered_histograms;
      histogram_bits += victim.histogram_bits;
      extra_bits += victim.extra_bits;
      total_bits += victim.total_bits;
      clustered_entropy += victim.clustered_entropy;
    }
    void Print(size_t num_inputs) const;

    size_t num_clustered_histograms = 0;
    size_t extra_bits = 0;

    // Set via BitsWritten below
    size_t histogram_bits = 0;
    size_t total_bits = 0;

    double clustered_entropy = 0.0;
  };

 public:
  AuxOut() = default;
  AuxOut(const AuxOut&) = default;

  void Assimilate(const AuxOut& victim);

  void Print(size_t num_inputs) const;

  size_t TotalBits() const {
    size_t total = 0;
    for (const auto& layer : layers) {
      total += layer.total_bits;
    }
    return total;
  }

  std::array<LayerTotals, kNumImageLayers> layers;

  const LayerTotals& layer(LayerType idx) const {
    return layers[static_cast<uint8_t>(idx)];
  }
  LayerTotals& layer(LayerType idx) {
    return layers[static_cast<uint8_t>(idx)];
  }

  size_t num_blocks = 0;

  // Number of blocks that use larger DCT (set by ac_strategy).
  size_t num_small_blocks = 0;
  size_t num_dct4x8_blocks = 0;
  size_t num_afv_blocks = 0;
  size_t num_dct8_blocks = 0;
  size_t num_dct8x16_blocks = 0;
  size_t num_dct8x32_blocks = 0;
  size_t num_dct16_blocks = 0;
  size_t num_dct16x32_blocks = 0;
  size_t num_dct32_blocks = 0;
  size_t num_dct32x64_blocks = 0;
  size_t num_dct64_blocks = 0;

  int num_butteraugli_iters = 0;
};
}  // namespace jxl

#endif  // LIB_JXL_AUX_OUT_H_
