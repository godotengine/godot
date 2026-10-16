// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_aux_out.h"

#include <cstddef>
#include <cstdio>

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"

namespace jxl {

const char* LayerName(LayerType layer) {
  switch (layer) {
    case LayerType::Header:
      return "Headers";
    case LayerType::Toc:
      return "TOC";
    case LayerType::Dictionary:
      return "Patches";
    case LayerType::Splines:
      return "Splines";
    case LayerType::Noise:
      return "Noise";
    case LayerType::Quant:
      return "Quantizer";
    case LayerType::ModularTree:
      return "ModularTree";
    case LayerType::ModularGlobal:
      return "ModularGlobal";
    case LayerType::Dc:
      return "DC";
    case LayerType::ModularDcGroup:
      return "ModularDcGroup";
    case LayerType::ControlFields:
      return "ControlFields";
    case LayerType::Order:
      return "CoeffOrder";
    case LayerType::Ac:
      return "ACHistograms";
    case LayerType::AcTokens:
      return "ACTokens";
    case LayerType::ModularAcGroup:
      return "ModularAcGroup";
  }
  JXL_DEBUG_ABORT("internal: unexpected LayerType: %d",
                  static_cast<int>(layer));
  return "Invalid";
}

void AuxOut::LayerTotals::Print(size_t num_inputs) const {
  if (JXL_DEBUG_V_LEVEL > 0) {
    printf("%10" PRIuS, total_bits);
    if (histogram_bits != 0) {
      printf("   [c/i:%6.2f | hst:%8" PRIuS " | ex:%8" PRIuS " | h+c+e:%12.3f",
             num_clustered_histograms * 1.0 / num_inputs, histogram_bits >> 3,
             extra_bits >> 3,
             (histogram_bits + clustered_entropy + extra_bits) / 8.0);
      printf("]");
    }
    printf("\n");
  }
}

void AuxOut::Assimilate(const AuxOut& victim) {
  for (size_t i = 0; i < kNumImageLayers; ++i) {
    LayerType l = static_cast<LayerType>(i);
    layer(l).Assimilate(victim.layer(l));
  }
  num_blocks += victim.num_blocks;
  num_small_blocks += victim.num_small_blocks;
  num_dct4x8_blocks += victim.num_dct4x8_blocks;
  num_afv_blocks += victim.num_afv_blocks;
  num_dct8_blocks += victim.num_dct8_blocks;
  num_dct8x16_blocks += victim.num_dct8x16_blocks;
  num_dct8x32_blocks += victim.num_dct8x32_blocks;
  num_dct16_blocks += victim.num_dct16_blocks;
  num_dct16x32_blocks += victim.num_dct16x32_blocks;
  num_dct32_blocks += victim.num_dct32_blocks;
  num_dct32x64_blocks += victim.num_dct32x64_blocks;
  num_dct64_blocks += victim.num_dct64_blocks;
  num_butteraugli_iters += victim.num_butteraugli_iters;
}

void AuxOut::Print(size_t num_inputs) const {
  if (JXL_DEBUG_V_LEVEL > 0) {
    if (num_inputs == 0) return;

    LayerTotals all_layers;
    for (const auto& layer : layers) {
      all_layers.Assimilate(layer);
    }

    printf("Average butteraugli iters: %10.2f\n",
           num_butteraugli_iters * 1.0 / num_inputs);

    for (size_t i = 0; i < kNumImageLayers; ++i) {
      LayerType l = static_cast<LayerType>(i);
      if (layer(l).total_bits != 0) {
        printf("Total layer bits %-10s\t", LayerName(l));
        printf("%10f%%", 100.0 * layer(l).total_bits / all_layers.total_bits);
        layer(l).Print(num_inputs);
      }
    }
    printf("Total image size           ");
    all_layers.Print(num_inputs);

    size_t total_blocks = 0;
    size_t total_positions = 0;
    if (total_blocks != 0 && total_positions != 0) {
      printf("\n\t\t  Blocks\t\tPositions\t\t\tBlocks/Position\n");
      printf(" Total:\t\t    %7" PRIuS "\t\t     %7" PRIuS " \t\t\t%10f%%\n\n",
             total_blocks, total_positions,
             100.0 * total_blocks / total_positions);
    }
  }
}

}  // namespace jxl
