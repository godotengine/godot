// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_PASSES_STATE_H_
#define LIB_JXL_PASSES_STATE_H_

#include <jxl/memory_manager.h>

#include <array>
#include <cstddef>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/dec_patch_dictionary.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/noise.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/splines.h"

// Structures that hold the (en/de)coder state for a JPEG XL kVarDCT
// (en/de)coder.

namespace jxl {

struct ImageFeatures {
  explicit ImageFeatures(JxlMemoryManager* memory_manager_)
      : patches(memory_manager_) {}
  NoiseParams noise_params;
  PatchDictionary patches;
  Splines splines;
};

// State common to both encoder and decoder.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct PassesSharedState {
  explicit PassesSharedState(JxlMemoryManager* memory_manager_)
      : memory_manager(memory_manager_), image_features(memory_manager_) {
    for (auto& reference_frame : reference_frames) {
      reference_frame.frame = jxl::make_unique<ImageBundle>(memory_manager_);
    }
  }

  JxlMemoryManager* memory_manager;
  const CodecMetadata* metadata;

  FrameDimensions frame_dim;

  // Control fields and parameters.
  AcStrategyImage ac_strategy;

  // Dequant matrices + quantizer.
  DequantMatrices matrices;
  Quantizer quantizer{matrices};
  ImageI raw_quant_field;

  // Per-block side information for EPF detail preservation.
  ImageB epf_sharpness;

  ColorCorrelationMap cmap;

  ImageFeatures image_features;

  // Memory area for storing coefficient orders.
  // `coeff_order_size` is the size used by *one* set of coefficient orders (at
  // most kMaxCoeffOrderSize). A set of coefficient orders is present for each
  // pass.
  size_t coeff_order_size = 0;
  std::vector<coeff_order_t> coeff_orders;

  // Decoder-side DC and quantized DC.
  ImageB quant_dc;
  Image3F dc_storage;
  const Image3F* JXL_RESTRICT dc = &dc_storage;

  BlockCtxMap block_ctx_map;

  Image3F dc_frames[4];

  std::array<ReferenceFrame, 4> reference_frames;

  // Number of pre-clustered set of histograms (with the same ctx map), per
  // pass. Encoded as num_histograms_ - 1.
  size_t num_histograms = 0;
};

// Initialized the state information that is shared between encoder and decoder.
Status InitializePassesSharedState(const FrameHeader& frame_header,
                                   PassesSharedState* JXL_RESTRICT shared,
                                   bool encoder = false);

}  // namespace jxl

#endif  // LIB_JXL_PASSES_STATE_H_
