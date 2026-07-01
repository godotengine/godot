// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_OPTIONS_H_
#define LIB_JXL_MODULAR_OPTIONS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/enc_ans_params.h"

namespace jxl {

using PropertyVal = int32_t;
using Properties = std::vector<PropertyVal>;

enum class Predictor : uint32_t {
  Zero = 0,
  Left = 1,
  Top = 2,
  Average0 = 3,
  Select = 4,
  Gradient = 5,
  Weighted = 6,
  TopRight = 7,
  TopLeft = 8,
  LeftLeft = 9,
  Average1 = 10,
  Average2 = 11,
  Average3 = 12,
  Average4 = 13,
  // The following predictors are encoder-only.
  Best = 14,  // Best of Gradient and Weighted
  Variable =
      15,  // Find the best decision tree for predictors/predictor per row
};

constexpr Predictor kUndefinedPredictor = static_cast<Predictor>(~0u);

constexpr size_t kNumModularPredictors =
    static_cast<size_t>(Predictor::Average4) + 1;
constexpr size_t kNumModularEncoderPredictors =
    static_cast<size_t>(Predictor::Variable) + 1;

static constexpr ssize_t kNumStaticProperties = 2;  // channel, group_id.

using StaticPropRange =
    std::array<std::array<uint32_t, 2>, kNumStaticProperties>;

struct ModularMultiplierInfo {
  StaticPropRange range;
  uint32_t multiplier;
};

struct ModularOptions {
  /// Used in both encode and decode:

  // Stop encoding/decoding when reaching a (non-meta) channel that has a
  // dimension bigger than max_chan_size.
  size_t max_chan_size = 0xFFFFFF;

  // Used during decoding for validation of transforms (sqeeezing) scheme.
  size_t group_dim = 0x1FFFFFFF;

  /// Encode options:
  // Fraction of pixels to look at to learn a MA tree
  // Number of iterations to do to learn a MA tree
  // (if zero there is no MA context model)
  float nb_repeats = .5f;

  // Maximum number of (previous channel) properties to use in the MA trees
  int max_properties = 0;  // no previous channels

  // Alternative heuristic tweaks.
  // Properties default to channel, group, weighted, gradient residual, W-NW,
  // NW-N, N-NE, N-NN
  std::vector<uint32_t> splitting_heuristics_properties = {0,  1,  15, 9,
                                                           10, 11, 12, 13};
  float splitting_heuristics_node_threshold = 96;
  size_t max_property_values = 32;

  // Predictor to use for each channel.
  Predictor predictor = kUndefinedPredictor;

  int wp_mode = 0;

  float fast_decode_multiplier = 1.01f;

  // Forces the encoder to produce a tree that is compatible with the WP-only
  // decode path (or with the no-wp path, or the gradient-only path).
  enum class TreeMode { kGradientOnly, kWPOnly, kNoWP, kDefault };
  TreeMode wp_tree_mode = TreeMode::kDefault;

  // Skip fast paths in the encoder.
  bool skip_encoder_fast_path = false;

  // Kind of tree to use.
  // TODO(veluca): add tree kinds for JPEG recompression with CfL enabled,
  // general AC metadata, different DC qualities, and others.
  enum class TreeKind {
    kTrivialTreeNoPredictor,
    kLearn,
    kJpegTranscodeACMeta,
    kFalconACMeta,
    kACMeta,
    kWPFixedDC,
    kGradientFixedDC,
  };
  TreeKind tree_kind = TreeKind::kLearn;

  HistogramParams histogram_params;

  // Ignore the image and just pretend all tokens are zeroes
  bool zero_tokens = false;
};

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_OPTIONS_H_
