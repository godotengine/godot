// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/enc_ma.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/encoding/ma_common.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/pack_signed.h"

namespace jxl {

namespace {
// Plot tree (if enabled) and predictor usage map.
constexpr bool kWantDebug = true;
// constexpr bool kPrintTree = false;

inline std::array<uint8_t, 3> PredictorColor(Predictor p) {
  switch (p) {
    case Predictor::Zero:
      return {{0, 0, 0}};
    case Predictor::Left:
      return {{255, 0, 0}};
    case Predictor::Top:
      return {{0, 255, 0}};
    case Predictor::Average0:
      return {{0, 0, 255}};
    case Predictor::Average4:
      return {{192, 128, 128}};
    case Predictor::Select:
      return {{255, 255, 0}};
    case Predictor::Gradient:
      return {{255, 0, 255}};
    case Predictor::Weighted:
      return {{0, 255, 255}};
      // TODO(jon)
    default:
      return {{255, 255, 255}};
  };
}

// `cutoffs` must be sorted.
Tree MakeFixedTree(int property, const std::vector<int32_t> &cutoffs,
                   Predictor pred, size_t num_pixels, int bitdepth) {
  size_t log_px = CeilLog2Nonzero(num_pixels);
  size_t min_gap = 0;
  // Reduce fixed tree height when encoding small images.
  if (log_px < 14) {
    min_gap = 8 * (14 - log_px);
  }
  const int shift = bitdepth > 11 ? std::min(4, bitdepth - 11) : 0;
  const int mul = 1 << shift;
  Tree tree;
  struct NodeInfo {
    size_t begin, end, pos;
  };
  std::queue<NodeInfo> q;
  // Leaf IDs will be set by roundtrip decoding the tree.
  tree.push_back(PropertyDecisionNode::Leaf(pred));
  q.push(NodeInfo{0, cutoffs.size(), 0});
  while (!q.empty()) {
    NodeInfo info = q.front();
    q.pop();
    if (info.begin + min_gap >= info.end) continue;
    uint32_t split = (info.begin + info.end) / 2;
    int32_t cutoff = cutoffs[split] * mul;
    tree[info.pos] = PropertyDecisionNode::Split(property, cutoff, tree.size());
    q.push(NodeInfo{split + 1, info.end, tree.size()});
    tree.push_back(PropertyDecisionNode::Leaf(pred));
    q.push(NodeInfo{info.begin, split, tree.size()});
    tree.push_back(PropertyDecisionNode::Leaf(pred));
  }
  return tree;
}

Status GatherTreeData(const Image &image, pixel_type chan, size_t group_id,
                      const weighted::Header &wp_header,
                      const ModularOptions &options, TreeSamples &tree_samples,
                      size_t *total_pixels) {
  const Channel &channel = image.channel[chan];
  JxlMemoryManager *memory_manager = channel.memory_manager();

  JXL_DEBUG_V(7, "Learning %" PRIuS "x%" PRIuS " channel %d", channel.w,
              channel.h, chan);

  std::array<pixel_type, kNumStaticProperties> static_props = {
      {chan, static_cast<int>(group_id)}};
  Properties properties(kNumNonrefProperties +
                        kExtraPropsPerChannel * options.max_properties);
  double pixel_fraction = std::min(1.0f, options.nb_repeats);
  // a fraction of 0 is used to disable learning entirely.
  if (pixel_fraction > 0) {
    pixel_fraction = std::max(pixel_fraction,
                              std::min(1.0, 1024.0 / (channel.w * channel.h)));
  }
  uint64_t threshold =
      (std::numeric_limits<uint64_t>::max() >> 32) * pixel_fraction;
  uint64_t s[2] = {static_cast<uint64_t>(0x94D049BB133111EBull),
                   static_cast<uint64_t>(0xBF58476D1CE4E5B9ull)};
  // Xorshift128+ adapted from xorshift128+-inl.h
  auto use_sample = [&]() {
    auto s1 = s[0];
    const auto s0 = s[1];
    const auto bits = s1 + s0;  // b, c
    s[0] = s0;
    s1 ^= s1 << 23;
    s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
    s[1] = s1;
    return (bits >> 32) <= threshold;
  };

  const ptrdiff_t onerow = channel.plane.PixelsPerRow();
  JXL_ASSIGN_OR_RETURN(
      Channel references,
      Channel::Create(memory_manager, properties.size() - kNumNonrefProperties,
                      channel.w));
  weighted::State wp_state(wp_header, channel.w, channel.h);
  tree_samples.PrepareForSamples(pixel_fraction * channel.h * channel.w + 64);
  const bool multiple_predictors = tree_samples.NumPredictors() != 1;
  auto compute_sample = [&](const pixel_type *p, size_t x, size_t y) {
    pixel_type_w pred[kNumModularPredictors];
    if (multiple_predictors) {
      PredictLearnAll(&properties, channel.w, p + x, onerow, x, y, references,
                      &wp_state, pred);
    } else {
      pred[static_cast<int>(tree_samples.PredictorFromIndex(0))] =
          PredictLearn(&properties, channel.w, p + x, onerow, x, y,
                       tree_samples.PredictorFromIndex(0), references,
                       &wp_state)
              .guess;
    }
    (*total_pixels)++;
    if (use_sample()) {
      tree_samples.AddSample(p[x], properties, pred);
    }
    wp_state.UpdateErrors(p[x], x, y, channel.w);
  };

  for (size_t y = 0; y < channel.h; y++) {
    const pixel_type *JXL_RESTRICT p = channel.Row(y);
    PrecomputeReferences(channel, y, image, chan, &references);
    InitPropsRow(&properties, static_props, y);

    // TODO(veluca): avoid computing WP if we don't use its property or
    // predictions.
    if (y > 1 && channel.w > 8 && references.w == 0) {
      for (size_t x = 0; x < 2; x++) {
        compute_sample(p, x, y);
      }
      for (size_t x = 2; x < channel.w - 2; x++) {
        pixel_type_w pred[kNumModularPredictors];
        if (multiple_predictors) {
          PredictLearnAllNEC(&properties, channel.w, p + x, onerow, x, y,
                             references, &wp_state, pred);
        } else {
          pred[static_cast<int>(tree_samples.PredictorFromIndex(0))] =
              PredictLearnNEC(&properties, channel.w, p + x, onerow, x, y,
                              tree_samples.PredictorFromIndex(0), references,
                              &wp_state)
                  .guess;
        }
        (*total_pixels)++;
        if (use_sample()) {
          tree_samples.AddSample(p[x], properties, pred);
        }
        wp_state.UpdateErrors(p[x], x, y, channel.w);
      }
      for (size_t x = channel.w - 2; x < channel.w; x++) {
        compute_sample(p, x, y);
      }
    } else {
      for (size_t x = 0; x < channel.w; x++) {
        compute_sample(p, x, y);
      }
    }
  }
  return true;
}

StatusOr<Tree> LearnTree(
    TreeSamples &&tree_samples, size_t total_pixels,
    const ModularOptions &options,
    const std::vector<ModularMultiplierInfo> &multiplier_info = {},
    StaticPropRange static_prop_range = {}) {
  Tree tree;
  for (size_t i = 0; i < kNumStaticProperties; i++) {
    if (static_prop_range[i][1] == 0) {
      static_prop_range[i][1] = std::numeric_limits<uint32_t>::max();
    }
  }
  if (!tree_samples.HasSamples()) {
    tree.emplace_back();
    tree.back().predictor = tree_samples.PredictorFromIndex(0);
    tree.back().property = -1;
    tree.back().predictor_offset = 0;
    tree.back().multiplier = 1;
    return tree;
  }
  float pixel_fraction = tree_samples.NumSamples() * 1.0f / total_pixels;
  float required_cost = pixel_fraction * 0.9 + 0.1;
  tree_samples.AllSamplesDone();
  JXL_RETURN_IF_ERROR(ComputeBestTree(
      tree_samples, options.splitting_heuristics_node_threshold * required_cost,
      multiplier_info, static_prop_range, options.fast_decode_multiplier,
      &tree));
  return tree;
}

Status EncodeModularChannelMAANS(const Image &image, pixel_type chan,
                                 const weighted::Header &wp_header,
                                 const Tree &global_tree, Token **tokenpp,
                                 size_t group_id, bool skip_encoder_fast_path) {
  const Channel &channel = image.channel[chan];
  JxlMemoryManager *memory_manager = channel.memory_manager();
  Token *tokenp = *tokenpp;
  JXL_ENSURE(channel.w != 0 && channel.h != 0);

  Image3F predictor_img;
  if (kWantDebug) {
    JXL_ASSIGN_OR_RETURN(predictor_img,
                         Image3F::Create(memory_manager, channel.w, channel.h));
  }

  JXL_DEBUG_V(6,
              "Encoding %" PRIuS "x%" PRIuS
              " channel %d, "
              "(shift=%i,%i)",
              channel.w, channel.h, chan, channel.hshift, channel.vshift);

  std::array<pixel_type, kNumStaticProperties> static_props = {
      {chan, static_cast<int>(group_id)}};
  bool use_wp;
  bool is_wp_only;
  bool is_gradient_only;
  size_t num_props;
  FlatTree tree = FilterTree(global_tree, static_props, &num_props, &use_wp,
                             &is_wp_only, &is_gradient_only);
  MATreeLookup tree_lookup(tree);
  JXL_DEBUG_V(3, "Encoding using a MA tree with %" PRIuS " nodes", tree.size());

  // Check if this tree is a WP-only tree with a small enough property value
  // range.
  // Initialized to avoid clang-tidy complaining.
  auto tree_lut = jxl::make_unique<TreeLut<uint16_t, false, false>>();
  if (is_wp_only) {
    is_wp_only = TreeToLookupTable(tree, *tree_lut);
  }
  if (is_gradient_only) {
    is_gradient_only = TreeToLookupTable(tree, *tree_lut);
  }

  if (is_wp_only && !skip_encoder_fast_path) {
    for (size_t c = 0; c < 3; c++) {
      FillImage(static_cast<float>(PredictorColor(Predictor::Weighted)[c]),
                &predictor_img.Plane(c));
    }
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    weighted::State wp_state(wp_header, channel.w, channel.h);
    Properties properties(1);
    bool unhealthy = false;
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        size_t offset = 0;
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        pixel_type_w topright =
            (x + 1 < channel.w && y ? *(r + x + 1 - onerow) : top);
        pixel_type_w toptop = (y > 1 ? *(r + x - onerow - onerow) : top);
        int32_t guess = wp_state.Predict</*compute_properties=*/true>(
            x, y, channel.w, top, left, topright, topleft, toptop, &properties,
            offset);
        uint32_t pos =
            kPropRangeFast +
            jxl::Clamp1(properties[0], -kPropRangeFast, kPropRangeFast - 1);
        uint32_t ctx_id = tree_lut->context_lookup[pos];
        int32_t residual;
        unhealthy |= SubOverflow(r[x], guess, residual);
        *tokenp++ = Token(ctx_id, PackSigned(residual));
        wp_state.UpdateErrors(r[x], x, y, channel.w);
      }
    }
    if (unhealthy) {
      return JXL_FAILURE("Residual overflow");
    }
  } else if (tree.size() == 1 && tree[0].predictor == Predictor::Gradient &&
             tree[0].multiplier == 1 && tree[0].predictor_offset == 0 &&
             !skip_encoder_fast_path) {
    for (size_t c = 0; c < 3; c++) {
      FillImage(static_cast<float>(PredictorColor(Predictor::Gradient)[c]),
                &predictor_img.Plane(c));
    }
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    bool unhealthy = false;
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        int32_t guess = ClampedGradient(top, left, topleft);
        int32_t residual;
        unhealthy |= SubOverflow(r[x], guess, residual);
        *tokenp++ = Token(tree[0].childID, PackSigned(residual));
      }
    }
    if (unhealthy) {
      return JXL_FAILURE("Residual overflow");
    }
  } else if (is_gradient_only && !skip_encoder_fast_path) {
    for (size_t c = 0; c < 3; c++) {
      FillImage(static_cast<float>(PredictorColor(Predictor::Gradient)[c]),
                &predictor_img.Plane(c));
    }
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    bool unhealthy = false;
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type_w left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
        pixel_type_w top = (y ? *(r + x - onerow) : left);
        pixel_type_w topleft = (x && y ? *(r + x - 1 - onerow) : left);
        int32_t guess = ClampedGradient(top, left, topleft);
        uint32_t pos =
            kPropRangeFast +
            std::min<pixel_type_w>(
                std::max<pixel_type_w>(-kPropRangeFast, top + left - topleft),
                kPropRangeFast - 1);
        uint32_t ctx_id = tree_lut->context_lookup[pos];
        int32_t residual;
        unhealthy |= SubOverflow(r[x], guess, residual);
        *tokenp++ = Token(ctx_id, PackSigned(residual));
      }
    }
    if (unhealthy) {
      return JXL_FAILURE("Residual overflow");
    }
  } else if (tree.size() == 1 && tree[0].predictor == Predictor::Zero &&
             tree[0].multiplier == 1 && tree[0].predictor_offset == 0 &&
             !skip_encoder_fast_path) {
    for (size_t c = 0; c < 3; c++) {
      FillImage(static_cast<float>(PredictorColor(Predictor::Zero)[c]),
                &predictor_img.Plane(c));
    }
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        *tokenp++ = Token(tree[0].childID, PackSigned(p[x]));
      }
    }
  } else if (tree.size() == 1 && tree[0].predictor != Predictor::Weighted &&
             (tree[0].multiplier & (tree[0].multiplier - 1)) == 0 &&
             tree[0].predictor_offset == 0 && !skip_encoder_fast_path) {
    // multiplier is a power of 2.
    for (size_t c = 0; c < 3; c++) {
      FillImage(static_cast<float>(PredictorColor(tree[0].predictor)[c]),
                &predictor_img.Plane(c));
    }
    uint32_t mul_shift =
        FloorLog2Nonzero(static_cast<uint32_t>(tree[0].multiplier));
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult pred = PredictNoTreeNoWP(channel.w, r + x, onerow, x,
                                                  y, tree[0].predictor);
        pixel_type_w residual = r[x] - pred.guess;
        JXL_DASSERT((residual >> mul_shift) * tree[0].multiplier == residual);
        *tokenp++ = Token(tree[0].childID, PackSigned(residual >> mul_shift));
      }
    }

  } else if (!use_wp && !skip_encoder_fast_path) {
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    Properties properties(num_props);
    JXL_ASSIGN_OR_RETURN(
        Channel references,
        Channel::Create(memory_manager,
                        properties.size() - kNumNonrefProperties, channel.w));
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      PrecomputeReferences(channel, y, image, chan, &references);
      float *pred_img_row[3];
      if (kWantDebug) {
        for (size_t c = 0; c < 3; c++) {
          pred_img_row[c] = predictor_img.PlaneRow(c, y);
        }
      }
      InitPropsRow(&properties, static_props, y);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult res =
            PredictTreeNoWP(&properties, channel.w, p + x, onerow, x, y,
                            tree_lookup, references);
        if (kWantDebug) {
          for (size_t i = 0; i < 3; i++) {
            pred_img_row[i][x] = PredictorColor(res.predictor)[i];
          }
        }
        pixel_type_w residual = p[x] - res.guess;
        JXL_DASSERT(residual % res.multiplier == 0);
        *tokenp++ = Token(res.context, PackSigned(residual / res.multiplier));
      }
    }
  } else {
    const ptrdiff_t onerow = channel.plane.PixelsPerRow();
    Properties properties(num_props);
    JXL_ASSIGN_OR_RETURN(
        Channel references,
        Channel::Create(memory_manager,
                        properties.size() - kNumNonrefProperties, channel.w));
    weighted::State wp_state(wp_header, channel.w, channel.h);
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      PrecomputeReferences(channel, y, image, chan, &references);
      float *pred_img_row[3];
      if (kWantDebug) {
        for (size_t c = 0; c < 3; c++) {
          pred_img_row[c] = predictor_img.PlaneRow(c, y);
        }
      }
      InitPropsRow(&properties, static_props, y);
      for (size_t x = 0; x < channel.w; x++) {
        PredictionResult res =
            PredictTreeWP(&properties, channel.w, p + x, onerow, x, y,
                          tree_lookup, references, &wp_state);
        if (kWantDebug) {
          for (size_t i = 0; i < 3; i++) {
            pred_img_row[i][x] = PredictorColor(res.predictor)[i];
          }
        }
        pixel_type_w residual = p[x] - res.guess;
        JXL_DASSERT(residual % res.multiplier == 0);
        *tokenp++ = Token(res.context, PackSigned(residual / res.multiplier));
        wp_state.UpdateErrors(p[x], x, y, channel.w);
      }
    }
  }
  /* TODO(szabadka): Add cparams to the call stack here.
  if (kWantDebug && WantDebugOutput(cparams)) {
    DumpImage(
        cparams,
        ("pred_" + ToString(group_id) + "_" + ToString(chan)).c_str(),
        predictor_img);
  }
  */
  *tokenpp = tokenp;
  return true;
}

}  // namespace

Tree PredefinedTree(ModularOptions::TreeKind tree_kind, size_t total_pixels,
                    int bitdepth, int prevprop) {
  switch (tree_kind) {
    case ModularOptions::TreeKind::kJpegTranscodeACMeta:
      // All the data is 0, so no need for a fancy tree.
      return {PropertyDecisionNode::Leaf(Predictor::Zero)};
    case ModularOptions::TreeKind::kTrivialTreeNoPredictor:
      // All the data is 0, so no need for a fancy tree.
      return {PropertyDecisionNode::Leaf(Predictor::Zero)};
    case ModularOptions::TreeKind::kFalconACMeta:
      // All the data is 0 except the quant field. TODO(veluca): make that 0
      // too.
      return {PropertyDecisionNode::Leaf(Predictor::Left)};
    case ModularOptions::TreeKind::kACMeta: {
      // Small image.
      if (total_pixels < 1024) {
        return {PropertyDecisionNode::Leaf(Predictor::Left)};
      }
      Tree tree;
      // 0: c > 1
      tree.push_back(PropertyDecisionNode::Split(0, 1, 1));
      // 1: c > 2
      tree.push_back(PropertyDecisionNode::Split(0, 2, 3));
      // 2: c > 0
      tree.push_back(PropertyDecisionNode::Split(0, 0, 5));
      // 3: EPF control field (all 0 or 4), top > 3
      tree.push_back(PropertyDecisionNode::Split(6, 3, 21));
      // 4: ACS+QF, y > 0
      tree.push_back(PropertyDecisionNode::Split(2, 0, 7));
      // 5: CfL x
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Gradient));
      // 6: CfL b
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Gradient));
      // 7: QF: split according to the left quant value.
      tree.push_back(PropertyDecisionNode::Split(7, 5, 9));
      // 8: ACS: split in 4 segments (8x8 from 0 to 3, large square 4-5, large
      // rectangular 6-11, 8x8 12+), according to previous ACS value.
      tree.push_back(PropertyDecisionNode::Split(7, 5, 15));
      // QF
      tree.push_back(PropertyDecisionNode::Split(7, 11, 11));
      tree.push_back(PropertyDecisionNode::Split(7, 3, 13));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Left));
      // ACS
      tree.push_back(PropertyDecisionNode::Split(7, 11, 17));
      tree.push_back(PropertyDecisionNode::Split(7, 3, 19));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      // EPF, left > 3
      tree.push_back(PropertyDecisionNode::Split(7, 3, 23));
      tree.push_back(PropertyDecisionNode::Split(7, 3, 25));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      tree.push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
      return tree;
    }
    case ModularOptions::TreeKind::kWPFixedDC: {
      std::vector<int32_t> cutoffs = {
          -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15,
          -11,  -7,   -4,   -3,   -1,   0,   1,   3,   5,   7,   11,
          15,   23,   31,   47,   63,   95,  127, 191, 255, 392, 500};
      return MakeFixedTree(kWPProp, cutoffs, Predictor::Weighted, total_pixels,
                           bitdepth);
    }
    case ModularOptions::TreeKind::kGradientFixedDC: {
      std::vector<int32_t> cutoffs = {
          -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15,
          -11,  -7,   -4,   -3,   -1,   0,   1,   3,   5,   7,   11,
          15,   23,   31,   47,   63,   95,  127, 191, 255, 392, 500};
      return MakeFixedTree(
          prevprop > 0 ? kNumNonrefProperties + 2 : kGradientProp, cutoffs,
          Predictor::Gradient, total_pixels, bitdepth);
    }
    case ModularOptions::TreeKind::kLearn: {
      JXL_DEBUG_ABORT("internal: kLearn is not predefined tree");
      return {};
    }
  }
  JXL_DEBUG_ABORT("internal: unexpected TreeKind: %d",
                  static_cast<int>(tree_kind));
  return {};
}

StatusOr<Tree> LearnTree(
    const Image *images, const ModularOptions *options, const uint32_t start,
    const uint32_t stop,
    const std::vector<ModularMultiplierInfo> &multiplier_info = {}) {
  TreeSamples tree_samples;
  JXL_RETURN_IF_ERROR(tree_samples.SetPredictor(options[start].predictor,
                                                options[start].wp_tree_mode));
  JXL_RETURN_IF_ERROR(
      tree_samples.SetProperties(options[start].splitting_heuristics_properties,
                                 options[start].wp_tree_mode));
  uint32_t max_c = 0;
  std::vector<pixel_type> pixel_samples;
  std::vector<pixel_type> diff_samples;
  std::vector<uint32_t> group_pixel_count;
  std::vector<uint32_t> channel_pixel_count;
  for (uint32_t i = start; i < stop; i++) {
    max_c = std::max<uint32_t>(images[i].channel.size(), max_c);
    CollectPixelSamples(images[i], options[i], i, group_pixel_count,
                        channel_pixel_count, pixel_samples, diff_samples);
  }
  StaticPropRange range;
  range[0] = {{0, max_c}};
  range[1] = {{start, stop}};

  tree_samples.PreQuantizeProperties(
      range, multiplier_info, group_pixel_count, channel_pixel_count,
      pixel_samples, diff_samples, options[start].max_property_values);

  size_t total_pixels = 0;
  for (size_t i = 0; i < images[start].channel.size(); i++) {
    if (i >= images[start].nb_meta_channels &&
        (images[start].channel[i].w > options[start].max_chan_size ||
         images[start].channel[i].h > options[start].max_chan_size)) {
      break;
    }
    total_pixels += images[start].channel[i].w * images[start].channel[i].h;
  }
  total_pixels = std::max<size_t>(total_pixels, 1);

  weighted::Header wp_header;

  for (size_t i = start; i < stop; i++) {
    size_t nb_channels = images[i].channel.size();

    if (images[i].w == 0 || images[i].h == 0 || nb_channels < 1)
      continue;  // is there any use for a zero-channel image?
    if (images[i].error) return JXL_FAILURE("Invalid image");
    JXL_ENSURE(options[i].tree_kind == ModularOptions::TreeKind::kLearn);

    JXL_DEBUG_V(
        2, "Encoding %" PRIuS "-channel, %i-bit, %" PRIuS "x%" PRIuS " image.",
        nb_channels, images[i].bitdepth, images[i].w, images[i].h);

    // encode transforms
    Bundle::Init(&wp_header);
    if (PredictorHasWeighted(options[i].predictor)) {
      weighted::PredictorMode(options[i].wp_mode, &wp_header);
    }

    // Gather tree data
    for (size_t c = 0; c < nb_channels; c++) {
      if (c >= images[i].nb_meta_channels &&
          (images[i].channel[c].w > options[i].max_chan_size ||
           images[i].channel[c].h > options[i].max_chan_size)) {
        break;
      }
      if (!images[i].channel[c].w || !images[i].channel[c].h) {
        continue;  // skip empty channels
      }
      JXL_RETURN_IF_ERROR(GatherTreeData(images[i], c, i, wp_header, options[i],
                                         tree_samples, &total_pixels));
    }
  }

  // TODO(veluca): parallelize more.
  JXL_ASSIGN_OR_RETURN(Tree tree,
                       LearnTree(std::move(tree_samples), total_pixels,
                                 options[start], multiplier_info, range));
  return tree;
}

Status ModularCompress(const Image &image, const ModularOptions &options,
                       size_t group_id, const Tree &tree, GroupHeader &header,
                       std::vector<Token> &tokens, size_t *width) {
  size_t nb_channels = image.channel.size();

  if (image.w == 0 || image.h == 0 || nb_channels < 1)
    return true;  // is there any use for a zero-channel image?
  if (image.error) return JXL_FAILURE("Invalid image");

  JXL_DEBUG_V(
      2, "Encoding %" PRIuS "-channel, %i-bit, %" PRIuS "x%" PRIuS " image.",
      nb_channels, image.bitdepth, image.w, image.h);

  // encode transforms
  Bundle::Init(&header);
  if (PredictorHasWeighted(options.predictor)) {
    weighted::PredictorMode(options.wp_mode, &header.wp_header);
  }
  header.transforms = image.transform;
  header.use_global_tree = true;

  size_t image_width = 0;
  size_t total_tokens = 0;
  for (size_t i = 0; i < nb_channels; i++) {
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options.max_chan_size ||
         image.channel[i].h > options.max_chan_size)) {
      break;
    }
    if (image.channel[i].w > image_width) image_width = image.channel[i].w;
    total_tokens += image.channel[i].w * image.channel[i].h;
  }
  if (options.zero_tokens) {
    tokens.resize(tokens.size() + total_tokens, {0, 0});
  } else {
    // Do one big allocation for all the tokens we'll need,
    // to avoid reallocs that might require copying.
    size_t pos = tokens.size();
    tokens.resize(pos + total_tokens);
    Token *tokenp = tokens.data() + pos;
    for (size_t i = 0; i < nb_channels; i++) {
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options.max_chan_size ||
           image.channel[i].h > options.max_chan_size)) {
        break;
      }
      if (!image.channel[i].w || !image.channel[i].h) {
        continue;  // skip empty channels
      }
      JXL_RETURN_IF_ERROR(
          EncodeModularChannelMAANS(image, i, header.wp_header, tree, &tokenp,
                                    group_id, options.skip_encoder_fast_path));
    }
    // Make sure we actually wrote all tokens
    JXL_ENSURE(tokenp == tokens.data() + tokens.size());
  }

  *width = image_width;

  return true;
}

Status ModularGenericCompress(const Image &image, const ModularOptions &opts,
                              BitWriter &writer, AuxOut *aux_out,
                              LayerType layer, size_t group_id) {
  size_t nb_channels = image.channel.size();

  if (image.w == 0 || image.h == 0 || nb_channels < 1)
    return true;  // is there any use for a zero-channel image?
  if (image.error) return JXL_FAILURE("Invalid image");

  ModularOptions options = opts;  // Make a copy to modify it.
  if (options.predictor == kUndefinedPredictor) {
    options.predictor = Predictor::Gradient;
  }

  size_t bits = writer.BitsWritten();

  JxlMemoryManager *memory_manager = image.memory_manager();
  JXL_DEBUG_V(
      2, "Encoding %" PRIuS "-channel, %i-bit, %" PRIuS "x%" PRIuS " image.",
      nb_channels, image.bitdepth, image.w, image.h);

  // encode transforms
  GroupHeader header;
  Bundle::Init(&header);
  if (PredictorHasWeighted(options.predictor)) {
    weighted::PredictorMode(options.wp_mode, &header.wp_header);
  }
  header.transforms = image.transform;

  JXL_RETURN_IF_ERROR(Bundle::Write(header, &writer, layer, aux_out));

  // Compute tree.
  Tree tree;
  if (options.tree_kind == ModularOptions::TreeKind::kLearn) {
    JXL_ASSIGN_OR_RETURN(tree, LearnTree(&image, &options, 0, 1));
  } else {
    size_t total_pixels = 0;
    for (size_t i = 0; i < nb_channels; i++) {
      if (i >= image.nb_meta_channels &&
          (image.channel[i].w > options.max_chan_size ||
           image.channel[i].h > options.max_chan_size)) {
        break;
      }
      total_pixels += image.channel[i].w * image.channel[i].h;
    }
    total_pixels = std::max<size_t>(total_pixels, 1);

    tree = PredefinedTree(options.tree_kind, total_pixels, image.bitdepth,
                          options.max_properties);
  }

  Tree decoded_tree;
  std::vector<std::vector<Token>> tree_tokens(1);
  JXL_RETURN_IF_ERROR(TokenizeTree(tree, tree_tokens.data(), &decoded_tree));
  JXL_ENSURE(tree.size() == decoded_tree.size());
  tree = std::move(decoded_tree);

  /* TODO(szabadka) Add text output callback
  if (kWantDebug && kPrintTree && WantDebugOutput(aux_out)) {
    PrintTree(*tree, aux_out->debug_prefix + "/tree_" + ToString(group_id));
  } */

  // Write tree
  EntropyEncodingData code;
  JXL_ASSIGN_OR_RETURN(
      size_t cost,
      BuildAndEncodeHistograms(memory_manager, options.histogram_params,
                               kNumTreeContexts, tree_tokens, &code, &writer,
                               LayerType::ModularTree, aux_out));
  JXL_RETURN_IF_ERROR(WriteTokens(tree_tokens[0], code, 0, &writer,
                                  LayerType::ModularTree, aux_out));

  size_t image_width = 0;
  std::vector<std::vector<Token>> tokens(1);
  // it puts `use_global_tree = true` in the header, but this is not used
  // further
  JXL_RETURN_IF_ERROR(ModularCompress(image, options, group_id, tree, header,
                                      tokens[0], &image_width));

  // Write data
  code = {};
  HistogramParams histo_params = options.histogram_params;
  histo_params.image_widths.push_back(image_width);
  JXL_ASSIGN_OR_RETURN(
      cost, BuildAndEncodeHistograms(memory_manager, histo_params,
                                     (tree.size() + 1) / 2, tokens, &code,
                                     &writer, layer, aux_out));
  (void)cost;
  JXL_RETURN_IF_ERROR(WriteTokens(tokens[0], code, 0, &writer, layer, aux_out));

  bits = writer.BitsWritten() - bits;
  JXL_DEBUG_V(4,
              "Modular-encoded a %" PRIuS "x%" PRIuS
              " bitdepth=%i nbchans=%" PRIuS " image in %" PRIuS " bytes",
              image.w, image.h, image.bitdepth, image.channel.size(), bits / 8);
  (void)bits;

  return true;
}

}  // namespace jxl
