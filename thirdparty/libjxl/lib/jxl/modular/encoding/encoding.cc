// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/encoding/encoding.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <queue>
#include <utility>
#include <vector>

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/scope_guard.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/pack_signed.h"

namespace jxl {

// Removes all nodes that use a static property (i.e. channel or group ID) from
// the tree and collapses each node on even levels with its two children to
// produce a flatter tree. Also computes whether the resulting tree requires
// using the weighted predictor.
FlatTree FilterTree(const Tree &global_tree,
                    std::array<pixel_type, kNumStaticProperties> &static_props,
                    size_t *num_props, bool *use_wp, bool *wp_only,
                    bool *gradient_only) {
  *num_props = 0;
  bool has_wp = false;
  bool has_non_wp = false;
  *gradient_only = true;
  const auto mark_property = [&](int32_t p) {
    if (p == kWPProp) {
      has_wp = true;
    } else if (p >= kNumStaticProperties) {
      has_non_wp = true;
    }
    if (p >= kNumStaticProperties && p != kGradientProp) {
      *gradient_only = false;
    }
  };
  FlatTree output;
  std::queue<size_t> nodes;
  nodes.push(0);
  // Produces a trimmed and flattened tree by doing a BFS visit of the original
  // tree, ignoring branches that are known to be false and proceeding two
  // levels at a time to collapse nodes in a flatter tree; if an inner parent
  // node has a leaf as a child, the leaf is duplicated and an implicit fake
  // node is added. This allows to reduce the number of branches when traversing
  // the resulting flat tree.
  while (!nodes.empty()) {
    size_t cur = nodes.front();
    nodes.pop();
    // Skip nodes that we can decide now, by jumping directly to their children.
    while (global_tree[cur].property < kNumStaticProperties &&
           global_tree[cur].property != -1) {
      if (static_props[global_tree[cur].property] > global_tree[cur].splitval) {
        cur = global_tree[cur].lchild;
      } else {
        cur = global_tree[cur].rchild;
      }
    }
    FlatDecisionNode flat;
    if (global_tree[cur].property == -1) {
      flat.property0 = -1;
      flat.childID = global_tree[cur].lchild;
      flat.predictor = global_tree[cur].predictor;
      flat.predictor_offset = global_tree[cur].predictor_offset;
      flat.multiplier = global_tree[cur].multiplier;
      *gradient_only &= flat.predictor == Predictor::Gradient;
      has_wp |= flat.predictor == Predictor::Weighted;
      has_non_wp |= flat.predictor != Predictor::Weighted;
      output.push_back(flat);
      continue;
    }
    flat.childID = output.size() + nodes.size() + 1;

    flat.property0 = global_tree[cur].property;
    *num_props = std::max<size_t>(flat.property0 + 1, *num_props);
    flat.splitval0 = global_tree[cur].splitval;

    for (size_t i = 0; i < 2; i++) {
      size_t cur_child =
          i == 0 ? global_tree[cur].lchild : global_tree[cur].rchild;
      // Skip nodes that we can decide now.
      while (global_tree[cur_child].property < kNumStaticProperties &&
             global_tree[cur_child].property != -1) {
        if (static_props[global_tree[cur_child].property] >
            global_tree[cur_child].splitval) {
          cur_child = global_tree[cur_child].lchild;
        } else {
          cur_child = global_tree[cur_child].rchild;
        }
      }
      // We ended up in a leaf, add a placeholder decision and two copies of the
      // leaf.
      if (global_tree[cur_child].property == -1) {
        flat.properties[i] = 0;
        flat.splitvals[i] = 0;
        nodes.push(cur_child);
        nodes.push(cur_child);
      } else {
        flat.properties[i] = global_tree[cur_child].property;
        flat.splitvals[i] = global_tree[cur_child].splitval;
        nodes.push(global_tree[cur_child].lchild);
        nodes.push(global_tree[cur_child].rchild);
        *num_props = std::max<size_t>(flat.properties[i] + 1, *num_props);
      }
    }

    for (int16_t property : flat.properties) mark_property(property);
    mark_property(flat.property0);
    output.push_back(flat);
  }
  if (*num_props > kNumNonrefProperties) {
    *num_props =
        DivCeil(*num_props - kNumNonrefProperties, kExtraPropsPerChannel) *
            kExtraPropsPerChannel +
        kNumNonrefProperties;
  } else {
    *num_props = kNumNonrefProperties;
  }
  *use_wp = has_wp;
  *wp_only = has_wp && !has_non_wp;

  return output;
}

namespace detail {
template <bool uses_lz77>
Status DecodeModularChannelMAANS(BitReader *br, ANSSymbolReader *reader,
                                 const std::vector<uint8_t> &context_map,
                                 const Tree &global_tree,
                                 const weighted::Header &wp_header,
                                 pixel_type chan, size_t group_id,
                                 TreeLut<uint8_t, false, false> &tree_lut,
                                 Image *image, uint32_t &fl_run,
                                 uint32_t &fl_v) {
  JxlMemoryManager *memory_manager = image->memory_manager();
  Channel &channel = image->channel[chan];

  std::array<pixel_type, kNumStaticProperties> static_props = {
      {chan, static_cast<int>(group_id)}};
  // TODO(veluca): filter the tree according to static_props.

  // zero pixel channel? could happen
  if (channel.w == 0 || channel.h == 0) return true;

  bool tree_has_wp_prop_or_pred = false;
  bool is_wp_only = false;
  bool is_gradient_only = false;
  size_t num_props;
  FlatTree tree =
      FilterTree(global_tree, static_props, &num_props,
                 &tree_has_wp_prop_or_pred, &is_wp_only, &is_gradient_only);

  // From here on, tree lookup returns a *clustered* context ID.
  // This avoids an extra memory lookup after tree traversal.
  for (auto &node : tree) {
    if (node.property0 == -1) {
      node.childID = context_map[node.childID];
    }
  }

  JXL_DEBUG_V(3, "Decoded MA tree with %" PRIuS " nodes", tree.size());

  // MAANS decode
  const auto make_pixel = [](uint64_t v, pixel_type multiplier,
                             pixel_type_w offset) -> pixel_type {
    JXL_DASSERT((v & 0xFFFFFFFF) == v);
    pixel_type_w val = UnpackSigned(v);
    // if it overflows, it overflows, and we have a problem anyway
    return val * multiplier + offset;
  };

  if (tree.size() == 1) {
    // special optimized case: no meta-adaptation, so no need
    // to compute properties.
    Predictor predictor = tree[0].predictor;
    int64_t offset = tree[0].predictor_offset;
    int32_t multiplier = tree[0].multiplier;
    size_t ctx_id = tree[0].childID;
    if (predictor == Predictor::Zero) {
      uint32_t value;
      if (reader->IsSingleValueAndAdvance(ctx_id, &value,
                                          channel.w * channel.h)) {
        // Special-case: histogram has a single symbol, with no extra bits, and
        // we use ANS mode.
        JXL_DEBUG_V(8, "Fastest track.");
        pixel_type v = make_pixel(value, multiplier, offset);
        for (size_t y = 0; y < channel.h; y++) {
          pixel_type *JXL_RESTRICT r = channel.Row(y);
          std::fill(r, r + channel.w, v);
        }
      } else {
        JXL_DEBUG_V(8, "Fast track.");
        if (multiplier == 1 && offset == 0) {
          for (size_t y = 0; y < channel.h; y++) {
            pixel_type *JXL_RESTRICT r = channel.Row(y);
            for (size_t x = 0; x < channel.w; x++) {
              uint32_t v =
                  reader->ReadHybridUintClusteredInlined<uses_lz77>(ctx_id, br);
              r[x] = UnpackSigned(v);
            }
          }
        } else {
          for (size_t y = 0; y < channel.h; y++) {
            pixel_type *JXL_RESTRICT r = channel.Row(y);
            for (size_t x = 0; x < channel.w; x++) {
              uint32_t v =
                  reader->ReadHybridUintClusteredMaybeInlined<uses_lz77>(ctx_id,
                                                                         br);
              r[x] = make_pixel(v, multiplier, offset);
            }
          }
        }
      }
      return true;
    } else if (uses_lz77 && predictor == Predictor::Gradient && offset == 0 &&
               multiplier == 1 && reader->IsHuffRleOnly()) {
      JXL_DEBUG_V(8, "Gradient RLE (fjxl) very fast track.");
      pixel_type_w sv = UnpackSigned(fl_v);
      for (size_t y = 0; y < channel.h; y++) {
        pixel_type *JXL_RESTRICT r = channel.Row(y);
        const pixel_type *JXL_RESTRICT rtop = (y ? channel.Row(y - 1) : r - 1);
        const pixel_type *JXL_RESTRICT rtopleft =
            (y ? channel.Row(y - 1) - 1 : r - 1);
        pixel_type_w guess = (y ? rtop[0] : 0);
        if (fl_run == 0) {
          reader->ReadHybridUintClusteredHuffRleOnly(ctx_id, br, &fl_v,
                                                     &fl_run);
          sv = UnpackSigned(fl_v);
        } else {
          fl_run--;
        }
        r[0] = sv + guess;
        for (size_t x = 1; x < channel.w; x++) {
          pixel_type left = r[x - 1];
          pixel_type top = rtop[x];
          pixel_type topleft = rtopleft[x];
          pixel_type_w guess = ClampedGradient(top, left, topleft);
          if (!fl_run) {
            reader->ReadHybridUintClusteredHuffRleOnly(ctx_id, br, &fl_v,
                                                       &fl_run);
            sv = UnpackSigned(fl_v);
          } else {
            fl_run--;
          }
          r[x] = sv + guess;
        }
      }
      return true;
    } else if (predictor == Predictor::Gradient && offset == 0 &&
               multiplier == 1) {
      JXL_DEBUG_V(8, "Gradient very fast track.");
      const intptr_t onerow = channel.plane.PixelsPerRow();
      for (size_t y = 0; y < channel.h; y++) {
        pixel_type *JXL_RESTRICT r = channel.Row(y);
        for (size_t x = 0; x < channel.w; x++) {
          pixel_type left = (x ? r[x - 1] : y ? *(r + x - onerow) : 0);
          pixel_type top = (y ? *(r + x - onerow) : left);
          pixel_type topleft = (x && y ? *(r + x - 1 - onerow) : left);
          pixel_type guess = ClampedGradient(top, left, topleft);
          uint64_t v = reader->ReadHybridUintClusteredMaybeInlined<uses_lz77>(
              ctx_id, br);
          r[x] = make_pixel(v, 1, guess);
        }
      }
      return true;
    }
  }

  // Check if this tree is a WP-only tree with a small enough property value
  // range.
  if (is_wp_only) {
    is_wp_only = TreeToLookupTable(tree, tree_lut);
  }
  if (is_gradient_only) {
    is_gradient_only = TreeToLookupTable(tree, tree_lut);
  }

  if (is_gradient_only) {
    JXL_DEBUG_V(8, "Gradient fast track.");
    const intptr_t onerow = channel.plane.PixelsPerRow();
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type *JXL_RESTRICT r = channel.Row(y);
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
        uint32_t ctx_id = tree_lut.context_lookup[pos];
        uint64_t v =
            reader->ReadHybridUintClusteredMaybeInlined<uses_lz77>(ctx_id, br);
        r[x] = make_pixel(v, 1, guess);
      }
    }
  } else if (!uses_lz77 && is_wp_only && channel.w > 8) {
    JXL_DEBUG_V(8, "WP fast track.");
    weighted::State wp_state(wp_header, channel.w, channel.h);
    Properties properties(1);
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type *JXL_RESTRICT r = channel.Row(y);
      const pixel_type *JXL_RESTRICT rtop = (y ? channel.Row(y - 1) : r - 1);
      const pixel_type *JXL_RESTRICT rtoptop =
          (y > 1 ? channel.Row(y - 2) : rtop);
      const pixel_type *JXL_RESTRICT rtopleft =
          (y ? channel.Row(y - 1) - 1 : r - 1);
      const pixel_type *JXL_RESTRICT rtopright =
          (y ? channel.Row(y - 1) + 1 : r - 1);
      size_t x = 0;
      {
        size_t offset = 0;
        pixel_type_w left = y ? rtop[x] : 0;
        pixel_type_w toptop = y ? rtoptop[x] : 0;
        pixel_type_w topright = (x + 1 < channel.w && y ? rtop[x + 1] : left);
        int32_t guess = wp_state.Predict</*compute_properties=*/true>(
            x, y, channel.w, left, left, topright, left, toptop, &properties,
            offset);
        uint32_t pos =
            kPropRangeFast + std::min(std::max(-kPropRangeFast, properties[0]),
                                      kPropRangeFast - 1);
        uint32_t ctx_id = tree_lut.context_lookup[pos];
        uint64_t v =
            reader->ReadHybridUintClusteredInlined<uses_lz77>(ctx_id, br);
        r[x] = make_pixel(v, 1, guess);
        wp_state.UpdateErrors(r[x], x, y, channel.w);
      }
      for (x = 1; x + 1 < channel.w; x++) {
        size_t offset = 0;
        int32_t guess = wp_state.Predict</*compute_properties=*/true>(
            x, y, channel.w, rtop[x], r[x - 1], rtopright[x], rtopleft[x],
            rtoptop[x], &properties, offset);
        uint32_t pos =
            kPropRangeFast + std::min(std::max(-kPropRangeFast, properties[0]),
                                      kPropRangeFast - 1);
        uint32_t ctx_id = tree_lut.context_lookup[pos];
        uint64_t v =
            reader->ReadHybridUintClusteredInlined<uses_lz77>(ctx_id, br);
        r[x] = make_pixel(v, 1, guess);
        wp_state.UpdateErrors(r[x], x, y, channel.w);
      }
      {
        size_t offset = 0;
        int32_t guess = wp_state.Predict</*compute_properties=*/true>(
            x, y, channel.w, rtop[x], r[x - 1], rtop[x], rtopleft[x],
            rtoptop[x], &properties, offset);
        uint32_t pos =
            kPropRangeFast + std::min(std::max(-kPropRangeFast, properties[0]),
                                      kPropRangeFast - 1);
        uint32_t ctx_id = tree_lut.context_lookup[pos];
        uint64_t v =
            reader->ReadHybridUintClusteredInlined<uses_lz77>(ctx_id, br);
        r[x] = make_pixel(v, 1, guess);
        wp_state.UpdateErrors(r[x], x, y, channel.w);
      }
    }
  } else if (!tree_has_wp_prop_or_pred) {
    // special optimized case: the weighted predictor and its properties are not
    // used, so no need to compute weights and properties.
    JXL_DEBUG_V(8, "Slow track.");
    MATreeLookup tree_lookup(tree);
    Properties properties = Properties(num_props);
    const intptr_t onerow = channel.plane.PixelsPerRow();
    JXL_ASSIGN_OR_RETURN(
        Channel references,
        Channel::Create(memory_manager,
                        properties.size() - kNumNonrefProperties, channel.w));
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type *JXL_RESTRICT p = channel.Row(y);
      PrecomputeReferences(channel, y, *image, chan, &references);
      InitPropsRow(&properties, static_props, y);
      if (y > 1 && channel.w > 8 && references.w == 0) {
        for (size_t x = 0; x < 2; x++) {
          PredictionResult res =
              PredictTreeNoWP(&properties, channel.w, p + x, onerow, x, y,
                              tree_lookup, references);
          uint64_t v =
              reader->ReadHybridUintClustered<uses_lz77>(res.context, br);
          p[x] = make_pixel(v, res.multiplier, res.guess);
        }
        for (size_t x = 2; x < channel.w - 2; x++) {
          PredictionResult res =
              PredictTreeNoWPNEC(&properties, channel.w, p + x, onerow, x, y,
                                 tree_lookup, references);
          uint64_t v = reader->ReadHybridUintClusteredInlined<uses_lz77>(
              res.context, br);
          p[x] = make_pixel(v, res.multiplier, res.guess);
        }
        for (size_t x = channel.w - 2; x < channel.w; x++) {
          PredictionResult res =
              PredictTreeNoWP(&properties, channel.w, p + x, onerow, x, y,
                              tree_lookup, references);
          uint64_t v =
              reader->ReadHybridUintClustered<uses_lz77>(res.context, br);
          p[x] = make_pixel(v, res.multiplier, res.guess);
        }
      } else {
        for (size_t x = 0; x < channel.w; x++) {
          PredictionResult res =
              PredictTreeNoWP(&properties, channel.w, p + x, onerow, x, y,
                              tree_lookup, references);
          uint64_t v = reader->ReadHybridUintClusteredMaybeInlined<uses_lz77>(
              res.context, br);
          p[x] = make_pixel(v, res.multiplier, res.guess);
        }
      }
    }
  } else {
    JXL_DEBUG_V(8, "Slowest track.");
    MATreeLookup tree_lookup(tree);
    Properties properties = Properties(num_props);
    const intptr_t onerow = channel.plane.PixelsPerRow();
    JXL_ASSIGN_OR_RETURN(
        Channel references,
        Channel::Create(memory_manager,
                        properties.size() - kNumNonrefProperties, channel.w));
    weighted::State wp_state(wp_header, channel.w, channel.h);
    for (size_t y = 0; y < channel.h; y++) {
      pixel_type *JXL_RESTRICT p = channel.Row(y);
      InitPropsRow(&properties, static_props, y);
      PrecomputeReferences(channel, y, *image, chan, &references);
      if (!uses_lz77 && y > 1 && channel.w > 8 && references.w == 0) {
        for (size_t x = 0; x < 2; x++) {
          PredictionResult res =
              PredictTreeWP(&properties, channel.w, p + x, onerow, x, y,
                            tree_lookup, references, &wp_state);
          uint64_t v =
              reader->ReadHybridUintClustered<uses_lz77>(res.context, br);
          p[x] = make_pixel(v, res.multiplier, res.guess);
          wp_state.UpdateErrors(p[x], x, y, channel.w);
        }
        for (size_t x = 2; x < channel.w - 2; x++) {
          PredictionResult res =
              PredictTreeWPNEC(&properties, channel.w, p + x, onerow, x, y,
                               tree_lookup, references, &wp_state);
          uint64_t v = reader->ReadHybridUintClusteredInlined<uses_lz77>(
              res.context, br);
          p[x] = make_pixel(v, res.multiplier, res.guess);
          wp_state.UpdateErrors(p[x], x, y, channel.w);
        }
        for (size_t x = channel.w - 2; x < channel.w; x++) {
          PredictionResult res =
              PredictTreeWP(&properties, channel.w, p + x, onerow, x, y,
                            tree_lookup, references, &wp_state);
          uint64_t v =
              reader->ReadHybridUintClustered<uses_lz77>(res.context, br);
          p[x] = make_pixel(v, res.multiplier, res.guess);
          wp_state.UpdateErrors(p[x], x, y, channel.w);
        }
      } else {
        for (size_t x = 0; x < channel.w; x++) {
          PredictionResult res =
              PredictTreeWP(&properties, channel.w, p + x, onerow, x, y,
                            tree_lookup, references, &wp_state);
          uint64_t v =
              reader->ReadHybridUintClustered<uses_lz77>(res.context, br);
          p[x] = make_pixel(v, res.multiplier, res.guess);
          wp_state.UpdateErrors(p[x], x, y, channel.w);
        }
      }
    }
  }
  return true;
}
}  // namespace detail

Status DecodeModularChannelMAANS(BitReader *br, ANSSymbolReader *reader,
                                 const std::vector<uint8_t> &context_map,
                                 const Tree &global_tree,
                                 const weighted::Header &wp_header,
                                 pixel_type chan, size_t group_id,
                                 TreeLut<uint8_t, false, false> &tree_lut,
                                 Image *image, uint32_t &fl_run,
                                 uint32_t &fl_v) {
  if (reader->UsesLZ77()) {
    return detail::DecodeModularChannelMAANS</*uses_lz77=*/true>(
        br, reader, context_map, global_tree, wp_header, chan, group_id,
        tree_lut, image, fl_run, fl_v);
  } else {
    return detail::DecodeModularChannelMAANS</*uses_lz77=*/false>(
        br, reader, context_map, global_tree, wp_header, chan, group_id,
        tree_lut, image, fl_run, fl_v);
  }
}

GroupHeader::GroupHeader() { Bundle::Init(this); }

Status ValidateChannelDimensions(const Image &image,
                                 const ModularOptions &options) {
  size_t nb_channels = image.channel.size();
  for (bool is_dc : {true, false}) {
    size_t group_dim = options.group_dim * (is_dc ? kBlockDim : 1);
    size_t c = image.nb_meta_channels;
    for (; c < nb_channels; c++) {
      const Channel &ch = image.channel[c];
      if (ch.w > options.group_dim || ch.h > options.group_dim) break;
    }
    for (; c < nb_channels; c++) {
      const Channel &ch = image.channel[c];
      if (ch.w == 0 || ch.h == 0) continue;  // skip empty
      bool is_dc_channel = std::min(ch.hshift, ch.vshift) >= 3;
      if (is_dc_channel != is_dc) continue;
      size_t tile_dim = group_dim >> std::max(ch.hshift, ch.vshift);
      if (tile_dim == 0) {
        return JXL_FAILURE("Inconsistent transforms");
      }
    }
  }
  return true;
}

Status ModularDecode(BitReader *br, Image &image, GroupHeader &header,
                     size_t group_id, ModularOptions *options,
                     const Tree *global_tree, const ANSCode *global_code,
                     const std::vector<uint8_t> *global_ctx_map,
                     const bool allow_truncated_group) {
  if (image.channel.empty()) return true;
  JxlMemoryManager *memory_manager = image.memory_manager();

  // decode transforms
  Status status = Bundle::Read(br, &header);
  if (!allow_truncated_group) JXL_RETURN_IF_ERROR(status);
  if (status.IsFatalError()) return status;
  if (!br->AllReadsWithinBounds()) {
    // Don't do/undo transforms if header is incomplete.
    header.transforms.clear();
    image.transform = header.transforms;
    for (auto &ch : image.channel) {
      ZeroFillImage(&ch.plane);
    }
    return Status(StatusCode::kNotEnoughBytes);
  }

  JXL_DEBUG_V(3, "Image data underwent %" PRIuS " transformations: ",
              header.transforms.size());
  image.transform = header.transforms;
  for (Transform &transform : image.transform) {
    JXL_RETURN_IF_ERROR(transform.MetaApply(image));
  }
  if (image.error) {
    return JXL_FAILURE("Corrupt file. Aborting.");
  }
  JXL_RETURN_IF_ERROR(ValidateChannelDimensions(image, *options));

  size_t nb_channels = image.channel.size();

  size_t num_chans = 0;
  size_t distance_multiplier = 0;
  for (size_t i = 0; i < nb_channels; i++) {
    Channel &channel = image.channel[i];
    if (!channel.w || !channel.h) {
      continue;  // skip empty channels
    }
    if (i >= image.nb_meta_channels && (channel.w > options->max_chan_size ||
                                        channel.h > options->max_chan_size)) {
      break;
    }
    if (channel.w > distance_multiplier) {
      distance_multiplier = channel.w;
    }
    num_chans++;
  }
  if (num_chans == 0) return true;

  size_t next_channel = 0;
  auto scope_guard = MakeScopeGuard([&]() {
    for (size_t c = next_channel; c < image.channel.size(); c++) {
      ZeroFillImage(&image.channel[c].plane);
    }
  });
  // Do not do anything if truncated groups are not allowed.
  if (allow_truncated_group) scope_guard.Disarm();

  // Read tree.
  Tree tree_storage;
  std::vector<uint8_t> context_map_storage;
  ANSCode code_storage;
  const Tree *tree = &tree_storage;
  const ANSCode *code = &code_storage;
  const std::vector<uint8_t> *context_map = &context_map_storage;
  if (!header.use_global_tree) {
    uint64_t max_tree_size = 1024;
    for (size_t i = 0; i < nb_channels; i++) {
      Channel &channel = image.channel[i];
      if (i >= image.nb_meta_channels && (channel.w > options->max_chan_size ||
                                          channel.h > options->max_chan_size)) {
        break;
      }
      uint64_t pixels = channel.w * channel.h;
      max_tree_size += pixels;
    }
    max_tree_size = std::min(static_cast<uint64_t>(1 << 20), max_tree_size);
    JXL_RETURN_IF_ERROR(
        DecodeTree(memory_manager, br, &tree_storage, max_tree_size));
    JXL_RETURN_IF_ERROR(DecodeHistograms(memory_manager, br,
                                         (tree_storage.size() + 1) / 2,
                                         &code_storage, &context_map_storage));
  } else {
    if (!global_tree || !global_code || !global_ctx_map ||
        global_tree->empty()) {
      return JXL_FAILURE("No global tree available but one was requested");
    }
    tree = global_tree;
    code = global_code;
    context_map = global_ctx_map;
  }

  // Read channels
  JXL_ASSIGN_OR_RETURN(ANSSymbolReader reader,
                       ANSSymbolReader::Create(code, br, distance_multiplier));
  auto tree_lut = jxl::make_unique<TreeLut<uint8_t, false, false>>();
  uint32_t fl_run = 0;
  uint32_t fl_v = 0;
  for (; next_channel < nb_channels; next_channel++) {
    Channel &channel = image.channel[next_channel];
    if (!channel.w || !channel.h) {
      continue;  // skip empty channels
    }
    if (next_channel >= image.nb_meta_channels &&
        (channel.w > options->max_chan_size ||
         channel.h > options->max_chan_size)) {
      break;
    }
    JXL_RETURN_IF_ERROR(DecodeModularChannelMAANS(
        br, &reader, *context_map, *tree, header.wp_header, next_channel,
        group_id, *tree_lut, &image, fl_run, fl_v));

    // Truncated group.
    if (!br->AllReadsWithinBounds()) {
      if (!allow_truncated_group) return JXL_FAILURE("Truncated input");
      return Status(StatusCode::kNotEnoughBytes);
    }
  }

  // Make sure no zero-filling happens even if next_channel < nb_channels.
  scope_guard.Disarm();

  if (!reader.CheckANSFinalState()) {
    return JXL_FAILURE("ANS decode final state failed");
  }
  return true;
}

Status ModularGenericDecompress(BitReader *br, Image &image,
                                GroupHeader *header, size_t group_id,
                                ModularOptions *options, bool undo_transforms,
                                const Tree *tree, const ANSCode *code,
                                const std::vector<uint8_t> *ctx_map,
                                bool allow_truncated_group) {
  std::vector<std::pair<uint32_t, uint32_t>> req_sizes;
  req_sizes.reserve(image.channel.size());
  for (const auto &c : image.channel) {
    req_sizes.emplace_back(c.w, c.h);
  }
  GroupHeader local_header;
  if (header == nullptr) header = &local_header;
  size_t bit_pos = br->TotalBitsConsumed();
  auto dec_status = ModularDecode(br, image, *header, group_id, options, tree,
                                  code, ctx_map, allow_truncated_group);
  if (!allow_truncated_group) JXL_RETURN_IF_ERROR(dec_status);
  if (dec_status.IsFatalError()) return dec_status;
  if (undo_transforms) image.undo_transforms(header->wp_header);
  if (image.error) return JXL_FAILURE("Corrupt file. Aborting.");
  JXL_DEBUG_V(4,
              "Modular-decoded a %" PRIuS "x%" PRIuS " nbchans=%" PRIuS
              " image from %" PRIuS " bytes",
              image.w, image.h, image.channel.size(),
              (br->TotalBitsConsumed() - bit_pos) / 8);
  JXL_DEBUG_V(5, "Modular image: %s", image.DebugString().c_str());
  (void)bit_pos;
  // Check that after applying all transforms we are back to the requested
  // image sizes, otherwise there's a programming error with the
  // transformations.
  if (undo_transforms) {
    JXL_ENSURE(image.channel.size() == req_sizes.size());
    for (size_t c = 0; c < req_sizes.size(); c++) {
      JXL_ENSURE(req_sizes[c].first == image.channel[c].w);
      JXL_ENSURE(req_sizes[c].second == image.channel[c].h);
    }
  }
  return dec_status;
}

}  // namespace jxl
