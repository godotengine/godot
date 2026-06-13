// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/encoding/dec_ma.h"

#include <jxl/memory_manager.h>

#include <limits>
#include <vector>

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/modular/encoding/ma_common.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/pack_signed.h"

namespace jxl {

namespace {

Status ValidateTree(const Tree &tree) {
  int num_properties = 0;
  for (auto node : tree) {
    if (node.property >= num_properties) {
      num_properties = node.property + 1;
    }
  }
  std::vector<int> height(tree.size());
  std::vector<std::pair<pixel_type, pixel_type>> property_ranges(
      num_properties * tree.size());
  for (int i = 0; i < num_properties; i++) {
    property_ranges[i].first = std::numeric_limits<pixel_type>::min();
    property_ranges[i].second = std::numeric_limits<pixel_type>::max();
  }
  const int kHeightLimit = 2048;
  for (size_t i = 0; i < tree.size(); i++) {
    if (height[i] > kHeightLimit) {
      return JXL_FAILURE("Tree too tall: %d", height[i]);
    }
    if (tree[i].property == -1) continue;
    height[tree[i].lchild] = height[i] + 1;
    height[tree[i].rchild] = height[i] + 1;
    for (size_t p = 0; p < static_cast<size_t>(num_properties); p++) {
      if (p == static_cast<size_t>(tree[i].property)) {
        pixel_type l = property_ranges[i * num_properties + p].first;
        pixel_type u = property_ranges[i * num_properties + p].second;
        pixel_type val = tree[i].splitval;
        if (l > val || u <= val) {
          return JXL_FAILURE("Invalid tree");
        }
        property_ranges[tree[i].lchild * num_properties + p] =
            std::make_pair(val + 1, u);
        property_ranges[tree[i].rchild * num_properties + p] =
            std::make_pair(l, val);
      } else {
        property_ranges[tree[i].lchild * num_properties + p] =
            property_ranges[i * num_properties + p];
        property_ranges[tree[i].rchild * num_properties + p] =
            property_ranges[i * num_properties + p];
      }
    }
  }
  return true;
}

Status DecodeTree(BitReader *br, ANSSymbolReader *reader,
                  const std::vector<uint8_t> &context_map, Tree *tree,
                  size_t tree_size_limit) {
  size_t leaf_id = 0;
  size_t to_decode = 1;
  tree->clear();
  while (to_decode > 0) {
    JXL_RETURN_IF_ERROR(br->AllReadsWithinBounds());
    if (tree->size() > tree_size_limit) {
      return JXL_FAILURE("Tree is too large: %" PRIuS " nodes vs %" PRIuS
                         " max nodes",
                         tree->size(), tree_size_limit);
    }
    to_decode--;
    uint32_t prop1 = reader->ReadHybridUint(kPropertyContext, br, context_map);
    if (prop1 > 256) return JXL_FAILURE("Invalid tree property value");
    int property = prop1 - 1;
    if (property == -1) {
      size_t predictor =
          reader->ReadHybridUint(kPredictorContext, br, context_map);
      if (predictor >= kNumModularPredictors) {
        return JXL_FAILURE("Invalid predictor");
      }
      int64_t predictor_offset =
          UnpackSigned(reader->ReadHybridUint(kOffsetContext, br, context_map));
      uint32_t mul_log =
          reader->ReadHybridUint(kMultiplierLogContext, br, context_map);
      if (mul_log >= 31) {
        return JXL_FAILURE("Invalid multiplier logarithm");
      }
      uint32_t mul_bits =
          reader->ReadHybridUint(kMultiplierBitsContext, br, context_map);
      if (mul_bits >= (1u << (31u - mul_log)) - 1u) {
        return JXL_FAILURE("Invalid multiplier");
      }
      uint32_t multiplier = (mul_bits + 1U) << mul_log;
      tree->emplace_back(-1, 0, leaf_id++, 0, static_cast<Predictor>(predictor),
                         predictor_offset, multiplier);
      continue;
    }
    int splitval =
        UnpackSigned(reader->ReadHybridUint(kSplitValContext, br, context_map));
    tree->emplace_back(property, splitval, tree->size() + to_decode + 1,
                       tree->size() + to_decode + 2, Predictor::Zero, 0, 1);
    to_decode += 2;
  }
  return ValidateTree(*tree);
}
}  // namespace

Status DecodeTree(JxlMemoryManager *memory_manager, BitReader *br, Tree *tree,
                  size_t tree_size_limit) {
  std::vector<uint8_t> tree_context_map;
  ANSCode tree_code;
  JXL_RETURN_IF_ERROR(DecodeHistograms(memory_manager, br, kNumTreeContexts,
                                       &tree_code, &tree_context_map));
  // TODO(eustas): investigate more infinite tree cases.
  if (tree_code.degenerate_symbols[tree_context_map[kPropertyContext]] > 0) {
    return JXL_FAILURE("Infinite tree");
  }
  JXL_ASSIGN_OR_RETURN(ANSSymbolReader reader,
                       ANSSymbolReader::Create(&tree_code, br));
  JXL_RETURN_IF_ERROR(DecodeTree(br, &reader, tree_context_map, tree,
                                 std::min(tree_size_limit, kMaxTreeSize)));
  if (!reader.CheckANSFinalState()) {
    return JXL_FAILURE("ANS decode final state failed");
  }
  return true;
}

}  // namespace jxl
