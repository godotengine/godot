// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/encoding/dec_ma.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/modular/encoding/ma_common.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/pack_signed.h"

namespace jxl {

namespace {

enum class NextAction { CHECK_AND_GO_LEFT, GO_RIGHT, POP };

struct WorkItem {
  size_t node_index;
  pixel_type orig_l;
  pixel_type orig_u;
  NextAction action;
};

Status ValidateTree(const Tree& tree) {
  if (tree.empty()) return true;
  // TODO(eustas): or invalid?

  int num_properties = 0;
  for (auto node : tree) {
    if (node.property >= num_properties) {
      num_properties = node.property + 1;
    }
  }

  std::vector<std::pair<pixel_type, pixel_type>> property_ranges(
      num_properties);
  for (int i = 0; i < num_properties; i++) {
    property_ranges[i].first = std::numeric_limits<pixel_type>::min();
    property_ranges[i].second = std::numeric_limits<pixel_type>::max();
  }

  constexpr size_t kHeightLimit = 2048;

  std::vector<WorkItem> stack;
  stack.push_back({/*node_index=*/0, /*orig_l=*/0, /*orig_u=*/0,
                   NextAction::CHECK_AND_GO_LEFT});

  while (!stack.empty()) {
    if (stack.size() >= kHeightLimit) return JXL_FAILURE("Tree too tall");
    WorkItem& item = stack.back();
    const auto& node = tree[item.node_index];
    switch (item.action) {
      case NextAction::CHECK_AND_GO_LEFT: {
        int16_t p = node.property;
        if (p == -1) {
          stack.pop_back();
          continue;
        }
        PropertyVal v = node.splitval;
        pixel_type l = property_ranges[p].first;
        pixel_type u = property_ranges[p].second;
        if (l > v || u <= v) {
          return JXL_FAILURE("Invalid tree");
        }
        item.orig_l = l;
        item.orig_u = u;
        item.action = NextAction::GO_RIGHT;
        property_ranges[node.property].first = node.splitval + 1;
        stack.push_back({/*node_index=*/node.lchild,
                         /*orig_l=*/0, /*orig_u=*/0,
                         NextAction::CHECK_AND_GO_LEFT});
        continue;
      }

      case NextAction::GO_RIGHT:
        item.action = NextAction::POP;
        property_ranges[node.property].first = item.orig_l;
        property_ranges[node.property].second = node.splitval;
        stack.push_back({/*node_index=*/node.rchild,
                         /*orig_l=*/0, /*orig_u=*/0,
                         NextAction::CHECK_AND_GO_LEFT});
        continue;

      case NextAction::POP:
        property_ranges[node.property].second = item.orig_u;
        stack.pop_back();
        continue;
    }
  }

  return true;
}

Status DecodeTree(BitReader* br, ANSSymbolReader* reader,
                  const std::vector<uint8_t>& context_map, Tree* tree,
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
      Predictor p = static_cast<Predictor>(static_cast<uint32_t>(predictor));
      tree->emplace_back(-1, 0, static_cast<int>(leaf_id), 0, p,
                         predictor_offset, multiplier);
      leaf_id++;
      continue;
    }
    int splitval =
        UnpackSigned(reader->ReadHybridUint(kSplitValContext, br, context_map));
    tree->emplace_back(
        property, splitval, static_cast<int>(tree->size() + to_decode + 1),
        static_cast<int>(tree->size() + to_decode + 2), Predictor::Zero, 0, 1);
    to_decode += 2;
  }
  return ValidateTree(*tree);
}
}  // namespace

Status DecodeTree(JxlMemoryManager* memory_manager, BitReader* br, Tree* tree,
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
