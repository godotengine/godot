// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_ENCODING_ENCODING_H_
#define LIB_JXL_MODULAR_ENCODING_ENCODING_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

struct ANSCode;
class BitReader;

// Valid range of properties for using lookup tables instead of trees.
constexpr int32_t kPropRangeFast = 512 << 4;

struct GroupHeader : public Fields {
  GroupHeader();

  JXL_FIELDS_NAME(GroupHeader)

  Status VisitFields(Visitor *JXL_RESTRICT visitor) override {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &use_global_tree));
    JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&wp_header));
    uint32_t num_transforms = static_cast<uint32_t>(transforms.size());
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(0), Val(1), BitsOffset(4, 2),
                                           BitsOffset(8, 18), 0,
                                           &num_transforms));
    if (visitor->IsReading()) transforms.resize(num_transforms);
    for (size_t i = 0; i < num_transforms; i++) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&transforms[i]));
    }
    return true;
  }

  bool use_global_tree;
  weighted::Header wp_header;

  std::vector<Transform> transforms;
};

FlatTree FilterTree(const Tree &global_tree,
                    std::array<pixel_type, kNumStaticProperties> &static_props,
                    size_t *num_props, bool *use_wp, bool *wp_only,
                    bool *gradient_only);

template <typename T, bool HAS_OFFSETS, bool HAS_MULTIPLIERS>
struct TreeLut {
  std::array<T, 2 * kPropRangeFast> context_lookup;
  std::array<int8_t, HAS_OFFSETS ? (2 * kPropRangeFast) : 0> offsets;
  std::array<int8_t, HAS_MULTIPLIERS ? (2 * kPropRangeFast) : 0> multipliers;
};

template <typename T, bool HAS_OFFSETS, bool HAS_MULTIPLIERS>
bool TreeToLookupTable(const FlatTree &tree,
                       TreeLut<T, HAS_OFFSETS, HAS_MULTIPLIERS> &lut) {
  struct TreeRange {
    // Begin *excluded*, end *included*. This works best with > vs <= decision
    // nodes.
    int begin, end;
    size_t pos;
  };
  std::vector<TreeRange> ranges;
  ranges.push_back(TreeRange{-kPropRangeFast - 1, kPropRangeFast - 1, 0});
  while (!ranges.empty()) {
    TreeRange cur = ranges.back();
    ranges.pop_back();
    if (cur.begin < -kPropRangeFast - 1 || cur.begin >= kPropRangeFast - 1 ||
        cur.end > kPropRangeFast - 1) {
      // Tree is outside the allowed range, exit.
      return false;
    }
    auto &node = tree[cur.pos];
    // Leaf.
    if (node.property0 == -1) {
      if (node.predictor_offset < std::numeric_limits<int8_t>::min() ||
          node.predictor_offset > std::numeric_limits<int8_t>::max()) {
        return false;
      }
      if (node.multiplier < std::numeric_limits<int8_t>::min() ||
          node.multiplier > std::numeric_limits<int8_t>::max()) {
        return false;
      }
      if (!HAS_MULTIPLIERS && node.multiplier != 1) {
        return false;
      }
      if (!HAS_OFFSETS && node.predictor_offset != 0) {
        return false;
      }
      for (int i = cur.begin + 1; i < cur.end + 1; i++) {
        lut.context_lookup[i + kPropRangeFast] = node.childID;
        if (HAS_MULTIPLIERS) {
          lut.multipliers[i + kPropRangeFast] = node.multiplier;
        }
        if (HAS_OFFSETS) {
          lut.offsets[i + kPropRangeFast] = node.predictor_offset;
        }
      }
      continue;
    }
    // > side of top node.
    if (node.properties[0] >= kNumStaticProperties) {
      ranges.push_back(TreeRange({node.splitvals[0], cur.end, node.childID}));
      ranges.push_back(
          TreeRange({node.splitval0, node.splitvals[0], node.childID + 1}));
    } else {
      ranges.push_back(TreeRange({node.splitval0, cur.end, node.childID}));
    }
    // <= side
    if (node.properties[1] >= kNumStaticProperties) {
      ranges.push_back(
          TreeRange({node.splitvals[1], node.splitval0, node.childID + 2}));
      ranges.push_back(
          TreeRange({cur.begin, node.splitvals[1], node.childID + 3}));
    } else {
      ranges.push_back(
          TreeRange({cur.begin, node.splitval0, node.childID + 2}));
    }
  }
  return true;
}
// TODO(veluca): make cleaner interfaces.

Status ValidateChannelDimensions(const Image &image,
                                 const ModularOptions &options);

Status ModularGenericDecompress(BitReader *br, Image &image,
                                GroupHeader *header, size_t group_id,
                                ModularOptions *options,
                                bool undo_transforms = true,
                                const Tree *tree = nullptr,
                                const ANSCode *code = nullptr,
                                const std::vector<uint8_t> *ctx_map = nullptr,
                                bool allow_truncated_group = false);
}  // namespace jxl

#endif  // LIB_JXL_MODULAR_ENCODING_ENCODING_H_
