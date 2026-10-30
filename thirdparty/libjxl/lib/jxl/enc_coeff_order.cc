// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/lehmer_code.h"
#include "lib/jxl/memory_manager_internal.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;

std::pair<uint32_t, uint32_t> ComputeUsedOrders(
    const SpeedTier speed, const AcStrategyImage& ac_strategy,
    const Rect& rect) {
  // No coefficient reordering in Falcon or faster.
  // Only uses DCT8 = 0, so bitfield = 1.
  if (speed >= SpeedTier::kFalcon) return {1, 1};

  uint32_t ret = 0;
  uint32_t ret_customize = 0;
  size_t xsize_blocks = rect.xsize();
  size_t ysize_blocks = rect.ysize();
  // TODO(veluca): precompute when doing DCT.
  for (size_t by = 0; by < ysize_blocks; ++by) {
    AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      int ord = kStrategyOrder[acs_row[bx].RawStrategy()];
      // Do not customize coefficient orders for blocks bigger than 32x32.
      ret |= 1u << ord;
      if (ord > 6) {
        continue;
      }
      ret_customize |= 1u << ord;
    }
  }
  // Use default orders for small images.
  if (ac_strategy.xsize() < 5 && ac_strategy.ysize() < 5) return {ret, 0};
  return {ret, ret_customize};
}

Status ComputeCoeffOrder(SpeedTier speed, const ACImage& ac_image,
                         const AcStrategyImage& ac_strategy,
                         const FrameDimensions& frame_dim,
                         uint32_t& all_used_orders, uint32_t prev_used_acs,
                         uint32_t current_used_acs,
                         uint32_t current_used_orders,
                         coeff_order_t* JXL_RESTRICT order) {
  JxlMemoryManager* memory_manager = ac_strategy.memory_manager();
  std::vector<int64_t> num_zeros(kCoeffOrderMaxSize);
  // If compressing at high speed and only using 8x8 DCTs, only consider a
  // subset of blocks.
  double block_fraction = 1.0f;
  // TODO(veluca): figure out why sampling blocks if non-8x8s are used makes
  // encoding significantly less dense.
  if (speed >= SpeedTier::kSquirrel && current_used_orders == 1) {
    block_fraction = 0.5f;
  }
  // No need to compute number of zero coefficients if all orders are the
  // default.
  if (current_used_orders != 0) {
    uint64_t threshold =
        (std::numeric_limits<uint64_t>::max() >> 32) * block_fraction;
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

    // Count number of zero coefficients, separately for each DCT band.
    // TODO(veluca): precompute when doing DCT.
    for (size_t group_index = 0; group_index < frame_dim.num_groups;
         group_index++) {
      const size_t gx = group_index % frame_dim.xsize_groups;
      const size_t gy = group_index / frame_dim.xsize_groups;
      const Rect rect(gx * kGroupDimInBlocks, gy * kGroupDimInBlocks,
                      kGroupDimInBlocks, kGroupDimInBlocks,
                      frame_dim.xsize_blocks, frame_dim.ysize_blocks);
      ConstACPtr rows[3];
      ACType type = ac_image.Type();
      for (size_t c = 0; c < 3; c++) {
        rows[c] = ac_image.PlaneRow(c, group_index, 0);
      }
      size_t ac_offset = 0;

      // TODO(veluca): SIMDfy.
      for (size_t by = 0; by < rect.ysize(); ++by) {
        AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
        for (size_t bx = 0; bx < rect.xsize(); ++bx) {
          AcStrategy acs = acs_row[bx];
          if (!acs.IsFirstBlock()) continue;
          if (!use_sample()) continue;
          size_t size = kDCTBlockSize << acs.log2_covered_blocks();
          for (size_t c = 0; c < 3; ++c) {
            const size_t order_offset =
                CoeffOrderOffset(kStrategyOrder[acs.RawStrategy()], c);
            if (type == ACType::k16) {
              for (size_t k = 0; k < size; k++) {
                bool is_zero = rows[c].ptr16[ac_offset + k] == 0;
                num_zeros[order_offset + k] += is_zero ? 1 : 0;
              }
            } else {
              for (size_t k = 0; k < size; k++) {
                bool is_zero = rows[c].ptr32[ac_offset + k] == 0;
                num_zeros[order_offset + k] += is_zero ? 1 : 0;
              }
            }
            // Ensure LLFs are first in the order.
            size_t cx = acs.covered_blocks_x();
            size_t cy = acs.covered_blocks_y();
            CoefficientLayout(&cy, &cx);
            for (size_t iy = 0; iy < cy; iy++) {
              for (size_t ix = 0; ix < cx; ix++) {
                num_zeros[order_offset + iy * kBlockDim * cx + ix] = -1;
              }
            }
          }
          ac_offset += size;
        }
      }
    }
  }
  struct PosAndCount {
    uint32_t pos;
    // Saving index breaks the ties for non-stable sort
    uint64_t count_and_idx;
  };
  size_t mem_bytes = AcStrategy::kMaxCoeffArea * sizeof(PosAndCount);
  JXL_ASSIGN_OR_RETURN(auto mem,
                       AlignedMemory::Create(memory_manager, mem_bytes));

  std::vector<coeff_order_t> natural_order_buffer;

  uint16_t computed = 0;
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    size_t sz = kDCTBlockSize * acs.covered_blocks_x() * acs.covered_blocks_y();
    // Expected maximal size is 256 x 256.
    JXL_DASSERT(sz <= (1 << 16));

    // Do nothing for transforms that don't appear.
    if ((1 << ord) & ~current_used_acs) continue;

    // Do nothing if we already committed to this custom order previously.
    if ((1 << ord) & prev_used_acs) continue;
    if ((1 << ord) & all_used_orders) continue;

    if (natural_order_buffer.size() < sz) natural_order_buffer.resize(sz);
    acs.ComputeNaturalCoeffOrder(natural_order_buffer.data());

    // Ensure natural coefficient order is not permuted if the order is
    // not transmitted.
    if ((1 << ord) & ~current_used_orders) {
      for (size_t c = 0; c < 3; c++) {
        size_t offset = CoeffOrderOffset(ord, c);
        JXL_ENSURE(CoeffOrderOffset(ord, c + 1) - offset == sz);
        memcpy(&order[offset], natural_order_buffer.data(),
               sz * sizeof(*order));
      }
      continue;
    }

    bool is_nondefault = false;
    for (uint8_t c = 0; c < 3; c++) {
      // Apply zig-zag order.
      PosAndCount* pos_and_val = mem.address<PosAndCount>();
      size_t offset = CoeffOrderOffset(ord, c);
      JXL_ENSURE(CoeffOrderOffset(ord, c + 1) - offset == sz);
      float inv_sqrt_sz = 1.0f / std::sqrt(sz);
      for (size_t i = 0; i < sz; ++i) {
        size_t pos = natural_order_buffer[i];
        pos_and_val[i].pos = pos;
        // We don't care for the exact number -> quantize number of zeros,
        // to get less permuted order.
        uint64_t count = num_zeros[offset + pos] * inv_sqrt_sz + 0.1f;
        // Worst case: all dct8x8, all zeroes: count <= nb_pixels/64/8
        // nb_pixels is limited to 2^40 (Level 10 limit)
        // so count is limited to 2^31
        JXL_DASSERT(count < (uint64_t{1} << 48));
        pos_and_val[i].count_and_idx = (count << 16) | i;
      }

      // Stable-sort -> elements with same number of zeros will preserve their
      // order.
      auto comparator = [](const PosAndCount& a, const PosAndCount& b) -> bool {
        return a.count_and_idx < b.count_and_idx;
      };
      std::sort(pos_and_val, pos_and_val + sz, comparator);

      // Grab indices.
      for (size_t i = 0; i < sz; ++i) {
        order[offset + i] = pos_and_val[i].pos;
        is_nondefault |= natural_order_buffer[i] != pos_and_val[i].pos;
      }
    }
    if (!is_nondefault) {
      current_used_orders &= ~(1 << ord);
    }
  }
  all_used_orders |= current_used_orders;
  return true;
}

namespace {

Status TokenizePermutation(const coeff_order_t* JXL_RESTRICT order, size_t skip,
                           size_t size, std::vector<Token>* tokens) {
  std::vector<LehmerT> lehmer(size);
  std::vector<uint32_t> temp(size + 1);
  JXL_RETURN_IF_ERROR(
      ComputeLehmerCode(order, temp.data(), size, lehmer.data()));
  size_t end = size;
  while (end > skip && lehmer[end - 1] == 0) {
    --end;
  }
  tokens->emplace_back(CoeffOrderContext(size),
                       static_cast<uint32_t>(end - skip));
  uint32_t last = 0;
  for (size_t i = skip; i < end; ++i) {
    tokens->emplace_back(CoeffOrderContext(last), lehmer[i]);
    last = lehmer[i];
  }
  return true;
}

}  // namespace

Status EncodePermutation(const coeff_order_t* JXL_RESTRICT order, size_t skip,
                         size_t size, BitWriter* writer, LayerType layer,
                         AuxOut* aux_out) {
  JxlMemoryManager* memory_manager = writer->memory_manager();
  std::vector<std::vector<Token>> tokens(1);
  JXL_RETURN_IF_ERROR(TokenizePermutation(order, skip, size, tokens.data()));
  EntropyEncodingData codes;
  JXL_ASSIGN_OR_RETURN(
      size_t cost, BuildAndEncodeHistograms(memory_manager, HistogramParams(),
                                            kPermutationContexts, tokens,
                                            &codes, writer, layer, aux_out));
  (void)cost;
  JXL_RETURN_IF_ERROR(WriteTokens(tokens[0], codes, 0, writer, layer, aux_out));
  return true;
}

namespace {
Status EncodeCoeffOrder(const coeff_order_t* JXL_RESTRICT order, AcStrategy acs,
                        std::vector<Token>* tokens, coeff_order_t* order_zigzag,
                        std::vector<coeff_order_t>& natural_order_lut) {
  const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
  const size_t size = kDCTBlockSize * llf;
  for (size_t i = 0; i < size; ++i) {
    order_zigzag[i] = natural_order_lut[order[i]];
  }
  JXL_RETURN_IF_ERROR(TokenizePermutation(order_zigzag, llf, size, tokens));
  return true;
}
}  // namespace

Status EncodeCoeffOrders(uint16_t used_orders,
                         const coeff_order_t* JXL_RESTRICT order,
                         BitWriter* writer, LayerType layer,
                         AuxOut* JXL_RESTRICT aux_out) {
  JxlMemoryManager* memory_manager = writer->memory_manager();
  size_t mem_bytes = AcStrategy::kMaxCoeffArea * sizeof(coeff_order_t);
  JXL_ASSIGN_OR_RETURN(auto mem,
                       AlignedMemory::Create(memory_manager, mem_bytes));
  uint16_t computed = 0;
  std::vector<std::vector<Token>> tokens(1);
  std::vector<coeff_order_t> natural_order_lut;
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    if ((used_orders & (1 << ord)) == 0) continue;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
    const size_t size = kDCTBlockSize * llf;
    if (natural_order_lut.size() < size) natural_order_lut.resize(size);
    acs.ComputeNaturalCoeffOrderLut(natural_order_lut.data());
    for (size_t c = 0; c < 3; c++) {
      JXL_RETURN_IF_ERROR(
          EncodeCoeffOrder(&order[CoeffOrderOffset(ord, c)], acs, tokens.data(),
                           mem.address<coeff_order_t>(), natural_order_lut));
    }
  }
  // Do not write anything if no order is used.
  if (used_orders != 0) {
    EntropyEncodingData codes;
    JXL_ASSIGN_OR_RETURN(
        size_t cost, BuildAndEncodeHistograms(memory_manager, HistogramParams(),
                                              kPermutationContexts, tokens,
                                              &codes, writer, layer, aux_out));
    (void)cost;
    JXL_RETURN_IF_ERROR(
        WriteTokens(tokens[0], codes, 0, writer, layer, aux_out));
  }
  return true;
}

}  // namespace jxl
