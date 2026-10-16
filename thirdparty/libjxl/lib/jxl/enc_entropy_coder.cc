// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_entropy_coder.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_entropy_coder.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/image.h"
#include "lib/jxl/pack_signed.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::AndNot;
using hwy::HWY_NAMESPACE::Eq;
using hwy::HWY_NAMESPACE::GetLane;

// Returns number of non-zero coefficients (but skip LLF).
// We cannot rely on block[] being all-zero bits, so first truncate to integer.
// Also writes the per-8x8 block nzeros starting at nzeros_pos.
int32_t NumNonZeroExceptLLF(const size_t cx, const size_t cy,
                            const AcStrategy acs, const size_t covered_blocks,
                            const size_t log2_covered_blocks,
                            const int32_t* JXL_RESTRICT block,
                            const size_t nzeros_stride,
                            int32_t* JXL_RESTRICT nzeros_pos) {
  const HWY_CAPPED(int32_t, kBlockDim) di;

  const auto zero = Zero(di);
  // Add FF..FF for every zero coefficient, negate to get #zeros.
  auto neg_sum_zero = zero;

  {
    // Mask sufficient for one row of coefficients.
    HWY_ALIGN const int32_t
        llf_mask_lanes[AcStrategy::kMaxCoeffBlocks * (1 + kBlockDim)] = {
            -1, -1, -1, -1};
    // First cx=1,2,4 elements are FF..FF, others 0.
    const int32_t* llf_mask_pos =
        llf_mask_lanes + AcStrategy::kMaxCoeffBlocks - cx;

    // Rows with LLF: mask out the LLF
    for (size_t y = 0; y < cy; y++) {
      for (size_t x = 0; x < cx * kBlockDim; x += Lanes(di)) {
        const auto llf_mask = LoadU(di, llf_mask_pos + x);

        // LLF counts as zero so we don't include it in nzeros.
        const auto coef =
            AndNot(llf_mask, Load(di, &block[y * cx * kBlockDim + x]));

        neg_sum_zero = Add(neg_sum_zero, VecFromMask(di, Eq(coef, zero)));
      }
    }
  }

  // Remaining rows: no mask
  for (size_t y = cy; y < cy * kBlockDim; y++) {
    for (size_t x = 0; x < cx * kBlockDim; x += Lanes(di)) {
      const auto coef = Load(di, &block[y * cx * kBlockDim + x]);
      neg_sum_zero = Add(neg_sum_zero, VecFromMask(di, Eq(coef, zero)));
    }
  }

  // We want area - sum_zero, add because neg_sum_zero is already negated.
  const int32_t nzeros = static_cast<int32_t>(cx * cy * kDCTBlockSize) +
                         GetLane(SumOfLanes(di, neg_sum_zero));

  const int32_t shifted_nzeros = static_cast<int32_t>(
      (nzeros + covered_blocks - 1) >> log2_covered_blocks);
  // Need non-canonicalized dimensions!
  for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
    for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
      nzeros_pos[x + y * nzeros_stride] = shifted_nzeros;
    }
  }

  return nzeros;
}

// Specialization for 8x8, where only top-left is LLF/DC.
// About 1% overall speedup vs. NumNonZeroExceptLLF.
int32_t NumNonZero8x8ExceptDC(const int32_t* JXL_RESTRICT block,
                              int32_t* JXL_RESTRICT nzeros_pos) {
  const HWY_CAPPED(int32_t, kBlockDim) di;

  const auto zero = Zero(di);
  // Add FF..FF for every zero coefficient, negate to get #zeros.
  auto neg_sum_zero = zero;

  {
    // First row has DC, so mask
    const size_t y = 0;
    HWY_ALIGN const int32_t dc_mask_lanes[kBlockDim] = {-1};

    for (size_t x = 0; x < kBlockDim; x += Lanes(di)) {
      const auto dc_mask = Load(di, dc_mask_lanes + x);

      // DC counts as zero so we don't include it in nzeros.
      const auto coef = AndNot(dc_mask, Load(di, &block[y * kBlockDim + x]));

      neg_sum_zero = Add(neg_sum_zero, VecFromMask(di, Eq(coef, zero)));
    }
  }

  // Remaining rows: no mask
  for (size_t y = 1; y < kBlockDim; y++) {
    for (size_t x = 0; x < kBlockDim; x += Lanes(di)) {
      const auto coef = Load(di, &block[y * kBlockDim + x]);
      neg_sum_zero = Add(neg_sum_zero, VecFromMask(di, Eq(coef, zero)));
    }
  }

  // We want 64 - sum_zero, add because neg_sum_zero is already negated.
  const int32_t nzeros = static_cast<int32_t>(kDCTBlockSize) +
                         GetLane(SumOfLanes(di, neg_sum_zero));

  *nzeros_pos = nzeros;

  return nzeros;
}

// The number of nonzeros of each block is predicted from the top and the left
// blocks, with opportune scaling to take into account the number of blocks of
// each strategy.  The predicted number of nonzeros divided by two is used as a
// context; if this number is above 63, a specific context is used.  If the
// number of nonzeros of a strategy is above 63, it is written directly using a
// fixed number of bits (that depends on the size of the strategy).
Status TokenizeCoefficients(const coeff_order_t* JXL_RESTRICT orders,
                            const Rect& rect,
                            const int32_t* JXL_RESTRICT* JXL_RESTRICT ac_rows,
                            const AcStrategyImage& ac_strategy,
                            const YCbCrChromaSubsampling& cs,
                            Image3I* JXL_RESTRICT tmp_num_nzeroes,
                            std::vector<Token>* JXL_RESTRICT output,
                            const ImageB& qdc, const ImageI& qf,
                            const BlockCtxMap& block_ctx_map) {
  const size_t xsize_blocks = rect.xsize();
  const size_t ysize_blocks = rect.ysize();
  output->clear();
  // TODO(user): update the estimate: usually less coefficients are used.
  output->reserve(3 * xsize_blocks * ysize_blocks * kDCTBlockSize);

  size_t offset[3] = {};
  const size_t nzeros_stride = tmp_num_nzeroes->PixelsPerRow();
  for (size_t by = 0; by < ysize_blocks; ++by) {
    size_t sby[3] = {by >> cs.VShift(0), by >> cs.VShift(1),
                     by >> cs.VShift(2)};
    int32_t* JXL_RESTRICT row_nzeros[3] = {
        tmp_num_nzeroes->PlaneRow(0, sby[0]),
        tmp_num_nzeroes->PlaneRow(1, sby[1]),
        tmp_num_nzeroes->PlaneRow(2, sby[2]),
    };
    const int32_t* JXL_RESTRICT row_nzeros_top[3] = {
        sby[0] == 0 ? nullptr : tmp_num_nzeroes->ConstPlaneRow(0, sby[0] - 1),
        sby[1] == 0 ? nullptr : tmp_num_nzeroes->ConstPlaneRow(1, sby[1] - 1),
        sby[2] == 0 ? nullptr : tmp_num_nzeroes->ConstPlaneRow(2, sby[2] - 1),
    };
    const uint8_t* JXL_RESTRICT row_qdc =
        qdc.ConstRow(rect.y0() + by) + rect.x0();
    const int32_t* JXL_RESTRICT row_qf = rect.ConstRow(qf, by);
    AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      AcStrategy acs = acs_row[bx];
      if (!acs.IsFirstBlock()) continue;
      size_t sbx[3] = {bx >> cs.HShift(0), bx >> cs.HShift(1),
                       bx >> cs.HShift(2)};
      size_t cx = acs.covered_blocks_x();
      size_t cy = acs.covered_blocks_y();
      const size_t covered_blocks = cx * cy;  // = #LLF coefficients
      const size_t log2_covered_blocks =
          Num0BitsBelowLS1Bit_Nonzero(covered_blocks);
      const size_t size = covered_blocks * kDCTBlockSize;

      CoefficientLayout(&cy, &cx);  // swap cx/cy to canonical order

      for (int c : {1, 0, 2}) {
        if (sbx[c] << cs.HShift(c) != bx) continue;
        if (sby[c] << cs.VShift(c) != by) continue;
        const int32_t* JXL_RESTRICT block = ac_rows[c] + offset[c];

        int32_t nzeros =
            (covered_blocks == 1)
                ? NumNonZero8x8ExceptDC(block, row_nzeros[c] + sbx[c])
                : NumNonZeroExceptLLF(cx, cy, acs, covered_blocks,
                                      log2_covered_blocks, block, nzeros_stride,
                                      row_nzeros[c] + sbx[c]);

        int ord = kStrategyOrder[acs.RawStrategy()];
        const coeff_order_t* JXL_RESTRICT order =
            &orders[CoeffOrderOffset(ord, c)];

        int32_t predicted_nzeros =
            PredictFromTopAndLeft(row_nzeros_top[c], row_nzeros[c], sbx[c], 32);
        size_t block_ctx =
            block_ctx_map.Context(row_qdc[bx], row_qf[sbx[c]], ord, c);
        const int32_t nzero_ctx =
            block_ctx_map.NonZeroContext(predicted_nzeros, block_ctx);

        output->emplace_back(nzero_ctx, nzeros);
        const size_t histo_offset =
            block_ctx_map.ZeroDensityContextsOffset(block_ctx);
        // Skip LLF.
        size_t prev = (nzeros > static_cast<ptrdiff_t>(size / 16) ? 0 : 1);
        for (size_t k = covered_blocks; k < size && nzeros != 0; ++k) {
          int32_t coeff = block[order[k]];
          size_t ctx =
              histo_offset + ZeroDensityContext(nzeros, k, covered_blocks,
                                                log2_covered_blocks, prev);
          uint32_t u_coeff = PackSigned(coeff);
          output->emplace_back(static_cast<uint32_t>(ctx), u_coeff);
          prev = (coeff != 0) ? 1 : 0;
          nzeros -= prev;
        }
        JXL_ENSURE(nzeros == 0);
        offset[c] += size;
      }
    }
  }
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(TokenizeCoefficients);
Status TokenizeCoefficients(const coeff_order_t* JXL_RESTRICT orders,
                            const Rect& rect,
                            const int32_t* JXL_RESTRICT* JXL_RESTRICT ac_rows,
                            const AcStrategyImage& ac_strategy,
                            const YCbCrChromaSubsampling& cs,
                            Image3I* JXL_RESTRICT tmp_num_nzeroes,
                            std::vector<Token>* JXL_RESTRICT output,
                            const ImageB& qdc, const ImageI& qf,
                            const BlockCtxMap& block_ctx_map) {
  return HWY_DYNAMIC_DISPATCH(TokenizeCoefficients)(
      orders, rect, ac_rows, ac_strategy, cs, tmp_num_nzeroes, output, qdc, qf,
      block_ctx_map);
}

}  // namespace jxl
#endif  // HWY_ONCE
