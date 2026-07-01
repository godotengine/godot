// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/ac_strategy.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstring>
#include <utility>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"

namespace jxl {

// Tries to generalize zig-zag order to non-square blocks. Surprisingly, in
// square block frequency along the (i + j == const) diagonals is roughly the
// same. For historical reasons, consecutive diagonals are traversed
// in alternating directions - so called "zig-zag" (or "snake") order.
template <bool is_lut>
static void CoeffOrderAndLut(AcStrategy acs, coeff_order_t* out) {
  size_t cx = acs.covered_blocks_x();
  size_t cy = acs.covered_blocks_y();
  CoefficientLayout(&cy, &cx);

  // CoefficientLayout ensures cx >= cy.
  // We compute the zigzag order for a cx x cx block, then discard all the
  // lines that are not multiple of the ratio between cx and cy.
  size_t xs = cx / cy;
  size_t xsm = xs - 1;
  size_t xss = CeilLog2Nonzero(xs);
  // First half of the block
  size_t cur = cx * cy;
  for (size_t i = 0; i < cx * kBlockDim; i++) {
    for (size_t j = 0; j <= i; j++) {
      size_t x = j;
      size_t y = i - j;
      if (i % 2) std::swap(x, y);
      if ((y & xsm) != 0) continue;
      y >>= xss;
      size_t val = 0;
      if (x < cx && y < cy) {
        val = y * cx + x;
      } else {
        val = cur++;
      }
      if (is_lut) {
        out[y * cx * kBlockDim + x] = val;
      } else {
        out[val] = y * cx * kBlockDim + x;
      }
    }
  }
  // Second half
  for (size_t ip = cx * kBlockDim - 1; ip > 0; ip--) {
    size_t i = ip - 1;
    for (size_t j = 0; j <= i; j++) {
      size_t x = cx * kBlockDim - 1 - (i - j);
      size_t y = cx * kBlockDim - 1 - j;
      if (i % 2) std::swap(x, y);
      if ((y & xsm) != 0) continue;
      y >>= xss;
      size_t val = cur++;
      if (is_lut) {
        out[y * cx * kBlockDim + x] = val;
      } else {
        out[val] = y * cx * kBlockDim + x;
      }
    }
  }
}

void AcStrategy::ComputeNaturalCoeffOrder(coeff_order_t* order) const {
  CoeffOrderAndLut</*is_lut=*/false>(*this, order);
}
void AcStrategy::ComputeNaturalCoeffOrderLut(coeff_order_t* lut) const {
  CoeffOrderAndLut</*is_lut=*/true>(*this, lut);
}

#if JXL_CXX_LANG < JXL_CXX_17
constexpr size_t AcStrategy::kMaxCoeffBlocks;
constexpr size_t AcStrategy::kMaxBlockDim;
constexpr size_t AcStrategy::kMaxCoeffArea;
#endif

StatusOr<AcStrategyImage> AcStrategyImage::Create(
    JxlMemoryManager* memory_manager, size_t xsize, size_t ysize) {
  AcStrategyImage img;
  JXL_ASSIGN_OR_RETURN(img.layers_,
                       ImageB::Create(memory_manager, xsize, ysize));
  img.row_ = img.layers_.Row(0);
  img.stride_ = img.layers_.PixelsPerRow();
  return img;
}

size_t AcStrategyImage::CountBlocks(AcStrategyType type) const {
  size_t ret = 0;
  for (size_t y = 0; y < layers_.ysize(); y++) {
    const uint8_t* JXL_RESTRICT row = layers_.ConstRow(y);
    for (size_t x = 0; x < layers_.xsize(); x++) {
      if (row[x] == ((static_cast<uint8_t>(type) << 1) | 1)) ret++;
    }
  }
  return ret;
}

}  // namespace jxl
