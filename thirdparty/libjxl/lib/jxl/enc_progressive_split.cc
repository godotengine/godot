// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_progressive_split.h"

#include <string.h>

#include <cstdint>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/common.h"  // kMaxNumPasses
#include "lib/jxl/frame_dimensions.h"

namespace jxl {

template <typename T>
void ProgressiveSplitter::SplitACCoefficients(
    const T* JXL_RESTRICT block, const AcStrategy& acs, size_t bx, size_t by,
    T* JXL_RESTRICT output[kMaxNumPasses]) {
  size_t size = acs.covered_blocks_x() * acs.covered_blocks_y() * kDCTBlockSize;
  auto shift_right_round0 = [&](T v, int shift) {
    T one_if_negative = static_cast<uint32_t>(v) >> 31;
    T add = (one_if_negative << shift) - one_if_negative;
    return (v + add) >> shift;
  };
  // Early quit for the simple case of only one pass.
  if (mode_.num_passes == 1) {
    memcpy(output[0], block, sizeof(T) * size);
    return;
  }
  size_t ncoeffs_all_done_from_earlier_passes = 1;

  int previous_pass_shift = 0;
  for (size_t num_pass = 0; num_pass < mode_.num_passes; num_pass++) {  // pass
    // Zero out output block.
    memset(output[num_pass], 0, size * sizeof(T));
    const int pass_shift = mode_.passes[num_pass].shift;
    size_t frame_ncoeffs = mode_.passes[num_pass].num_coefficients;
    size_t xsize = acs.covered_blocks_x();
    size_t ysize = acs.covered_blocks_y();
    CoefficientLayout(&ysize, &xsize);
    for (size_t y = 0; y < ysize * frame_ncoeffs; y++) {    // superblk-y
      for (size_t x = 0; x < xsize * frame_ncoeffs; x++) {  // superblk-x
        size_t pos = y * xsize * kBlockDim + x;
        if (x < xsize * ncoeffs_all_done_from_earlier_passes &&
            y < ysize * ncoeffs_all_done_from_earlier_passes) {
          // This coefficient was already included in an earlier pass,
          // which included a genuinely smaller set of coefficients.
          continue;
        }
        T v = block[pos];
        // Previous pass discarded some bits: do not encode them again.
        if (previous_pass_shift != 0) {
          T previous_v = shift_right_round0(v, previous_pass_shift) *
                         (1 << previous_pass_shift);
          v -= previous_v;
        }
        output[num_pass][pos] = shift_right_round0(v, pass_shift);
      }  // superblk-x
    }  // superblk-y
    // We just finished a pass.
    // Hence, we are now guaranteed to have included all coeffs up to
    // frame_ncoeffs in every block, unless the current pass is shifted.
    if (mode_.passes[num_pass].shift == 0) {
      ncoeffs_all_done_from_earlier_passes = frame_ncoeffs;
    }
    previous_pass_shift = mode_.passes[num_pass].shift;
  }  // num_pass
}

template void ProgressiveSplitter::SplitACCoefficients<int32_t>(
    const int32_t* JXL_RESTRICT, const AcStrategy&, size_t, size_t,
    int32_t* JXL_RESTRICT[kMaxNumPasses]);

template void ProgressiveSplitter::SplitACCoefficients<int16_t>(
    const int16_t* JXL_RESTRICT, const AcStrategy&, size_t, size_t,
    int16_t* JXL_RESTRICT[kMaxNumPasses]);

}  // namespace jxl
