// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Edge-preserving smoothing: weighted average based on L1 patch similarity.

#include "lib/jxl/epf.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/loop_filter.h"
#include "lib/jxl/quantizer.h"

namespace jxl {

// Mirror n floats starting at *p and store them before p.
JXL_INLINE void LeftMirror(float* p, size_t n) {
  for (size_t i = 0; i < n; i++) {
    *(p - 1 - i) = p[i];
  }
}

// Mirror n floats starting at *(p - n) and store them at *p.
JXL_INLINE void RightMirror(float* p, size_t n) {
  for (size_t i = 0; i < n; i++) {
    p[i] = *(p - 1 - i);
  }
}

Status ComputeSigma(const LoopFilter& lf, const Rect& block_rect,
                    PassesDecoderState* state) {
  JXL_ENSURE(lf.epf_iters > 0);
  const AcStrategyImage& ac_strategy = state->shared->ac_strategy;
  const float quant_scale = state->shared->quantizer.Scale();

  const size_t sigma_stride = state->sigma.PixelsPerRow();
  const size_t sharpness_stride = state->shared->epf_sharpness.PixelsPerRow();

  for (size_t by = 0; by < block_rect.ysize(); ++by) {
    float* JXL_RESTRICT sigma_row = block_rect.Row(&state->sigma, by);
    const uint8_t* JXL_RESTRICT sharpness_row =
        block_rect.ConstRow(state->shared->epf_sharpness, by);
    AcStrategyRow acs_row = ac_strategy.ConstRow(block_rect, by);
    const int32_t* const JXL_RESTRICT row_quant =
        block_rect.ConstRow(state->shared->raw_quant_field, by);

    for (size_t bx = 0; bx < block_rect.xsize(); bx++) {
      AcStrategy acs = acs_row[bx];
      size_t llf_x = acs.covered_blocks_x();
      if (!acs.IsFirstBlock()) continue;
      // quant_scale is smaller for low quality.
      // quant_scale is roughly 0.08 / butteraugli score.
      //
      // row_quant is smaller for low quality.
      // row_quant is a quantization multiplier of form 1.0 /
      // row_quant[bx]
      //
      // lf.epf_quant_mul is a parameter in the format
      // kInvSigmaNum is a constant
      float sigma_quant =
          lf.epf_quant_mul / (quant_scale * row_quant[bx] * kInvSigmaNum);
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          float sigma =
              sigma_quant *
              lf.epf_sharp_lut[sharpness_row[bx + ix + iy * sharpness_stride]];
          // Avoid infinities.
          sigma = std::min(-1e-4f, sigma);  // TODO(veluca): remove this.
          sigma_row[bx + ix + kSigmaPadding +
                    (iy + kSigmaPadding) * sigma_stride] = 1.0f / sigma;
        }
      }
      // TODO(veluca): remove this padding.
      // Left padding with mirroring.
      if (bx + block_rect.x0() == 0) {
        for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
          LeftMirror(
              sigma_row + kSigmaPadding + (iy + kSigmaPadding) * sigma_stride,
              kSigmaBorder);
        }
      }
      // Right padding with mirroring.
      if (bx + block_rect.x0() + llf_x ==
          state->shared->frame_dim.xsize_blocks) {
        for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
          RightMirror(sigma_row + kSigmaPadding + bx + llf_x +
                          (iy + kSigmaPadding) * sigma_stride,
                      kSigmaBorder);
        }
      }
      // Offsets for row copying, in blocks.
      size_t offset_before = bx + block_rect.x0() == 0 ? 1 : bx + kSigmaPadding;
      size_t offset_after =
          bx + block_rect.x0() + llf_x == state->shared->frame_dim.xsize_blocks
              ? kSigmaPadding + llf_x + bx + kSigmaBorder
              : kSigmaPadding + llf_x + bx;
      size_t num = offset_after - offset_before;
      // Above
      if (by + block_rect.y0() == 0) {
        for (size_t iy = 0; iy < kSigmaBorder; iy++) {
          memcpy(
              sigma_row + offset_before +
                  (kSigmaPadding - 1 - iy) * sigma_stride,
              sigma_row + offset_before + (kSigmaPadding + iy) * sigma_stride,
              num * sizeof(*sigma_row));
        }
      }
      // Below
      if (by + block_rect.y0() + acs.covered_blocks_y() ==
          state->shared->frame_dim.ysize_blocks) {
        for (size_t iy = 0; iy < kSigmaBorder; iy++) {
          memcpy(
              sigma_row + offset_before +
                  sigma_stride * (acs.covered_blocks_y() + kSigmaPadding + iy),
              sigma_row + offset_before +
                  sigma_stride *
                      (acs.covered_blocks_y() + kSigmaPadding - 1 - iy),
              num * sizeof(*sigma_row));
        }
      }
    }
  }
  return true;
}

}  // namespace jxl
