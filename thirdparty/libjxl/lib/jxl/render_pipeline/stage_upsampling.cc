// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_upsampling.h"

#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/status.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_upsampling.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/simd_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Clamp;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Min;
using hwy::HWY_NAMESPACE::MulAdd;

class UpsamplingStage : public RenderPipelineStage {
 public:
  explicit UpsamplingStage(const CustomTransformData& ups_factors, size_t c,
                           size_t shift)
      : RenderPipelineStage(RenderPipelineStage::Settings::Symmetric(
            /*shift=*/shift, /*border=*/2)),
        c_(c) {
    const float* weights = shift == 1   ? ups_factors.upsampling2_weights
                           : shift == 2 ? ups_factors.upsampling4_weights
                                        : ups_factors.upsampling8_weights;
    size_t N = 1 << (shift - 1);
    for (size_t i = 0; i < 5 * N; i++) {
      for (size_t j = 0; j < 5 * N; j++) {
        size_t y = std::min(i, j);
        size_t x = std::max(i, j);
        kernel_[j / 5][i / 5][j % 5][i % 5] =
            weights[5 * N * y - y * (y - 1) / 2 + x - y];
      }
    }
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    static HWY_FULL(float) df;
    size_t shift = settings_.shift_x;
    size_t N = 1 << shift;
    const size_t xsize_v = RoundUpTo(xsize, Lanes(df));
    for (ssize_t iy = -2; iy <= 2; iy++) {
      msan::UnpoisonMemory(GetInputRow(input_rows, c_, iy) + xsize + 2,
                           sizeof(float) * (xsize_v - xsize));
    }
    JXL_ENSURE(xextra == 0);
    ssize_t x0 = 0;
    ssize_t x1 = xsize;
    if (N == 2) {
      ProcessRowImpl<2>(input_rows, output_rows, x0, x1);
    }
    if (N == 4) {
      ProcessRowImpl<4>(input_rows, output_rows, x0, x1);
    }
    if (N == 8) {
      ProcessRowImpl<8>(input_rows, output_rows, x0, x1);
    }
    for (size_t oy = 0; oy < N; oy++) {
      float* dst_row = GetOutputRow(output_rows, c_, oy);
      msan::PoisonMemory(dst_row + xsize * N,
                         sizeof(float) * (xsize_v - xsize) * N);
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c == c_ ? RenderPipelineChannelMode::kInOut
                   : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "Upsample"; }

 private:
  template <size_t N>
  JXL_INLINE float Kernel(size_t x, size_t y, ssize_t ix, ssize_t iy) const {
    static_assert(N == 2 || N == 4 || N == 8, "N must be 2, 4, or 8");
    ix += 2;
    iy += 2;
    if (N == 2) {
      return kernel_[0][0][y % 2 ? 4 - iy : iy][x % 2 ? 4 - ix : ix];
    }
    if (N == 4) {
      return kernel_[y % 4 < 2 ? y % 2 : 1 - y % 2]
                    [x % 4 < 2 ? x % 2 : 1 - x % 2][y % 4 < 2 ? iy : 4 - iy]
                    [x % 4 < 2 ? ix : 4 - ix];
    }
    if (N == 8) {
      return kernel_[y % 8 < 4 ? y % 4 : 3 - y % 4]
                    [x % 8 < 4 ? x % 4 : 3 - x % 4][y % 8 < 4 ? iy : 4 - iy]
                    [x % 8 < 4 ? ix : 4 - ix];
    }
  }

  template <ssize_t N>
  void ProcessRowImpl(const RowInfo& input_rows, const RowInfo& output_rows,
                      ssize_t x0, ssize_t x1) const {
    static HWY_FULL(float) df;
    using V = hwy::HWY_NAMESPACE::Vec<HWY_FULL(float)>;
    V ups0, ups1, ups2, ups3, ups4, ups5, ups6, ups7;  // NOLINT
    (void)ups2, (void)ups3, (void)ups4, (void)ups5, (void)ups6, (void)ups7;
    // Once we have C++17 available, change this back to `V* ups[N]` and
    // initialize using `if constexpr` below.
    V* ups[8] = {};
    static_assert(N == 2 || N == 4 || N == 8, "N must be 2, 4, or 8");
    if (N >= 2) {
      ups[0] = &ups0;
      ups[1] = &ups1;
    }
    if (N >= 4) {
      ups[2] = &ups2;
      ups[3] = &ups3;
    }
    if (N == 8) {
      ups[4] = &ups4;
      ups[5] = &ups5;
      ups[6] = &ups6;
      ups[7] = &ups7;
    }

    for (size_t oy = 0; oy < N; oy++) {
      float* dst_row = GetOutputRow(output_rows, c_, oy);
      for (ssize_t x = x0; x < x1; x += Lanes(df)) {
        for (size_t ox = 0; ox < N; ox++) {
          auto result = Zero(df);
          auto min = LoadU(df, GetInputRow(input_rows, c_, 0) + x);
          auto max = min;
          for (ssize_t iy = -2; iy <= 2; iy++) {
            for (ssize_t ix = -2; ix <= 2; ix++) {
              auto v = LoadU(df, GetInputRow(input_rows, c_, iy) + x + ix);
              result = MulAdd(Set(df, Kernel<N>(ox, oy, ix, iy)), v, result);
              min = Min(v, min);
              max = Max(v, max);
            }
          }
          // Avoid overshooting.
          *ups[ox] = Clamp(result, min, max);
        }
        if (N == 2) {
          StoreInterleaved(df, ups0, ups1, dst_row + x * N);
        }
        if (N == 4) {
          StoreInterleaved(df, ups0, ups1, ups2, ups3, dst_row + x * N);
        }
        if (N == 8) {
          StoreInterleaved(df, ups0, ups1, ups2, ups3, ups4, ups5, ups6, ups7,
                           dst_row + x * N);
        }
      }
    }
  }

  size_t c_;
  float kernel_[4][4][5][5];
};

std::unique_ptr<RenderPipelineStage> GetUpsamplingStage(
    const CustomTransformData& ups_factors, size_t c, size_t shift) {
  return jxl::make_unique<UpsamplingStage>(ups_factors, c, shift);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(GetUpsamplingStage);

std::unique_ptr<RenderPipelineStage> GetUpsamplingStage(
    const CustomTransformData& ups_factors, size_t c, size_t shift) {
  if ((shift < 1) || (shift > 3)) {
    JXL_DEBUG_ABORT("internal: (shift != 0) && (shift <= 3)");
    return nullptr;
  }
  return HWY_DYNAMIC_DISPATCH(GetUpsamplingStage)(ups_factors, c, shift);
}

}  // namespace jxl
#endif
