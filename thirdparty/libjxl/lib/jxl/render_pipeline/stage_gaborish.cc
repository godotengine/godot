// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_gaborish.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_gaborish.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;

class GaborishStage : public RenderPipelineStage {
 public:
  explicit GaborishStage(const LoopFilter& lf)
      : RenderPipelineStage(RenderPipelineStage::Settings::Symmetric(
            /*shift=*/0, /*border=*/1)) {
    weights_[0] = 1;
    weights_[1] = lf.gab_x_weight1;
    weights_[2] = lf.gab_x_weight2;
    weights_[3] = 1;
    weights_[4] = lf.gab_y_weight1;
    weights_[5] = lf.gab_y_weight2;
    weights_[6] = 1;
    weights_[7] = lf.gab_b_weight1;
    weights_[8] = lf.gab_b_weight2;
    // Normalize
    for (size_t c = 0; c < 3; c++) {
      const float div =
          weights_[3 * c] + 4 * (weights_[3 * c + 1] + weights_[3 * c + 2]);
      const float mul = 1.0f / div;
      weights_[3 * c] *= mul;
      weights_[3 * c + 1] *= mul;
      weights_[3 * c + 2] *= mul;
    }
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    const HWY_FULL(float) d;
    for (size_t c = 0; c < 3; c++) {
      float* JXL_RESTRICT row_t = GetInputRow(input_rows, c, -1);
      float* JXL_RESTRICT row_m = GetInputRow(input_rows, c, 0);
      float* JXL_RESTRICT row_b = GetInputRow(input_rows, c, 1);
      float* JXL_RESTRICT row_out = GetOutputRow(output_rows, c, 0);
      const auto w0 = Set(d, weights_[3 * c + 0]);
      const auto w1 = Set(d, weights_[3 * c + 1]);
      const auto w2 = Set(d, weights_[3 * c + 2]);
// Group data need only be aligned to a block; for >=512 bit vectors, this may
// result in unaligned loads.
#if HWY_CAP_GE512
#define LoadMaybeU LoadU
#else
#define LoadMaybeU Load
#endif
      // Since GetInputRow(input_rows, c, {-1, 0, 1}) is aligned, rounding
      // xextra up to Lanes(d) doesn't access anything problematic.
      for (ssize_t x = -RoundUpTo(xextra, Lanes(d));
           x < static_cast<ssize_t>(xsize + xextra); x += Lanes(d)) {
        const auto t = LoadMaybeU(d, row_t + x);
        const auto tl = LoadU(d, row_t + x - 1);
        const auto tr = LoadU(d, row_t + x + 1);
        const auto m = LoadMaybeU(d, row_m + x);
        const auto l = LoadU(d, row_m + x - 1);
        const auto r = LoadU(d, row_m + x + 1);
        const auto b = LoadMaybeU(d, row_b + x);
        const auto bl = LoadU(d, row_b + x - 1);
        const auto br = LoadU(d, row_b + x + 1);
        const auto sum0 = m;
        const auto sum1 = Add(Add(l, r), Add(t, b));
        const auto sum2 = Add(Add(tl, tr), Add(bl, br));
        auto pixels = MulAdd(sum2, w2, MulAdd(sum1, w1, Mul(sum0, w0)));
        Store(pixels, d, row_out + x);
      }
    }
    return true;
  }
#undef LoadMaybeU

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInOut
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "Gab"; }

 private:
  float weights_[9];
};

std::unique_ptr<RenderPipelineStage> GetGaborishStage(const LoopFilter& lf) {
  return jxl::make_unique<GaborishStage>(lf);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(GetGaborishStage);

std::unique_ptr<RenderPipelineStage> GetGaborishStage(const LoopFilter& lf) {
  if (lf.gab != 1) return nullptr;
  return HWY_DYNAMIC_DISPATCH(GetGaborishStage)(lf);
}

}  // namespace jxl
#endif
