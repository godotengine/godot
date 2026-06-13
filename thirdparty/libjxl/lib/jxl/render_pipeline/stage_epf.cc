// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_epf.h"

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"  // JXL_HIGH_PRECISION
#include "lib/jxl/epf.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_epf.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
// TODO(veluca): In principle, vectors could be not capped, if we want to deal
// with having two different sigma values in a single vector.
using DF = HWY_CAPPED(float, 8);

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::AbsDiff;
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Div;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Vec;
using hwy::HWY_NAMESPACE::VFromD;
using hwy::HWY_NAMESPACE::ZeroIfNegative;

JXL_INLINE Vec<DF> Weight(Vec<DF> sad, Vec<DF> inv_sigma, Vec<DF> thres) {
  auto v = MulAdd(sad, inv_sigma, Set(DF(), 1.0f));
  return ZeroIfNegative(v);
}

// 5x5 plus-shaped kernel with 5 SADs per pixel (3x3 plus-shaped). So this makes
// this filter a 7x7 filter.
class EPF0Stage : public RenderPipelineStage {
 public:
  EPF0Stage(LoopFilter lf, const ImageF& sigma)
      : RenderPipelineStage(RenderPipelineStage::Settings::Symmetric(
            /*shift=*/0, /*border=*/3)),
        lf_(std::move(lf)),
        sigma_(&sigma) {}

  template <bool aligned>
  JXL_INLINE void AddPixel(int row, float* JXL_RESTRICT rows[3][7], ssize_t x,
                           Vec<DF> sad, Vec<DF> inv_sigma,
                           Vec<DF>* JXL_RESTRICT X, Vec<DF>* JXL_RESTRICT Y,
                           Vec<DF>* JXL_RESTRICT B,
                           Vec<DF>* JXL_RESTRICT w) const {
    auto cx = aligned ? Load(DF(), rows[0][3 + row] + x)
                      : LoadU(DF(), rows[0][3 + row] + x);
    auto cy = aligned ? Load(DF(), rows[1][3 + row] + x)
                      : LoadU(DF(), rows[1][3 + row] + x);
    auto cb = aligned ? Load(DF(), rows[2][3 + row] + x)
                      : LoadU(DF(), rows[2][3 + row] + x);

    auto weight = Weight(sad, inv_sigma, Set(DF(), lf_.epf_pass1_zeroflush));
    *w = Add(*w, weight);
    *X = MulAdd(weight, cx, *X);
    *Y = MulAdd(weight, cy, *Y);
    *B = MulAdd(weight, cb, *B);
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    DF df;

    using V = decltype(Zero(df));
    V t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, tA, tB;  // NOLINT
    V* sads[12] = {&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, &tA, &tB};

    xextra = RoundUpTo(xextra, Lanes(df));
    const float* JXL_RESTRICT row_sigma =
        sigma_->Row(ypos / kBlockDim + kSigmaPadding);

    float sm = lf_.epf_pass0_sigma_scale * 1.65;
    float bsm = sm * lf_.epf_border_sad_mul;

    HWY_ALIGN float sad_mul_center[kBlockDim] = {bsm, sm, sm, sm,
                                                 sm,  sm, sm, bsm};
    HWY_ALIGN float sad_mul_border[kBlockDim] = {bsm, bsm, bsm, bsm,
                                                 bsm, bsm, bsm, bsm};
    float* JXL_RESTRICT rows[3][7];
    for (size_t c = 0; c < 3; c++) {
      for (int i = 0; i < 7; i++) {
        rows[c][i] = GetInputRow(input_rows, c, i - 3);
      }
    }

    const float* sad_mul =
        (ypos % kBlockDim == 0 || ypos % kBlockDim == kBlockDim - 1)
            ? sad_mul_border
            : sad_mul_center;

    for (ssize_t x = -xextra; x < static_cast<ssize_t>(xsize + xextra);
         x += Lanes(df)) {
      size_t bx = (x + xpos + kSigmaPadding * kBlockDim) / kBlockDim;
      size_t ix = (x + xpos) % kBlockDim;

      if (row_sigma[bx] < kMinSigma) {
        for (size_t c = 0; c < 3; c++) {
          auto px = Load(df, rows[c][3 + 0] + x);
          StoreU(px, df, GetOutputRow(output_rows, c, 0) + x);
        }
        continue;
      }

      const auto sm = Load(df, sad_mul + ix);
      const auto inv_sigma = Mul(Set(df, row_sigma[bx]), sm);

      for (auto& sad : sads) *sad = Zero(df);
      constexpr std::array<int, 2> sads_off[12] = {
          {{-2, 0}}, {{-1, -1}}, {{-1, 0}}, {{-1, 1}}, {{0, -2}}, {{0, -1}},
          {{0, 1}},  {{0, 2}},   {{1, -1}}, {{1, 0}},  {{1, 1}},  {{2, 0}},
      };

      // compute sads
      // TODO(veluca): consider unrolling and optimizing this.
      for (size_t c = 0; c < 3; c++) {
        auto scale = Set(df, lf_.epf_channel_scale[c]);
        for (size_t i = 0; i < 12; i++) {
          auto sad = Zero(df);
          constexpr std::array<int, 2> plus_off[] = {
              {{0, 0}}, {{-1, 0}}, {{0, -1}}, {{1, 0}}, {{0, 1}}};
          for (const auto& off : plus_off) {
            const auto r11 = LoadU(df, rows[c][3 + off[0]] + x + off[1]);
            const auto c11 = LoadU(df, rows[c][3 + sads_off[i][0] + off[0]] +
                                           x + sads_off[i][1] + off[1]);
            sad = Add(sad, AbsDiff(r11, c11));
          }
          *sads[i] = MulAdd(sad, scale, *sads[i]);
        }
      }
      const auto x_cc = Load(df, rows[0][3 + 0] + x);
      const auto y_cc = Load(df, rows[1][3 + 0] + x);
      const auto b_cc = Load(df, rows[2][3 + 0] + x);

      auto w = Set(df, 1);
      auto X = x_cc;
      auto Y = y_cc;
      auto B = b_cc;

      for (size_t i = 0; i < 12; i++) {
        AddPixel</*aligned=*/false>(/*row=*/sads_off[i][0], rows,
                                    x + sads_off[i][1], *sads[i], inv_sigma, &X,
                                    &Y, &B, &w);
      }
#if JXL_HIGH_PRECISION
      auto inv_w = Div(Set(df, 1.0f), w);
#else
      auto inv_w = ApproximateReciprocal(w);
#endif
      StoreU(Mul(X, inv_w), df, GetOutputRow(output_rows, 0, 0) + x);
      StoreU(Mul(Y, inv_w), df, GetOutputRow(output_rows, 1, 0) + x);
      StoreU(Mul(B, inv_w), df, GetOutputRow(output_rows, 2, 0) + x);
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInOut
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "EPF0"; }

 private:
  LoopFilter lf_;
  const ImageF* sigma_;
};

// 3x3 plus-shaped kernel with 5 SADs per pixel (also 3x3 plus-shaped). So this
// makes this filter a 5x5 filter.
class EPF1Stage : public RenderPipelineStage {
 public:
  EPF1Stage(LoopFilter lf, const ImageF& sigma)
      : RenderPipelineStage(RenderPipelineStage::Settings::Symmetric(
            /*shift=*/0, /*border=*/2)),
        lf_(std::move(lf)),
        sigma_(&sigma) {}

  template <bool aligned>
  JXL_INLINE void AddPixel(int row, float* JXL_RESTRICT rows[3][5], ssize_t x,
                           Vec<DF> sad, Vec<DF> inv_sigma,
                           Vec<DF>* JXL_RESTRICT X, Vec<DF>* JXL_RESTRICT Y,
                           Vec<DF>* JXL_RESTRICT B,
                           Vec<DF>* JXL_RESTRICT w) const {
    auto cx = aligned ? Load(DF(), rows[0][2 + row] + x)
                      : LoadU(DF(), rows[0][2 + row] + x);
    auto cy = aligned ? Load(DF(), rows[1][2 + row] + x)
                      : LoadU(DF(), rows[1][2 + row] + x);
    auto cb = aligned ? Load(DF(), rows[2][2 + row] + x)
                      : LoadU(DF(), rows[2][2 + row] + x);

    auto weight = Weight(sad, inv_sigma, Set(DF(), lf_.epf_pass1_zeroflush));
    *w = Add(*w, weight);
    *X = MulAdd(weight, cx, *X);
    *Y = MulAdd(weight, cy, *Y);
    *B = MulAdd(weight, cb, *B);
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    DF df;
    xextra = RoundUpTo(xextra, Lanes(df));
    const float* JXL_RESTRICT row_sigma =
        sigma_->Row(ypos / kBlockDim + kSigmaPadding);

    float sm = 1.65f;
    float bsm = sm * lf_.epf_border_sad_mul;

    HWY_ALIGN float sad_mul_center[kBlockDim] = {bsm, sm, sm, sm,
                                                 sm,  sm, sm, bsm};
    HWY_ALIGN float sad_mul_border[kBlockDim] = {bsm, bsm, bsm, bsm,
                                                 bsm, bsm, bsm, bsm};

    float* JXL_RESTRICT rows[3][5];
    for (size_t c = 0; c < 3; c++) {
      for (int i = 0; i < 5; i++) {
        rows[c][i] = GetInputRow(input_rows, c, i - 2);
      }
    }

    const float* sad_mul =
        (ypos % kBlockDim == 0 || ypos % kBlockDim == kBlockDim - 1)
            ? sad_mul_border
            : sad_mul_center;

    for (ssize_t x = -xextra; x < static_cast<ssize_t>(xsize + xextra);
         x += Lanes(df)) {
      size_t bx = (x + xpos + kSigmaPadding * kBlockDim) / kBlockDim;
      size_t ix = (x + xpos) % kBlockDim;

      if (row_sigma[bx] < kMinSigma) {
        for (size_t c = 0; c < 3; c++) {
          auto px = Load(df, rows[c][2 + 0] + x);
          Store(px, df, GetOutputRow(output_rows, c, 0) + x);
        }
        continue;
      }

      const auto sm = Load(df, sad_mul + ix);
      const auto inv_sigma = Mul(Set(df, row_sigma[bx]), sm);
      auto sad0 = Zero(df);
      auto sad1 = Zero(df);
      auto sad2 = Zero(df);
      auto sad3 = Zero(df);

      // compute sads
      for (size_t c = 0; c < 3; c++) {
        // center px = 22, px above = 21
        auto t = Undefined(df);

        const auto p20 = Load(df, rows[c][2 + -2] + x);
        const auto p21 = Load(df, rows[c][2 + -1] + x);
        auto sad0c = AbsDiff(p20, p21);  // SAD 2, 1

        const auto p11 = LoadU(df, rows[c][2 + -1] + x - 1);
        auto sad1c = AbsDiff(p11, p21);  // SAD 1, 2

        const auto p31 = LoadU(df, rows[c][2 + -1] + x + 1);
        auto sad2c = AbsDiff(p31, p21);  // SAD 3, 2

        const auto p02 = LoadU(df, rows[c][2 + 0] + x - 2);
        const auto p12 = LoadU(df, rows[c][2 + 0] + x - 1);
        sad1c = Add(sad1c, AbsDiff(p02, p12));  // SAD 1, 2
        sad0c = Add(sad0c, AbsDiff(p11, p12));  // SAD 2, 1

        const auto p22 = LoadU(df, rows[c][2 + 0] + x);
        t = AbsDiff(p12, p22);
        sad1c = Add(sad1c, t);  // SAD 1, 2
        sad2c = Add(sad2c, t);  // SAD 3, 2
        t = AbsDiff(p22, p21);
        auto sad3c = t;  // SAD 2, 3
        sad0c = Add(sad0c, t);  // SAD 2, 1

        const auto p32 = LoadU(df, rows[c][2 + 0] + x + 1);
        sad0c = Add(sad0c, AbsDiff(p31, p32));  // SAD 2, 1
        t = AbsDiff(p22, p32);
        sad1c = Add(sad1c, t);  // SAD 1, 2
        sad2c = Add(sad2c, t);  // SAD 3, 2

        const auto p42 = LoadU(df, rows[c][2 + 0] + x + 2);
        sad2c = Add(sad2c, AbsDiff(p42, p32));  // SAD 3, 2

        const auto p13 = LoadU(df, rows[c][2 + 1] + x - 1);
        sad3c = Add(sad3c, AbsDiff(p13, p12));  // SAD 2, 3

        const auto p23 = Load(df, rows[c][2 + 1] + x);
        t = AbsDiff(p22, p23);
        sad0c = Add(sad0c, t);                  // SAD 2, 1
        sad3c = Add(sad3c, t);                  // SAD 2, 3
        sad1c = Add(sad1c, AbsDiff(p13, p23));  // SAD 1, 2

        const auto p33 = LoadU(df, rows[c][2 + 1] + x + 1);
        sad2c = Add(sad2c, AbsDiff(p33, p23));  // SAD 3, 2
        sad3c = Add(sad3c, AbsDiff(p33, p32));  // SAD 2, 3

        const auto p24 = Load(df, rows[c][2 + 2] + x);
        sad3c = Add(sad3c, AbsDiff(p24, p23));  // SAD 2, 3

        auto scale = Set(df, lf_.epf_channel_scale[c]);
        sad0 = MulAdd(sad0c, scale, sad0);
        sad1 = MulAdd(sad1c, scale, sad1);
        sad2 = MulAdd(sad2c, scale, sad2);
        sad3 = MulAdd(sad3c, scale, sad3);
      }
      const auto x_cc = Load(df, rows[0][2 + 0] + x);
      const auto y_cc = Load(df, rows[1][2 + 0] + x);
      const auto b_cc = Load(df, rows[2][2 + 0] + x);

      auto w = Set(df, 1);
      auto X = x_cc;
      auto Y = y_cc;
      auto B = b_cc;

      // Top row
      AddPixel</*aligned=*/true>(/*row=*/-1, rows, x, sad0, inv_sigma, &X, &Y,
                                 &B, &w);
      // Center
      AddPixel</*aligned=*/false>(/*row=*/0, rows, x - 1, sad1, inv_sigma, &X,
                                  &Y, &B, &w);
      AddPixel</*aligned=*/false>(/*row=*/0, rows, x + 1, sad2, inv_sigma, &X,
                                  &Y, &B, &w);
      // Bottom
      AddPixel</*aligned=*/true>(/*row=*/1, rows, x, sad3, inv_sigma, &X, &Y,
                                 &B, &w);
#if JXL_HIGH_PRECISION
      auto inv_w = Div(Set(df, 1.0f), w);
#else
      auto inv_w = ApproximateReciprocal(w);
#endif
      Store(Mul(X, inv_w), df, GetOutputRow(output_rows, 0, 0) + x);
      Store(Mul(Y, inv_w), df, GetOutputRow(output_rows, 1, 0) + x);
      Store(Mul(B, inv_w), df, GetOutputRow(output_rows, 2, 0) + x);
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInOut
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "EPF1"; }

 private:
  LoopFilter lf_;
  const ImageF* sigma_;
};

// 3x3 plus-shaped kernel with 1 SAD per pixel. So this makes this filter a 3x3
// filter.
class EPF2Stage : public RenderPipelineStage {
 public:
  EPF2Stage(LoopFilter lf, const ImageF& sigma)
      : RenderPipelineStage(RenderPipelineStage::Settings::Symmetric(
            /*shift=*/0, /*border=*/1)),
        lf_(std::move(lf)),
        sigma_(&sigma) {}

  template <bool aligned>
  JXL_INLINE void AddPixel(int row, float* JXL_RESTRICT rows[3][3], ssize_t x,
                           Vec<DF> rx, Vec<DF> ry, Vec<DF> rb,
                           Vec<DF> inv_sigma, Vec<DF>* JXL_RESTRICT X,
                           Vec<DF>* JXL_RESTRICT Y, Vec<DF>* JXL_RESTRICT B,
                           Vec<DF>* JXL_RESTRICT w) const {
    auto cx = aligned ? Load(DF(), rows[0][1 + row] + x)
                      : LoadU(DF(), rows[0][1 + row] + x);
    auto cy = aligned ? Load(DF(), rows[1][1 + row] + x)
                      : LoadU(DF(), rows[1][1 + row] + x);
    auto cb = aligned ? Load(DF(), rows[2][1 + row] + x)
                      : LoadU(DF(), rows[2][1 + row] + x);

    auto sad = Mul(AbsDiff(cx, rx), Set(DF(), lf_.epf_channel_scale[0]));
    sad = MulAdd(AbsDiff(cy, ry), Set(DF(), lf_.epf_channel_scale[1]), sad);
    sad = MulAdd(AbsDiff(cb, rb), Set(DF(), lf_.epf_channel_scale[2]), sad);

    auto weight = Weight(sad, inv_sigma, Set(DF(), lf_.epf_pass2_zeroflush));

    *w = Add(*w, weight);
    *X = MulAdd(weight, cx, *X);
    *Y = MulAdd(weight, cy, *Y);
    *B = MulAdd(weight, cb, *B);
  }

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                    size_t thread_id) const final {
    DF df;
    xextra = RoundUpTo(xextra, Lanes(df));
    const float* JXL_RESTRICT row_sigma =
        sigma_->Row(ypos / kBlockDim + kSigmaPadding);

    float sm = lf_.epf_pass2_sigma_scale * 1.65;
    float bsm = sm * lf_.epf_border_sad_mul;

    HWY_ALIGN float sad_mul_center[kBlockDim] = {bsm, sm, sm, sm,
                                                 sm,  sm, sm, bsm};
    HWY_ALIGN float sad_mul_border[kBlockDim] = {bsm, bsm, bsm, bsm,
                                                 bsm, bsm, bsm, bsm};

    float* JXL_RESTRICT rows[3][3];
    for (size_t c = 0; c < 3; c++) {
      for (int i = 0; i < 3; i++) {
        rows[c][i] = GetInputRow(input_rows, c, i - 1);
      }
    }

    const float* sad_mul =
        (ypos % kBlockDim == 0 || ypos % kBlockDim == kBlockDim - 1)
            ? sad_mul_border
            : sad_mul_center;

    for (ssize_t x = -xextra; x < static_cast<ssize_t>(xsize + xextra);
         x += Lanes(df)) {
      size_t bx = (x + xpos + kSigmaPadding * kBlockDim) / kBlockDim;
      size_t ix = (x + xpos) % kBlockDim;

      if (row_sigma[bx] < kMinSigma) {
        for (size_t c = 0; c < 3; c++) {
          auto px = Load(df, rows[c][1 + 0] + x);
          Store(px, df, GetOutputRow(output_rows, c, 0) + x);
        }
        continue;
      }

      const auto sm = Load(df, sad_mul + ix);
      const auto inv_sigma = Mul(Set(df, row_sigma[bx]), sm);

      const auto x_cc = Load(df, rows[0][1 + 0] + x);
      const auto y_cc = Load(df, rows[1][1 + 0] + x);
      const auto b_cc = Load(df, rows[2][1 + 0] + x);

      auto w = Set(df, 1);
      auto X = x_cc;
      auto Y = y_cc;
      auto B = b_cc;

      // Top row
      AddPixel</*aligned=*/true>(/*row=*/-1, rows, x, x_cc, y_cc, b_cc,
                                 inv_sigma, &X, &Y, &B, &w);
      // Center
      AddPixel</*aligned=*/false>(/*row=*/0, rows, x - 1, x_cc, y_cc, b_cc,
                                  inv_sigma, &X, &Y, &B, &w);
      AddPixel</*aligned=*/false>(/*row=*/0, rows, x + 1, x_cc, y_cc, b_cc,
                                  inv_sigma, &X, &Y, &B, &w);
      // Bottom
      AddPixel</*aligned=*/true>(/*row=*/1, rows, x, x_cc, y_cc, b_cc,
                                 inv_sigma, &X, &Y, &B, &w);
#if JXL_HIGH_PRECISION
      auto inv_w = Div(Set(df, 1.0f), w);
#else
      auto inv_w = ApproximateReciprocal(w);
#endif
      Store(Mul(X, inv_w), df, GetOutputRow(output_rows, 0, 0) + x);
      Store(Mul(Y, inv_w), df, GetOutputRow(output_rows, 1, 0) + x);
      Store(Mul(B, inv_w), df, GetOutputRow(output_rows, 2, 0) + x);
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInOut
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "EPF2"; }

 private:
  LoopFilter lf_;
  const ImageF* sigma_;
};

std::unique_ptr<RenderPipelineStage> GetEPFStage0(const LoopFilter& lf,
                                                  const ImageF& sigma) {
  return jxl::make_unique<EPF0Stage>(lf, sigma);
}

std::unique_ptr<RenderPipelineStage> GetEPFStage1(const LoopFilter& lf,
                                                  const ImageF& sigma) {
  return jxl::make_unique<EPF1Stage>(lf, sigma);
}

std::unique_ptr<RenderPipelineStage> GetEPFStage2(const LoopFilter& lf,
                                                  const ImageF& sigma) {
  return jxl::make_unique<EPF2Stage>(lf, sigma);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(GetEPFStage0);
HWY_EXPORT(GetEPFStage1);
HWY_EXPORT(GetEPFStage2);

std::unique_ptr<RenderPipelineStage> GetEPFStage(const LoopFilter& lf,
                                                 const ImageF& sigma,
                                                 EpfStage epf_stage) {
  if (lf.epf_iters == 0) return nullptr;
  switch (epf_stage) {
    case EpfStage::Zero:
      return HWY_DYNAMIC_DISPATCH(GetEPFStage0)(lf, sigma);
    case EpfStage::One:
      return HWY_DYNAMIC_DISPATCH(GetEPFStage1)(lf, sigma);
    case EpfStage::Two:
      return HWY_DYNAMIC_DISPATCH(GetEPFStage2)(lf, sigma);
  }
  JXL_DEBUG_ABORT("internal: unexpected EpfStage: %d",
                  static_cast<int>(epf_stage));
  return nullptr;
}

}  // namespace jxl
#endif
