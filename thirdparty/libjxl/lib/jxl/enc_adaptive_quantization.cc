// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_adaptive_quantization.h"

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lib/jxl/cms/opsin_params.h"
#include "lib/jxl/common.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/render_pipeline/render_pipeline.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_adaptive_quantization.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/fast_math-inl.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_group.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_debug_image.h"
#include "lib/jxl/enc_group.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/epf.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/quant_weights.h"

// Set JXL_DEBUG_ADAPTIVE_QUANTIZATION to 1 to enable debugging.
#ifndef JXL_DEBUG_ADAPTIVE_QUANTIZATION
#define JXL_DEBUG_ADAPTIVE_QUANTIZATION 0
#endif

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::AbsDiff;
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::And;
using hwy::HWY_NAMESPACE::Gt;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Min;
using hwy::HWY_NAMESPACE::Rebind;
using hwy::HWY_NAMESPACE::Sqrt;
using hwy::HWY_NAMESPACE::ZeroIfNegative;

// The following functions modulate an exponent (out_val) and return the updated
// value. Their descriptor is limited to 8 lanes for 8x8 blocks.

// Hack for mask estimation. Eventually replace this code with butteraugli's
// masking.
float ComputeMaskForAcStrategyUse(const float out_val) {
  const float kMul = 1.0f;
  const float kOffset = 0.001f;
  return kMul / (out_val + kOffset);
}

template <class D, class V>
V ComputeMask(const D d, const V out_val) {
  const auto kBase = Set(d, -0.7647f);
  const auto kMul4 = Set(d, 9.4708735624378946f);
  const auto kMul2 = Set(d, 17.35036561631863f);
  const auto kOffset2 = Set(d, 302.59587815579727f);
  const auto kMul3 = Set(d, 6.7943250517376494f);
  const auto kOffset3 = Set(d, 3.7179635626140772f);
  const auto kOffset4 = Mul(Set(d, 0.25f), kOffset3);
  const auto kMul0 = Set(d, 0.80061762862741759f);
  const auto k1 = Set(d, 1.0f);

  // Avoid division by zero.
  const auto v1 = Max(Mul(out_val, kMul0), Set(d, 1e-3f));
  const auto v2 = Div(k1, Add(v1, kOffset2));
  const auto v3 = Div(k1, MulAdd(v1, v1, kOffset3));
  const auto v4 = Div(k1, MulAdd(v1, v1, kOffset4));
  // TODO(jyrki):
  // A log or two here could make sense. In butteraugli we have effectively
  // log(log(x + C)) for this kind of use, as a single log is used in
  // saturating visual masking and here the modulation values are exponential,
  // another log would counter that.
  return Add(kBase, MulAdd(kMul4, v4, MulAdd(kMul2, v2, Mul(kMul3, v3))));
}

// mul and mul2 represent a scaling difference between jxl and butteraugli.
const float kSGmul = 226.77216153508914f;
const float kSGmul2 = 1.0f / 73.377132366608819f;
// Includes correction factor for std::log -> log2.
const float kSGRetMul = kSGmul2 * 18.6580932135f * kInvLog2e;
const float kSGVOffset = 7.7825991679894591f;

template <bool invert, typename D, typename V>
V RatioOfDerivativesOfCubicRootToSimpleGamma(const D d, V v) {
  // The opsin space in jxl is the cubic root of photons, i.e., v * v * v
  // is related to the number of photons.
  //
  // SimpleGamma(v * v * v) is the psychovisual space in butteraugli.
  // This ratio allows quantization to move from jxl's opsin space to
  // butteraugli's log-gamma space.
  float kEpsilon = 1e-2;
  v = ZeroIfNegative(v);
  const auto kNumMul = Set(d, kSGRetMul * 3 * kSGmul);
  const auto kVOffset = Set(d, kSGVOffset * kInvLog2e + kEpsilon);
  const auto kDenMul = Set(d, kInvLog2e * kSGmul);

  const auto v2 = Mul(v, v);

  const auto num = MulAdd(kNumMul, v2, Set(d, kEpsilon));
  const auto den = MulAdd(Mul(kDenMul, v), v2, kVOffset);
  return invert ? Div(num, den) : Div(den, num);
}

template <bool invert = false>
float RatioOfDerivativesOfCubicRootToSimpleGamma(float v) {
  using DScalar = HWY_CAPPED(float, 1);
  auto vscalar = Load(DScalar(), &v);
  return GetLane(
      RatioOfDerivativesOfCubicRootToSimpleGamma<invert>(DScalar(), vscalar));
}

// TODO(veluca): this function computes an approximation of the derivative of
// SimpleGamma with (f(x+eps)-f(x))/eps. Consider two-sided approximation or
// exact derivatives. For reference, SimpleGamma was:
/*
template <typename D, typename V>
V SimpleGamma(const D d, V v) {
  // A simple HDR compatible gamma function.
  const auto mul = Set(d, kSGmul);
  const auto kRetMul = Set(d, kSGRetMul);
  const auto kRetAdd = Set(d, kSGmul2 * -20.2789020414f);
  const auto kVOffset = Set(d, kSGVOffset);

  v *= mul;

  // This should happen rarely, but may lead to a NaN, which is rather
  // undesirable. Since negative photons don't exist we solve the NaNs by
  // clamping here.
  // TODO(veluca): with FastLog2f, this no longer leads to NaNs.
  v = ZeroIfNegative(v);
  return kRetMul * FastLog2f(d, v + kVOffset) + kRetAdd;
}
*/

template <class D, class V>
V GammaModulation(const D d, const size_t x, const size_t y,
                  const ImageF& xyb_x, const ImageF& xyb_y, const Rect& rect,
                  const V out_val) {
  const float kBias = 0.16f;
  JXL_DASSERT(kBias > jxl::cms::kOpsinAbsorbanceBias[0]);
  JXL_DASSERT(kBias > jxl::cms::kOpsinAbsorbanceBias[1]);
  JXL_DASSERT(kBias > jxl::cms::kOpsinAbsorbanceBias[2]);
  auto overall_ratio = Zero(d);
  auto bias = Set(d, kBias);
  for (size_t dy = 0; dy < 8; ++dy) {
    const float* const JXL_RESTRICT row_in_x = rect.ConstRow(xyb_x, y + dy);
    const float* const JXL_RESTRICT row_in_y = rect.ConstRow(xyb_y, y + dy);
    for (size_t dx = 0; dx < 8; dx += Lanes(d)) {
      const auto iny = Add(Load(d, row_in_y + x + dx), bias);
      const auto inx = Load(d, row_in_x + x + dx);

      const auto r = Sub(iny, inx);
      const auto ratio_r =
          RatioOfDerivativesOfCubicRootToSimpleGamma</*invert=*/true>(d, r);
      overall_ratio = Add(overall_ratio, ratio_r);

      const auto g = Add(iny, inx);
      const auto ratio_g =
          RatioOfDerivativesOfCubicRootToSimpleGamma</*invert=*/true>(d, g);
      overall_ratio = Add(overall_ratio, ratio_g);
    }
  }
  overall_ratio = Mul(SumOfLanes(d, overall_ratio), Set(d, 0.5f / 64));
  // ideally -1.0, but likely optimal correction adds some entropy, so slightly
  // less than that.
  const auto kGamma = Set(d, 0.1005613337192697f);
  return MulAdd(kGamma, FastLog2f(d, overall_ratio), out_val);
}

// Change precision in 8x8 blocks that have significant amounts of blue
// content (but are not close to solid blue).
// This is based on the idea that M and L cone activations saturate the
// S (blue) receptors, and the S reception becomes more important when
// both M and L levels are low. In that case M and L receptors may be
// observing S-spectra instead and viewing them with higher spatial
// accuracy, justifying spending more bits here.
template <class D, class V>
V BlueModulation(const D d, const size_t x, const size_t y,
                 const ImageF& planex, const ImageF& planey,
                 const ImageF& planeb, const Rect& rect, const V out_val) {
  auto sum = Zero(d);
  static const float kLimit = 0.010474084867598155;
  static const float kOffset = 0.0031994768654636393;
  for (size_t dy = 0; dy < 8; ++dy) {
    const float* JXL_RESTRICT row_in_x = rect.ConstRow(planex, y + dy) + x;
    const float* JXL_RESTRICT row_in_y = rect.ConstRow(planey, y + dy) + x;
    const float* JXL_RESTRICT row_in_b = rect.ConstRow(planeb, y + dy) + x;
    for (size_t dx = 0; dx < 8; dx += Lanes(d)) {
      const auto p_x = Load(d, row_in_x + dx);
      const auto p_b = Load(d, row_in_b + dx);
      const auto p_y_raw = Add(Load(d, row_in_y + dx), Set(d, kOffset));
      const auto p_y_effective = Add(p_y_raw, Abs(p_x));
      sum = Add(sum,
                IfThenElseZero(Gt(p_b, p_y_effective),
                               Min(Sub(p_b, p_y_effective), Set(d, kLimit))));
    }
  }
  static const float kMul = 0.90590804735610064;
  sum = SumOfLanes(d, sum);
  float scalar_sum = GetLane(sum);
  // If it is all blue, don't boost the quantization.
  // All blue likely means low frequency blue. Let's not make the most
  // perfect sky ever.
  if (scalar_sum >= 32 * kLimit) {
    scalar_sum = 64 * kLimit - scalar_sum;
  }
  static const float kMaxLimit = 15.463398341612438;
  if (scalar_sum >= kMaxLimit * kLimit) {
    scalar_sum = kMaxLimit * kLimit;
  }
  scalar_sum *= kMul;
  return Add(Set(d, scalar_sum), out_val);
}

// Change precision in 8x8 blocks that have high frequency content.
template <class D, class V>
V HfModulation(const D d, const size_t x, const size_t y, const ImageF& xyb_y,
               const Rect& rect, const V out_val) {
  // Zero out the invalid differences for the rightmost value per row.
  const Rebind<uint32_t, D> du;
  HWY_ALIGN constexpr uint32_t kMaskRight[kBlockDim] = {~0u, ~0u, ~0u, ~0u,
                                                        ~0u, ~0u, ~0u, 0};

  // Sums of deltas of y and x components between (approximate)
  // 4-connected pixels.
  auto sum_y = Zero(d);
  static const float valmin_y = 0.0206;
  auto valminv_y = Set(d, valmin_y);
  for (size_t dy = 0; dy < 8; ++dy) {
    const float* JXL_RESTRICT row_in_y = rect.ConstRow(xyb_y, y + dy) + x;
    const float* JXL_RESTRICT row_in_y_next =
        dy == 7 ? row_in_y : rect.ConstRow(xyb_y, y + dy + 1) + x;

    // In SCALAR, there is no guarantee of having extra row padding.
    // Hence, we need to ensure we don't access pixels outside the row itself.
    // In SIMD modes, however, rows are padded, so it's safe to access one
    // garbage value after the row. The vector then gets masked with kMaskRight
    // to remove the influence of that value.
#if HWY_TARGET != HWY_SCALAR
    for (size_t dx = 0; dx < 8; dx += Lanes(d)) {
#else
    for (size_t dx = 0; dx < 7; dx += Lanes(d)) {
#endif
      const auto mask = BitCast(d, Load(du, kMaskRight + dx));
      {
        const auto p_y = Load(d, row_in_y + dx);
        const auto pr_y = LoadU(d, row_in_y + dx + 1);
        sum_y = Add(sum_y, And(mask, Min(valminv_y, AbsDiff(p_y, pr_y))));
        const auto pd_y = Load(d, row_in_y_next + dx);
        sum_y = Add(sum_y, Min(valminv_y, AbsDiff(p_y, pd_y)));
      }
    }
#if HWY_TARGET == HWY_SCALAR
    const auto p_y = Load(d, row_in_y + 7);
    const auto pd_y = Load(d, row_in_y_next + 7);
    sum_y = Add(sum_y, Min(valminv_y, AbsDiff(p_y, pd_y)));
#endif
  }
  static const float kMul_y = -0.38;
  sum_y = SumOfLanes(d, sum_y);

  float scalar_sum_y = GetLane(sum_y);
  scalar_sum_y *= kMul_y;

  // higher value -> more bpp
  float kOffset = 0.42;
  scalar_sum_y += kOffset;

  return Add(Set(d, scalar_sum_y), out_val);
}

void PerBlockModulations(const float butteraugli_target, const ImageF& xyb_x,
                         const ImageF& xyb_y, const ImageF& xyb_b,
                         const Rect& rect_in, const float scale,
                         const Rect& rect_out, ImageF* out) {
  float base_level = 0.48f * scale;
  float kDampenRampStart = 2.0f;
  float kDampenRampEnd = 14.0f;
  float dampen = 1.0f;
  if (butteraugli_target >= kDampenRampStart) {
    dampen = 1.0f - ((butteraugli_target - kDampenRampStart) /
                     (kDampenRampEnd - kDampenRampStart));
    if (dampen < 0) {
      dampen = 0;
    }
  }
  const float mul = scale * dampen;
  const float add = (1.0f - dampen) * base_level;
  for (size_t iy = rect_out.y0(); iy < rect_out.y1(); iy++) {
    const size_t y = iy * 8;
    float* const JXL_RESTRICT row_out = out->Row(iy);
    const HWY_CAPPED(float, kBlockDim) df;
    for (size_t ix = rect_out.x0(); ix < rect_out.x1(); ix++) {
      size_t x = ix * 8;
      auto mask_val = ComputeMask(df, Set(df, row_out[ix]));
      mask_val = GammaModulation(df, x, y, xyb_x, xyb_y, rect_in, mask_val);
      auto out_val = HfModulation(df, x, y, xyb_y, rect_in, mask_val);
      out_val = Min(out_val, BlueModulation(df, x, y, xyb_x, xyb_y, xyb_b,
                                            rect_in, mask_val));
      // We want multiplicative quantization field, so everything
      // until this point has been modulating the exponent.
      row_out[ix] = FastPow2f(GetLane(out_val) * 1.442695041f) * mul + add;
    }
  }
}

template <typename D, typename V>
V MaskingSqrt(const D d, V v) {
  static const float kLogOffset = 27.505837037000106f;
  static const float kMul = 211.66567973503678f;
  const auto mul_v = Set(d, kMul * 1e8);
  const auto offset_v = Set(d, kLogOffset);
  return Mul(Set(d, 0.25f), Sqrt(MulAdd(v, Sqrt(mul_v), offset_v)));
}

float MaskingSqrt(const float v) {
  using DScalar = HWY_CAPPED(float, 1);
  auto vscalar = Load(DScalar(), &v);
  return GetLane(MaskingSqrt(DScalar(), vscalar));
}

inline void StoreMin4(const float v, float& min0, float& min1, float& min2,
               float& min3) {
  if (v < min3) {
    if (v < min0) {
      min3 = min2;
      min2 = min1;
      min1 = min0;
      min0 = v;
    } else if (v < min1) {
      min3 = min2;
      min2 = min1;
      min1 = v;
    } else if (v < min2) {
      min3 = min2;
      min2 = v;
    } else {
      min3 = v;
    }
  }
}

// Look for smooth areas near the area of degradation.
// If the areas are generally smooth, don't do masking.
// Output is downsampled 2x.
Status FuzzyErosion(const float butteraugli_target, const Rect& from_rect,
                    const ImageF& from, const Rect& to_rect, ImageF* to) {
  const size_t xsize = from.xsize();
  const size_t ysize = from.ysize();
  constexpr int kStep = 1;
  static_assert(kStep == 1, "Step must be 1");
  JXL_ENSURE(to_rect.xsize() * 2 == from_rect.xsize());
  JXL_ENSURE(to_rect.ysize() * 2 == from_rect.ysize());
  static const float kMulBase[4] = { 0.125, 0.1, 0.09, 0.06 };
  static const float kMulAdd[4] = { 0.0, -0.1, -0.09, -0.06 };
  float mul = 0.0;
  if (butteraugli_target < 2.0f) {
    mul = (2.0f - butteraugli_target) * (1.0f / 2.0f);
  }
  float kMul[4] = { 0 };
  float norm_sum = 0.0;
  for (size_t ii = 0; ii < 4; ++ii) {
    kMul[ii] = kMulBase[ii] + mul * kMulAdd[ii];
    norm_sum += kMul[ii];
  }
  static const float kTotal = 0.29959705784054957;
  for (size_t ii = 0; ii < 4; ++ii) {
    kMul[ii] *= kTotal / norm_sum;
  }
  for (size_t fy = 0; fy < from_rect.ysize(); ++fy) {
    size_t y = fy + from_rect.y0();
    size_t ym1 = y >= kStep ? y - kStep : y;
    size_t yp1 = y + kStep < ysize ? y + kStep : y;
    const float* rowt = from.Row(ym1);
    const float* row = from.Row(y);
    const float* rowb = from.Row(yp1);
    float* row_out = to_rect.Row(to, fy / 2);
    for (size_t fx = 0; fx < from_rect.xsize(); ++fx) {
      size_t x = fx + from_rect.x0();
      size_t xm1 = x >= kStep ? x - kStep : x;
      size_t xp1 = x + kStep < xsize ? x + kStep : x;
      float min[4] = { row[x], row[xm1], row[xp1], rowt[xm1] };
      // Sort the first four values.
      if (min[0] > min[1]) std::swap(min[0], min[1]);
      if (min[0] > min[2]) std::swap(min[0], min[2]);
      if (min[0] > min[3]) std::swap(min[0], min[3]);
      if (min[1] > min[2]) std::swap(min[1], min[2]);
      if (min[1] > min[3]) std::swap(min[1], min[3]);
      if (min[2] > min[3]) std::swap(min[2], min[3]);
      // The remaining five values of a 3x3 neighbourhood.
      StoreMin4(rowt[x], min[0], min[1], min[2], min[3]);
      StoreMin4(rowt[xp1], min[0], min[1], min[2], min[3]);
      StoreMin4(rowb[xm1], min[0], min[1], min[2], min[3]);
      StoreMin4(rowb[x], min[0], min[1], min[2], min[3]);
      StoreMin4(rowb[xp1], min[0], min[1], min[2], min[3]);

      float v = kMul[0] * min[0] + kMul[1] * min[1] + kMul[2] * min[2] + kMul[3] * min[3];
      if (fx % 2 == 0 && fy % 2 == 0) {
        row_out[fx / 2] = v;
      } else {
        row_out[fx / 2] += v;
      }
    }
  }
  return true;
}

struct AdaptiveQuantizationImpl {
  Status PrepareBuffers(JxlMemoryManager* memory_manager, size_t num_threads) {
    JXL_ASSIGN_OR_RETURN(
        diff_buffer,
        ImageF::Create(memory_manager, kEncTileDim + 8, num_threads));
    for (size_t i = pre_erosion.size(); i < num_threads; i++) {
      JXL_ASSIGN_OR_RETURN(
          ImageF tmp,
          ImageF::Create(memory_manager, kEncTileDimInBlocks * 2 + 2,
                         kEncTileDimInBlocks * 2 + 2));
      pre_erosion.emplace_back(std::move(tmp));
    }
    return true;
  }

  Status ComputeTile(float butteraugli_target, float scale, const Image3F& xyb,
                     const Rect& rect_in, const Rect& rect_out,
                     const int thread, ImageF* mask, ImageF* mask1x1) {
    JXL_ENSURE(rect_in.x0() % kBlockDim == 0);
    JXL_ENSURE(rect_in.y0() % kBlockDim == 0);
    const size_t xsize = xyb.xsize();
    const size_t ysize = xyb.ysize();

    // The XYB gamma is 3.0 to be able to decode faster with two muls.
    // Butteraugli's gamma is matching the gamma of human eye, around 2.6.
    // We approximate the gamma difference by adding one cubic root into
    // the adaptive quantization. This gives us a total gamma of 2.6666
    // for quantization uses.
    const float match_gamma_offset = 0.019;

    const HWY_FULL(float) df;

    size_t y_start_1x1 = rect_in.y0() + rect_out.y0() * 8;
    size_t y_end_1x1 = y_start_1x1 + rect_out.ysize() * 8;

    size_t x_start_1x1 = rect_in.x0() + rect_out.x0() * 8;
    size_t x_end_1x1 = x_start_1x1 + rect_out.xsize() * 8;

    if (rect_in.x0() != 0 && rect_out.x0() == 0) x_start_1x1 -= 2;
    if (rect_in.x1() < xsize && rect_out.x1() * 8 == rect_in.xsize()) {
      x_end_1x1 += 2;
    }
    if (rect_in.y0() != 0 && rect_out.y0() == 0) y_start_1x1 -= 2;
    if (rect_in.y1() < ysize && rect_out.y1() * 8 == rect_in.ysize()) {
      y_end_1x1 += 2;
    }

    // Computes image (padded to multiple of 8x8) of local pixel differences.
    // Subsample both directions by 4.
    // 1x1 Laplacian of intensity.
    for (size_t y = y_start_1x1; y < y_end_1x1; ++y) {
      const size_t y2 = y + 1 < ysize ? y + 1 : y;
      const size_t y1 = y > 0 ? y - 1 : y;
      const float* row_in = xyb.ConstPlaneRow(1, y);
      const float* row_in1 = xyb.ConstPlaneRow(1, y1);
      const float* row_in2 = xyb.ConstPlaneRow(1, y2);
      float* mask1x1_out = mask1x1->Row(y);
      auto scalar_pixel1x1 = [&](size_t x) {
        const size_t x2 = x + 1 < xsize ? x + 1 : x;
        const size_t x1 = x > 0 ? x - 1 : x;
        const float base =
            0.25f * (row_in2[x] + row_in1[x] + row_in[x1] + row_in[x2]);
        const float gammac = RatioOfDerivativesOfCubicRootToSimpleGamma(
            row_in[x] + match_gamma_offset);
        float diff = std::abs(gammac * (row_in[x] - base));
        static const double kScaler = 1.0;
        diff *= kScaler;
        diff = std::log1p(diff);
        static const float kMul = 1.0;
        static const float kOffset = 0.01;
        mask1x1_out[x] = kMul / (diff + kOffset);
      };
      for (size_t x = x_start_1x1; x < x_end_1x1; ++x) {
        scalar_pixel1x1(x);
      }
    }

    size_t y_start = rect_in.y0() + rect_out.y0() * 8;
    size_t y_end = y_start + rect_out.ysize() * 8;

    size_t x_start = rect_in.x0() + rect_out.x0() * 8;
    size_t x_end = x_start + rect_out.xsize() * 8;

    if (x_start != 0) x_start -= 4;
    if (x_end != xsize) x_end += 4;
    if (y_start != 0) y_start -= 4;
    if (y_end != ysize) y_end += 4;
    JXL_RETURN_IF_ERROR(pre_erosion[thread].ShrinkTo((x_end - x_start) / 4,
                                                     (y_end - y_start) / 4));

    static const float limit = 0.2f;
    for (size_t y = y_start; y < y_end; ++y) {
      size_t y2 = y + 1 < ysize ? y + 1 : y;
      size_t y1 = y > 0 ? y - 1 : y;

      const float* row_in = xyb.ConstPlaneRow(1, y);
      const float* row_in1 = xyb.ConstPlaneRow(1, y1);
      const float* row_in2 = xyb.ConstPlaneRow(1, y2);
      float* JXL_RESTRICT row_out = diff_buffer.Row(thread);

      auto scalar_pixel = [&](size_t x) {
        const size_t x2 = x + 1 < xsize ? x + 1 : x;
        const size_t x1 = x > 0 ? x - 1 : x;
        const float base =
            0.25f * (row_in2[x] + row_in1[x] + row_in[x1] + row_in[x2]);
        const float gammac = RatioOfDerivativesOfCubicRootToSimpleGamma(
            row_in[x] + match_gamma_offset);
        float diff = gammac * (row_in[x] - base);
        diff *= diff;
        if (diff >= limit) {
          diff = limit;
        }
        diff = MaskingSqrt(diff);
        if ((y % 4) != 0) {
          row_out[x - x_start] += diff;
        } else {
          row_out[x - x_start] = diff;
        }
      };

      size_t x = x_start;
      // First pixel of the row.
      if (x_start == 0) {
        scalar_pixel(x_start);
        ++x;
      }
      // SIMD
      const auto match_gamma_offset_v = Set(df, match_gamma_offset);
      const auto quarter = Set(df, 0.25f);
      for (; x + 1 + Lanes(df) < x_end; x += Lanes(df)) {
        const auto in = LoadU(df, row_in + x);
        const auto in_r = LoadU(df, row_in + x + 1);
        const auto in_l = LoadU(df, row_in + x - 1);
        const auto in_t = LoadU(df, row_in2 + x);
        const auto in_b = LoadU(df, row_in1 + x);
        auto base = Mul(quarter, Add(Add(in_r, in_l), Add(in_t, in_b)));
        auto gammacv =
            RatioOfDerivativesOfCubicRootToSimpleGamma</*invert=*/false>(
                df, Add(in, match_gamma_offset_v));
        auto diff = Mul(gammacv, Sub(in, base));
        diff = Mul(diff, diff);
        diff = Min(diff, Set(df, limit));
        diff = MaskingSqrt(df, diff);
        if ((y & 3) != 0) {
          diff = Add(diff, LoadU(df, row_out + x - x_start));
        }
        StoreU(diff, df, row_out + x - x_start);
      }
      // Scalar
      for (; x < x_end; ++x) {
        scalar_pixel(x);
      }
      if (y % 4 == 3) {
        float* row_d_out = pre_erosion[thread].Row((y - y_start) / 4);
        for (size_t qx = 0; qx < (x_end - x_start) / 4; qx++) {
          row_d_out[qx] = (row_out[qx * 4] + row_out[qx * 4 + 1] +
                           row_out[qx * 4 + 2] + row_out[qx * 4 + 3]) *
                          0.25f;
        }
      }
    }
    JXL_ENSURE(x_start % (kBlockDim / 2) == 0);
    JXL_ENSURE(y_start % (kBlockDim / 2) == 0);
    Rect from_rect(x_start % 8 == 0 ? 0 : 1, y_start % 8 == 0 ? 0 : 1,
                   rect_out.xsize() * 2, rect_out.ysize() * 2);
    JXL_RETURN_IF_ERROR(FuzzyErosion(butteraugli_target, from_rect,
                                     pre_erosion[thread], rect_out, &aq_map));
    for (size_t y = 0; y < rect_out.ysize(); ++y) {
      const float* aq_map_row = rect_out.ConstRow(aq_map, y);
      float* mask_row = rect_out.Row(mask, y);
      for (size_t x = 0; x < rect_out.xsize(); ++x) {
        mask_row[x] = ComputeMaskForAcStrategyUse(aq_map_row[x]);
      }
    }
    PerBlockModulations(butteraugli_target, xyb.Plane(0), xyb.Plane(1),
                        xyb.Plane(2), rect_in, scale, rect_out, &aq_map);
    return true;
  }
  std::vector<ImageF> pre_erosion;
  ImageF aq_map;
  ImageF diff_buffer;
};

Status Blur1x1Masking(JxlMemoryManager* memory_manager, ThreadPool* pool,
                      ImageF* mask1x1, const Rect& rect) {
  // Blur the mask1x1 to obtain the masking image.
  // Before blurring it contains an image of absolute value of the
  // Laplacian of the intensity channel.
  static const float kFilterMask1x1[5] = {
    0.364911248, 0.05, 0.1688888021, 0.221069183, 0.306563504,
  };
  double sum =
      1.0 + 4 * (kFilterMask1x1[0] + kFilterMask1x1[1] + kFilterMask1x1[2] +
                 kFilterMask1x1[4] + 2 * kFilterMask1x1[3]);
  if (sum < 1e-5) {
    sum = 1e-5;
  }
  const float normalize = static_cast<float>(1.0 / sum);
  WeightsSymmetric5 weights =
      WeightsSymmetric5{{HWY_REP4(normalize)},
                        {HWY_REP4(normalize * kFilterMask1x1[0])},
                        {HWY_REP4(normalize * kFilterMask1x1[2])},
                        {HWY_REP4(normalize * kFilterMask1x1[1])},
                        {HWY_REP4(normalize * kFilterMask1x1[4])},
                        {HWY_REP4(normalize * kFilterMask1x1[3])}};
  JXL_ASSIGN_OR_RETURN(
      ImageF temp, ImageF::Create(memory_manager, rect.xsize(), rect.ysize()));
  JXL_RETURN_IF_ERROR(
      Symmetric5(*mask1x1, rect, weights, pool, &temp, Rect(temp)));
  *mask1x1 = std::move(temp);
  return true;
}

StatusOr<ImageF> AdaptiveQuantizationMap(const float butteraugli_target,
                                         const Image3F& xyb, const Rect& rect,
                                         float scale, ThreadPool* pool,
                                         ImageF* mask, ImageF* mask1x1) {
  JXL_ENSURE(rect.xsize() % kBlockDim == 0);
  JXL_ENSURE(rect.ysize() % kBlockDim == 0);
  AdaptiveQuantizationImpl impl;
  const size_t xsize_blocks = rect.xsize() / kBlockDim;
  const size_t ysize_blocks = rect.ysize() / kBlockDim;
  JxlMemoryManager* memory_manager = xyb.memory_manager();
  JXL_ASSIGN_OR_RETURN(
      impl.aq_map, ImageF::Create(memory_manager, xsize_blocks, ysize_blocks));
  JXL_ASSIGN_OR_RETURN(
      *mask, ImageF::Create(memory_manager, xsize_blocks, ysize_blocks));
  JXL_ASSIGN_OR_RETURN(
      *mask1x1, ImageF::Create(memory_manager, xyb.xsize(), xyb.ysize()));
  const auto prepare = [&](const size_t num_threads) -> Status {
    JXL_RETURN_IF_ERROR(impl.PrepareBuffers(memory_manager, num_threads));
    return true;
  };
  const auto process_tile = [&](const uint32_t tid,
                                const size_t thread) -> Status {
    size_t n_enc_tiles = DivCeil(xsize_blocks, kEncTileDimInBlocks);
    size_t tx = tid % n_enc_tiles;
    size_t ty = tid / n_enc_tiles;
    size_t by0 = ty * kEncTileDimInBlocks;
    size_t by1 = std::min((ty + 1) * kEncTileDimInBlocks, ysize_blocks);
    size_t bx0 = tx * kEncTileDimInBlocks;
    size_t bx1 = std::min((tx + 1) * kEncTileDimInBlocks, xsize_blocks);
    Rect rect_out(bx0, by0, bx1 - bx0, by1 - by0);
    JXL_RETURN_IF_ERROR(impl.ComputeTile(butteraugli_target, scale, xyb, rect,
                                         rect_out, thread, mask, mask1x1));
    return true;
  };
  size_t num_tiles = DivCeil(xsize_blocks, kEncTileDimInBlocks) *
                     DivCeil(ysize_blocks, kEncTileDimInBlocks);
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_tiles, prepare, process_tile,
                                "AQ DiffPrecompute"));

  JXL_RETURN_IF_ERROR(Blur1x1Masking(memory_manager, pool, mask1x1, rect));
  return std::move(impl).aq_map;
}

}  // namespace

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(AdaptiveQuantizationMap);

namespace {

// If true, prints the quantization maps at each iteration.
constexpr bool FLAGS_dump_quant_state = false;

Status DumpHeatmap(const CompressParams& cparams, const AuxOut* aux_out,
                   const std::string& label, const ImageF& image,
                   float good_threshold, float bad_threshold) {
  if (JXL_DEBUG_ADAPTIVE_QUANTIZATION) {
    JXL_ASSIGN_OR_RETURN(
        Image3F heatmap,
        CreateHeatMapImage(image, good_threshold, bad_threshold));
    char filename[200];
    snprintf(filename, sizeof(filename), "%s%05d", label.c_str(),
             aux_out->num_butteraugli_iters);
    JXL_RETURN_IF_ERROR(DumpImage(cparams, filename, heatmap));
  }
  return true;
}

Status DumpHeatmaps(const CompressParams& cparams, const AuxOut* aux_out,
                    float butteraugli_target, const ImageF& quant_field,
                    const ImageF& tile_heatmap, const ImageF& bt_diffmap) {
  if (JXL_DEBUG_ADAPTIVE_QUANTIZATION) {
    JxlMemoryManager* memory_manager = quant_field.memory_manager();
    if (!WantDebugOutput(cparams)) return true;
    JXL_ASSIGN_OR_RETURN(ImageF inv_qmap,
                         ImageF::Create(memory_manager, quant_field.xsize(),
                                        quant_field.ysize()));
    for (size_t y = 0; y < quant_field.ysize(); ++y) {
      const float* JXL_RESTRICT row_q = quant_field.ConstRow(y);
      float* JXL_RESTRICT row_inv_q = inv_qmap.Row(y);
      for (size_t x = 0; x < quant_field.xsize(); ++x) {
        row_inv_q[x] = 1.0f / row_q[x];  // never zero
      }
    }
    JXL_RETURN_IF_ERROR(DumpHeatmap(cparams, aux_out, "quant_heatmap", inv_qmap,
                                    4.0f * butteraugli_target,
                                    6.0f * butteraugli_target));
    JXL_RETURN_IF_ERROR(DumpHeatmap(cparams, aux_out, "tile_heatmap",
                                    tile_heatmap, butteraugli_target,
                                    1.5f * butteraugli_target));
    // matches heat maps produced by the command line tool.
    JXL_RETURN_IF_ERROR(DumpHeatmap(cparams, aux_out, "bt_diffmap", bt_diffmap,
                                    ButteraugliFuzzyInverse(1.5),
                                    ButteraugliFuzzyInverse(0.5)));
  }
  return true;
}

StatusOr<ImageF> TileDistMap(const ImageF& distmap, int tile_size, int margin,
                             const AcStrategyImage& ac_strategy) {
  const int tile_xsize = (distmap.xsize() + tile_size - 1) / tile_size;
  const int tile_ysize = (distmap.ysize() + tile_size - 1) / tile_size;
  JxlMemoryManager* memory_manager = distmap.memory_manager();
  JXL_ASSIGN_OR_RETURN(ImageF tile_distmap,
                       ImageF::Create(memory_manager, tile_xsize, tile_ysize));
  size_t distmap_stride = tile_distmap.PixelsPerRow();
  for (int tile_y = 0; tile_y < tile_ysize; ++tile_y) {
    AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(tile_y);
    float* JXL_RESTRICT dist_row = tile_distmap.Row(tile_y);
    for (int tile_x = 0; tile_x < tile_xsize; ++tile_x) {
      AcStrategy acs = ac_strategy_row[tile_x];
      if (!acs.IsFirstBlock()) continue;
      int this_tile_xsize = acs.covered_blocks_x() * tile_size;
      int this_tile_ysize = acs.covered_blocks_y() * tile_size;
      int y_begin = std::max<int>(0, tile_size * tile_y - margin);
      int y_end = std::min<int>(distmap.ysize(),
                                tile_size * tile_y + this_tile_ysize + margin);
      int x_begin = std::max<int>(0, tile_size * tile_x - margin);
      int x_end = std::min<int>(distmap.xsize(),
                                tile_size * tile_x + this_tile_xsize + margin);
      float dist_norm = 0.0;
      double pixels = 0;
      for (int y = y_begin; y < y_end; ++y) {
        float ymul = 1.0;
        constexpr float kBorderMul = 0.98f;
        constexpr float kCornerMul = 0.7f;
        if (margin != 0 && (y == y_begin || y == y_end - 1)) {
          ymul = kBorderMul;
        }
        const float* const JXL_RESTRICT row = distmap.Row(y);
        for (int x = x_begin; x < x_end; ++x) {
          float xmul = ymul;
          if (margin != 0 && (x == x_begin || x == x_end - 1)) {
            if (xmul == 1.0) {
              xmul = kBorderMul;
            } else {
              xmul = kCornerMul;
            }
          }
          float v = row[x];
          v *= v;
          v *= v;
          v *= v;
          v *= v;
          dist_norm += xmul * v;
          pixels += xmul;
        }
      }
      if (pixels == 0) pixels = 1;
      // 16th norm is less than the max norm, we reduce the difference
      // with this normalization factor.
      constexpr float kTileNorm = 1.2f;
      const float tile_dist =
          kTileNorm * std::pow(dist_norm / pixels, 1.0f / 16.0f);
      dist_row[tile_x] = tile_dist;
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          dist_row[tile_x + distmap_stride * iy + ix] = tile_dist;
        }
      }
    }
  }
  return tile_distmap;
}

const float kDcQuantPow = 0.83f;
const float kDcQuant = 1.095924047623553f;
const float kAcQuant = 0.765f;

// Computes the decoded image for a given set of compression parameters.
StatusOr<ImageBundle> RoundtripImage(const FrameHeader& frame_header,
                                     const Image3F& opsin,
                                     PassesEncoderState* enc_state,
                                     const JxlCmsInterface& cms,
                                     ThreadPool* pool) {
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  std::unique_ptr<PassesDecoderState> dec_state =
      jxl::make_unique<PassesDecoderState>(memory_manager);
  JXL_RETURN_IF_ERROR(dec_state->output_encoding_info.SetFromMetadata(
      *enc_state->shared.metadata));
  dec_state->shared = &enc_state->shared;
  JXL_ENSURE(opsin.ysize() % kBlockDim == 0);

  const size_t xsize_groups = DivCeil(opsin.xsize(), kGroupDim);
  const size_t ysize_groups = DivCeil(opsin.ysize(), kGroupDim);
  const size_t num_groups = xsize_groups * ysize_groups;

  size_t num_special_frames = enc_state->special_frames.size();
  size_t num_passes = enc_state->progressive_splitter.GetNumPasses();
  JXL_ASSIGN_OR_RETURN(auto modular_frame_encoder,
                       ModularFrameEncoder::Create(memory_manager, frame_header,
                                                   enc_state->cparams, false));
  JXL_RETURN_IF_ERROR(
      InitializePassesEncoder(frame_header, opsin, Rect(opsin), cms, pool,
                              enc_state, modular_frame_encoder.get(), nullptr));
  JXL_RETURN_IF_ERROR(dec_state->Init(frame_header));
  JXL_RETURN_IF_ERROR(dec_state->InitForAC(num_passes, pool));

  ImageBundle decoded(memory_manager, &enc_state->shared.metadata->m);
  decoded.origin = frame_header.frame_origin;
  JXL_ASSIGN_OR_RETURN(
      Image3F tmp,
      Image3F::Create(memory_manager, opsin.xsize(), opsin.ysize()));
  JXL_RETURN_IF_ERROR(decoded.SetFromImage(
      std::move(tmp), dec_state->output_encoding_info.color_encoding));

  PassesDecoderState::PipelineOptions options;
  options.use_slow_render_pipeline = false;
  options.coalescing = false;
  options.render_spotcolors = false;
  options.render_noise = false;

  // Same as frame_header.nonserialized_metadata->m
  const ImageMetadata& metadata = *decoded.metadata();

  JXL_RETURN_IF_ERROR(dec_state->PreparePipeline(
      frame_header, &enc_state->shared.metadata->m, &decoded, options));

  AlignedArray<GroupDecCache> group_dec_caches;
  const auto allocate_storage = [&](const size_t num_threads) -> Status {
    JXL_RETURN_IF_ERROR(
        dec_state->render_pipeline->PrepareForThreads(num_threads,
                                                      /*use_group_ids=*/false));
    JXL_ASSIGN_OR_RETURN(group_dec_caches, AlignedArray<GroupDecCache>::Create(
                                               memory_manager, num_threads));
    return true;
  };
  const auto process_group = [&](const uint32_t group_index,
                                 const size_t thread) -> Status {
    if (frame_header.loop_filter.epf_iters > 0) {
      JXL_RETURN_IF_ERROR(
          ComputeSigma(frame_header.loop_filter,
                       dec_state->shared->frame_dim.BlockGroupRect(group_index),
                       dec_state.get()));
    }
    RenderPipelineInput input =
        dec_state->render_pipeline->GetInputBuffers(group_index, thread);
    JXL_RETURN_IF_ERROR(DecodeGroupForRoundtrip(
        frame_header, enc_state->coeffs, group_index, dec_state.get(),
        &group_dec_caches[thread], thread, input, nullptr, nullptr));
    for (size_t c = 0; c < metadata.num_extra_channels; c++) {
      std::pair<ImageF*, Rect> ri = input.GetBuffer(3 + c);
      FillPlane(0.0f, ri.first, ri.second);
    }
    JXL_RETURN_IF_ERROR(input.Done());
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, num_groups, allocate_storage,
                                process_group, "AQ loop"));

  // Ensure we don't create any new special frames.
  enc_state->special_frames.resize(num_special_frames);

  return decoded;
}

constexpr int kDefaultButteraugliIters = 2;
constexpr int kMaxButteraugliIters = 4;

Status FindBestQuantization(const FrameHeader& frame_header,
                            const Image3F& linear, const Image3F& opsin,
                            ImageF& quant_field, PassesEncoderState* enc_state,
                            const JxlCmsInterface& cms, ThreadPool* pool,
                            AuxOut* aux_out) {
  const CompressParams& cparams = enc_state->cparams;
  if (cparams.resampling > 1 &&
      cparams.original_butteraugli_distance <= 4.0 * cparams.resampling) {
    // For downsampled opsin image, the butteraugli based adaptive quantization
    // loop would only make the size bigger without improving the distance much,
    // so in this case we enable it only for very high butteraugli targets.
    return true;
  }
  JxlMemoryManager* memory_manager = enc_state->memory_manager();
  Quantizer& quantizer = enc_state->shared.quantizer;
  ImageI& raw_quant_field = enc_state->shared.raw_quant_field;

  const float butteraugli_target = cparams.butteraugli_distance;
  const float original_butteraugli = cparams.original_butteraugli_distance;
  ButteraugliParams params;
  const auto& tf = frame_header.nonserialized_metadata->m.color_encoding.Tf();
  params.intensity_target =
      tf.IsPQ() || tf.IsHLG()
          ? frame_header.nonserialized_metadata->m.IntensityTarget()
          : 80.f;
  JxlButteraugliComparator comparator(params, cms);
  JXL_RETURN_IF_ERROR(comparator.SetLinearReferenceImage(linear));
  bool lower_is_better =
      (comparator.GoodQualityScore() < comparator.BadQualityScore());
  const float initial_quant_dc = InitialQuantDC(butteraugli_target);
  JXL_RETURN_IF_ERROR(AdjustQuantField(enc_state->shared.ac_strategy,
                                       Rect(quant_field), original_butteraugli,
                                       &quant_field));
  ImageF tile_distmap;
  JXL_ASSIGN_OR_RETURN(
      ImageF initial_quant_field,
      ImageF::Create(memory_manager, quant_field.xsize(), quant_field.ysize()));
  JXL_RETURN_IF_ERROR(CopyImageTo(quant_field, &initial_quant_field));

  float initial_qf_min;
  float initial_qf_max;
  ImageMinMax(initial_quant_field, &initial_qf_min, &initial_qf_max);
  float initial_qf_ratio = initial_qf_max / initial_qf_min;
  float qf_max_deviation_low = std::sqrt(250 / initial_qf_ratio);
  float asymmetry = 2;
  if (qf_max_deviation_low < asymmetry) asymmetry = qf_max_deviation_low;
  float qf_lower = initial_qf_min / (asymmetry * qf_max_deviation_low);
  float qf_higher = initial_qf_max * (qf_max_deviation_low / asymmetry);

  JXL_ENSURE(qf_higher / qf_lower < 253);

  constexpr int kOriginalComparisonRound = 1;
  int iters = kDefaultButteraugliIters;
  if (cparams.speed_tier <= SpeedTier::kTortoise) {
    iters = kMaxButteraugliIters;
  }
  for (int i = 0; i < iters + 1; ++i) {
    if (JXL_DEBUG_ADAPTIVE_QUANTIZATION) {
      printf("\nQuantization field:\n");
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          printf(" %.5f", quant_field.Row(y)[x]);
        }
        printf("\n");
      }
    }
    JXL_RETURN_IF_ERROR(quantizer.SetQuantField(initial_quant_dc, quant_field,
                                                &raw_quant_field));
    JXL_ASSIGN_OR_RETURN(
        ImageBundle dec_linear,
        RoundtripImage(frame_header, opsin, enc_state, cms, pool));
    float score;
    ImageF diffmap;
    JXL_RETURN_IF_ERROR(comparator.CompareWith(dec_linear, &diffmap, &score));
    if (!lower_is_better) {
      score = -score;
      ScaleImage(-1.0f, &diffmap);
    }
    JXL_ASSIGN_OR_RETURN(tile_distmap,
                         TileDistMap(diffmap, 8 * cparams.resampling, 0,
                                     enc_state->shared.ac_strategy));
    if (JXL_DEBUG_ADAPTIVE_QUANTIZATION && WantDebugOutput(cparams)) {
      JXL_RETURN_IF_ERROR(DumpImage(cparams, ("dec" + ToString(i)).c_str(),
                                    *dec_linear.color()));
      JXL_RETURN_IF_ERROR(DumpHeatmaps(cparams, aux_out, butteraugli_target,
                                       quant_field, tile_distmap, diffmap));
    }
    if (aux_out != nullptr) ++aux_out->num_butteraugli_iters;
    if (JXL_DEBUG_ADAPTIVE_QUANTIZATION) {
      float minval;
      float maxval;
      ImageMinMax(quant_field, &minval, &maxval);
      printf("\nButteraugli iter: %d/%d\n", i, iters);
      printf("Butteraugli distance: %f  (target = %f)\n", score,
             original_butteraugli);
      printf("quant range: %f ... %f  DC quant: %f\n", minval, maxval,
             initial_quant_dc);
      if (FLAGS_dump_quant_state) {
        quantizer.DumpQuantizationMap(raw_quant_field);
      }
    }

    if (i == iters) break;

    double kPow[8] = {
        0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    double kPowMod[8] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    if (i == kOriginalComparisonRound) {
      // Don't allow optimization to make the quant field a lot worse than
      // what the initial guess was. This allows the AC field to have enough
      // precision to reduce the oscillations due to the dc reconstruction.
      double kInitMul = 0.6;
      const double kOneMinusInitMul = 1.0 - kInitMul;
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        float* const JXL_RESTRICT row_q = quant_field.Row(y);
        const float* const JXL_RESTRICT row_init = initial_quant_field.Row(y);
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          double clamp = kOneMinusInitMul * row_q[x] + kInitMul * row_init[x];
          if (row_q[x] < clamp) {
            row_q[x] = clamp;
            if (row_q[x] > qf_higher) row_q[x] = qf_higher;
            if (row_q[x] < qf_lower) row_q[x] = qf_lower;
          }
        }
      }
    }

    double cur_pow = 0.0;
    if (i < 7) {
      cur_pow = kPow[i] + (original_butteraugli - 1.0) * kPowMod[i];
      if (cur_pow < 0) {
        cur_pow = 0;
      }
    }
    if (cur_pow == 0.0) {
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        const float* const JXL_RESTRICT row_dist = tile_distmap.Row(y);
        float* const JXL_RESTRICT row_q = quant_field.Row(y);
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          const float diff = row_dist[x] / original_butteraugli;
          if (diff > 1.0f) {
            float old = row_q[x];
            row_q[x] *= diff;
            int qf_old =
                static_cast<int>(std::lround(old * quantizer.InvGlobalScale()));
            int qf_new = static_cast<int>(
                std::lround(row_q[x] * quantizer.InvGlobalScale()));
            if (qf_old == qf_new) {
              row_q[x] = old + quantizer.Scale();
            }
          }
          if (row_q[x] > qf_higher) row_q[x] = qf_higher;
          if (row_q[x] < qf_lower) row_q[x] = qf_lower;
        }
      }
    } else {
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        const float* const JXL_RESTRICT row_dist = tile_distmap.Row(y);
        float* const JXL_RESTRICT row_q = quant_field.Row(y);
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          const float diff = row_dist[x] / original_butteraugli;
          if (diff <= 1.0f) {
            row_q[x] *= std::pow(diff, cur_pow);
          } else {
            float old = row_q[x];
            row_q[x] *= diff;
            int qf_old =
                static_cast<int>(std::lround(old * quantizer.InvGlobalScale()));
            int qf_new = static_cast<int>(
                std::lround(row_q[x] * quantizer.InvGlobalScale()));
            if (qf_old == qf_new) {
              row_q[x] = old + quantizer.Scale();
            }
          }
          if (row_q[x] > qf_higher) row_q[x] = qf_higher;
          if (row_q[x] < qf_lower) row_q[x] = qf_lower;
        }
      }
    }
  }
  JXL_RETURN_IF_ERROR(
      quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field));
  return true;
}

Status FindBestQuantizationMaxError(const FrameHeader& frame_header,
                                    const Image3F& opsin, ImageF& quant_field,
                                    PassesEncoderState* enc_state,
                                    const JxlCmsInterface& cms,
                                    ThreadPool* pool, AuxOut* aux_out) {
  // TODO(szabadka): Make this work for non-opsin color spaces.
  const CompressParams& cparams = enc_state->cparams;
  Quantizer& quantizer = enc_state->shared.quantizer;
  ImageI& raw_quant_field = enc_state->shared.raw_quant_field;

  // TODO(veluca): better choice of this value.
  const float initial_quant_dc =
      16 * std::sqrt(0.1f / cparams.butteraugli_distance);
  JXL_RETURN_IF_ERROR(
      AdjustQuantField(enc_state->shared.ac_strategy, Rect(quant_field),
                       cparams.original_butteraugli_distance, &quant_field));

  const float inv_max_err[3] = {1.0f / enc_state->cparams.max_error[0],
                                1.0f / enc_state->cparams.max_error[1],
                                1.0f / enc_state->cparams.max_error[2]};

  for (int i = 0; i < kMaxButteraugliIters + 1; ++i) {
    JXL_RETURN_IF_ERROR(quantizer.SetQuantField(initial_quant_dc, quant_field,
                                                &raw_quant_field));
    if (JXL_DEBUG_ADAPTIVE_QUANTIZATION && aux_out) {
      JXL_RETURN_IF_ERROR(
          DumpXybImage(cparams, ("ops" + ToString(i)).c_str(), opsin));
    }
    JXL_ASSIGN_OR_RETURN(
        ImageBundle decoded,
        RoundtripImage(frame_header, opsin, enc_state, cms, pool));
    if (JXL_DEBUG_ADAPTIVE_QUANTIZATION && aux_out) {
      JXL_RETURN_IF_ERROR(DumpXybImage(cparams, ("dec" + ToString(i)).c_str(),
                                       *decoded.color()));
    }
    for (size_t by = 0; by < enc_state->shared.frame_dim.ysize_blocks; by++) {
      AcStrategyRow ac_strategy_row =
          enc_state->shared.ac_strategy.ConstRow(by);
      for (size_t bx = 0; bx < enc_state->shared.frame_dim.xsize_blocks; bx++) {
        AcStrategy acs = ac_strategy_row[bx];
        if (!acs.IsFirstBlock()) continue;
        float max_error = 0;
        for (size_t c = 0; c < 3; c++) {
          for (size_t y = by * kBlockDim;
               y < (by + acs.covered_blocks_y()) * kBlockDim; y++) {
            if (y >= decoded.ysize()) continue;
            const float* JXL_RESTRICT in_row = opsin.ConstPlaneRow(c, y);
            const float* JXL_RESTRICT dec_row =
                decoded.color()->ConstPlaneRow(c, y);
            for (size_t x = bx * kBlockDim;
                 x < (bx + acs.covered_blocks_x()) * kBlockDim; x++) {
              if (x >= decoded.xsize()) continue;
              max_error = std::max(
                  std::abs(in_row[x] - dec_row[x]) * inv_max_err[c], max_error);
            }
          }
        }
        // Target an error between max_error/2 and max_error.
        // If the error in the varblock is above the target, increase the qf to
        // compensate. If the error is below the target, decrease the qf.
        // However, to avoid an excessive increase of the qf, only do so if the
        // error is less than half the maximum allowed error.
        const float qf_mul = (max_error < 0.5f)   ? max_error * 2.0f
                             : (max_error > 1.0f) ? max_error
                                                  : 1.0f;
        for (size_t qy = by; qy < by + acs.covered_blocks_y(); qy++) {
          float* JXL_RESTRICT quant_field_row = quant_field.Row(qy);
          for (size_t qx = bx; qx < bx + acs.covered_blocks_x(); qx++) {
            quant_field_row[qx] *= qf_mul;
          }
        }
      }
    }
  }
  JXL_RETURN_IF_ERROR(
      quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field));
  return true;
}

}  // namespace

Status AdjustQuantField(const AcStrategyImage& ac_strategy, const Rect& rect,
                        float butteraugli_target, ImageF* quant_field) {
  // Replace the whole quant_field in non-8x8 blocks with the maximum of each
  // 8x8 block.
  size_t stride = quant_field->PixelsPerRow();

  // At low distances it is great to use max, but mean works better
  // at high distances. We interpolate between them for a distance
  // range.
  float mean_max_mixer = 1.0f;
  {
    static const float kLimit = 1.54138f;
    static const float kMul = 0.56391f;
    static const float kMin = 0.0f;
    if (butteraugli_target > kLimit) {
      mean_max_mixer -= (butteraugli_target - kLimit) * kMul;
      if (mean_max_mixer < kMin) {
        mean_max_mixer = kMin;
      }
    }
  }
  for (size_t y = 0; y < rect.ysize(); ++y) {
    AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(rect, y);
    float* JXL_RESTRICT quant_row = rect.Row(quant_field, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      AcStrategy acs = ac_strategy_row[x];
      if (!acs.IsFirstBlock()) continue;
      JXL_ENSURE(x + acs.covered_blocks_x() <= quant_field->xsize());
      JXL_ENSURE(y + acs.covered_blocks_y() <= quant_field->ysize());
      float max = quant_row[x];
      float mean = 0.0;
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          mean += quant_row[x + ix + iy * stride];
          max = std::max(quant_row[x + ix + iy * stride], max);
        }
      }
      mean /= acs.covered_blocks_y() * acs.covered_blocks_x();
      if (acs.covered_blocks_y() * acs.covered_blocks_x() >= 4) {
        max *= mean_max_mixer;
        max += (1.0f - mean_max_mixer) * mean;
      }
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          quant_row[x + ix + iy * stride] = max;
        }
      }
    }
  }
  return true;
}

float InitialQuantDC(float butteraugli_target) {
  const float kDcMul = 0.3;  // Butteraugli target where non-linearity kicks in.
  const float butteraugli_target_dc = std::max<float>(
      0.5f * butteraugli_target,
      std::min<float>(butteraugli_target,
                      kDcMul * std::pow((1.0f / kDcMul) * butteraugli_target,
                                        kDcQuantPow)));
  // We want the maximum DC value to be at most 2**15 * kInvDCQuant / quant_dc.
  // The maximum DC value might not be in the kXybRange because of inverse
  // gaborish, so we add some slack to the maximum theoretical quant obtained
  // this way (64).
  return std::min(kDcQuant / butteraugli_target_dc, 50.f);
}

StatusOr<ImageF> InitialQuantField(const float butteraugli_target,
                                   const Image3F& opsin, const Rect& rect,
                                   ThreadPool* pool, float rescale,
                                   ImageF* mask, ImageF* mask1x1) {
  const float quant_ac = kAcQuant / butteraugli_target;
  return HWY_DYNAMIC_DISPATCH(AdaptiveQuantizationMap)(
      butteraugli_target, opsin, rect, quant_ac * rescale, pool, mask, mask1x1);
}

Status FindBestQuantizer(const FrameHeader& frame_header, const Image3F* linear,
                         const Image3F& opsin, ImageF& quant_field,
                         PassesEncoderState* enc_state,
                         const JxlCmsInterface& cms, ThreadPool* pool,
                         AuxOut* aux_out, double rescale) {
  const CompressParams& cparams = enc_state->cparams;
  if (cparams.max_error_mode) {
    JXL_RETURN_IF_ERROR(FindBestQuantizationMaxError(
        frame_header, opsin, quant_field, enc_state, cms, pool, aux_out));
  } else if (linear && cparams.speed_tier <= SpeedTier::kKitten) {
    // Normal encoding to a butteraugli score.
    JXL_RETURN_IF_ERROR(FindBestQuantization(frame_header, *linear, opsin,
                                             quant_field, enc_state, cms, pool,
                                             aux_out));
  }
  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
