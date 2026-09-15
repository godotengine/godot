// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/compressed_dc.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/compressed_dc.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

using D = HWY_FULL(float);
using DScalar = HWY_CAPPED(float, 1);

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Div;
using hwy::HWY_NAMESPACE::Max;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Rebind;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::Vec;
using hwy::HWY_NAMESPACE::ZeroIfNegative;

// TODO(veluca): optimize constants.
const float w1 = 0.20345139757231578f;
const float w2 = 0.0334829185968739f;
const float w0 = 1.0f - 4.0f * (w1 + w2);

template <class V>
V MaxWorkaround(V a, V b) {
#if (HWY_TARGET == HWY_AVX3) && HWY_COMPILER_CLANG <= 800
  // Prevents "Do not know how to split the result of this operator" error
  return IfThenElse(a > b, a, b);
#else
  return Max(a, b);
#endif
}

template <typename D>
JXL_INLINE void ComputePixelChannel(const D d, const float dc_factor,
                                    const float* JXL_RESTRICT row_top,
                                    const float* JXL_RESTRICT row,
                                    const float* JXL_RESTRICT row_bottom,
                                    Vec<D>* JXL_RESTRICT mc,
                                    Vec<D>* JXL_RESTRICT sm,
                                    Vec<D>* JXL_RESTRICT gap, size_t x) {
  const auto tl = LoadU(d, row_top + x - 1);
  const auto tc = Load(d, row_top + x);
  const auto tr = LoadU(d, row_top + x + 1);

  const auto ml = LoadU(d, row + x - 1);
  *mc = Load(d, row + x);
  const auto mr = LoadU(d, row + x + 1);

  const auto bl = LoadU(d, row_bottom + x - 1);
  const auto bc = Load(d, row_bottom + x);
  const auto br = LoadU(d, row_bottom + x + 1);

  const auto w_center = Set(d, w0);
  const auto w_side = Set(d, w1);
  const auto w_corner = Set(d, w2);

  const auto corner = Add(Add(tl, tr), Add(bl, br));
  const auto side = Add(Add(ml, mr), Add(tc, bc));
  *sm = MulAdd(corner, w_corner, MulAdd(side, w_side, Mul(*mc, w_center)));

  const auto dc_quant = Set(d, dc_factor);
  *gap = MaxWorkaround(*gap, Abs(Div(Sub(*mc, *sm), dc_quant)));
}

template <typename D>
JXL_INLINE void ComputePixel(
    const float* JXL_RESTRICT dc_factors,
    const float* JXL_RESTRICT* JXL_RESTRICT rows_top,
    const float* JXL_RESTRICT* JXL_RESTRICT rows,
    const float* JXL_RESTRICT* JXL_RESTRICT rows_bottom,
    float* JXL_RESTRICT* JXL_RESTRICT out_rows, size_t x) {
  const D d;
  auto mc_x = Undefined(d);
  auto mc_y = Undefined(d);
  auto mc_b = Undefined(d);
  auto sm_x = Undefined(d);
  auto sm_y = Undefined(d);
  auto sm_b = Undefined(d);
  auto gap = Set(d, 0.5f);
  ComputePixelChannel(d, dc_factors[0], rows_top[0], rows[0], rows_bottom[0],
                      &mc_x, &sm_x, &gap, x);
  ComputePixelChannel(d, dc_factors[1], rows_top[1], rows[1], rows_bottom[1],
                      &mc_y, &sm_y, &gap, x);
  ComputePixelChannel(d, dc_factors[2], rows_top[2], rows[2], rows_bottom[2],
                      &mc_b, &sm_b, &gap, x);
  auto factor = MulAdd(Set(d, -4.0f), gap, Set(d, 3.0f));
  factor = ZeroIfNegative(factor);

  auto out = MulAdd(Sub(sm_x, mc_x), factor, mc_x);
  Store(out, d, out_rows[0] + x);
  out = MulAdd(Sub(sm_y, mc_y), factor, mc_y);
  Store(out, d, out_rows[1] + x);
  out = MulAdd(Sub(sm_b, mc_b), factor, mc_b);
  Store(out, d, out_rows[2] + x);
}

Status AdaptiveDCSmoothing(JxlMemoryManager* memory_manager,
                           const float* dc_factors, Image3F* dc,
                           ThreadPool* pool) {
  const size_t xsize = dc->xsize();
  const size_t ysize = dc->ysize();
  if (ysize <= 2 || xsize <= 2) return true;

  // TODO(veluca): use tile-based processing?
  // TODO(veluca): decide if changes to the y channel should be propagated to
  // the x and b channels through color correlation.
  JXL_ENSURE(w1 + w2 < 0.25f);

  JXL_ASSIGN_OR_RETURN(Image3F smoothed,
                       Image3F::Create(memory_manager, xsize, ysize));
  // Fill in borders that the loop below will not. First and last are unused.
  for (size_t c = 0; c < 3; c++) {
    for (size_t y : {static_cast<size_t>(0), ysize - 1}) {
      memcpy(smoothed.PlaneRow(c, y), dc->PlaneRow(c, y),
             xsize * sizeof(float));
    }
  }
  auto process_row = [&](const uint32_t y, size_t /*thread*/) -> Status {
    const float* JXL_RESTRICT rows_top[3]{
        dc->ConstPlaneRow(0, y - 1),
        dc->ConstPlaneRow(1, y - 1),
        dc->ConstPlaneRow(2, y - 1),
    };
    const float* JXL_RESTRICT rows[3] = {
        dc->ConstPlaneRow(0, y),
        dc->ConstPlaneRow(1, y),
        dc->ConstPlaneRow(2, y),
    };
    const float* JXL_RESTRICT rows_bottom[3] = {
        dc->ConstPlaneRow(0, y + 1),
        dc->ConstPlaneRow(1, y + 1),
        dc->ConstPlaneRow(2, y + 1),
    };
    float* JXL_RESTRICT rows_out[3] = {
        smoothed.PlaneRow(0, y),
        smoothed.PlaneRow(1, y),
        smoothed.PlaneRow(2, y),
    };
    for (size_t x : {static_cast<size_t>(0), xsize - 1}) {
      for (size_t c = 0; c < 3; c++) {
        rows_out[c][x] = rows[c][x];
      }
    }

    size_t x = 1;
    // First pixels
    const size_t N = Lanes(D());
    for (; x < std::min(N, xsize - 1); x++) {
      ComputePixel<DScalar>(dc_factors, rows_top, rows, rows_bottom, rows_out,
                            x);
    }
    // Full vectors.
    for (; x + N <= xsize - 1; x += N) {
      ComputePixel<D>(dc_factors, rows_top, rows, rows_bottom, rows_out, x);
    }
    // Last pixels.
    for (; x < xsize - 1; x++) {
      ComputePixel<DScalar>(dc_factors, rows_top, rows, rows_bottom, rows_out,
                            x);
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 1, ysize - 1, ThreadPool::NoInit,
                                process_row, "DCSmoothingRow"));
  dc->Swap(smoothed);
  return true;
}

// DC dequantization.
void DequantDC(const Rect& r, Image3F* dc, ImageB* quant_dc, const Image& in,
               const float* dc_factors, float mul, const float* cfl_factors,
               const YCbCrChromaSubsampling& chroma_subsampling,
               const BlockCtxMap& bctx) {
  const HWY_FULL(float) df;
  const Rebind<pixel_type, HWY_FULL(float)> di;  // assumes pixel_type <= float
  if (chroma_subsampling.Is444()) {
    const auto fac_x = Set(df, dc_factors[0] * mul);
    const auto fac_y = Set(df, dc_factors[1] * mul);
    const auto fac_b = Set(df, dc_factors[2] * mul);
    const auto cfl_fac_x = Set(df, cfl_factors[0]);
    const auto cfl_fac_b = Set(df, cfl_factors[2]);
    for (size_t y = 0; y < r.ysize(); y++) {
      float* dec_row_x = r.PlaneRow(dc, 0, y);
      float* dec_row_y = r.PlaneRow(dc, 1, y);
      float* dec_row_b = r.PlaneRow(dc, 2, y);
      const int32_t* quant_row_x = in.channel[1].plane.Row(y);
      const int32_t* quant_row_y = in.channel[0].plane.Row(y);
      const int32_t* quant_row_b = in.channel[2].plane.Row(y);
      for (size_t x = 0; x < r.xsize(); x += Lanes(di)) {
        const auto in_q_x = Load(di, quant_row_x + x);
        const auto in_q_y = Load(di, quant_row_y + x);
        const auto in_q_b = Load(di, quant_row_b + x);
        const auto in_x = Mul(ConvertTo(df, in_q_x), fac_x);
        const auto in_y = Mul(ConvertTo(df, in_q_y), fac_y);
        const auto in_b = Mul(ConvertTo(df, in_q_b), fac_b);
        Store(in_y, df, dec_row_y + x);
        Store(MulAdd(in_y, cfl_fac_x, in_x), df, dec_row_x + x);
        Store(MulAdd(in_y, cfl_fac_b, in_b), df, dec_row_b + x);
      }
    }
  } else {
    for (size_t c : {1, 0, 2}) {
      Rect rect(r.x0() >> chroma_subsampling.HShift(c),
                r.y0() >> chroma_subsampling.VShift(c),
                r.xsize() >> chroma_subsampling.HShift(c),
                r.ysize() >> chroma_subsampling.VShift(c));
      const auto fac = Set(df, dc_factors[c] * mul);
      const Channel& ch = in.channel[c < 2 ? c ^ 1 : c];
      for (size_t y = 0; y < rect.ysize(); y++) {
        const int32_t* quant_row = ch.plane.Row(y);
        float* row = rect.PlaneRow(dc, c, y);
        for (size_t x = 0; x < rect.xsize(); x += Lanes(di)) {
          const auto in_q = Load(di, quant_row + x);
          const auto in = Mul(ConvertTo(df, in_q), fac);
          Store(in, df, row + x);
        }
      }
    }
  }
  if (bctx.num_dc_ctxs <= 1) {
    for (size_t y = 0; y < r.ysize(); y++) {
      uint8_t* qdc_row = r.Row(quant_dc, y);
      memset(qdc_row, 0, sizeof(*qdc_row) * r.xsize());
    }
  } else {
    for (size_t y = 0; y < r.ysize(); y++) {
      uint8_t* qdc_row_val = r.Row(quant_dc, y);
      const int32_t* quant_row_x =
          in.channel[1].plane.Row(y >> chroma_subsampling.VShift(0));
      const int32_t* quant_row_y =
          in.channel[0].plane.Row(y >> chroma_subsampling.VShift(1));
      const int32_t* quant_row_b =
          in.channel[2].plane.Row(y >> chroma_subsampling.VShift(2));
      for (size_t x = 0; x < r.xsize(); x++) {
        int bucket_x = 0;
        int bucket_y = 0;
        int bucket_b = 0;
        for (int t : bctx.dc_thresholds[0]) {
          if (quant_row_x[x >> chroma_subsampling.HShift(0)] > t) bucket_x++;
        }
        for (int t : bctx.dc_thresholds[1]) {
          if (quant_row_y[x >> chroma_subsampling.HShift(1)] > t) bucket_y++;
        }
        for (int t : bctx.dc_thresholds[2]) {
          if (quant_row_b[x >> chroma_subsampling.HShift(2)] > t) bucket_b++;
        }
        int bucket = bucket_x;
        bucket *= bctx.dc_thresholds[2].size() + 1;
        bucket += bucket_b;
        bucket *= bctx.dc_thresholds[1].size() + 1;
        bucket += bucket_y;
        qdc_row_val[x] = bucket;
      }
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(DequantDC);
HWY_EXPORT(AdaptiveDCSmoothing);
Status AdaptiveDCSmoothing(JxlMemoryManager* memory_manager,
                           const float* dc_factors, Image3F* dc,
                           ThreadPool* pool) {
  return HWY_DYNAMIC_DISPATCH(AdaptiveDCSmoothing)(memory_manager, dc_factors,
                                                   dc, pool);
}

void DequantDC(const Rect& r, Image3F* dc, ImageB* quant_dc, const Image& in,
               const float* dc_factors, float mul, const float* cfl_factors,
               const YCbCrChromaSubsampling& chroma_subsampling,
               const BlockCtxMap& bctx) {
  HWY_DYNAMIC_DISPATCH(DequantDC)
  (r, dc, quant_dc, in, dc_factors, mul, cfl_factors, chroma_subsampling, bctx);
}

}  // namespace jxl
#endif  // HWY_ONCE
