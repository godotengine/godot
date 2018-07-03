// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// distortion calculation
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>  // for abs()

#include "src/dsp/dsp.h"

#if !defined(WEBP_REDUCE_SIZE)

//------------------------------------------------------------------------------
// SSIM / PSNR

// hat-shaped filter. Sum of coefficients is equal to 16.
static const uint32_t kWeight[2 * VP8_SSIM_KERNEL + 1] = {
  1, 2, 3, 4, 3, 2, 1
};
static const uint32_t kWeightSum = 16 * 16;   // sum{kWeight}^2

static WEBP_INLINE double SSIMCalculation(
    const VP8DistoStats* const stats, uint32_t N  /*num samples*/) {
  const uint32_t w2 =  N * N;
  const uint32_t C1 = 20 * w2;
  const uint32_t C2 = 60 * w2;
  const uint32_t C3 = 8 * 8 * w2;   // 'dark' limit ~= 6
  const uint64_t xmxm = (uint64_t)stats->xm * stats->xm;
  const uint64_t ymym = (uint64_t)stats->ym * stats->ym;
  if (xmxm + ymym >= C3) {
    const int64_t xmym = (int64_t)stats->xm * stats->ym;
    const int64_t sxy = (int64_t)stats->xym * N - xmym;    // can be negative
    const uint64_t sxx = (uint64_t)stats->xxm * N - xmxm;
    const uint64_t syy = (uint64_t)stats->yym * N - ymym;
    // we descale by 8 to prevent overflow during the fnum/fden multiply.
    const uint64_t num_S = (2 * (uint64_t)(sxy < 0 ? 0 : sxy) + C2) >> 8;
    const uint64_t den_S = (sxx + syy + C2) >> 8;
    const uint64_t fnum = (2 * xmym + C1) * num_S;
    const uint64_t fden = (xmxm + ymym + C1) * den_S;
    const double r = (double)fnum / fden;
    assert(r >= 0. && r <= 1.0);
    return r;
  }
  return 1.;   // area is too dark to contribute meaningfully
}

double VP8SSIMFromStats(const VP8DistoStats* const stats) {
  return SSIMCalculation(stats, kWeightSum);
}

double VP8SSIMFromStatsClipped(const VP8DistoStats* const stats) {
  return SSIMCalculation(stats, stats->w);
}

static double SSIMGetClipped_C(const uint8_t* src1, int stride1,
                               const uint8_t* src2, int stride2,
                               int xo, int yo, int W, int H) {
  VP8DistoStats stats = { 0, 0, 0, 0, 0, 0 };
  const int ymin = (yo - VP8_SSIM_KERNEL < 0) ? 0 : yo - VP8_SSIM_KERNEL;
  const int ymax = (yo + VP8_SSIM_KERNEL > H - 1) ? H - 1
                                                  : yo + VP8_SSIM_KERNEL;
  const int xmin = (xo - VP8_SSIM_KERNEL < 0) ? 0 : xo - VP8_SSIM_KERNEL;
  const int xmax = (xo + VP8_SSIM_KERNEL > W - 1) ? W - 1
                                                  : xo + VP8_SSIM_KERNEL;
  int x, y;
  src1 += ymin * stride1;
  src2 += ymin * stride2;
  for (y = ymin; y <= ymax; ++y, src1 += stride1, src2 += stride2) {
    for (x = xmin; x <= xmax; ++x) {
      const uint32_t w = kWeight[VP8_SSIM_KERNEL + x - xo]
                       * kWeight[VP8_SSIM_KERNEL + y - yo];
      const uint32_t s1 = src1[x];
      const uint32_t s2 = src2[x];
      stats.w   += w;
      stats.xm  += w * s1;
      stats.ym  += w * s2;
      stats.xxm += w * s1 * s1;
      stats.xym += w * s1 * s2;
      stats.yym += w * s2 * s2;
    }
  }
  return VP8SSIMFromStatsClipped(&stats);
}

static double SSIMGet_C(const uint8_t* src1, int stride1,
                        const uint8_t* src2, int stride2) {
  VP8DistoStats stats = { 0, 0, 0, 0, 0, 0 };
  int x, y;
  for (y = 0; y <= 2 * VP8_SSIM_KERNEL; ++y, src1 += stride1, src2 += stride2) {
    for (x = 0; x <= 2 * VP8_SSIM_KERNEL; ++x) {
      const uint32_t w = kWeight[x] * kWeight[y];
      const uint32_t s1 = src1[x];
      const uint32_t s2 = src2[x];
      stats.xm  += w * s1;
      stats.ym  += w * s2;
      stats.xxm += w * s1 * s1;
      stats.xym += w * s1 * s2;
      stats.yym += w * s2 * s2;
    }
  }
  return VP8SSIMFromStats(&stats);
}

#endif  // !defined(WEBP_REDUCE_SIZE)

//------------------------------------------------------------------------------

#if !defined(WEBP_DISABLE_STATS)
static uint32_t AccumulateSSE_C(const uint8_t* src1,
                                const uint8_t* src2, int len) {
  int i;
  uint32_t sse2 = 0;
  assert(len <= 65535);  // to ensure that accumulation fits within uint32_t
  for (i = 0; i < len; ++i) {
    const int32_t diff = src1[i] - src2[i];
    sse2 += diff * diff;
  }
  return sse2;
}
#endif

//------------------------------------------------------------------------------

#if !defined(WEBP_REDUCE_SIZE)
VP8SSIMGetFunc VP8SSIMGet;
VP8SSIMGetClippedFunc VP8SSIMGetClipped;
#endif
#if !defined(WEBP_DISABLE_STATS)
VP8AccumulateSSEFunc VP8AccumulateSSE;
#endif

extern void VP8SSIMDspInitSSE2(void);

WEBP_DSP_INIT_FUNC(VP8SSIMDspInit) {
#if !defined(WEBP_REDUCE_SIZE)
  VP8SSIMGetClipped = SSIMGetClipped_C;
  VP8SSIMGet = SSIMGet_C;
#endif

#if !defined(WEBP_DISABLE_STATS)
  VP8AccumulateSSE = AccumulateSSE_C;
#endif

  if (VP8GetCPUInfo != NULL) {
#if defined(WEBP_USE_SSE2)
    if (VP8GetCPUInfo(kSSE2)) {
      VP8SSIMDspInitSSE2();
    }
#endif
  }
}
