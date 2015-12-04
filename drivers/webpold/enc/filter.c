// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Selecting filter level
//
// Author: somnath@google.com (Somnath Banerjee)

#include "./vp8enci.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

// NOTE: clip1, tables and InitTables are repeated entries of dsp.c
static uint8_t abs0[255 + 255 + 1];     // abs(i)
static uint8_t abs1[255 + 255 + 1];     // abs(i)>>1
static int8_t sclip1[1020 + 1020 + 1];  // clips [-1020, 1020] to [-128, 127]
static int8_t sclip2[112 + 112 + 1];    // clips [-112, 112] to [-16, 15]
static uint8_t clip1[255 + 510 + 1];    // clips [-255,510] to [0,255]

static int tables_ok = 0;

static void InitTables(void) {
  if (!tables_ok) {
    int i;
    for (i = -255; i <= 255; ++i) {
      abs0[255 + i] = (i < 0) ? -i : i;
      abs1[255 + i] = abs0[255 + i] >> 1;
    }
    for (i = -1020; i <= 1020; ++i) {
      sclip1[1020 + i] = (i < -128) ? -128 : (i > 127) ? 127 : i;
    }
    for (i = -112; i <= 112; ++i) {
      sclip2[112 + i] = (i < -16) ? -16 : (i > 15) ? 15 : i;
    }
    for (i = -255; i <= 255 + 255; ++i) {
      clip1[255 + i] = (i < 0) ? 0 : (i > 255) ? 255 : i;
    }
    tables_ok = 1;
  }
}

//------------------------------------------------------------------------------
// Edge filtering functions

// 4 pixels in, 2 pixels out
static WEBP_INLINE void do_filter2(uint8_t* p, int step) {
  const int p1 = p[-2*step], p0 = p[-step], q0 = p[0], q1 = p[step];
  const int a = 3 * (q0 - p0) + sclip1[1020 + p1 - q1];
  const int a1 = sclip2[112 + ((a + 4) >> 3)];
  const int a2 = sclip2[112 + ((a + 3) >> 3)];
  p[-step] = clip1[255 + p0 + a2];
  p[    0] = clip1[255 + q0 - a1];
}

// 4 pixels in, 4 pixels out
static WEBP_INLINE void do_filter4(uint8_t* p, int step) {
  const int p1 = p[-2*step], p0 = p[-step], q0 = p[0], q1 = p[step];
  const int a = 3 * (q0 - p0);
  const int a1 = sclip2[112 + ((a + 4) >> 3)];
  const int a2 = sclip2[112 + ((a + 3) >> 3)];
  const int a3 = (a1 + 1) >> 1;
  p[-2*step] = clip1[255 + p1 + a3];
  p[-  step] = clip1[255 + p0 + a2];
  p[      0] = clip1[255 + q0 - a1];
  p[   step] = clip1[255 + q1 - a3];
}

// high edge-variance
static WEBP_INLINE int hev(const uint8_t* p, int step, int thresh) {
  const int p1 = p[-2*step], p0 = p[-step], q0 = p[0], q1 = p[step];
  return (abs0[255 + p1 - p0] > thresh) || (abs0[255 + q1 - q0] > thresh);
}

static WEBP_INLINE int needs_filter(const uint8_t* p, int step, int thresh) {
  const int p1 = p[-2*step], p0 = p[-step], q0 = p[0], q1 = p[step];
  return (2 * abs0[255 + p0 - q0] + abs1[255 + p1 - q1]) <= thresh;
}

static WEBP_INLINE int needs_filter2(const uint8_t* p,
                                     int step, int t, int it) {
  const int p3 = p[-4*step], p2 = p[-3*step], p1 = p[-2*step], p0 = p[-step];
  const int q0 = p[0], q1 = p[step], q2 = p[2*step], q3 = p[3*step];
  if ((2 * abs0[255 + p0 - q0] + abs1[255 + p1 - q1]) > t)
    return 0;
  return abs0[255 + p3 - p2] <= it && abs0[255 + p2 - p1] <= it &&
         abs0[255 + p1 - p0] <= it && abs0[255 + q3 - q2] <= it &&
         abs0[255 + q2 - q1] <= it && abs0[255 + q1 - q0] <= it;
}

//------------------------------------------------------------------------------
// Simple In-loop filtering (Paragraph 15.2)

static void SimpleVFilter16(uint8_t* p, int stride, int thresh) {
  int i;
  for (i = 0; i < 16; ++i) {
    if (needs_filter(p + i, stride, thresh)) {
      do_filter2(p + i, stride);
    }
  }
}

static void SimpleHFilter16(uint8_t* p, int stride, int thresh) {
  int i;
  for (i = 0; i < 16; ++i) {
    if (needs_filter(p + i * stride, 1, thresh)) {
      do_filter2(p + i * stride, 1);
    }
  }
}

static void SimpleVFilter16i(uint8_t* p, int stride, int thresh) {
  int k;
  for (k = 3; k > 0; --k) {
    p += 4 * stride;
    SimpleVFilter16(p, stride, thresh);
  }
}

static void SimpleHFilter16i(uint8_t* p, int stride, int thresh) {
  int k;
  for (k = 3; k > 0; --k) {
    p += 4;
    SimpleHFilter16(p, stride, thresh);
  }
}

//------------------------------------------------------------------------------
// Complex In-loop filtering (Paragraph 15.3)

static WEBP_INLINE void FilterLoop24(uint8_t* p,
                                     int hstride, int vstride, int size,
                                     int thresh, int ithresh, int hev_thresh) {
  while (size-- > 0) {
    if (needs_filter2(p, hstride, thresh, ithresh)) {
      if (hev(p, hstride, hev_thresh)) {
        do_filter2(p, hstride);
      } else {
        do_filter4(p, hstride);
      }
    }
    p += vstride;
  }
}

// on three inner edges
static void VFilter16i(uint8_t* p, int stride,
                       int thresh, int ithresh, int hev_thresh) {
  int k;
  for (k = 3; k > 0; --k) {
    p += 4 * stride;
    FilterLoop24(p, stride, 1, 16, thresh, ithresh, hev_thresh);
  }
}

static void HFilter16i(uint8_t* p, int stride,
                       int thresh, int ithresh, int hev_thresh) {
  int k;
  for (k = 3; k > 0; --k) {
    p += 4;
    FilterLoop24(p, 1, stride, 16, thresh, ithresh, hev_thresh);
  }
}

static void VFilter8i(uint8_t* u, uint8_t* v, int stride,
                      int thresh, int ithresh, int hev_thresh) {
  FilterLoop24(u + 4 * stride, stride, 1, 8, thresh, ithresh, hev_thresh);
  FilterLoop24(v + 4 * stride, stride, 1, 8, thresh, ithresh, hev_thresh);
}

static void HFilter8i(uint8_t* u, uint8_t* v, int stride,
                      int thresh, int ithresh, int hev_thresh) {
  FilterLoop24(u + 4, 1, stride, 8, thresh, ithresh, hev_thresh);
  FilterLoop24(v + 4, 1, stride, 8, thresh, ithresh, hev_thresh);
}

//------------------------------------------------------------------------------

void (*VP8EncVFilter16i)(uint8_t*, int, int, int, int) = VFilter16i;
void (*VP8EncHFilter16i)(uint8_t*, int, int, int, int) = HFilter16i;
void (*VP8EncVFilter8i)(uint8_t*, uint8_t*, int, int, int, int) = VFilter8i;
void (*VP8EncHFilter8i)(uint8_t*, uint8_t*, int, int, int, int) = HFilter8i;

void (*VP8EncSimpleVFilter16i)(uint8_t*, int, int) = SimpleVFilter16i;
void (*VP8EncSimpleHFilter16i)(uint8_t*, int, int) = SimpleHFilter16i;

//------------------------------------------------------------------------------
// Paragraph 15.4: compute the inner-edge filtering strength

static int GetILevel(int sharpness, int level) {
  if (sharpness > 0) {
    if (sharpness > 4) {
      level >>= 2;
    } else {
      level >>= 1;
    }
    if (level > 9 - sharpness) {
      level = 9 - sharpness;
    }
  }
  if (level < 1) level = 1;
  return level;
}

static void DoFilter(const VP8EncIterator* const it, int level) {
  const VP8Encoder* const enc = it->enc_;
  const int ilevel = GetILevel(enc->config_->filter_sharpness, level);
  const int limit = 2 * level + ilevel;

  uint8_t* const y_dst = it->yuv_out2_ + Y_OFF;
  uint8_t* const u_dst = it->yuv_out2_ + U_OFF;
  uint8_t* const v_dst = it->yuv_out2_ + V_OFF;

  // copy current block to yuv_out2_
  memcpy(y_dst, it->yuv_out_, YUV_SIZE * sizeof(uint8_t));

  if (enc->filter_hdr_.simple_ == 1) {   // simple
    VP8EncSimpleHFilter16i(y_dst, BPS, limit);
    VP8EncSimpleVFilter16i(y_dst, BPS, limit);
  } else {    // complex
    const int hev_thresh = (level >= 40) ? 2 : (level >= 15) ? 1 : 0;
    VP8EncHFilter16i(y_dst, BPS, limit, ilevel, hev_thresh);
    VP8EncHFilter8i(u_dst, v_dst, BPS, limit, ilevel, hev_thresh);
    VP8EncVFilter16i(y_dst, BPS, limit, ilevel, hev_thresh);
    VP8EncVFilter8i(u_dst, v_dst, BPS, limit, ilevel, hev_thresh);
  }
}

//------------------------------------------------------------------------------
// SSIM metric

enum { KERNEL = 3 };
static const double kMinValue = 1.e-10;  // minimal threshold

void VP8SSIMAddStats(const DistoStats* const src, DistoStats* const dst) {
  dst->w   += src->w;
  dst->xm  += src->xm;
  dst->ym  += src->ym;
  dst->xxm += src->xxm;
  dst->xym += src->xym;
  dst->yym += src->yym;
}

static void VP8SSIMAccumulate(const uint8_t* src1, int stride1,
                              const uint8_t* src2, int stride2,
                              int xo, int yo, int W, int H,
                              DistoStats* const stats) {
  const int ymin = (yo - KERNEL < 0) ? 0 : yo - KERNEL;
  const int ymax = (yo + KERNEL > H - 1) ? H - 1 : yo + KERNEL;
  const int xmin = (xo - KERNEL < 0) ? 0 : xo - KERNEL;
  const int xmax = (xo + KERNEL > W - 1) ? W - 1 : xo + KERNEL;
  int x, y;
  src1 += ymin * stride1;
  src2 += ymin * stride2;
  for (y = ymin; y <= ymax; ++y, src1 += stride1, src2 += stride2) {
    for (x = xmin; x <= xmax; ++x) {
      const int s1 = src1[x];
      const int s2 = src2[x];
      stats->w   += 1;
      stats->xm  += s1;
      stats->ym  += s2;
      stats->xxm += s1 * s1;
      stats->xym += s1 * s2;
      stats->yym += s2 * s2;
    }
  }
}

double VP8SSIMGet(const DistoStats* const stats) {
  const double xmxm = stats->xm * stats->xm;
  const double ymym = stats->ym * stats->ym;
  const double xmym = stats->xm * stats->ym;
  const double w2 = stats->w * stats->w;
  double sxx = stats->xxm * stats->w - xmxm;
  double syy = stats->yym * stats->w - ymym;
  double sxy = stats->xym * stats->w - xmym;
  double C1, C2;
  double fnum;
  double fden;
  // small errors are possible, due to rounding. Clamp to zero.
  if (sxx < 0.) sxx = 0.;
  if (syy < 0.) syy = 0.;
  C1 = 6.5025 * w2;
  C2 = 58.5225 * w2;
  fnum = (2 * xmym + C1) * (2 * sxy + C2);
  fden = (xmxm + ymym + C1) * (sxx + syy + C2);
  return (fden != 0.) ? fnum / fden : kMinValue;
}

double VP8SSIMGetSquaredError(const DistoStats* const s) {
  if (s->w > 0.) {
    const double iw2 = 1. / (s->w * s->w);
    const double sxx = s->xxm * s->w - s->xm * s->xm;
    const double syy = s->yym * s->w - s->ym * s->ym;
    const double sxy = s->xym * s->w - s->xm * s->ym;
    const double SSE = iw2 * (sxx + syy - 2. * sxy);
    if (SSE > kMinValue) return SSE;
  }
  return kMinValue;
}

void VP8SSIMAccumulatePlane(const uint8_t* src1, int stride1,
                            const uint8_t* src2, int stride2,
                            int W, int H, DistoStats* const stats) {
  int x, y;
  for (y = 0; y < H; ++y) {
    for (x = 0; x < W; ++x) {
      VP8SSIMAccumulate(src1, stride1, src2, stride2, x, y, W, H, stats);
    }
  }
}

static double GetMBSSIM(const uint8_t* yuv1, const uint8_t* yuv2) {
  int x, y;
  DistoStats s = { .0, .0, .0, .0, .0, .0 };

  // compute SSIM in a 10 x 10 window
  for (x = 3; x < 13; x++) {
    for (y = 3; y < 13; y++) {
      VP8SSIMAccumulate(yuv1 + Y_OFF, BPS, yuv2 + Y_OFF, BPS, x, y, 16, 16, &s);
    }
  }
  for (x = 1; x < 7; x++) {
    for (y = 1; y < 7; y++) {
      VP8SSIMAccumulate(yuv1 + U_OFF, BPS, yuv2 + U_OFF, BPS, x, y, 8, 8, &s);
      VP8SSIMAccumulate(yuv1 + V_OFF, BPS, yuv2 + V_OFF, BPS, x, y, 8, 8, &s);
    }
  }
  return VP8SSIMGet(&s);
}

//------------------------------------------------------------------------------
// Exposed APIs: Encoder should call the following 3 functions to adjust
// loop filter strength

void VP8InitFilter(VP8EncIterator* const it) {
  int s, i;
  if (!it->lf_stats_) return;

  InitTables();
  for (s = 0; s < NUM_MB_SEGMENTS; s++) {
    for (i = 0; i < MAX_LF_LEVELS; i++) {
      (*it->lf_stats_)[s][i] = 0;
    }
  }
}

void VP8StoreFilterStats(VP8EncIterator* const it) {
  int d;
  const int s = it->mb_->segment_;
  const int level0 = it->enc_->dqm_[s].fstrength_;  // TODO: ref_lf_delta[]

  // explore +/-quant range of values around level0
  const int delta_min = -it->enc_->dqm_[s].quant_;
  const int delta_max = it->enc_->dqm_[s].quant_;
  const int step_size = (delta_max - delta_min >= 4) ? 4 : 1;

  if (!it->lf_stats_) return;

  // NOTE: Currently we are applying filter only across the sublock edges
  // There are two reasons for that.
  // 1. Applying filter on macro block edges will change the pixels in
  // the left and top macro blocks. That will be hard to restore
  // 2. Macro Blocks on the bottom and right are not yet compressed. So we
  // cannot apply filter on the right and bottom macro block edges.
  if (it->mb_->type_ == 1 && it->mb_->skip_) return;

  // Always try filter level  zero
  (*it->lf_stats_)[s][0] += GetMBSSIM(it->yuv_in_, it->yuv_out_);

  for (d = delta_min; d <= delta_max; d += step_size) {
    const int level = level0 + d;
    if (level <= 0 || level >= MAX_LF_LEVELS) {
      continue;
    }
    DoFilter(it, level);
    (*it->lf_stats_)[s][level] += GetMBSSIM(it->yuv_in_, it->yuv_out2_);
  }
}

void VP8AdjustFilterStrength(VP8EncIterator* const it) {
  int s;
  VP8Encoder* const enc = it->enc_;

  if (!it->lf_stats_) {
    return;
  }
  for (s = 0; s < NUM_MB_SEGMENTS; s++) {
    int i, best_level = 0;
    // Improvement over filter level 0 should be at least 1e-5 (relatively)
    double best_v = 1.00001 * (*it->lf_stats_)[s][0];
    for (i = 1; i < MAX_LF_LEVELS; i++) {
      const double v = (*it->lf_stats_)[s][i];
      if (v > best_v) {
        best_v = v;
        best_level = i;
      }
    }
    enc->dqm_[s].fstrength_ = best_level;
  }
}

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
