/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 *
 *  This code was originally written by: Nathan E. Egge, at the Daala
 *  project.
 */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/ssim.h"
#include "vpx_ports/system_state.h"

typedef struct fs_level fs_level;
typedef struct fs_ctx fs_ctx;

#define SSIM_C1 (255 * 255 * 0.01 * 0.01)
#define SSIM_C2 (255 * 255 * 0.03 * 0.03)
#if CONFIG_VP9_HIGHBITDEPTH
#define SSIM_C1_10 (1023 * 1023 * 0.01 * 0.01)
#define SSIM_C1_12 (4095 * 4095 * 0.01 * 0.01)
#define SSIM_C2_10 (1023 * 1023 * 0.03 * 0.03)
#define SSIM_C2_12 (4095 * 4095 * 0.03 * 0.03)
#endif
#define FS_MINI(_a, _b) ((_a) < (_b) ? (_a) : (_b))
#define FS_MAXI(_a, _b) ((_a) > (_b) ? (_a) : (_b))

struct fs_level {
  uint32_t *im1;
  uint32_t *im2;
  double *ssim;
  int w;
  int h;
};

struct fs_ctx {
  fs_level *level;
  int nlevels;
  unsigned *col_buf;
};

static int fs_ctx_init(fs_ctx *_ctx, int _w, int _h, int _nlevels) {
  unsigned char *data;
  size_t data_size;
  int lw;
  int lh;
  int l;
  lw = (_w + 1) >> 1;
  lh = (_h + 1) >> 1;
  data_size =
      _nlevels * sizeof(fs_level) + 2 * (lw + 8) * 8 * sizeof(*_ctx->col_buf);
  for (l = 0; l < _nlevels; l++) {
    size_t im_size;
    size_t level_size;
    im_size = lw * (size_t)lh;
    level_size = 2 * im_size * sizeof(*_ctx->level[l].im1);
    level_size += sizeof(*_ctx->level[l].ssim) - 1;
    level_size /= sizeof(*_ctx->level[l].ssim);
    level_size += im_size;
    level_size *= sizeof(*_ctx->level[l].ssim);
    data_size += level_size;
    lw = (lw + 1) >> 1;
    lh = (lh + 1) >> 1;
  }
  data = (unsigned char *)malloc(data_size);
  if (!data) return -1;
  _ctx->level = (fs_level *)data;
  _ctx->nlevels = _nlevels;
  data += _nlevels * sizeof(*_ctx->level);
  lw = (_w + 1) >> 1;
  lh = (_h + 1) >> 1;
  for (l = 0; l < _nlevels; l++) {
    size_t im_size;
    size_t level_size;
    _ctx->level[l].w = lw;
    _ctx->level[l].h = lh;
    im_size = lw * (size_t)lh;
    level_size = 2 * im_size * sizeof(*_ctx->level[l].im1);
    level_size += sizeof(*_ctx->level[l].ssim) - 1;
    level_size /= sizeof(*_ctx->level[l].ssim);
    level_size *= sizeof(*_ctx->level[l].ssim);
    _ctx->level[l].im1 = (uint32_t *)data;
    _ctx->level[l].im2 = _ctx->level[l].im1 + im_size;
    data += level_size;
    _ctx->level[l].ssim = (double *)data;
    data += im_size * sizeof(*_ctx->level[l].ssim);
    lw = (lw + 1) >> 1;
    lh = (lh + 1) >> 1;
  }
  _ctx->col_buf = (unsigned *)data;
  return 0;
}

static void fs_ctx_clear(fs_ctx *_ctx) { free(_ctx->level); }

static void fs_downsample_level(fs_ctx *_ctx, int _l) {
  const uint32_t *src1;
  const uint32_t *src2;
  uint32_t *dst1;
  uint32_t *dst2;
  int w2;
  int h2;
  int w;
  int h;
  int i;
  int j;
  w = _ctx->level[_l].w;
  h = _ctx->level[_l].h;
  dst1 = _ctx->level[_l].im1;
  dst2 = _ctx->level[_l].im2;
  w2 = _ctx->level[_l - 1].w;
  h2 = _ctx->level[_l - 1].h;
  src1 = _ctx->level[_l - 1].im1;
  src2 = _ctx->level[_l - 1].im2;
  for (j = 0; j < h; j++) {
    int j0offs;
    int j1offs;
    j0offs = 2 * j * w2;
    j1offs = FS_MINI(2 * j + 1, h2) * w2;
    for (i = 0; i < w; i++) {
      int i0;
      int i1;
      i0 = 2 * i;
      i1 = FS_MINI(i0 + 1, w2);
      dst1[j * w + i] =
          (uint32_t)((int64_t)src1[j0offs + i0] + src1[j0offs + i1] +
                     src1[j1offs + i0] + src1[j1offs + i1]);
      dst2[j * w + i] =
          (uint32_t)((int64_t)src2[j0offs + i0] + src2[j0offs + i1] +
                     src2[j1offs + i0] + src2[j1offs + i1]);
    }
  }
}

static void fs_downsample_level0(fs_ctx *_ctx, const uint8_t *_src1,
                                 int _s1ystride, const uint8_t *_src2,
                                 int _s2ystride, int _w, int _h, uint32_t bd,
                                 uint32_t shift) {
  uint32_t *dst1;
  uint32_t *dst2;
  int w;
  int h;
  int i;
  int j;
  w = _ctx->level[0].w;
  h = _ctx->level[0].h;
  dst1 = _ctx->level[0].im1;
  dst2 = _ctx->level[0].im2;
  for (j = 0; j < h; j++) {
    int j0;
    int j1;
    j0 = 2 * j;
    j1 = FS_MINI(j0 + 1, _h);
    for (i = 0; i < w; i++) {
      int i0;
      int i1;
      i0 = 2 * i;
      i1 = FS_MINI(i0 + 1, _w);
      if (bd == 8 && shift == 0) {
        dst1[j * w + i] =
            _src1[j0 * _s1ystride + i0] + _src1[j0 * _s1ystride + i1] +
            _src1[j1 * _s1ystride + i0] + _src1[j1 * _s1ystride + i1];
        dst2[j * w + i] =
            _src2[j0 * _s2ystride + i0] + _src2[j0 * _s2ystride + i1] +
            _src2[j1 * _s2ystride + i0] + _src2[j1 * _s2ystride + i1];
      } else {
        uint16_t *src1s = CONVERT_TO_SHORTPTR(_src1);
        uint16_t *src2s = CONVERT_TO_SHORTPTR(_src2);
        dst1[j * w + i] = (src1s[j0 * _s1ystride + i0] >> shift) +
                          (src1s[j0 * _s1ystride + i1] >> shift) +
                          (src1s[j1 * _s1ystride + i0] >> shift) +
                          (src1s[j1 * _s1ystride + i1] >> shift);
        dst2[j * w + i] = (src2s[j0 * _s2ystride + i0] >> shift) +
                          (src2s[j0 * _s2ystride + i1] >> shift) +
                          (src2s[j1 * _s2ystride + i0] >> shift) +
                          (src2s[j1 * _s2ystride + i1] >> shift);
      }
    }
  }
}

static void fs_apply_luminance(fs_ctx *_ctx, int _l, int bit_depth) {
  unsigned *col_sums_x;
  unsigned *col_sums_y;
  uint32_t *im1;
  uint32_t *im2;
  double *ssim;
  double c1;
  int w;
  int h;
  int j0offs;
  int j1offs;
  int i;
  int j;
  double ssim_c1 = SSIM_C1;
#if CONFIG_VP9_HIGHBITDEPTH
  if (bit_depth == 10) ssim_c1 = SSIM_C1_10;
  if (bit_depth == 12) ssim_c1 = SSIM_C1_12;
#else
  assert(bit_depth == 8);
  (void)bit_depth;
#endif
  w = _ctx->level[_l].w;
  h = _ctx->level[_l].h;
  col_sums_x = _ctx->col_buf;
  col_sums_y = col_sums_x + w;
  im1 = _ctx->level[_l].im1;
  im2 = _ctx->level[_l].im2;
  for (i = 0; i < w; i++) col_sums_x[i] = 5 * im1[i];
  for (i = 0; i < w; i++) col_sums_y[i] = 5 * im2[i];
  for (j = 1; j < 4; j++) {
    j1offs = FS_MINI(j, h - 1) * w;
    for (i = 0; i < w; i++) col_sums_x[i] += im1[j1offs + i];
    for (i = 0; i < w; i++) col_sums_y[i] += im2[j1offs + i];
  }
  ssim = _ctx->level[_l].ssim;
  c1 = (double)(ssim_c1 * 4096 * (1 << 4 * _l));
  for (j = 0; j < h; j++) {
    int64_t mux;
    int64_t muy;
    int i0;
    int i1;
    mux = (int64_t)5 * col_sums_x[0];
    muy = (int64_t)5 * col_sums_y[0];
    for (i = 1; i < 4; i++) {
      i1 = FS_MINI(i, w - 1);
      mux += col_sums_x[i1];
      muy += col_sums_y[i1];
    }
    for (i = 0; i < w; i++) {
      ssim[j * w + i] *= (2 * mux * (double)muy + c1) /
                         (mux * (double)mux + muy * (double)muy + c1);
      if (i + 1 < w) {
        i0 = FS_MAXI(0, i - 4);
        i1 = FS_MINI(i + 4, w - 1);
        mux += (int)col_sums_x[i1] - (int)col_sums_x[i0];
        muy += (int)col_sums_x[i1] - (int)col_sums_x[i0];
      }
    }
    if (j + 1 < h) {
      j0offs = FS_MAXI(0, j - 4) * w;
      for (i = 0; i < w; i++) col_sums_x[i] -= im1[j0offs + i];
      for (i = 0; i < w; i++) col_sums_y[i] -= im2[j0offs + i];
      j1offs = FS_MINI(j + 4, h - 1) * w;
      for (i = 0; i < w; i++)
        col_sums_x[i] = (uint32_t)((int64_t)col_sums_x[i] + im1[j1offs + i]);
      for (i = 0; i < w; i++)
        col_sums_y[i] = (uint32_t)((int64_t)col_sums_y[i] + im2[j1offs + i]);
    }
  }
}

#define FS_COL_SET(_col, _joffs, _ioffs)                       \
  do {                                                         \
    unsigned gx;                                               \
    unsigned gy;                                               \
    gx = gx_buf[((j + (_joffs)) & 7) * stride + i + (_ioffs)]; \
    gy = gy_buf[((j + (_joffs)) & 7) * stride + i + (_ioffs)]; \
    col_sums_gx2[(_col)] = gx * (double)gx;                    \
    col_sums_gy2[(_col)] = gy * (double)gy;                    \
    col_sums_gxgy[(_col)] = gx * (double)gy;                   \
  } while (0)

#define FS_COL_ADD(_col, _joffs, _ioffs)                       \
  do {                                                         \
    unsigned gx;                                               \
    unsigned gy;                                               \
    gx = gx_buf[((j + (_joffs)) & 7) * stride + i + (_ioffs)]; \
    gy = gy_buf[((j + (_joffs)) & 7) * stride + i + (_ioffs)]; \
    col_sums_gx2[(_col)] += gx * (double)gx;                   \
    col_sums_gy2[(_col)] += gy * (double)gy;                   \
    col_sums_gxgy[(_col)] += gx * (double)gy;                  \
  } while (0)

#define FS_COL_SUB(_col, _joffs, _ioffs)                       \
  do {                                                         \
    unsigned gx;                                               \
    unsigned gy;                                               \
    gx = gx_buf[((j + (_joffs)) & 7) * stride + i + (_ioffs)]; \
    gy = gy_buf[((j + (_joffs)) & 7) * stride + i + (_ioffs)]; \
    col_sums_gx2[(_col)] -= gx * (double)gx;                   \
    col_sums_gy2[(_col)] -= gy * (double)gy;                   \
    col_sums_gxgy[(_col)] -= gx * (double)gy;                  \
  } while (0)

#define FS_COL_COPY(_col1, _col2)                    \
  do {                                               \
    col_sums_gx2[(_col1)] = col_sums_gx2[(_col2)];   \
    col_sums_gy2[(_col1)] = col_sums_gy2[(_col2)];   \
    col_sums_gxgy[(_col1)] = col_sums_gxgy[(_col2)]; \
  } while (0)

#define FS_COL_HALVE(_col1, _col2)                         \
  do {                                                     \
    col_sums_gx2[(_col1)] = col_sums_gx2[(_col2)] * 0.5;   \
    col_sums_gy2[(_col1)] = col_sums_gy2[(_col2)] * 0.5;   \
    col_sums_gxgy[(_col1)] = col_sums_gxgy[(_col2)] * 0.5; \
  } while (0)

#define FS_COL_DOUBLE(_col1, _col2)                      \
  do {                                                   \
    col_sums_gx2[(_col1)] = col_sums_gx2[(_col2)] * 2;   \
    col_sums_gy2[(_col1)] = col_sums_gy2[(_col2)] * 2;   \
    col_sums_gxgy[(_col1)] = col_sums_gxgy[(_col2)] * 2; \
  } while (0)

static void fs_calc_structure(fs_ctx *_ctx, int _l, int bit_depth) {
  uint32_t *im1;
  uint32_t *im2;
  unsigned *gx_buf;
  unsigned *gy_buf;
  double *ssim;
  double col_sums_gx2[8];
  double col_sums_gy2[8];
  double col_sums_gxgy[8];
  double c2;
  int stride;
  int w;
  int h;
  int i;
  int j;
  double ssim_c2 = SSIM_C2;
#if CONFIG_VP9_HIGHBITDEPTH
  if (bit_depth == 10) ssim_c2 = SSIM_C2_10;
  if (bit_depth == 12) ssim_c2 = SSIM_C2_12;
#else
  assert(bit_depth == 8);
  (void)bit_depth;
#endif

  w = _ctx->level[_l].w;
  h = _ctx->level[_l].h;
  im1 = _ctx->level[_l].im1;
  im2 = _ctx->level[_l].im2;
  ssim = _ctx->level[_l].ssim;
  gx_buf = _ctx->col_buf;
  stride = w + 8;
  gy_buf = gx_buf + 8 * stride;
  memset(gx_buf, 0, 2 * 8 * stride * sizeof(*gx_buf));
  c2 = ssim_c2 * (1 << 4 * _l) * 16 * 104;
  for (j = 0; j < h + 4; j++) {
    if (j < h - 1) {
      for (i = 0; i < w - 1; i++) {
        int64_t g1;
        int64_t g2;
        int64_t gx;
        int64_t gy;
        g1 = labs((int64_t)im1[(j + 1) * w + i + 1] - (int64_t)im1[j * w + i]);
        g2 = labs((int64_t)im1[(j + 1) * w + i] - (int64_t)im1[j * w + i + 1]);
        gx = 4 * FS_MAXI(g1, g2) + FS_MINI(g1, g2);
        g1 = labs((int64_t)im2[(j + 1) * w + i + 1] - (int64_t)im2[j * w + i]);
        g2 = labs((int64_t)im2[(j + 1) * w + i] - (int64_t)im2[j * w + i + 1]);
        gy = ((int64_t)4 * FS_MAXI(g1, g2) + FS_MINI(g1, g2));
        gx_buf[(j & 7) * stride + i + 4] = (uint32_t)gx;
        gy_buf[(j & 7) * stride + i + 4] = (uint32_t)gy;
      }
    } else {
      memset(gx_buf + (j & 7) * stride, 0, stride * sizeof(*gx_buf));
      memset(gy_buf + (j & 7) * stride, 0, stride * sizeof(*gy_buf));
    }
    if (j >= 4) {
      int k;
      col_sums_gx2[3] = col_sums_gx2[2] = col_sums_gx2[1] = col_sums_gx2[0] = 0;
      col_sums_gy2[3] = col_sums_gy2[2] = col_sums_gy2[1] = col_sums_gy2[0] = 0;
      col_sums_gxgy[3] = col_sums_gxgy[2] = col_sums_gxgy[1] =
          col_sums_gxgy[0] = 0;
      for (i = 4; i < 8; i++) {
        FS_COL_SET(i, -1, 0);
        FS_COL_ADD(i, 0, 0);
        for (k = 1; k < 8 - i; k++) {
          FS_COL_DOUBLE(i, i);
          FS_COL_ADD(i, -k - 1, 0);
          FS_COL_ADD(i, k, 0);
        }
      }
      for (i = 0; i < w; i++) {
        double mugx2;
        double mugy2;
        double mugxgy;
        mugx2 = col_sums_gx2[0];
        for (k = 1; k < 8; k++) mugx2 += col_sums_gx2[k];
        mugy2 = col_sums_gy2[0];
        for (k = 1; k < 8; k++) mugy2 += col_sums_gy2[k];
        mugxgy = col_sums_gxgy[0];
        for (k = 1; k < 8; k++) mugxgy += col_sums_gxgy[k];
        ssim[(j - 4) * w + i] = (2 * mugxgy + c2) / (mugx2 + mugy2 + c2);
        if (i + 1 < w) {
          FS_COL_SET(0, -1, 1);
          FS_COL_ADD(0, 0, 1);
          FS_COL_SUB(2, -3, 2);
          FS_COL_SUB(2, 2, 2);
          FS_COL_HALVE(1, 2);
          FS_COL_SUB(3, -4, 3);
          FS_COL_SUB(3, 3, 3);
          FS_COL_HALVE(2, 3);
          FS_COL_COPY(3, 4);
          FS_COL_DOUBLE(4, 5);
          FS_COL_ADD(4, -4, 5);
          FS_COL_ADD(4, 3, 5);
          FS_COL_DOUBLE(5, 6);
          FS_COL_ADD(5, -3, 6);
          FS_COL_ADD(5, 2, 6);
          FS_COL_DOUBLE(6, 7);
          FS_COL_ADD(6, -2, 7);
          FS_COL_ADD(6, 1, 7);
          FS_COL_SET(7, -1, 8);
          FS_COL_ADD(7, 0, 8);
        }
      }
    }
  }
}

#define FS_NLEVELS (4)

/*These weights were derived from the default weights found in Wang's original
 Matlab implementation: {0.0448, 0.2856, 0.2363, 0.1333}.
 We drop the finest scale and renormalize the rest to sum to 1.*/

static const double FS_WEIGHTS[FS_NLEVELS] = {
  0.2989654541015625, 0.3141326904296875, 0.2473602294921875, 0.1395416259765625
};

static double fs_average(fs_ctx *_ctx, int _l) {
  double *ssim;
  double ret;
  int w;
  int h;
  int i;
  int j;
  w = _ctx->level[_l].w;
  h = _ctx->level[_l].h;
  ssim = _ctx->level[_l].ssim;
  ret = 0;
  for (j = 0; j < h; j++)
    for (i = 0; i < w; i++) ret += ssim[j * w + i];
  return pow(ret / (w * h), FS_WEIGHTS[_l]);
}

static double convert_ssim_db(double _ssim, double _weight) {
  assert(_weight >= _ssim);
  if ((_weight - _ssim) < 1e-10) return MAX_SSIM_DB;
  return 10 * (log10(_weight) - log10(_weight - _ssim));
}

static double calc_ssim(const uint8_t *_src, int _systride, const uint8_t *_dst,
                        int _dystride, int _w, int _h, uint32_t _bd,
                        uint32_t _shift) {
  fs_ctx ctx;
  double ret;
  int l;
  ret = 1;
  if (fs_ctx_init(&ctx, _w, _h, FS_NLEVELS)) return 99.0;
  fs_downsample_level0(&ctx, _src, _systride, _dst, _dystride, _w, _h, _bd,
                       _shift);
  for (l = 0; l < FS_NLEVELS - 1; l++) {
    fs_calc_structure(&ctx, l, _bd);
    ret *= fs_average(&ctx, l);
    fs_downsample_level(&ctx, l + 1);
  }
  fs_calc_structure(&ctx, l, _bd);
  fs_apply_luminance(&ctx, l, _bd);
  ret *= fs_average(&ctx, l);
  fs_ctx_clear(&ctx);
  return ret;
}

double vpx_calc_fastssim(const YV12_BUFFER_CONFIG *source,
                         const YV12_BUFFER_CONFIG *dest, double *ssim_y,
                         double *ssim_u, double *ssim_v, uint32_t bd,
                         uint32_t in_bd) {
  double ssimv;
  uint32_t bd_shift = 0;
  vpx_clear_system_state();
  assert(bd >= in_bd);
  bd_shift = bd - in_bd;

  *ssim_y = calc_ssim(source->y_buffer, source->y_stride, dest->y_buffer,
                      dest->y_stride, source->y_crop_width,
                      source->y_crop_height, in_bd, bd_shift);
  *ssim_u = calc_ssim(source->u_buffer, source->uv_stride, dest->u_buffer,
                      dest->uv_stride, source->uv_crop_width,
                      source->uv_crop_height, in_bd, bd_shift);
  *ssim_v = calc_ssim(source->v_buffer, source->uv_stride, dest->v_buffer,
                      dest->uv_stride, source->uv_crop_width,
                      source->uv_crop_height, in_bd, bd_shift);

  ssimv = (*ssim_y) * .8 + .1 * ((*ssim_u) + (*ssim_v));
  return convert_ssim_db(ssimv, 1.0);
}
