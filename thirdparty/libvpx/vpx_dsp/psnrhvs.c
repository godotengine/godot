/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 *
 *  This code was originally written by: Gregory Maxwell, at the Daala
 *  project.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/ssim.h"
#include "vpx_ports/system_state.h"
#include "vpx_dsp/psnr.h"

#if !defined(M_PI)
#define M_PI (3.141592653589793238462643)
#endif
#include <string.h>

static void od_bin_fdct8x8(tran_low_t *y, int ystride, const int16_t *x,
                           int xstride) {
  int i, j;
  (void)xstride;
  vpx_fdct8x8(x, y, ystride);
  for (i = 0; i < 8; i++)
    for (j = 0; j < 8; j++)
      *(y + ystride * i + j) = (*(y + ystride * i + j) + 4) >> 3;
}
#if CONFIG_VP9_HIGHBITDEPTH
static void hbd_od_bin_fdct8x8(tran_low_t *y, int ystride, const int16_t *x,
                               int xstride) {
  int i, j;
  (void)xstride;
  vpx_highbd_fdct8x8(x, y, ystride);
  for (i = 0; i < 8; i++)
    for (j = 0; j < 8; j++)
      *(y + ystride * i + j) = (*(y + ystride * i + j) + 4) >> 3;
}
#endif

/* Normalized inverse quantization matrix for 8x8 DCT at the point of
 * transparency. This is not the JPEG based matrix from the paper,
 this one gives a slightly higher MOS agreement.*/
static const double csf_y[8][8] = {
  { 1.6193873005, 2.2901594831, 2.08509755623, 1.48366094411, 1.00227514334,
    0.678296995242, 0.466224900598, 0.3265091542 },
  { 2.2901594831, 1.94321815382, 2.04793073064, 1.68731108984, 1.2305666963,
    0.868920337363, 0.61280991668, 0.436405793551 },
  { 2.08509755623, 2.04793073064, 1.34329019223, 1.09205635862, 0.875748795257,
    0.670882927016, 0.501731932449, 0.372504254596 },
  { 1.48366094411, 1.68731108984, 1.09205635862, 0.772819797575, 0.605636379554,
    0.48309405692, 0.380429446972, 0.295774038565 },
  { 1.00227514334, 1.2305666963, 0.875748795257, 0.605636379554, 0.448996256676,
    0.352889268808, 0.283006984131, 0.226951348204 },
  { 0.678296995242, 0.868920337363, 0.670882927016, 0.48309405692,
    0.352889268808, 0.27032073436, 0.215017739696, 0.17408067321 },
  { 0.466224900598, 0.61280991668, 0.501731932449, 0.380429446972,
    0.283006984131, 0.215017739696, 0.168869545842, 0.136153931001 },
  { 0.3265091542, 0.436405793551, 0.372504254596, 0.295774038565,
    0.226951348204, 0.17408067321, 0.136153931001, 0.109083846276 }
};
static const double csf_cb420[8][8] = {
  { 1.91113096927, 2.46074210438, 1.18284184739, 1.14982565193, 1.05017074788,
    0.898018824055, 0.74725392039, 0.615105596242 },
  { 2.46074210438, 1.58529308355, 1.21363250036, 1.38190029285, 1.33100189972,
    1.17428548929, 0.996404342439, 0.830890433625 },
  { 1.18284184739, 1.21363250036, 0.978712413627, 1.02624506078, 1.03145147362,
    0.960060382087, 0.849823426169, 0.731221236837 },
  { 1.14982565193, 1.38190029285, 1.02624506078, 0.861317501629, 0.801821139099,
    0.751437590932, 0.685398513368, 0.608694761374 },
  { 1.05017074788, 1.33100189972, 1.03145147362, 0.801821139099, 0.676555426187,
    0.605503172737, 0.55002013668, 0.495804539034 },
  { 0.898018824055, 1.17428548929, 0.960060382087, 0.751437590932,
    0.605503172737, 0.514674450957, 0.454353482512, 0.407050308965 },
  { 0.74725392039, 0.996404342439, 0.849823426169, 0.685398513368,
    0.55002013668, 0.454353482512, 0.389234902883, 0.342353999733 },
  { 0.615105596242, 0.830890433625, 0.731221236837, 0.608694761374,
    0.495804539034, 0.407050308965, 0.342353999733, 0.295530605237 }
};
static const double csf_cr420[8][8] = {
  { 2.03871978502, 2.62502345193, 1.26180942886, 1.11019789803, 1.01397751469,
    0.867069376285, 0.721500455585, 0.593906509971 },
  { 2.62502345193, 1.69112867013, 1.17180569821, 1.3342742857, 1.28513006198,
    1.13381474809, 0.962064122248, 0.802254508198 },
  { 1.26180942886, 1.17180569821, 0.944981930573, 0.990876405848,
    0.995903384143, 0.926972725286, 0.820534991409, 0.706020324706 },
  { 1.11019789803, 1.3342742857, 0.990876405848, 0.831632933426, 0.77418706195,
    0.725539939514, 0.661776842059, 0.587716619023 },
  { 1.01397751469, 1.28513006198, 0.995903384143, 0.77418706195, 0.653238524286,
    0.584635025748, 0.531064164893, 0.478717061273 },
  { 0.867069376285, 1.13381474809, 0.926972725286, 0.725539939514,
    0.584635025748, 0.496936637883, 0.438694579826, 0.393021669543 },
  { 0.721500455585, 0.962064122248, 0.820534991409, 0.661776842059,
    0.531064164893, 0.438694579826, 0.375820256136, 0.330555063063 },
  { 0.593906509971, 0.802254508198, 0.706020324706, 0.587716619023,
    0.478717061273, 0.393021669543, 0.330555063063, 0.285345396658 }
};

static double convert_score_db(double _score, double _weight, int bit_depth) {
  int16_t pix_max = 255;
  assert(_score * _weight >= 0.0);
  if (bit_depth == 10)
    pix_max = 1023;
  else if (bit_depth == 12)
    pix_max = 4095;

  if (_weight * _score < pix_max * pix_max * 1e-10) return MAX_PSNR;
  return 10 * (log10(pix_max * pix_max) - log10(_weight * _score));
}

static double calc_psnrhvs(const unsigned char *src, int _systride,
                           const unsigned char *dst, int _dystride, double _par,
                           int _w, int _h, int _step, const double _csf[8][8],
                           uint32_t bit_depth, uint32_t _shift) {
  double ret;
  const uint8_t *_src8 = src;
  const uint8_t *_dst8 = dst;
  const uint16_t *_src16 = CONVERT_TO_SHORTPTR(src);
  const uint16_t *_dst16 = CONVERT_TO_SHORTPTR(dst);
  DECLARE_ALIGNED(16, int16_t, dct_s[8 * 8]);
  DECLARE_ALIGNED(16, int16_t, dct_d[8 * 8]);
  DECLARE_ALIGNED(16, tran_low_t, dct_s_coef[8 * 8]);
  DECLARE_ALIGNED(16, tran_low_t, dct_d_coef[8 * 8]);
  double mask[8][8];
  int pixels;
  int x;
  int y;
  (void)_par;
  ret = pixels = 0;

  /*In the PSNR-HVS-M paper[1] the authors describe the construction of
   their masking table as "we have used the quantization table for the
   color component Y of JPEG [6] that has been also obtained on the
   basis of CSF. Note that the values in quantization table JPEG have
   been normalized and then squared." Their CSF matrix (from PSNR-HVS)
   was also constructed from the JPEG matrices. I can not find any obvious
   scheme of normalizing to produce their table, but if I multiply their
   CSF by 0.3885746225901003 and square the result I get their masking table.
   I have no idea where this constant comes from, but deviating from it
   too greatly hurts MOS agreement.

   [1] Nikolay Ponomarenko, Flavia Silvestri, Karen Egiazarian, Marco Carli,
   Jaakko Astola, Vladimir Lukin, "On between-coefficient contrast masking
   of DCT basis functions", CD-ROM Proceedings of the Third
   International Workshop on Video Processing and Quality Metrics for Consumer
   Electronics VPQM-07, Scottsdale, Arizona, USA, 25-26 January, 2007, 4 p.

   Suggested in aomedia issue #2363:
   0.3885746225901003 is a reciprocal of the maximum coefficient (2.573509)
   of the old JPEG based matrix from the paper. Since you are not using that,
   divide by actual maximum coefficient. */
  for (x = 0; x < 8; x++)
    for (y = 0; y < 8; y++)
      mask[x][y] = (_csf[x][y] / _csf[1][0]) * (_csf[x][y] / _csf[1][0]);
  for (y = 0; y < _h - 7; y += _step) {
    for (x = 0; x < _w - 7; x += _step) {
      int i;
      int j;
      double s_means[4];
      double d_means[4];
      double s_vars[4];
      double d_vars[4];
      double s_gmean = 0;
      double d_gmean = 0;
      double s_gvar = 0;
      double d_gvar = 0;
      double s_mask = 0;
      double d_mask = 0;
      for (i = 0; i < 4; i++)
        s_means[i] = d_means[i] = s_vars[i] = d_vars[i] = 0;
      for (i = 0; i < 8; i++) {
        for (j = 0; j < 8; j++) {
          int sub = ((i & 12) >> 2) + ((j & 12) >> 1);
          if (bit_depth == 8 && _shift == 0) {
            dct_s[i * 8 + j] = _src8[(y + i) * _systride + (j + x)];
            dct_d[i * 8 + j] = _dst8[(y + i) * _dystride + (j + x)];
          } else if (bit_depth == 10 || bit_depth == 12) {
            dct_s[i * 8 + j] = _src16[(y + i) * _systride + (j + x)] >> _shift;
            dct_d[i * 8 + j] = _dst16[(y + i) * _dystride + (j + x)] >> _shift;
          }
          s_gmean += dct_s[i * 8 + j];
          d_gmean += dct_d[i * 8 + j];
          s_means[sub] += dct_s[i * 8 + j];
          d_means[sub] += dct_d[i * 8 + j];
        }
      }
      s_gmean /= 64.f;
      d_gmean /= 64.f;
      for (i = 0; i < 4; i++) s_means[i] /= 16.f;
      for (i = 0; i < 4; i++) d_means[i] /= 16.f;
      for (i = 0; i < 8; i++) {
        for (j = 0; j < 8; j++) {
          int sub = ((i & 12) >> 2) + ((j & 12) >> 1);
          s_gvar += (dct_s[i * 8 + j] - s_gmean) * (dct_s[i * 8 + j] - s_gmean);
          d_gvar += (dct_d[i * 8 + j] - d_gmean) * (dct_d[i * 8 + j] - d_gmean);
          s_vars[sub] += (dct_s[i * 8 + j] - s_means[sub]) *
                         (dct_s[i * 8 + j] - s_means[sub]);
          d_vars[sub] += (dct_d[i * 8 + j] - d_means[sub]) *
                         (dct_d[i * 8 + j] - d_means[sub]);
        }
      }
      s_gvar *= 1 / 63.f * 64;
      d_gvar *= 1 / 63.f * 64;
      for (i = 0; i < 4; i++) s_vars[i] *= 1 / 15.f * 16;
      for (i = 0; i < 4; i++) d_vars[i] *= 1 / 15.f * 16;
      if (s_gvar > 0)
        s_gvar = (s_vars[0] + s_vars[1] + s_vars[2] + s_vars[3]) / s_gvar;
      if (d_gvar > 0)
        d_gvar = (d_vars[0] + d_vars[1] + d_vars[2] + d_vars[3]) / d_gvar;
#if CONFIG_VP9_HIGHBITDEPTH
      if (bit_depth == 10 || bit_depth == 12) {
        hbd_od_bin_fdct8x8(dct_s_coef, 8, dct_s, 8);
        hbd_od_bin_fdct8x8(dct_d_coef, 8, dct_d, 8);
      }
#endif
      if (bit_depth == 8) {
        od_bin_fdct8x8(dct_s_coef, 8, dct_s, 8);
        od_bin_fdct8x8(dct_d_coef, 8, dct_d, 8);
      }
      for (i = 0; i < 8; i++)
        for (j = (i == 0); j < 8; j++)
          s_mask += dct_s_coef[i * 8 + j] * dct_s_coef[i * 8 + j] * mask[i][j];
      for (i = 0; i < 8; i++)
        for (j = (i == 0); j < 8; j++)
          d_mask += dct_d_coef[i * 8 + j] * dct_d_coef[i * 8 + j] * mask[i][j];
      s_mask = sqrt(s_mask * s_gvar) / 32.f;
      d_mask = sqrt(d_mask * d_gvar) / 32.f;
      if (d_mask > s_mask) s_mask = d_mask;
      for (i = 0; i < 8; i++) {
        for (j = 0; j < 8; j++) {
          double err;
          err = fabs((double)(dct_s_coef[i * 8 + j] - dct_d_coef[i * 8 + j]));
          if (i != 0 || j != 0)
            err = err < s_mask / mask[i][j] ? 0 : err - s_mask / mask[i][j];
          ret += (err * _csf[i][j]) * (err * _csf[i][j]);
          pixels++;
        }
      }
    }
  }
  if (pixels <= 0) return 0;
  ret /= pixels;
  return ret;
}

double vpx_psnrhvs(const YV12_BUFFER_CONFIG *src,
                   const YV12_BUFFER_CONFIG *dest, double *y_psnrhvs,
                   double *u_psnrhvs, double *v_psnrhvs, uint32_t bd,
                   uint32_t in_bd) {
  double psnrhvs;
  const double par = 1.0;
  const int step = 7;
  uint32_t bd_shift = 0;
  vpx_clear_system_state();

  assert(bd == 8 || bd == 10 || bd == 12);
  assert(bd >= in_bd);

  bd_shift = bd - in_bd;

  *y_psnrhvs = calc_psnrhvs(src->y_buffer, src->y_stride, dest->y_buffer,
                            dest->y_stride, par, src->y_crop_width,
                            src->y_crop_height, step, csf_y, bd, bd_shift);
  *u_psnrhvs = calc_psnrhvs(src->u_buffer, src->uv_stride, dest->u_buffer,
                            dest->uv_stride, par, src->uv_crop_width,
                            src->uv_crop_height, step, csf_cb420, bd, bd_shift);
  *v_psnrhvs = calc_psnrhvs(src->v_buffer, src->uv_stride, dest->v_buffer,
                            dest->uv_stride, par, src->uv_crop_width,
                            src->uv_crop_height, step, csf_cr420, bd, bd_shift);
  psnrhvs = (*y_psnrhvs) * .8 + .1 * ((*u_psnrhvs) + (*v_psnrhvs));
  return convert_score_db(psnrhvs, 1.0, in_bd);
}
