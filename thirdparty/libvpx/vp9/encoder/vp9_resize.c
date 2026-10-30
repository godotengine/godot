/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./vpx_config.h"
#if CONFIG_VP9_HIGHBITDEPTH
#include "vpx_dsp/vpx_dsp_common.h"
#endif  // CONFIG_VP9_HIGHBITDEPTH
#include "vpx_ports/mem.h"
#include "vp9/common/vp9_common.h"
#include "vp9/encoder/vp9_resize.h"

#define FILTER_BITS 7

#define INTERP_TAPS 8
#define SUBPEL_BITS 5
#define SUBPEL_MASK ((1 << SUBPEL_BITS) - 1)
#define INTERP_PRECISION_BITS 32

typedef int16_t interp_kernel[INTERP_TAPS];

// Filters for interpolation (0.5-band) - note this also filters integer pels.
static const interp_kernel filteredinterp_filters500[(1 << SUBPEL_BITS)] = {
  { -3, 0, 35, 64, 35, 0, -3, 0 },    { -3, -1, 34, 64, 36, 1, -3, 0 },
  { -3, -1, 32, 64, 38, 1, -3, 0 },   { -2, -2, 31, 63, 39, 2, -3, 0 },
  { -2, -2, 29, 63, 41, 2, -3, 0 },   { -2, -2, 28, 63, 42, 3, -4, 0 },
  { -2, -3, 27, 63, 43, 4, -4, 0 },   { -2, -3, 25, 62, 45, 5, -4, 0 },
  { -2, -3, 24, 62, 46, 5, -4, 0 },   { -2, -3, 23, 61, 47, 6, -4, 0 },
  { -2, -3, 21, 60, 49, 7, -4, 0 },   { -1, -4, 20, 60, 50, 8, -4, -1 },
  { -1, -4, 19, 59, 51, 9, -4, -1 },  { -1, -4, 17, 58, 52, 10, -4, 0 },
  { -1, -4, 16, 57, 53, 12, -4, -1 }, { -1, -4, 15, 56, 54, 13, -4, -1 },
  { -1, -4, 14, 55, 55, 14, -4, -1 }, { -1, -4, 13, 54, 56, 15, -4, -1 },
  { -1, -4, 12, 53, 57, 16, -4, -1 }, { 0, -4, 10, 52, 58, 17, -4, -1 },
  { -1, -4, 9, 51, 59, 19, -4, -1 },  { -1, -4, 8, 50, 60, 20, -4, -1 },
  { 0, -4, 7, 49, 60, 21, -3, -2 },   { 0, -4, 6, 47, 61, 23, -3, -2 },
  { 0, -4, 5, 46, 62, 24, -3, -2 },   { 0, -4, 5, 45, 62, 25, -3, -2 },
  { 0, -4, 4, 43, 63, 27, -3, -2 },   { 0, -4, 3, 42, 63, 28, -2, -2 },
  { 0, -3, 2, 41, 63, 29, -2, -2 },   { 0, -3, 2, 39, 63, 31, -2, -2 },
  { 0, -3, 1, 38, 64, 32, -1, -3 },   { 0, -3, 1, 36, 64, 34, -1, -3 }
};

// Filters for interpolation (0.625-band) - note this also filters integer pels.
static const interp_kernel filteredinterp_filters625[(1 << SUBPEL_BITS)] = {
  { -1, -8, 33, 80, 33, -8, -1, 0 }, { -1, -8, 30, 80, 35, -8, -1, 1 },
  { -1, -8, 28, 80, 37, -7, -2, 1 }, { 0, -8, 26, 79, 39, -7, -2, 1 },
  { 0, -8, 24, 79, 41, -7, -2, 1 },  { 0, -8, 22, 78, 43, -6, -2, 1 },
  { 0, -8, 20, 78, 45, -5, -3, 1 },  { 0, -8, 18, 77, 48, -5, -3, 1 },
  { 0, -8, 16, 76, 50, -4, -3, 1 },  { 0, -8, 15, 75, 52, -3, -4, 1 },
  { 0, -7, 13, 74, 54, -3, -4, 1 },  { 0, -7, 11, 73, 56, -2, -4, 1 },
  { 0, -7, 10, 71, 58, -1, -4, 1 },  { 1, -7, 8, 70, 60, 0, -5, 1 },
  { 1, -6, 6, 68, 62, 1, -5, 1 },    { 1, -6, 5, 67, 63, 2, -5, 1 },
  { 1, -6, 4, 65, 65, 4, -6, 1 },    { 1, -5, 2, 63, 67, 5, -6, 1 },
  { 1, -5, 1, 62, 68, 6, -6, 1 },    { 1, -5, 0, 60, 70, 8, -7, 1 },
  { 1, -4, -1, 58, 71, 10, -7, 0 },  { 1, -4, -2, 56, 73, 11, -7, 0 },
  { 1, -4, -3, 54, 74, 13, -7, 0 },  { 1, -4, -3, 52, 75, 15, -8, 0 },
  { 1, -3, -4, 50, 76, 16, -8, 0 },  { 1, -3, -5, 48, 77, 18, -8, 0 },
  { 1, -3, -5, 45, 78, 20, -8, 0 },  { 1, -2, -6, 43, 78, 22, -8, 0 },
  { 1, -2, -7, 41, 79, 24, -8, 0 },  { 1, -2, -7, 39, 79, 26, -8, 0 },
  { 1, -2, -7, 37, 80, 28, -8, -1 }, { 1, -1, -8, 35, 80, 30, -8, -1 },
};

// Filters for interpolation (0.75-band) - note this also filters integer pels.
static const interp_kernel filteredinterp_filters750[(1 << SUBPEL_BITS)] = {
  { 2, -11, 25, 96, 25, -11, 2, 0 }, { 2, -11, 22, 96, 28, -11, 2, 0 },
  { 2, -10, 19, 95, 31, -11, 2, 0 }, { 2, -10, 17, 95, 34, -12, 2, 0 },
  { 2, -9, 14, 94, 37, -12, 2, 0 },  { 2, -8, 12, 93, 40, -12, 1, 0 },
  { 2, -8, 9, 92, 43, -12, 1, 1 },   { 2, -7, 7, 91, 46, -12, 1, 0 },
  { 2, -7, 5, 90, 49, -12, 1, 0 },   { 2, -6, 3, 88, 52, -12, 0, 1 },
  { 2, -5, 1, 86, 55, -12, 0, 1 },   { 2, -5, -1, 84, 58, -11, 0, 1 },
  { 2, -4, -2, 82, 61, -11, -1, 1 }, { 2, -4, -4, 80, 64, -10, -1, 1 },
  { 1, -3, -5, 77, 67, -9, -1, 1 },  { 1, -3, -6, 75, 70, -8, -2, 1 },
  { 1, -2, -7, 72, 72, -7, -2, 1 },  { 1, -2, -8, 70, 75, -6, -3, 1 },
  { 1, -1, -9, 67, 77, -5, -3, 1 },  { 1, -1, -10, 64, 80, -4, -4, 2 },
  { 1, -1, -11, 61, 82, -2, -4, 2 }, { 1, 0, -11, 58, 84, -1, -5, 2 },
  { 1, 0, -12, 55, 86, 1, -5, 2 },   { 1, 0, -12, 52, 88, 3, -6, 2 },
  { 0, 1, -12, 49, 90, 5, -7, 2 },   { 0, 1, -12, 46, 91, 7, -7, 2 },
  { 1, 1, -12, 43, 92, 9, -8, 2 },   { 0, 1, -12, 40, 93, 12, -8, 2 },
  { 0, 2, -12, 37, 94, 14, -9, 2 },  { 0, 2, -12, 34, 95, 17, -10, 2 },
  { 0, 2, -11, 31, 95, 19, -10, 2 }, { 0, 2, -11, 28, 96, 22, -11, 2 }
};

// Filters for interpolation (0.875-band) - note this also filters integer pels.
static const interp_kernel filteredinterp_filters875[(1 << SUBPEL_BITS)] = {
  { 3, -8, 13, 112, 13, -8, 3, 0 },   { 3, -7, 10, 112, 17, -9, 3, -1 },
  { 2, -6, 7, 111, 21, -9, 3, -1 },   { 2, -5, 4, 111, 24, -10, 3, -1 },
  { 2, -4, 1, 110, 28, -11, 3, -1 },  { 1, -3, -1, 108, 32, -12, 4, -1 },
  { 1, -2, -3, 106, 36, -13, 4, -1 }, { 1, -1, -6, 105, 40, -14, 4, -1 },
  { 1, -1, -7, 102, 44, -14, 4, -1 }, { 1, 0, -9, 100, 48, -15, 4, -1 },
  { 1, 1, -11, 97, 53, -16, 4, -1 },  { 0, 1, -12, 95, 57, -16, 4, -1 },
  { 0, 2, -13, 91, 61, -16, 4, -1 },  { 0, 2, -14, 88, 65, -16, 4, -1 },
  { 0, 3, -15, 84, 69, -17, 4, 0 },   { 0, 3, -16, 81, 73, -16, 3, 0 },
  { 0, 3, -16, 77, 77, -16, 3, 0 },   { 0, 3, -16, 73, 81, -16, 3, 0 },
  { 0, 4, -17, 69, 84, -15, 3, 0 },   { -1, 4, -16, 65, 88, -14, 2, 0 },
  { -1, 4, -16, 61, 91, -13, 2, 0 },  { -1, 4, -16, 57, 95, -12, 1, 0 },
  { -1, 4, -16, 53, 97, -11, 1, 1 },  { -1, 4, -15, 48, 100, -9, 0, 1 },
  { -1, 4, -14, 44, 102, -7, -1, 1 }, { -1, 4, -14, 40, 105, -6, -1, 1 },
  { -1, 4, -13, 36, 106, -3, -2, 1 }, { -1, 4, -12, 32, 108, -1, -3, 1 },
  { -1, 3, -11, 28, 110, 1, -4, 2 },  { -1, 3, -10, 24, 111, 4, -5, 2 },
  { -1, 3, -9, 21, 111, 7, -6, 2 },   { -1, 3, -9, 17, 112, 10, -7, 3 }
};

// Filters for interpolation (full-band) - no filtering for integer pixels
static const interp_kernel filteredinterp_filters1000[(1 << SUBPEL_BITS)] = {
  { 0, 0, 0, 128, 0, 0, 0, 0 },        { 0, 1, -3, 128, 3, -1, 0, 0 },
  { -1, 2, -6, 127, 7, -2, 1, 0 },     { -1, 3, -9, 126, 12, -4, 1, 0 },
  { -1, 4, -12, 125, 16, -5, 1, 0 },   { -1, 4, -14, 123, 20, -6, 2, 0 },
  { -1, 5, -15, 120, 25, -8, 2, 0 },   { -1, 5, -17, 118, 30, -9, 3, -1 },
  { -1, 6, -18, 114, 35, -10, 3, -1 }, { -1, 6, -19, 111, 41, -12, 3, -1 },
  { -1, 6, -20, 107, 46, -13, 4, -1 }, { -1, 6, -21, 103, 52, -14, 4, -1 },
  { -1, 6, -21, 99, 57, -16, 5, -1 },  { -1, 6, -21, 94, 63, -17, 5, -1 },
  { -1, 6, -20, 89, 68, -18, 5, -1 },  { -1, 6, -20, 84, 73, -19, 6, -1 },
  { -1, 6, -20, 79, 79, -20, 6, -1 },  { -1, 6, -19, 73, 84, -20, 6, -1 },
  { -1, 5, -18, 68, 89, -20, 6, -1 },  { -1, 5, -17, 63, 94, -21, 6, -1 },
  { -1, 5, -16, 57, 99, -21, 6, -1 },  { -1, 4, -14, 52, 103, -21, 6, -1 },
  { -1, 4, -13, 46, 107, -20, 6, -1 }, { -1, 3, -12, 41, 111, -19, 6, -1 },
  { -1, 3, -10, 35, 114, -18, 6, -1 }, { -1, 3, -9, 30, 118, -17, 5, -1 },
  { 0, 2, -8, 25, 120, -15, 5, -1 },   { 0, 2, -6, 20, 123, -14, 4, -1 },
  { 0, 1, -5, 16, 125, -12, 4, -1 },   { 0, 1, -4, 12, 126, -9, 3, -1 },
  { 0, 1, -2, 7, 127, -6, 2, -1 },     { 0, 0, -1, 3, 128, -3, 1, 0 }
};

// Filters for factor of 2 downsampling.
static const int16_t vp9_down2_symeven_half_filter[] = { 56, 12, -3, -1 };
static const int16_t vp9_down2_symodd_half_filter[] = { 64, 35, 0, -3 };

static const interp_kernel *choose_interp_filter(int inlength, int outlength) {
  int outlength16 = outlength * 16;
  if (outlength16 >= inlength * 16)
    return filteredinterp_filters1000;
  else if (outlength16 >= inlength * 13)
    return filteredinterp_filters875;
  else if (outlength16 >= inlength * 11)
    return filteredinterp_filters750;
  else if (outlength16 >= inlength * 9)
    return filteredinterp_filters625;
  else
    return filteredinterp_filters500;
}

static void interpolate(const uint8_t *const input, int inlength,
                        uint8_t *output, int outlength) {
  const int64_t delta =
      (((uint64_t)inlength << 32) + outlength / 2) / outlength;
  const int64_t offset =
      inlength > outlength
          ? (((int64_t)(inlength - outlength) << 31) + outlength / 2) /
                outlength
          : -(((int64_t)(outlength - inlength) << 31) + outlength / 2) /
                outlength;
  uint8_t *optr = output;
  int x, x1, x2, sum, k, int_pel, sub_pel;
  int64_t y;

  const interp_kernel *interp_filters =
      choose_interp_filter(inlength, outlength);

  x = 0;
  y = offset;
  while ((y >> INTERP_PRECISION_BITS) < (INTERP_TAPS / 2 - 1)) {
    x++;
    y += delta;
  }
  x1 = x;
  x = outlength - 1;
  y = delta * x + offset;
  while ((y >> INTERP_PRECISION_BITS) + (int64_t)(INTERP_TAPS / 2) >=
         inlength) {
    x--;
    y -= delta;
  }
  x2 = x;
  if (x1 > x2) {
    for (x = 0, y = offset; x < outlength; ++x, y += delta) {
      const int16_t *filter;
      int_pel = y >> INTERP_PRECISION_BITS;
      sub_pel = (y >> (INTERP_PRECISION_BITS - SUBPEL_BITS)) & SUBPEL_MASK;
      filter = interp_filters[sub_pel];
      sum = 0;
      for (k = 0; k < INTERP_TAPS; ++k) {
        const int pk = int_pel - INTERP_TAPS / 2 + 1 + k;
        sum += filter[k] *
               input[(pk < 0 ? 0 : (pk >= inlength ? inlength - 1 : pk))];
      }
      *optr++ = clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
    }
  } else {
    // Initial part.
    for (x = 0, y = offset; x < x1; ++x, y += delta) {
      const int16_t *filter;
      int_pel = y >> INTERP_PRECISION_BITS;
      sub_pel = (y >> (INTERP_PRECISION_BITS - SUBPEL_BITS)) & SUBPEL_MASK;
      filter = interp_filters[sub_pel];
      sum = 0;
      for (k = 0; k < INTERP_TAPS; ++k)
        sum += filter[k] * input[(int_pel - INTERP_TAPS / 2 + 1 + k < 0
                                      ? 0
                                      : int_pel - INTERP_TAPS / 2 + 1 + k)];
      *optr++ = clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
    }
    // Middle part.
    for (; x <= x2; ++x, y += delta) {
      const int16_t *filter;
      int_pel = y >> INTERP_PRECISION_BITS;
      sub_pel = (y >> (INTERP_PRECISION_BITS - SUBPEL_BITS)) & SUBPEL_MASK;
      filter = interp_filters[sub_pel];
      sum = 0;
      for (k = 0; k < INTERP_TAPS; ++k)
        sum += filter[k] * input[int_pel - INTERP_TAPS / 2 + 1 + k];
      *optr++ = clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
    }
    // End part.
    for (; x < outlength; ++x, y += delta) {
      const int16_t *filter;
      int_pel = y >> INTERP_PRECISION_BITS;
      sub_pel = (y >> (INTERP_PRECISION_BITS - SUBPEL_BITS)) & SUBPEL_MASK;
      filter = interp_filters[sub_pel];
      sum = 0;
      for (k = 0; k < INTERP_TAPS; ++k)
        sum += filter[k] * input[(int_pel - INTERP_TAPS / 2 + 1 + k >= inlength
                                      ? inlength - 1
                                      : int_pel - INTERP_TAPS / 2 + 1 + k)];
      *optr++ = clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
    }
  }
}

static void down2_symeven(const uint8_t *const input, int length,
                          uint8_t *output) {
  // Actual filter len = 2 * filter_len_half.
  const int16_t *filter = vp9_down2_symeven_half_filter;
  const int filter_len_half = sizeof(vp9_down2_symeven_half_filter) / 2;
  int i, j;
  uint8_t *optr = output;
  int l1 = filter_len_half;
  int l2 = (length - filter_len_half);
  l1 += (l1 & 1);
  l2 += (l2 & 1);
  if (l1 > l2) {
    // Short input length.
    for (i = 0; i < length; i += 2) {
      int sum = (1 << (FILTER_BITS - 1));
      for (j = 0; j < filter_len_half; ++j) {
        sum += (input[(i - j < 0 ? 0 : i - j)] +
                input[(i + 1 + j >= length ? length - 1 : i + 1 + j)]) *
               filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel(sum);
    }
  } else {
    // Initial part.
    for (i = 0; i < l1; i += 2) {
      int sum = (1 << (FILTER_BITS - 1));
      for (j = 0; j < filter_len_half; ++j) {
        sum += (input[(i - j < 0 ? 0 : i - j)] + input[i + 1 + j]) * filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel(sum);
    }
    // Middle part.
    for (; i < l2; i += 2) {
      int sum = (1 << (FILTER_BITS - 1));
      for (j = 0; j < filter_len_half; ++j) {
        sum += (input[i - j] + input[i + 1 + j]) * filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel(sum);
    }
    // End part.
    for (; i < length; i += 2) {
      int sum = (1 << (FILTER_BITS - 1));
      for (j = 0; j < filter_len_half; ++j) {
        sum += (input[i - j] +
                input[(i + 1 + j >= length ? length - 1 : i + 1 + j)]) *
               filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel(sum);
    }
  }
}

static void down2_symodd(const uint8_t *const input, int length,
                         uint8_t *output) {
  // Actual filter len = 2 * filter_len_half - 1.
  const int16_t *filter = vp9_down2_symodd_half_filter;
  const int filter_len_half = sizeof(vp9_down2_symodd_half_filter) / 2;
  int i, j;
  uint8_t *optr = output;
  int l1 = filter_len_half - 1;
  int l2 = (length - filter_len_half + 1);
  l1 += (l1 & 1);
  l2 += (l2 & 1);
  if (l1 > l2) {
    // Short input length.
    for (i = 0; i < length; i += 2) {
      int sum = (1 << (FILTER_BITS - 1)) + input[i] * filter[0];
      for (j = 1; j < filter_len_half; ++j) {
        sum += (input[(i - j < 0 ? 0 : i - j)] +
                input[(i + j >= length ? length - 1 : i + j)]) *
               filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel(sum);
    }
  } else {
    // Initial part.
    for (i = 0; i < l1; i += 2) {
      int sum = (1 << (FILTER_BITS - 1)) + input[i] * filter[0];
      for (j = 1; j < filter_len_half; ++j) {
        sum += (input[(i - j < 0 ? 0 : i - j)] + input[i + j]) * filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel(sum);
    }
    // Middle part.
    for (; i < l2; i += 2) {
      int sum = (1 << (FILTER_BITS - 1)) + input[i] * filter[0];
      for (j = 1; j < filter_len_half; ++j) {
        sum += (input[i - j] + input[i + j]) * filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel(sum);
    }
    // End part.
    for (; i < length; i += 2) {
      int sum = (1 << (FILTER_BITS - 1)) + input[i] * filter[0];
      for (j = 1; j < filter_len_half; ++j) {
        sum += (input[i - j] + input[(i + j >= length ? length - 1 : i + j)]) *
               filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel(sum);
    }
  }
}

static int get_down2_length(int length, int steps) {
  int s;
  for (s = 0; s < steps; ++s) length = (length + 1) >> 1;
  return length;
}

static int get_down2_steps(int in_length, int out_length) {
  int steps = 0;
  int proj_in_length;
  while ((proj_in_length = get_down2_length(in_length, 1)) >= out_length) {
    ++steps;
    in_length = proj_in_length;
    if (in_length == 1) {
      // Special case: we break because any further calls to get_down2_length()
      // with be with length == 1, which return 1, resulting in an infinite
      // loop.
      break;
    }
  }
  return steps;
}

static void resize_multistep(const uint8_t *const input, int length,
                             uint8_t *output, int olength, uint8_t *otmp) {
  int steps;
  if (length == olength) {
    memcpy(output, input, sizeof(output[0]) * length);
    return;
  }
  steps = get_down2_steps(length, olength);

  if (steps > 0) {
    int s;
    uint8_t *out = NULL;
    uint8_t *otmp2;
    int filteredlength = length;

    assert(otmp != NULL);
    otmp2 = otmp + get_down2_length(length, 1);
    for (s = 0; s < steps; ++s) {
      const int proj_filteredlength = get_down2_length(filteredlength, 1);
      const uint8_t *const in = (s == 0 ? input : out);
      if (s == steps - 1 && proj_filteredlength == olength)
        out = output;
      else
        out = (s & 1 ? otmp2 : otmp);
      if (filteredlength & 1)
        down2_symodd(in, filteredlength, out);
      else
        down2_symeven(in, filteredlength, out);
      filteredlength = proj_filteredlength;
    }
    if (filteredlength != olength) {
      interpolate(out, filteredlength, output, olength);
    }
  } else {
    interpolate(input, length, output, olength);
  }
}

static void fill_col_to_arr(uint8_t *img, int stride, int len, uint8_t *arr) {
  int i;
  uint8_t *iptr = img;
  uint8_t *aptr = arr;
  for (i = 0; i < len; ++i, iptr += stride) {
    *aptr++ = *iptr;
  }
}

static void fill_arr_to_col(uint8_t *img, int stride, int len, uint8_t *arr) {
  int i;
  uint8_t *iptr = img;
  uint8_t *aptr = arr;
  for (i = 0; i < len; ++i, iptr += stride) {
    *iptr = *aptr++;
  }
}

void vp9_resize_plane(const uint8_t *const input, int height, int width,
                      int in_stride, uint8_t *output, int height2, int width2,
                      int out_stride) {
  int i;
  uint8_t *intbuf = (uint8_t *)calloc(width2 * height, sizeof(*intbuf));
  uint8_t *tmpbuf =
      (uint8_t *)calloc(width < height ? height : width, sizeof(*tmpbuf));
  uint8_t *arrbuf = (uint8_t *)calloc(height, sizeof(*arrbuf));
  uint8_t *arrbuf2 = (uint8_t *)calloc(height2, sizeof(*arrbuf2));
  if (intbuf == NULL || tmpbuf == NULL || arrbuf == NULL || arrbuf2 == NULL)
    goto Error;
  assert(width > 0);
  assert(height > 0);
  assert(width2 > 0);
  assert(height2 > 0);
  for (i = 0; i < height; ++i)
    resize_multistep(input + in_stride * i, width, intbuf + width2 * i, width2,
                     tmpbuf);
  for (i = 0; i < width2; ++i) {
    fill_col_to_arr(intbuf + i, width2, height, arrbuf);
    resize_multistep(arrbuf, height, arrbuf2, height2, tmpbuf);
    fill_arr_to_col(output + i, out_stride, height2, arrbuf2);
  }

Error:
  free(intbuf);
  free(tmpbuf);
  free(arrbuf);
  free(arrbuf2);
}

#if CONFIG_VP9_HIGHBITDEPTH
static void highbd_interpolate(const uint16_t *const input, int inlength,
                               uint16_t *output, int outlength, int bd) {
  const int64_t delta =
      (((uint64_t)inlength << 32) + outlength / 2) / outlength;
  const int64_t offset =
      inlength > outlength
          ? (((int64_t)(inlength - outlength) << 31) + outlength / 2) /
                outlength
          : -(((int64_t)(outlength - inlength) << 31) + outlength / 2) /
                outlength;
  uint16_t *optr = output;
  int x, x1, x2, sum, k, int_pel, sub_pel;
  int64_t y;

  const interp_kernel *interp_filters =
      choose_interp_filter(inlength, outlength);

  x = 0;
  y = offset;
  while ((y >> INTERP_PRECISION_BITS) < (INTERP_TAPS / 2 - 1)) {
    x++;
    y += delta;
  }
  x1 = x;
  x = outlength - 1;
  y = delta * x + offset;
  while ((y >> INTERP_PRECISION_BITS) + (int64_t)(INTERP_TAPS / 2) >=
         inlength) {
    x--;
    y -= delta;
  }
  x2 = x;
  if (x1 > x2) {
    for (x = 0, y = offset; x < outlength; ++x, y += delta) {
      const int16_t *filter;
      int_pel = y >> INTERP_PRECISION_BITS;
      sub_pel = (y >> (INTERP_PRECISION_BITS - SUBPEL_BITS)) & SUBPEL_MASK;
      filter = interp_filters[sub_pel];
      sum = 0;
      for (k = 0; k < INTERP_TAPS; ++k) {
        const int pk = int_pel - INTERP_TAPS / 2 + 1 + k;
        sum += filter[k] *
               input[(pk < 0 ? 0 : (pk >= inlength ? inlength - 1 : pk))];
      }
      *optr++ = clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
    }
  } else {
    // Initial part.
    for (x = 0, y = offset; x < x1; ++x, y += delta) {
      const int16_t *filter;
      int_pel = y >> INTERP_PRECISION_BITS;
      sub_pel = (y >> (INTERP_PRECISION_BITS - SUBPEL_BITS)) & SUBPEL_MASK;
      filter = interp_filters[sub_pel];
      sum = 0;
      for (k = 0; k < INTERP_TAPS; ++k) {
        assert(int_pel - INTERP_TAPS / 2 + 1 + k < inlength);
        sum += filter[k] * input[(int_pel - INTERP_TAPS / 2 + 1 + k < 0
                                      ? 0
                                      : int_pel - INTERP_TAPS / 2 + 1 + k)];
      }
      *optr++ = clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
    }
    // Middle part.
    for (; x <= x2; ++x, y += delta) {
      const int16_t *filter;
      int_pel = y >> INTERP_PRECISION_BITS;
      sub_pel = (y >> (INTERP_PRECISION_BITS - SUBPEL_BITS)) & SUBPEL_MASK;
      filter = interp_filters[sub_pel];
      sum = 0;
      for (k = 0; k < INTERP_TAPS; ++k)
        sum += filter[k] * input[int_pel - INTERP_TAPS / 2 + 1 + k];
      *optr++ = clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
    }
    // End part.
    for (; x < outlength; ++x, y += delta) {
      const int16_t *filter;
      int_pel = y >> INTERP_PRECISION_BITS;
      sub_pel = (y >> (INTERP_PRECISION_BITS - SUBPEL_BITS)) & SUBPEL_MASK;
      filter = interp_filters[sub_pel];
      sum = 0;
      for (k = 0; k < INTERP_TAPS; ++k)
        sum += filter[k] * input[(int_pel - INTERP_TAPS / 2 + 1 + k >= inlength
                                      ? inlength - 1
                                      : int_pel - INTERP_TAPS / 2 + 1 + k)];
      *optr++ = clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
    }
  }
}

static void highbd_down2_symeven(const uint16_t *const input, int length,
                                 uint16_t *output, int bd) {
  // Actual filter len = 2 * filter_len_half.
  static const int16_t *filter = vp9_down2_symeven_half_filter;
  const int filter_len_half = sizeof(vp9_down2_symeven_half_filter) / 2;
  int i, j;
  uint16_t *optr = output;
  int l1 = filter_len_half;
  int l2 = (length - filter_len_half);
  l1 += (l1 & 1);
  l2 += (l2 & 1);
  if (l1 > l2) {
    // Short input length.
    for (i = 0; i < length; i += 2) {
      int sum = (1 << (FILTER_BITS - 1));
      for (j = 0; j < filter_len_half; ++j) {
        sum += (input[(i - j < 0 ? 0 : i - j)] +
                input[(i + 1 + j >= length ? length - 1 : i + 1 + j)]) *
               filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel_highbd(sum, bd);
    }
  } else {
    // Initial part.
    for (i = 0; i < l1; i += 2) {
      int sum = (1 << (FILTER_BITS - 1));
      for (j = 0; j < filter_len_half; ++j) {
        sum += (input[(i - j < 0 ? 0 : i - j)] + input[i + 1 + j]) * filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel_highbd(sum, bd);
    }
    // Middle part.
    for (; i < l2; i += 2) {
      int sum = (1 << (FILTER_BITS - 1));
      for (j = 0; j < filter_len_half; ++j) {
        sum += (input[i - j] + input[i + 1 + j]) * filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel_highbd(sum, bd);
    }
    // End part.
    for (; i < length; i += 2) {
      int sum = (1 << (FILTER_BITS - 1));
      for (j = 0; j < filter_len_half; ++j) {
        sum += (input[i - j] +
                input[(i + 1 + j >= length ? length - 1 : i + 1 + j)]) *
               filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel_highbd(sum, bd);
    }
  }
}

static void highbd_down2_symodd(const uint16_t *const input, int length,
                                uint16_t *output, int bd) {
  // Actual filter len = 2 * filter_len_half - 1.
  static const int16_t *filter = vp9_down2_symodd_half_filter;
  const int filter_len_half = sizeof(vp9_down2_symodd_half_filter) / 2;
  int i, j;
  uint16_t *optr = output;
  int l1 = filter_len_half - 1;
  int l2 = (length - filter_len_half + 1);
  l1 += (l1 & 1);
  l2 += (l2 & 1);
  if (l1 > l2) {
    // Short input length.
    for (i = 0; i < length; i += 2) {
      int sum = (1 << (FILTER_BITS - 1)) + input[i] * filter[0];
      for (j = 1; j < filter_len_half; ++j) {
        sum += (input[(i - j < 0 ? 0 : i - j)] +
                input[(i + j >= length ? length - 1 : i + j)]) *
               filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel_highbd(sum, bd);
    }
  } else {
    // Initial part.
    for (i = 0; i < l1; i += 2) {
      int sum = (1 << (FILTER_BITS - 1)) + input[i] * filter[0];
      for (j = 1; j < filter_len_half; ++j) {
        sum += (input[(i - j < 0 ? 0 : i - j)] + input[i + j]) * filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel_highbd(sum, bd);
    }
    // Middle part.
    for (; i < l2; i += 2) {
      int sum = (1 << (FILTER_BITS - 1)) + input[i] * filter[0];
      for (j = 1; j < filter_len_half; ++j) {
        sum += (input[i - j] + input[i + j]) * filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel_highbd(sum, bd);
    }
    // End part.
    for (; i < length; i += 2) {
      int sum = (1 << (FILTER_BITS - 1)) + input[i] * filter[0];
      for (j = 1; j < filter_len_half; ++j) {
        sum += (input[i - j] + input[(i + j >= length ? length - 1 : i + j)]) *
               filter[j];
      }
      sum >>= FILTER_BITS;
      *optr++ = clip_pixel_highbd(sum, bd);
    }
  }
}

static void highbd_resize_multistep(const uint16_t *const input, int length,
                                    uint16_t *output, int olength,
                                    uint16_t *otmp, int bd) {
  int steps;
  if (length == olength) {
    memcpy(output, input, sizeof(output[0]) * length);
    return;
  }
  steps = get_down2_steps(length, olength);

  if (steps > 0) {
    int s;
    uint16_t *out = NULL;
    uint16_t *otmp2;
    int filteredlength = length;

    assert(otmp != NULL);
    otmp2 = otmp + get_down2_length(length, 1);
    for (s = 0; s < steps; ++s) {
      const int proj_filteredlength = get_down2_length(filteredlength, 1);
      const uint16_t *const in = (s == 0 ? input : out);
      if (s == steps - 1 && proj_filteredlength == olength)
        out = output;
      else
        out = (s & 1 ? otmp2 : otmp);
      if (filteredlength & 1)
        highbd_down2_symodd(in, filteredlength, out, bd);
      else
        highbd_down2_symeven(in, filteredlength, out, bd);
      filteredlength = proj_filteredlength;
    }
    if (filteredlength != olength) {
      highbd_interpolate(out, filteredlength, output, olength, bd);
    }
  } else {
    highbd_interpolate(input, length, output, olength, bd);
  }
}

static void highbd_fill_col_to_arr(uint16_t *img, int stride, int len,
                                   uint16_t *arr) {
  int i;
  uint16_t *iptr = img;
  uint16_t *aptr = arr;
  for (i = 0; i < len; ++i, iptr += stride) {
    *aptr++ = *iptr;
  }
}

static void highbd_fill_arr_to_col(uint16_t *img, int stride, int len,
                                   uint16_t *arr) {
  int i;
  uint16_t *iptr = img;
  uint16_t *aptr = arr;
  for (i = 0; i < len; ++i, iptr += stride) {
    *iptr = *aptr++;
  }
}

void vp9_highbd_resize_plane(const uint8_t *const input, int height, int width,
                             int in_stride, uint8_t *output, int height2,
                             int width2, int out_stride, int bd) {
  int i;
  uint16_t *intbuf = (uint16_t *)malloc(sizeof(uint16_t) * width2 * height);
  uint16_t *tmpbuf =
      (uint16_t *)malloc(sizeof(uint16_t) * (width < height ? height : width));
  uint16_t *arrbuf = (uint16_t *)malloc(sizeof(uint16_t) * height);
  uint16_t *arrbuf2 = (uint16_t *)malloc(sizeof(uint16_t) * height2);
  if (intbuf == NULL || tmpbuf == NULL || arrbuf == NULL || arrbuf2 == NULL)
    goto Error;
  assert(width > 0);
  assert(height > 0);
  assert(width2 > 0);
  assert(height2 > 0);
  for (i = 0; i < height; ++i) {
    highbd_resize_multistep(CONVERT_TO_SHORTPTR(input + in_stride * i), width,
                            intbuf + width2 * i, width2, tmpbuf, bd);
  }
  for (i = 0; i < width2; ++i) {
    highbd_fill_col_to_arr(intbuf + i, width2, height, arrbuf);
    highbd_resize_multistep(arrbuf, height, arrbuf2, height2, tmpbuf, bd);
    highbd_fill_arr_to_col(CONVERT_TO_SHORTPTR(output + i), out_stride, height2,
                           arrbuf2);
  }

Error:
  free(intbuf);
  free(tmpbuf);
  free(arrbuf);
  free(arrbuf2);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH
