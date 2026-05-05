/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <emmintrin.h>  // SSE2

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_ports/mem.h"

typedef uint32_t (*high_variance_fn_t)(const uint16_t *src, int src_stride,
                                       const uint16_t *ref, int ref_stride,
                                       uint32_t *sse, int *sum);

uint32_t vpx_highbd_calc8x8var_sse2(const uint16_t *src, int src_stride,
                                    const uint16_t *ref, int ref_stride,
                                    uint32_t *sse, int *sum);

uint32_t vpx_highbd_calc16x16var_sse2(const uint16_t *src, int src_stride,
                                      const uint16_t *ref, int ref_stride,
                                      uint32_t *sse, int *sum);

static void highbd_8_variance_sse2(const uint16_t *src, int src_stride,
                                   const uint16_t *ref, int ref_stride, int w,
                                   int h, uint32_t *sse, int *sum,
                                   high_variance_fn_t var_fn, int block_size) {
  int i, j;

  *sse = 0;
  *sum = 0;

  for (i = 0; i < h; i += block_size) {
    for (j = 0; j < w; j += block_size) {
      unsigned int sse0;
      int sum0;
      var_fn(src + src_stride * i + j, src_stride, ref + ref_stride * i + j,
             ref_stride, &sse0, &sum0);
      *sse += sse0;
      *sum += sum0;
    }
  }
}

static void highbd_10_variance_sse2(const uint16_t *src, int src_stride,
                                    const uint16_t *ref, int ref_stride, int w,
                                    int h, uint32_t *sse, int *sum,
                                    high_variance_fn_t var_fn, int block_size) {
  int i, j;
  uint64_t sse_long = 0;
  int32_t sum_long = 0;

  for (i = 0; i < h; i += block_size) {
    for (j = 0; j < w; j += block_size) {
      unsigned int sse0;
      int sum0;
      var_fn(src + src_stride * i + j, src_stride, ref + ref_stride * i + j,
             ref_stride, &sse0, &sum0);
      sse_long += sse0;
      sum_long += sum0;
    }
  }
  *sum = ROUND_POWER_OF_TWO(sum_long, 2);
  *sse = (uint32_t)ROUND_POWER_OF_TWO(sse_long, 4);
}

static void highbd_12_variance_sse2(const uint16_t *src, int src_stride,
                                    const uint16_t *ref, int ref_stride, int w,
                                    int h, uint32_t *sse, int *sum,
                                    high_variance_fn_t var_fn, int block_size) {
  int i, j;
  uint64_t sse_long = 0;
  int32_t sum_long = 0;

  for (i = 0; i < h; i += block_size) {
    for (j = 0; j < w; j += block_size) {
      unsigned int sse0;
      int sum0;
      var_fn(src + src_stride * i + j, src_stride, ref + ref_stride * i + j,
             ref_stride, &sse0, &sum0);
      sse_long += sse0;
      sum_long += sum0;
    }
  }
  *sum = ROUND_POWER_OF_TWO(sum_long, 4);
  *sse = (uint32_t)ROUND_POWER_OF_TWO(sse_long, 8);
}

#define HIGH_GET_VAR(S)                                                       \
  void vpx_highbd_8_get##S##x##S##var_sse2(                                   \
      const uint8_t *src8, int src_stride, const uint8_t *ref8,               \
      int ref_stride, uint32_t *sse, int *sum) {                              \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8);                                \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);                                \
    vpx_highbd_calc##S##x##S##var_sse2(src, src_stride, ref, ref_stride, sse, \
                                       sum);                                  \
  }                                                                           \
                                                                              \
  void vpx_highbd_10_get##S##x##S##var_sse2(                                  \
      const uint8_t *src8, int src_stride, const uint8_t *ref8,               \
      int ref_stride, uint32_t *sse, int *sum) {                              \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8);                                \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);                                \
    vpx_highbd_calc##S##x##S##var_sse2(src, src_stride, ref, ref_stride, sse, \
                                       sum);                                  \
    *sum = ROUND_POWER_OF_TWO(*sum, 2);                                       \
    *sse = ROUND_POWER_OF_TWO(*sse, 4);                                       \
  }                                                                           \
                                                                              \
  void vpx_highbd_12_get##S##x##S##var_sse2(                                  \
      const uint8_t *src8, int src_stride, const uint8_t *ref8,               \
      int ref_stride, uint32_t *sse, int *sum) {                              \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8);                                \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);                                \
    vpx_highbd_calc##S##x##S##var_sse2(src, src_stride, ref, ref_stride, sse, \
                                       sum);                                  \
    *sum = ROUND_POWER_OF_TWO(*sum, 4);                                       \
    *sse = ROUND_POWER_OF_TWO(*sse, 8);                                       \
  }

HIGH_GET_VAR(16)
HIGH_GET_VAR(8)

#undef HIGH_GET_VAR

#define VAR_FN(w, h, block_size, shift)                                    \
  uint32_t vpx_highbd_8_variance##w##x##h##_sse2(                          \
      const uint8_t *src8, int src_stride, const uint8_t *ref8,            \
      int ref_stride, uint32_t *sse) {                                     \
    int sum;                                                               \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8);                             \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);                             \
    highbd_8_variance_sse2(                                                \
        src, src_stride, ref, ref_stride, w, h, sse, &sum,                 \
        vpx_highbd_calc##block_size##x##block_size##var_sse2, block_size); \
    return *sse - (uint32_t)(((int64_t)sum * sum) >> (shift));             \
  }                                                                        \
                                                                           \
  uint32_t vpx_highbd_10_variance##w##x##h##_sse2(                         \
      const uint8_t *src8, int src_stride, const uint8_t *ref8,            \
      int ref_stride, uint32_t *sse) {                                     \
    int sum;                                                               \
    int64_t var;                                                           \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8);                             \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);                             \
    highbd_10_variance_sse2(                                               \
        src, src_stride, ref, ref_stride, w, h, sse, &sum,                 \
        vpx_highbd_calc##block_size##x##block_size##var_sse2, block_size); \
    var = (int64_t)(*sse) - (((int64_t)sum * sum) >> (shift));             \
    return (var >= 0) ? (uint32_t)var : 0;                                 \
  }                                                                        \
                                                                           \
  uint32_t vpx_highbd_12_variance##w##x##h##_sse2(                         \
      const uint8_t *src8, int src_stride, const uint8_t *ref8,            \
      int ref_stride, uint32_t *sse) {                                     \
    int sum;                                                               \
    int64_t var;                                                           \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8);                             \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);                             \
    highbd_12_variance_sse2(                                               \
        src, src_stride, ref, ref_stride, w, h, sse, &sum,                 \
        vpx_highbd_calc##block_size##x##block_size##var_sse2, block_size); \
    var = (int64_t)(*sse) - (((int64_t)sum * sum) >> (shift));             \
    return (var >= 0) ? (uint32_t)var : 0;                                 \
  }

VAR_FN(64, 64, 16, 12)
VAR_FN(64, 32, 16, 11)
VAR_FN(32, 64, 16, 11)
VAR_FN(32, 32, 16, 10)
VAR_FN(32, 16, 16, 9)
VAR_FN(16, 32, 16, 9)
VAR_FN(16, 16, 16, 8)
VAR_FN(16, 8, 8, 7)
VAR_FN(8, 16, 8, 7)
VAR_FN(8, 8, 8, 6)

#undef VAR_FN

unsigned int vpx_highbd_8_mse16x16_sse2(const uint8_t *src8, int src_stride,
                                        const uint8_t *ref8, int ref_stride,
                                        unsigned int *sse) {
  int sum;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  highbd_8_variance_sse2(src, src_stride, ref, ref_stride, 16, 16, sse, &sum,
                         vpx_highbd_calc16x16var_sse2, 16);
  return *sse;
}

unsigned int vpx_highbd_10_mse16x16_sse2(const uint8_t *src8, int src_stride,
                                         const uint8_t *ref8, int ref_stride,
                                         unsigned int *sse) {
  int sum;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  highbd_10_variance_sse2(src, src_stride, ref, ref_stride, 16, 16, sse, &sum,
                          vpx_highbd_calc16x16var_sse2, 16);
  return *sse;
}

unsigned int vpx_highbd_12_mse16x16_sse2(const uint8_t *src8, int src_stride,
                                         const uint8_t *ref8, int ref_stride,
                                         unsigned int *sse) {
  int sum;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  highbd_12_variance_sse2(src, src_stride, ref, ref_stride, 16, 16, sse, &sum,
                          vpx_highbd_calc16x16var_sse2, 16);
  return *sse;
}

unsigned int vpx_highbd_8_mse8x8_sse2(const uint8_t *src8, int src_stride,
                                      const uint8_t *ref8, int ref_stride,
                                      unsigned int *sse) {
  int sum;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  highbd_8_variance_sse2(src, src_stride, ref, ref_stride, 8, 8, sse, &sum,
                         vpx_highbd_calc8x8var_sse2, 8);
  return *sse;
}

unsigned int vpx_highbd_10_mse8x8_sse2(const uint8_t *src8, int src_stride,
                                       const uint8_t *ref8, int ref_stride,
                                       unsigned int *sse) {
  int sum;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  highbd_10_variance_sse2(src, src_stride, ref, ref_stride, 8, 8, sse, &sum,
                          vpx_highbd_calc8x8var_sse2, 8);
  return *sse;
}

unsigned int vpx_highbd_12_mse8x8_sse2(const uint8_t *src8, int src_stride,
                                       const uint8_t *ref8, int ref_stride,
                                       unsigned int *sse) {
  int sum;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  highbd_12_variance_sse2(src, src_stride, ref, ref_stride, 8, 8, sse, &sum,
                          vpx_highbd_calc8x8var_sse2, 8);
  return *sse;
}

// The 2 unused parameters are place holders for PIC enabled build.
// These definitions are for functions defined in
// highbd_subpel_variance_impl_sse2.asm
#define DECL(w, opt)                                                         \
  int vpx_highbd_sub_pixel_variance##w##xh_##opt(                            \
      const uint16_t *src, ptrdiff_t src_stride, int x_offset, int y_offset, \
      const uint16_t *ref, ptrdiff_t ref_stride, int height,                 \
      unsigned int *sse, void *unused0, void *unused);
#define DECLS(opt) \
  DECL(8, opt)     \
  DECL(16, opt)

DECLS(sse2)

#undef DECLS
#undef DECL

#define FN(w, h, wf, wlog2, hlog2, opt, cast)                                  \
  uint32_t vpx_highbd_8_sub_pixel_variance##w##x##h##_##opt(                   \
      const uint8_t *src8, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *ref8, int ref_stride, uint32_t *sse_ptr) {                \
    uint32_t sse;                                                              \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8);                                 \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);                                 \
    int se = vpx_highbd_sub_pixel_variance##wf##xh_##opt(                      \
        src, src_stride, x_offset, y_offset, ref, ref_stride, h, &sse, NULL,   \
        NULL);                                                                 \
    if (w > wf) {                                                              \
      unsigned int sse2;                                                       \
      int se2 = vpx_highbd_sub_pixel_variance##wf##xh_##opt(                   \
          src + 16, src_stride, x_offset, y_offset, ref + 16, ref_stride, h,   \
          &sse2, NULL, NULL);                                                  \
      se += se2;                                                               \
      sse += sse2;                                                             \
      if (w > wf * 2) {                                                        \
        se2 = vpx_highbd_sub_pixel_variance##wf##xh_##opt(                     \
            src + 32, src_stride, x_offset, y_offset, ref + 32, ref_stride, h, \
            &sse2, NULL, NULL);                                                \
        se += se2;                                                             \
        sse += sse2;                                                           \
        se2 = vpx_highbd_sub_pixel_variance##wf##xh_##opt(                     \
            src + 48, src_stride, x_offset, y_offset, ref + 48, ref_stride, h, \
            &sse2, NULL, NULL);                                                \
        se += se2;                                                             \
        sse += sse2;                                                           \
      }                                                                        \
    }                                                                          \
    *sse_ptr = sse;                                                            \
    return sse - (uint32_t)((cast se * se) >> (wlog2 + hlog2));                \
  }                                                                            \
                                                                               \
  uint32_t vpx_highbd_10_sub_pixel_variance##w##x##h##_##opt(                  \
      const uint8_t *src8, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *ref8, int ref_stride, uint32_t *sse_ptr) {                \
    int64_t var;                                                               \
    uint32_t sse;                                                              \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8);                                 \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);                                 \
    int se = vpx_highbd_sub_pixel_variance##wf##xh_##opt(                      \
        src, src_stride, x_offset, y_offset, ref, ref_stride, h, &sse, NULL,   \
        NULL);                                                                 \
    if (w > wf) {                                                              \
      uint32_t sse2;                                                           \
      int se2 = vpx_highbd_sub_pixel_variance##wf##xh_##opt(                   \
          src + 16, src_stride, x_offset, y_offset, ref + 16, ref_stride, h,   \
          &sse2, NULL, NULL);                                                  \
      se += se2;                                                               \
      sse += sse2;                                                             \
      if (w > wf * 2) {                                                        \
        se2 = vpx_highbd_sub_pixel_variance##wf##xh_##opt(                     \
            src + 32, src_stride, x_offset, y_offset, ref + 32, ref_stride, h, \
            &sse2, NULL, NULL);                                                \
        se += se2;                                                             \
        sse += sse2;                                                           \
        se2 = vpx_highbd_sub_pixel_variance##wf##xh_##opt(                     \
            src + 48, src_stride, x_offset, y_offset, ref + 48, ref_stride, h, \
            &sse2, NULL, NULL);                                                \
        se += se2;                                                             \
        sse += sse2;                                                           \
      }                                                                        \
    }                                                                          \
    se = ROUND_POWER_OF_TWO(se, 2);                                            \
    sse = ROUND_POWER_OF_TWO(sse, 4);                                          \
    *sse_ptr = sse;                                                            \
    var = (int64_t)(sse) - ((cast se * se) >> (wlog2 + hlog2));                \
    return (var >= 0) ? (uint32_t)var : 0;                                     \
  }                                                                            \
                                                                               \
  uint32_t vpx_highbd_12_sub_pixel_variance##w##x##h##_##opt(                  \
      const uint8_t *src8, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *ref8, int ref_stride, uint32_t *sse_ptr) {                \
    int start_row;                                                             \
    uint32_t sse;                                                              \
    int se = 0;                                                                \
    int64_t var;                                                               \
    uint64_t long_sse = 0;                                                     \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8);                                 \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);                                 \
    for (start_row = 0; start_row < h; start_row += 16) {                      \
      uint32_t sse2;                                                           \
      int height = h - start_row < 16 ? h - start_row : 16;                    \
      int se2 = vpx_highbd_sub_pixel_variance##wf##xh_##opt(                   \
          src + (start_row * src_stride), src_stride, x_offset, y_offset,      \
          ref + (start_row * ref_stride), ref_stride, height, &sse2, NULL,     \
          NULL);                                                               \
      se += se2;                                                               \
      long_sse += sse2;                                                        \
      if (w > wf) {                                                            \
        se2 = vpx_highbd_sub_pixel_variance##wf##xh_##opt(                     \
            src + 16 + (start_row * src_stride), src_stride, x_offset,         \
            y_offset, ref + 16 + (start_row * ref_stride), ref_stride, height, \
            &sse2, NULL, NULL);                                                \
        se += se2;                                                             \
        long_sse += sse2;                                                      \
        if (w > wf * 2) {                                                      \
          se2 = vpx_highbd_sub_pixel_variance##wf##xh_##opt(                   \
              src + 32 + (start_row * src_stride), src_stride, x_offset,       \
              y_offset, ref + 32 + (start_row * ref_stride), ref_stride,       \
              height, &sse2, NULL, NULL);                                      \
          se += se2;                                                           \
          long_sse += sse2;                                                    \
          se2 = vpx_highbd_sub_pixel_variance##wf##xh_##opt(                   \
              src + 48 + (start_row * src_stride), src_stride, x_offset,       \
              y_offset, ref + 48 + (start_row * ref_stride), ref_stride,       \
              height, &sse2, NULL, NULL);                                      \
          se += se2;                                                           \
          long_sse += sse2;                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    se = ROUND_POWER_OF_TWO(se, 4);                                            \
    sse = (uint32_t)ROUND_POWER_OF_TWO(long_sse, 8);                           \
    *sse_ptr = sse;                                                            \
    var = (int64_t)(sse) - ((cast se * se) >> (wlog2 + hlog2));                \
    return (var >= 0) ? (uint32_t)var : 0;                                     \
  }

#define FNS(opt)                       \
  FN(64, 64, 16, 6, 6, opt, (int64_t)) \
  FN(64, 32, 16, 6, 5, opt, (int64_t)) \
  FN(32, 64, 16, 5, 6, opt, (int64_t)) \
  FN(32, 32, 16, 5, 5, opt, (int64_t)) \
  FN(32, 16, 16, 5, 4, opt, (int64_t)) \
  FN(16, 32, 16, 4, 5, opt, (int64_t)) \
  FN(16, 16, 16, 4, 4, opt, (int64_t)) \
  FN(16, 8, 16, 4, 3, opt, (int64_t))  \
  FN(8, 16, 8, 3, 4, opt, (int64_t))   \
  FN(8, 8, 8, 3, 3, opt, (int64_t))    \
  FN(8, 4, 8, 3, 2, opt, (int64_t))

FNS(sse2)

#undef FNS
#undef FN

// The 2 unused parameters are place holders for PIC enabled build.
#define DECL(w, opt)                                                         \
  int vpx_highbd_sub_pixel_avg_variance##w##xh_##opt(                        \
      const uint16_t *src, ptrdiff_t src_stride, int x_offset, int y_offset, \
      const uint16_t *ref, ptrdiff_t ref_stride, const uint16_t *second,     \
      ptrdiff_t second_stride, int height, unsigned int *sse, void *unused0, \
      void *unused);
#define DECLS(opt1) \
  DECL(16, opt1)    \
  DECL(8, opt1)

DECLS(sse2)
#undef DECL
#undef DECLS

#define FN(w, h, wf, wlog2, hlog2, opt, cast)                                  \
  uint32_t vpx_highbd_8_sub_pixel_avg_variance##w##x##h##_##opt(               \
      const uint8_t *src8, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *ref8, int ref_stride, uint32_t *sse_ptr,                  \
      const uint8_t *sec8) {                                                   \
    uint32_t sse;                                                              \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8);                                 \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);                                 \
    uint16_t *sec = CONVERT_TO_SHORTPTR(sec8);                                 \
    int se = vpx_highbd_sub_pixel_avg_variance##wf##xh_##opt(                  \
        src, src_stride, x_offset, y_offset, ref, ref_stride, sec, w, h, &sse, \
        NULL, NULL);                                                           \
    if (w > wf) {                                                              \
      uint32_t sse2;                                                           \
      int se2 = vpx_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
          src + 16, src_stride, x_offset, y_offset, ref + 16, ref_stride,      \
          sec + 16, w, h, &sse2, NULL, NULL);                                  \
      se += se2;                                                               \
      sse += sse2;                                                             \
      if (w > wf * 2) {                                                        \
        se2 = vpx_highbd_sub_pixel_avg_variance##wf##xh_##opt(                 \
            src + 32, src_stride, x_offset, y_offset, ref + 32, ref_stride,    \
            sec + 32, w, h, &sse2, NULL, NULL);                                \
        se += se2;                                                             \
        sse += sse2;                                                           \
        se2 = vpx_highbd_sub_pixel_avg_variance##wf##xh_##opt(                 \
            src + 48, src_stride, x_offset, y_offset, ref + 48, ref_stride,    \
            sec + 48, w, h, &sse2, NULL, NULL);                                \
        se += se2;                                                             \
        sse += sse2;                                                           \
      }                                                                        \
    }                                                                          \
    *sse_ptr = sse;                                                            \
    return sse - (uint32_t)((cast se * se) >> (wlog2 + hlog2));                \
  }                                                                            \
                                                                               \
  uint32_t vpx_highbd_10_sub_pixel_avg_variance##w##x##h##_##opt(              \
      const uint8_t *src8, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *ref8, int ref_stride, uint32_t *sse_ptr,                  \
      const uint8_t *sec8) {                                                   \
    int64_t var;                                                               \
    uint32_t sse;                                                              \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8);                                 \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);                                 \
    uint16_t *sec = CONVERT_TO_SHORTPTR(sec8);                                 \
    int se = vpx_highbd_sub_pixel_avg_variance##wf##xh_##opt(                  \
        src, src_stride, x_offset, y_offset, ref, ref_stride, sec, w, h, &sse, \
        NULL, NULL);                                                           \
    if (w > wf) {                                                              \
      uint32_t sse2;                                                           \
      int se2 = vpx_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
          src + 16, src_stride, x_offset, y_offset, ref + 16, ref_stride,      \
          sec + 16, w, h, &sse2, NULL, NULL);                                  \
      se += se2;                                                               \
      sse += sse2;                                                             \
      if (w > wf * 2) {                                                        \
        se2 = vpx_highbd_sub_pixel_avg_variance##wf##xh_##opt(                 \
            src + 32, src_stride, x_offset, y_offset, ref + 32, ref_stride,    \
            sec + 32, w, h, &sse2, NULL, NULL);                                \
        se += se2;                                                             \
        sse += sse2;                                                           \
        se2 = vpx_highbd_sub_pixel_avg_variance##wf##xh_##opt(                 \
            src + 48, src_stride, x_offset, y_offset, ref + 48, ref_stride,    \
            sec + 48, w, h, &sse2, NULL, NULL);                                \
        se += se2;                                                             \
        sse += sse2;                                                           \
      }                                                                        \
    }                                                                          \
    se = ROUND_POWER_OF_TWO(se, 2);                                            \
    sse = ROUND_POWER_OF_TWO(sse, 4);                                          \
    *sse_ptr = sse;                                                            \
    var = (int64_t)(sse) - ((cast se * se) >> (wlog2 + hlog2));                \
    return (var >= 0) ? (uint32_t)var : 0;                                     \
  }                                                                            \
                                                                               \
  uint32_t vpx_highbd_12_sub_pixel_avg_variance##w##x##h##_##opt(              \
      const uint8_t *src8, int src_stride, int x_offset, int y_offset,         \
      const uint8_t *ref8, int ref_stride, uint32_t *sse_ptr,                  \
      const uint8_t *sec8) {                                                   \
    int start_row;                                                             \
    int64_t var;                                                               \
    uint32_t sse;                                                              \
    int se = 0;                                                                \
    uint64_t long_sse = 0;                                                     \
    uint16_t *src = CONVERT_TO_SHORTPTR(src8);                                 \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);                                 \
    uint16_t *sec = CONVERT_TO_SHORTPTR(sec8);                                 \
    for (start_row = 0; start_row < h; start_row += 16) {                      \
      uint32_t sse2;                                                           \
      int height = h - start_row < 16 ? h - start_row : 16;                    \
      int se2 = vpx_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
          src + (start_row * src_stride), src_stride, x_offset, y_offset,      \
          ref + (start_row * ref_stride), ref_stride, sec + (start_row * w),   \
          w, height, &sse2, NULL, NULL);                                       \
      se += se2;                                                               \
      long_sse += sse2;                                                        \
      if (w > wf) {                                                            \
        se2 = vpx_highbd_sub_pixel_avg_variance##wf##xh_##opt(                 \
            src + 16 + (start_row * src_stride), src_stride, x_offset,         \
            y_offset, ref + 16 + (start_row * ref_stride), ref_stride,         \
            sec + 16 + (start_row * w), w, height, &sse2, NULL, NULL);         \
        se += se2;                                                             \
        long_sse += sse2;                                                      \
        if (w > wf * 2) {                                                      \
          se2 = vpx_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
              src + 32 + (start_row * src_stride), src_stride, x_offset,       \
              y_offset, ref + 32 + (start_row * ref_stride), ref_stride,       \
              sec + 32 + (start_row * w), w, height, &sse2, NULL, NULL);       \
          se += se2;                                                           \
          long_sse += sse2;                                                    \
          se2 = vpx_highbd_sub_pixel_avg_variance##wf##xh_##opt(               \
              src + 48 + (start_row * src_stride), src_stride, x_offset,       \
              y_offset, ref + 48 + (start_row * ref_stride), ref_stride,       \
              sec + 48 + (start_row * w), w, height, &sse2, NULL, NULL);       \
          se += se2;                                                           \
          long_sse += sse2;                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    se = ROUND_POWER_OF_TWO(se, 4);                                            \
    sse = (uint32_t)ROUND_POWER_OF_TWO(long_sse, 8);                           \
    *sse_ptr = sse;                                                            \
    var = (int64_t)(sse) - ((cast se * se) >> (wlog2 + hlog2));                \
    return (var >= 0) ? (uint32_t)var : 0;                                     \
  }

#define FNS(opt1)                       \
  FN(64, 64, 16, 6, 6, opt1, (int64_t)) \
  FN(64, 32, 16, 6, 5, opt1, (int64_t)) \
  FN(32, 64, 16, 5, 6, opt1, (int64_t)) \
  FN(32, 32, 16, 5, 5, opt1, (int64_t)) \
  FN(32, 16, 16, 5, 4, opt1, (int64_t)) \
  FN(16, 32, 16, 4, 5, opt1, (int64_t)) \
  FN(16, 16, 16, 4, 4, opt1, (int64_t)) \
  FN(16, 8, 16, 4, 3, opt1, (int64_t))  \
  FN(8, 16, 8, 4, 3, opt1, (int64_t))   \
  FN(8, 8, 8, 3, 3, opt1, (int64_t))    \
  FN(8, 4, 8, 3, 2, opt1, (int64_t))

FNS(sse2)

#undef FNS
#undef FN

void vpx_highbd_comp_avg_pred_sse2(uint16_t *comp_pred, const uint16_t *pred,
                                   int width, int height, const uint16_t *ref,
                                   int ref_stride) {
  int i, j;
  if (width > 8) {
    for (i = 0; i < height; ++i) {
      for (j = 0; j < width; j += 16) {
        const __m128i p0 = _mm_loadu_si128((const __m128i *)&pred[j]);
        const __m128i p1 = _mm_loadu_si128((const __m128i *)&pred[j + 8]);
        const __m128i r0 = _mm_loadu_si128((const __m128i *)&ref[j]);
        const __m128i r1 = _mm_loadu_si128((const __m128i *)&ref[j + 8]);
        _mm_storeu_si128((__m128i *)&comp_pred[j], _mm_avg_epu16(p0, r0));
        _mm_storeu_si128((__m128i *)&comp_pred[j + 8], _mm_avg_epu16(p1, r1));
      }
      comp_pred += width;
      pred += width;
      ref += ref_stride;
    }
  } else if (width == 8) {
    for (i = 0; i < height; i += 2) {
      const __m128i p0 = _mm_loadu_si128((const __m128i *)&pred[0]);
      const __m128i p1 = _mm_loadu_si128((const __m128i *)&pred[8]);
      const __m128i r0 = _mm_loadu_si128((const __m128i *)&ref[0]);
      const __m128i r1 = _mm_loadu_si128((const __m128i *)&ref[ref_stride]);
      _mm_storeu_si128((__m128i *)&comp_pred[0], _mm_avg_epu16(p0, r0));
      _mm_storeu_si128((__m128i *)&comp_pred[8], _mm_avg_epu16(p1, r1));
      comp_pred += 8 << 1;
      pred += 8 << 1;
      ref += ref_stride << 1;
    }
  } else {
    assert(width == 4);
    for (i = 0; i < height; i += 2) {
      const __m128i p0 = _mm_loadl_epi64((const __m128i *)&pred[0]);
      const __m128i p1 = _mm_loadl_epi64((const __m128i *)&pred[4]);
      const __m128i r0 = _mm_loadl_epi64((const __m128i *)&ref[0]);
      const __m128i r1 = _mm_loadl_epi64((const __m128i *)&ref[ref_stride]);
      _mm_storel_epi64((__m128i *)&comp_pred[0], _mm_avg_epu16(p0, r0));
      _mm_storel_epi64((__m128i *)&comp_pred[4], _mm_avg_epu16(p1, r1));
      comp_pred += 4 << 1;
      pred += 4 << 1;
      ref += ref_stride << 1;
    }
  }
}
