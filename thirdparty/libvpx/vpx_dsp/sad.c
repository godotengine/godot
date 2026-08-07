/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"

/* Sum the difference between every corresponding element of the buffers. */
static INLINE unsigned int sad(const uint8_t *src_ptr, int src_stride,
                               const uint8_t *ref_ptr, int ref_stride,
                               int width, int height) {
  int y, x;
  unsigned int sad = 0;

  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) sad += abs(src_ptr[x] - ref_ptr[x]);

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }
  return sad;
}

#define sadMxN(m, n)                                                          \
  unsigned int vpx_sad##m##x##n##_c(const uint8_t *src_ptr, int src_stride,   \
                                    const uint8_t *ref_ptr, int ref_stride) { \
    return sad(src_ptr, src_stride, ref_ptr, ref_stride, m, n);               \
  }                                                                           \
  unsigned int vpx_sad##m##x##n##_avg_c(                                      \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,         \
      int ref_stride, const uint8_t *second_pred) {                           \
    DECLARE_ALIGNED(32, uint8_t, comp_pred[m * n]);                           \
    vpx_comp_avg_pred_c(comp_pred, second_pred, m, n, ref_ptr, ref_stride);   \
    return sad(src_ptr, src_stride, comp_pred, m, m, n);                      \
  }                                                                           \
  unsigned int vpx_sad_skip_##m##x##n##_c(                                    \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,         \
      int ref_stride) {                                                       \
    return 2 * sad(src_ptr, 2 * src_stride, ref_ptr, 2 * ref_stride, (m),     \
                   (n / 2));                                                  \
  }

// Compare |src_ptr| to 4 distinct references in |ref_array[4]|
#define sadMxNx4D(m, n)                                                        \
  void vpx_sad##m##x##n##x4d_c(const uint8_t *src_ptr, int src_stride,         \
                               const uint8_t *const ref_array[4],              \
                               int ref_stride, uint32_t sad_array[4]) {        \
    int i;                                                                     \
    for (i = 0; i < 4; ++i)                                                    \
      sad_array[i] =                                                           \
          vpx_sad##m##x##n##_c(src_ptr, src_stride, ref_array[i], ref_stride); \
  }                                                                            \
  void vpx_sad_skip_##m##x##n##x4d_c(const uint8_t *src_ptr, int src_stride,   \
                                     const uint8_t *const ref_array[4],        \
                                     int ref_stride, uint32_t sad_array[4]) {  \
    int i;                                                                     \
    for (i = 0; i < 4; ++i) {                                                  \
      sad_array[i] = 2 * sad(src_ptr, 2 * src_stride, ref_array[i],            \
                             2 * ref_stride, (m), (n / 2));                    \
    }                                                                          \
  }

/* clang-format off */
// 64x64
sadMxN(64, 64)
sadMxNx4D(64, 64)

// 64x32
sadMxN(64, 32)
sadMxNx4D(64, 32)

// 32x64
sadMxN(32, 64)
sadMxNx4D(32, 64)

// 32x32
sadMxN(32, 32)
sadMxNx4D(32, 32)

// 32x16
sadMxN(32, 16)
sadMxNx4D(32, 16)

// 16x32
sadMxN(16, 32)
sadMxNx4D(16, 32)

// 16x16
sadMxN(16, 16)
sadMxNx4D(16, 16)

// 16x8
sadMxN(16, 8)
sadMxNx4D(16, 8)

// 8x16
sadMxN(8, 16)
sadMxNx4D(8, 16)

// 8x8
sadMxN(8, 8)
sadMxNx4D(8, 8)

// 8x4
sadMxN(8, 4)
sadMxNx4D(8, 4)

// 4x8
sadMxN(4, 8)
sadMxNx4D(4, 8)

// 4x4
sadMxN(4, 4)
sadMxNx4D(4, 4)
/* clang-format on */

#if CONFIG_VP9_HIGHBITDEPTH
        static INLINE
    unsigned int highbd_sad(const uint8_t *src8_ptr, int src_stride,
                            const uint8_t *ref8_ptr, int ref_stride, int width,
                            int height) {
  int y, x;
  unsigned int sad = 0;
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8_ptr);
  const uint16_t *ref_ptr = CONVERT_TO_SHORTPTR(ref8_ptr);
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) sad += abs(src[x] - ref_ptr[x]);

    src += src_stride;
    ref_ptr += ref_stride;
  }
  return sad;
}

static INLINE unsigned int highbd_sadb(const uint8_t *src8_ptr, int src_stride,
                                       const uint16_t *ref_ptr, int ref_stride,
                                       int width, int height) {
  int y, x;
  unsigned int sad = 0;
  const uint16_t *src = CONVERT_TO_SHORTPTR(src8_ptr);
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) sad += abs(src[x] - ref_ptr[x]);

    src += src_stride;
    ref_ptr += ref_stride;
  }
  return sad;
}

#define highbd_sadMxN(m, n)                                                    \
  unsigned int vpx_highbd_sad##m##x##n##_c(                                    \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride) {                                                        \
    return highbd_sad(src_ptr, src_stride, ref_ptr, ref_stride, m, n);         \
  }                                                                            \
  unsigned int vpx_highbd_sad##m##x##n##_avg_c(                                \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride, const uint8_t *second_pred) {                            \
    DECLARE_ALIGNED(16, uint16_t, comp_pred[m * n]);                           \
    vpx_highbd_comp_avg_pred_c(comp_pred, CONVERT_TO_SHORTPTR(second_pred), m, \
                               n, CONVERT_TO_SHORTPTR(ref_ptr), ref_stride);   \
    return highbd_sadb(src_ptr, src_stride, comp_pred, m, m, n);               \
  }                                                                            \
  unsigned int vpx_highbd_sad_skip_##m##x##n##_c(                              \
      const uint8_t *src, int src_stride, const uint8_t *ref,                  \
      int ref_stride) {                                                        \
    return 2 *                                                                 \
           highbd_sad(src, 2 * src_stride, ref, 2 * ref_stride, (m), (n / 2)); \
  }

#define highbd_sadMxNx4D(m, n)                                                 \
  void vpx_highbd_sad##m##x##n##x4d_c(const uint8_t *src_ptr, int src_stride,  \
                                      const uint8_t *const ref_array[4],       \
                                      int ref_stride, uint32_t sad_array[4]) { \
    int i;                                                                     \
    for (i = 0; i < 4; ++i) {                                                  \
      sad_array[i] = vpx_highbd_sad##m##x##n##_c(src_ptr, src_stride,          \
                                                 ref_array[i], ref_stride);    \
    }                                                                          \
  }                                                                            \
  void vpx_highbd_sad_skip_##m##x##n##x4d_c(                                   \
      const uint8_t *src, int src_stride, const uint8_t *const ref_array[4],   \
      int ref_stride, uint32_t sad_array[4]) {                                 \
    int i;                                                                     \
    for (i = 0; i < 4; ++i) {                                                  \
      sad_array[i] = vpx_highbd_sad_skip_##m##x##n##_c(                        \
          src, src_stride, ref_array[i], ref_stride);                          \
    }                                                                          \
  }

/* clang-format off */
// 64x64
highbd_sadMxN(64, 64)
highbd_sadMxNx4D(64, 64)

// 64x32
highbd_sadMxN(64, 32)
highbd_sadMxNx4D(64, 32)

// 32x64
highbd_sadMxN(32, 64)
highbd_sadMxNx4D(32, 64)

// 32x32
highbd_sadMxN(32, 32)
highbd_sadMxNx4D(32, 32)

// 32x16
highbd_sadMxN(32, 16)
highbd_sadMxNx4D(32, 16)

// 16x32
highbd_sadMxN(16, 32)
highbd_sadMxNx4D(16, 32)

// 16x16
highbd_sadMxN(16, 16)
highbd_sadMxNx4D(16, 16)

// 16x8
highbd_sadMxN(16, 8)
highbd_sadMxNx4D(16, 8)

// 8x16
highbd_sadMxN(8, 16)
highbd_sadMxNx4D(8, 16)

// 8x8
highbd_sadMxN(8, 8)
highbd_sadMxNx4D(8, 8)

// 8x4
highbd_sadMxN(8, 4)
highbd_sadMxNx4D(8, 4)

// 4x8
highbd_sadMxN(4, 8)
highbd_sadMxNx4D(4, 8)

// 4x4
highbd_sadMxN(4, 4)
highbd_sadMxNx4D(4, 4)
/* clang-format on */

#endif  // CONFIG_VP9_HIGHBITDEPTH
