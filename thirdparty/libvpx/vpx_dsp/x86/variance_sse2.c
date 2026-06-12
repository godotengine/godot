/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <emmintrin.h>  // SSE2

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_ports/mem.h"
#include "vpx_dsp/x86/mem_sse2.h"

static INLINE unsigned int add32x4_sse2(__m128i val) {
  val = _mm_add_epi32(val, _mm_srli_si128(val, 8));
  val = _mm_add_epi32(val, _mm_srli_si128(val, 4));
  return (unsigned int)_mm_cvtsi128_si32(val);
}

unsigned int vpx_get_mb_ss_sse2(const int16_t *src_ptr) {
  __m128i vsum = _mm_setzero_si128();
  int i;

  for (i = 0; i < 32; ++i) {
    const __m128i v = _mm_loadu_si128((const __m128i *)src_ptr);
    vsum = _mm_add_epi32(vsum, _mm_madd_epi16(v, v));
    src_ptr += 8;
  }

  return add32x4_sse2(vsum);
}

static INLINE __m128i load4x2_sse2(const uint8_t *const p, const int stride) {
  const __m128i p0 = _mm_cvtsi32_si128(loadu_int32(p + 0 * stride));
  const __m128i p1 = _mm_cvtsi32_si128(loadu_int32(p + 1 * stride));
  const __m128i p01 = _mm_unpacklo_epi32(p0, p1);
  return _mm_unpacklo_epi8(p01, _mm_setzero_si128());
}

static INLINE void variance_kernel_sse2(const __m128i src_ptr,
                                        const __m128i ref_ptr,
                                        __m128i *const sse,
                                        __m128i *const sum) {
  const __m128i diff = _mm_sub_epi16(src_ptr, ref_ptr);
  *sse = _mm_add_epi32(*sse, _mm_madd_epi16(diff, diff));
  *sum = _mm_add_epi16(*sum, diff);
}

// Can handle 128 pixels' diff sum (such as 8x16 or 16x8)
// Slightly faster than variance_final_256_pel_sse2()
static INLINE void variance_final_128_pel_sse2(__m128i vsse, __m128i vsum,
                                               unsigned int *const sse,
                                               int *const sum) {
  *sse = add32x4_sse2(vsse);

  vsum = _mm_add_epi16(vsum, _mm_srli_si128(vsum, 8));
  vsum = _mm_add_epi16(vsum, _mm_srli_si128(vsum, 4));
  vsum = _mm_add_epi16(vsum, _mm_srli_si128(vsum, 2));
  *sum = (int16_t)_mm_extract_epi16(vsum, 0);
}

// Can handle 256 pixels' diff sum (such as 16x16)
static INLINE void variance_final_256_pel_sse2(__m128i vsse, __m128i vsum,
                                               unsigned int *const sse,
                                               int *const sum) {
  *sse = add32x4_sse2(vsse);

  vsum = _mm_add_epi16(vsum, _mm_srli_si128(vsum, 8));
  vsum = _mm_add_epi16(vsum, _mm_srli_si128(vsum, 4));
  *sum = (int16_t)_mm_extract_epi16(vsum, 0);
  *sum += (int16_t)_mm_extract_epi16(vsum, 1);
}

// Can handle 512 pixels' diff sum (such as 16x32 or 32x16)
static INLINE void variance_final_512_pel_sse2(__m128i vsse, __m128i vsum,
                                               unsigned int *const sse,
                                               int *const sum) {
  *sse = add32x4_sse2(vsse);

  vsum = _mm_add_epi16(vsum, _mm_srli_si128(vsum, 8));
  vsum = _mm_unpacklo_epi16(vsum, vsum);
  vsum = _mm_srai_epi32(vsum, 16);
  *sum = (int)add32x4_sse2(vsum);
}

static INLINE __m128i sum_to_32bit_sse2(const __m128i sum) {
  const __m128i sum_lo = _mm_srai_epi32(_mm_unpacklo_epi16(sum, sum), 16);
  const __m128i sum_hi = _mm_srai_epi32(_mm_unpackhi_epi16(sum, sum), 16);
  return _mm_add_epi32(sum_lo, sum_hi);
}

// Can handle 1024 pixels' diff sum (such as 32x32)
static INLINE int sum_final_sse2(const __m128i sum) {
  const __m128i t = sum_to_32bit_sse2(sum);
  return (int)add32x4_sse2(t);
}

static INLINE void variance4_sse2(const uint8_t *src_ptr, const int src_stride,
                                  const uint8_t *ref_ptr, const int ref_stride,
                                  const int h, __m128i *const sse,
                                  __m128i *const sum) {
  int i;

  assert(h <= 256);  // May overflow for larger height.
  *sse = _mm_setzero_si128();
  *sum = _mm_setzero_si128();

  for (i = 0; i < h; i += 2) {
    const __m128i s = load4x2_sse2(src_ptr, src_stride);
    const __m128i r = load4x2_sse2(ref_ptr, ref_stride);

    variance_kernel_sse2(s, r, sse, sum);
    src_ptr += 2 * src_stride;
    ref_ptr += 2 * ref_stride;
  }
}

static INLINE void variance8_sse2(const uint8_t *src_ptr, const int src_stride,
                                  const uint8_t *ref_ptr, const int ref_stride,
                                  const int h, __m128i *const sse,
                                  __m128i *const sum) {
  const __m128i zero = _mm_setzero_si128();
  int i;

  assert(h <= 128);  // May overflow for larger height.
  *sse = _mm_setzero_si128();
  *sum = _mm_setzero_si128();

  for (i = 0; i < h; i++) {
    const __m128i s =
        _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)src_ptr), zero);
    const __m128i r =
        _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)ref_ptr), zero);

    variance_kernel_sse2(s, r, sse, sum);
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }
}

static INLINE void variance16_kernel_sse2(const uint8_t *const src_ptr,
                                          const uint8_t *const ref_ptr,
                                          __m128i *const sse,
                                          __m128i *const sum) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i s = _mm_loadu_si128((const __m128i *)src_ptr);
  const __m128i r = _mm_loadu_si128((const __m128i *)ref_ptr);
  const __m128i src0 = _mm_unpacklo_epi8(s, zero);
  const __m128i ref0 = _mm_unpacklo_epi8(r, zero);
  const __m128i src1 = _mm_unpackhi_epi8(s, zero);
  const __m128i ref1 = _mm_unpackhi_epi8(r, zero);

  variance_kernel_sse2(src0, ref0, sse, sum);
  variance_kernel_sse2(src1, ref1, sse, sum);
}

static INLINE void variance16_sse2(const uint8_t *src_ptr, const int src_stride,
                                   const uint8_t *ref_ptr, const int ref_stride,
                                   const int h, __m128i *const sse,
                                   __m128i *const sum) {
  int i;

  assert(h <= 64);  // May overflow for larger height.
  *sse = _mm_setzero_si128();
  *sum = _mm_setzero_si128();

  for (i = 0; i < h; ++i) {
    variance16_kernel_sse2(src_ptr, ref_ptr, sse, sum);
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }
}

static INLINE void variance32_sse2(const uint8_t *src_ptr, const int src_stride,
                                   const uint8_t *ref_ptr, const int ref_stride,
                                   const int h, __m128i *const sse,
                                   __m128i *const sum) {
  int i;

  assert(h <= 32);  // May overflow for larger height.
  // Don't initialize sse here since it's an accumulation.
  *sum = _mm_setzero_si128();

  for (i = 0; i < h; ++i) {
    variance16_kernel_sse2(src_ptr + 0, ref_ptr + 0, sse, sum);
    variance16_kernel_sse2(src_ptr + 16, ref_ptr + 16, sse, sum);
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }
}

static INLINE void variance64_sse2(const uint8_t *src_ptr, const int src_stride,
                                   const uint8_t *ref_ptr, const int ref_stride,
                                   const int h, __m128i *const sse,
                                   __m128i *const sum) {
  int i;

  assert(h <= 16);  // May overflow for larger height.
  // Don't initialize sse here since it's an accumulation.
  *sum = _mm_setzero_si128();

  for (i = 0; i < h; ++i) {
    variance16_kernel_sse2(src_ptr + 0, ref_ptr + 0, sse, sum);
    variance16_kernel_sse2(src_ptr + 16, ref_ptr + 16, sse, sum);
    variance16_kernel_sse2(src_ptr + 32, ref_ptr + 32, sse, sum);
    variance16_kernel_sse2(src_ptr + 48, ref_ptr + 48, sse, sum);
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }
}

void vpx_get8x8var_sse2(const uint8_t *src_ptr, int src_stride,
                        const uint8_t *ref_ptr, int ref_stride,
                        unsigned int *sse, int *sum) {
  __m128i vsse, vsum;
  variance8_sse2(src_ptr, src_stride, ref_ptr, ref_stride, 8, &vsse, &vsum);
  variance_final_128_pel_sse2(vsse, vsum, sse, sum);
}

void vpx_get16x16var_sse2(const uint8_t *src_ptr, int src_stride,
                          const uint8_t *ref_ptr, int ref_stride,
                          unsigned int *sse, int *sum) {
  __m128i vsse, vsum;
  variance16_sse2(src_ptr, src_stride, ref_ptr, ref_stride, 16, &vsse, &vsum);
  variance_final_256_pel_sse2(vsse, vsum, sse, sum);
}

unsigned int vpx_variance4x4_sse2(const uint8_t *src_ptr, int src_stride,
                                  const uint8_t *ref_ptr, int ref_stride,
                                  unsigned int *sse) {
  __m128i vsse, vsum;
  int sum;
  variance4_sse2(src_ptr, src_stride, ref_ptr, ref_stride, 4, &vsse, &vsum);
  variance_final_128_pel_sse2(vsse, vsum, sse, &sum);
  return *sse - ((sum * sum) >> 4);
}

unsigned int vpx_variance4x8_sse2(const uint8_t *src_ptr, int src_stride,
                                  const uint8_t *ref_ptr, int ref_stride,
                                  unsigned int *sse) {
  __m128i vsse, vsum;
  int sum;
  variance4_sse2(src_ptr, src_stride, ref_ptr, ref_stride, 8, &vsse, &vsum);
  variance_final_128_pel_sse2(vsse, vsum, sse, &sum);
  return *sse - ((sum * sum) >> 5);
}

unsigned int vpx_variance8x4_sse2(const uint8_t *src_ptr, int src_stride,
                                  const uint8_t *ref_ptr, int ref_stride,
                                  unsigned int *sse) {
  __m128i vsse, vsum;
  int sum;
  variance8_sse2(src_ptr, src_stride, ref_ptr, ref_stride, 4, &vsse, &vsum);
  variance_final_128_pel_sse2(vsse, vsum, sse, &sum);
  return *sse - ((sum * sum) >> 5);
}

unsigned int vpx_variance8x8_sse2(const uint8_t *src_ptr, int src_stride,
                                  const uint8_t *ref_ptr, int ref_stride,
                                  unsigned int *sse) {
  __m128i vsse, vsum;
  int sum;
  variance8_sse2(src_ptr, src_stride, ref_ptr, ref_stride, 8, &vsse, &vsum);
  variance_final_128_pel_sse2(vsse, vsum, sse, &sum);
  return *sse - ((sum * sum) >> 6);
}

unsigned int vpx_variance8x16_sse2(const uint8_t *src_ptr, int src_stride,
                                   const uint8_t *ref_ptr, int ref_stride,
                                   unsigned int *sse) {
  __m128i vsse, vsum;
  int sum;
  variance8_sse2(src_ptr, src_stride, ref_ptr, ref_stride, 16, &vsse, &vsum);
  variance_final_128_pel_sse2(vsse, vsum, sse, &sum);
  return *sse - ((sum * sum) >> 7);
}

unsigned int vpx_variance16x8_sse2(const uint8_t *src_ptr, int src_stride,
                                   const uint8_t *ref_ptr, int ref_stride,
                                   unsigned int *sse) {
  __m128i vsse, vsum;
  int sum;
  variance16_sse2(src_ptr, src_stride, ref_ptr, ref_stride, 8, &vsse, &vsum);
  variance_final_128_pel_sse2(vsse, vsum, sse, &sum);
  return *sse - ((sum * sum) >> 7);
}

unsigned int vpx_variance16x16_sse2(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    unsigned int *sse) {
  __m128i vsse, vsum;
  int sum;
  variance16_sse2(src_ptr, src_stride, ref_ptr, ref_stride, 16, &vsse, &vsum);
  variance_final_256_pel_sse2(vsse, vsum, sse, &sum);
  return *sse - (uint32_t)(((int64_t)sum * sum) >> 8);
}

unsigned int vpx_variance16x32_sse2(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    unsigned int *sse) {
  __m128i vsse, vsum;
  int sum;
  variance16_sse2(src_ptr, src_stride, ref_ptr, ref_stride, 32, &vsse, &vsum);
  variance_final_512_pel_sse2(vsse, vsum, sse, &sum);
  return *sse - (unsigned int)(((int64_t)sum * sum) >> 9);
}

unsigned int vpx_variance32x16_sse2(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    unsigned int *sse) {
  __m128i vsse = _mm_setzero_si128();
  __m128i vsum;
  int sum;
  variance32_sse2(src_ptr, src_stride, ref_ptr, ref_stride, 16, &vsse, &vsum);
  variance_final_512_pel_sse2(vsse, vsum, sse, &sum);
  return *sse - (unsigned int)(((int64_t)sum * sum) >> 9);
}

unsigned int vpx_variance32x32_sse2(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    unsigned int *sse) {
  __m128i vsse = _mm_setzero_si128();
  __m128i vsum;
  int sum;
  variance32_sse2(src_ptr, src_stride, ref_ptr, ref_stride, 32, &vsse, &vsum);
  *sse = add32x4_sse2(vsse);
  sum = sum_final_sse2(vsum);
  return *sse - (unsigned int)(((int64_t)sum * sum) >> 10);
}

unsigned int vpx_variance32x64_sse2(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    unsigned int *sse) {
  __m128i vsse = _mm_setzero_si128();
  __m128i vsum = _mm_setzero_si128();
  int sum;
  int i = 0;

  for (i = 0; i < 2; i++) {
    __m128i vsum16;
    variance32_sse2(src_ptr + 32 * i * src_stride, src_stride,
                    ref_ptr + 32 * i * ref_stride, ref_stride, 32, &vsse,
                    &vsum16);
    vsum = _mm_add_epi32(vsum, sum_to_32bit_sse2(vsum16));
  }
  *sse = add32x4_sse2(vsse);
  sum = (int)add32x4_sse2(vsum);
  return *sse - (unsigned int)(((int64_t)sum * sum) >> 11);
}

unsigned int vpx_variance64x32_sse2(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    unsigned int *sse) {
  __m128i vsse = _mm_setzero_si128();
  __m128i vsum = _mm_setzero_si128();
  int sum;
  int i = 0;

  for (i = 0; i < 2; i++) {
    __m128i vsum16;
    variance64_sse2(src_ptr + 16 * i * src_stride, src_stride,
                    ref_ptr + 16 * i * ref_stride, ref_stride, 16, &vsse,
                    &vsum16);
    vsum = _mm_add_epi32(vsum, sum_to_32bit_sse2(vsum16));
  }
  *sse = add32x4_sse2(vsse);
  sum = (int)add32x4_sse2(vsum);
  return *sse - (unsigned int)(((int64_t)sum * sum) >> 11);
}

unsigned int vpx_variance64x64_sse2(const uint8_t *src_ptr, int src_stride,
                                    const uint8_t *ref_ptr, int ref_stride,
                                    unsigned int *sse) {
  __m128i vsse = _mm_setzero_si128();
  __m128i vsum = _mm_setzero_si128();
  int sum;
  int i = 0;

  for (i = 0; i < 4; i++) {
    __m128i vsum16;
    variance64_sse2(src_ptr + 16 * i * src_stride, src_stride,
                    ref_ptr + 16 * i * ref_stride, ref_stride, 16, &vsse,
                    &vsum16);
    vsum = _mm_add_epi32(vsum, sum_to_32bit_sse2(vsum16));
  }
  *sse = add32x4_sse2(vsse);
  sum = (int)add32x4_sse2(vsum);
  return *sse - (unsigned int)(((int64_t)sum * sum) >> 12);
}

unsigned int vpx_mse8x8_sse2(const uint8_t *src_ptr, int src_stride,
                             const uint8_t *ref_ptr, int ref_stride,
                             unsigned int *sse) {
  vpx_variance8x8_sse2(src_ptr, src_stride, ref_ptr, ref_stride, sse);
  return *sse;
}

unsigned int vpx_mse8x16_sse2(const uint8_t *src_ptr, int src_stride,
                              const uint8_t *ref_ptr, int ref_stride,
                              unsigned int *sse) {
  vpx_variance8x16_sse2(src_ptr, src_stride, ref_ptr, ref_stride, sse);
  return *sse;
}

unsigned int vpx_mse16x8_sse2(const uint8_t *src_ptr, int src_stride,
                              const uint8_t *ref_ptr, int ref_stride,
                              unsigned int *sse) {
  vpx_variance16x8_sse2(src_ptr, src_stride, ref_ptr, ref_stride, sse);
  return *sse;
}

unsigned int vpx_mse16x16_sse2(const uint8_t *src_ptr, int src_stride,
                               const uint8_t *ref_ptr, int ref_stride,
                               unsigned int *sse) {
  vpx_variance16x16_sse2(src_ptr, src_stride, ref_ptr, ref_stride, sse);
  return *sse;
}

// The 2 unused parameters are place holders for PIC enabled build.
// These definitions are for functions defined in subpel_variance.asm
#define DECL(w, opt)                                                          \
  int vpx_sub_pixel_variance##w##xh_##opt(                                    \
      const uint8_t *src_ptr, ptrdiff_t src_stride, int x_offset,             \
      int y_offset, const uint8_t *ref_ptr, ptrdiff_t ref_stride, int height, \
      unsigned int *sse, void *unused0, void *unused)
#define DECLS(opt1, opt2) \
  DECL(4, opt1);          \
  DECL(8, opt1);          \
  DECL(16, opt1)

DECLS(sse2, sse2);
DECLS(ssse3, ssse3);
#undef DECLS
#undef DECL

#define FN(w, h, wf, wlog2, hlog2, opt, cast_prod, cast)                  \
  unsigned int vpx_sub_pixel_variance##w##x##h##_##opt(                   \
      const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, \
      const uint8_t *ref_ptr, int ref_stride, unsigned int *sse) {        \
    unsigned int sse_tmp;                                                 \
    int se = vpx_sub_pixel_variance##wf##xh_##opt(                        \
        src_ptr, src_stride, x_offset, y_offset, ref_ptr, ref_stride, h,  \
        &sse_tmp, NULL, NULL);                                            \
    if (w > wf) {                                                         \
      unsigned int sse2;                                                  \
      int se2 = vpx_sub_pixel_variance##wf##xh_##opt(                     \
          src_ptr + 16, src_stride, x_offset, y_offset, ref_ptr + 16,     \
          ref_stride, h, &sse2, NULL, NULL);                              \
      se += se2;                                                          \
      sse_tmp += sse2;                                                    \
      if (w > wf * 2) {                                                   \
        se2 = vpx_sub_pixel_variance##wf##xh_##opt(                       \
            src_ptr + 32, src_stride, x_offset, y_offset, ref_ptr + 32,   \
            ref_stride, h, &sse2, NULL, NULL);                            \
        se += se2;                                                        \
        sse_tmp += sse2;                                                  \
        se2 = vpx_sub_pixel_variance##wf##xh_##opt(                       \
            src_ptr + 48, src_stride, x_offset, y_offset, ref_ptr + 48,   \
            ref_stride, h, &sse2, NULL, NULL);                            \
        se += se2;                                                        \
        sse_tmp += sse2;                                                  \
      }                                                                   \
    }                                                                     \
    *sse = sse_tmp;                                                       \
    return sse_tmp -                                                      \
           (unsigned int)(cast_prod(cast se * se) >> (wlog2 + hlog2));    \
  }

#define FNS(opt1, opt2)                             \
  FN(64, 64, 16, 6, 6, opt1, (int64_t), (int64_t))  \
  FN(64, 32, 16, 6, 5, opt1, (int64_t), (int64_t))  \
  FN(32, 64, 16, 5, 6, opt1, (int64_t), (int64_t))  \
  FN(32, 32, 16, 5, 5, opt1, (int64_t), (int64_t))  \
  FN(32, 16, 16, 5, 4, opt1, (int64_t), (int64_t))  \
  FN(16, 32, 16, 4, 5, opt1, (int64_t), (int64_t))  \
  FN(16, 16, 16, 4, 4, opt1, (uint32_t), (int64_t)) \
  FN(16, 8, 16, 4, 3, opt1, (int32_t), (int32_t))   \
  FN(8, 16, 8, 3, 4, opt1, (int32_t), (int32_t))    \
  FN(8, 8, 8, 3, 3, opt1, (int32_t), (int32_t))     \
  FN(8, 4, 8, 3, 2, opt1, (int32_t), (int32_t))     \
  FN(4, 8, 4, 2, 3, opt1, (int32_t), (int32_t))     \
  FN(4, 4, 4, 2, 2, opt1, (int32_t), (int32_t))

FNS(sse2, sse2)
FNS(ssse3, ssse3)

#undef FNS
#undef FN

// The 2 unused parameters are place holders for PIC enabled build.
#define DECL(w, opt)                                                   \
  int vpx_sub_pixel_avg_variance##w##xh_##opt(                         \
      const uint8_t *src_ptr, ptrdiff_t src_stride, int x_offset,      \
      int y_offset, const uint8_t *ref_ptr, ptrdiff_t ref_stride,      \
      const uint8_t *second_pred, ptrdiff_t second_stride, int height, \
      unsigned int *sse, void *unused0, void *unused)
#define DECLS(opt1, opt2) \
  DECL(4, opt1);          \
  DECL(8, opt1);          \
  DECL(16, opt1)

DECLS(sse2, sse2);
DECLS(ssse3, ssse3);
#undef DECL
#undef DECLS

#define FN(w, h, wf, wlog2, hlog2, opt, cast_prod, cast)                  \
  unsigned int vpx_sub_pixel_avg_variance##w##x##h##_##opt(               \
      const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset, \
      const uint8_t *ref_ptr, int ref_stride, unsigned int *sse,          \
      const uint8_t *second_pred) {                                       \
    unsigned int sse_tmp;                                                 \
    int se = vpx_sub_pixel_avg_variance##wf##xh_##opt(                    \
        src_ptr, src_stride, x_offset, y_offset, ref_ptr, ref_stride,     \
        second_pred, w, h, &sse_tmp, NULL, NULL);                         \
    if (w > wf) {                                                         \
      unsigned int sse2;                                                  \
      int se2 = vpx_sub_pixel_avg_variance##wf##xh_##opt(                 \
          src_ptr + 16, src_stride, x_offset, y_offset, ref_ptr + 16,     \
          ref_stride, second_pred + 16, w, h, &sse2, NULL, NULL);         \
      se += se2;                                                          \
      sse_tmp += sse2;                                                    \
      if (w > wf * 2) {                                                   \
        se2 = vpx_sub_pixel_avg_variance##wf##xh_##opt(                   \
            src_ptr + 32, src_stride, x_offset, y_offset, ref_ptr + 32,   \
            ref_stride, second_pred + 32, w, h, &sse2, NULL, NULL);       \
        se += se2;                                                        \
        sse_tmp += sse2;                                                  \
        se2 = vpx_sub_pixel_avg_variance##wf##xh_##opt(                   \
            src_ptr + 48, src_stride, x_offset, y_offset, ref_ptr + 48,   \
            ref_stride, second_pred + 48, w, h, &sse2, NULL, NULL);       \
        se += se2;                                                        \
        sse_tmp += sse2;                                                  \
      }                                                                   \
    }                                                                     \
    *sse = sse_tmp;                                                       \
    return sse_tmp -                                                      \
           (unsigned int)(cast_prod(cast se * se) >> (wlog2 + hlog2));    \
  }

#define FNS(opt1, opt2)                             \
  FN(64, 64, 16, 6, 6, opt1, (int64_t), (int64_t))  \
  FN(64, 32, 16, 6, 5, opt1, (int64_t), (int64_t))  \
  FN(32, 64, 16, 5, 6, opt1, (int64_t), (int64_t))  \
  FN(32, 32, 16, 5, 5, opt1, (int64_t), (int64_t))  \
  FN(32, 16, 16, 5, 4, opt1, (int64_t), (int64_t))  \
  FN(16, 32, 16, 4, 5, opt1, (int64_t), (int64_t))  \
  FN(16, 16, 16, 4, 4, opt1, (uint32_t), (int64_t)) \
  FN(16, 8, 16, 4, 3, opt1, (uint32_t), (int32_t))  \
  FN(8, 16, 8, 3, 4, opt1, (uint32_t), (int32_t))   \
  FN(8, 8, 8, 3, 3, opt1, (uint32_t), (int32_t))    \
  FN(8, 4, 8, 3, 2, opt1, (uint32_t), (int32_t))    \
  FN(4, 8, 4, 2, 3, opt1, (uint32_t), (int32_t))    \
  FN(4, 4, 4, 2, 2, opt1, (uint32_t), (int32_t))

FNS(sse2, sse)
FNS(ssse3, ssse3)

#undef FNS
#undef FN
