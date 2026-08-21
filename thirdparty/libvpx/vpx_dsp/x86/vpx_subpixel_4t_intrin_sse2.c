/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <emmintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/x86/convolve.h"
#include "vpx_dsp/x86/convolve_sse2.h"
#include "vpx_ports/mem.h"

#define CONV8_ROUNDING_BITS (7)
#define CONV8_ROUNDING_NUM (1 << (CONV8_ROUNDING_BITS - 1))

static void vpx_filter_block1d16_h4_sse2(const uint8_t *src_ptr,
                                         ptrdiff_t src_stride, uint8_t *dst_ptr,
                                         ptrdiff_t dst_stride, uint32_t height,
                                         const int16_t *kernel) {
  __m128i kernel_reg;                         // Kernel
  __m128i kernel_reg_23, kernel_reg_45;       // Segments of the kernel used
  const __m128i reg_32 = _mm_set1_epi16(32);  // Used for rounding
  int h;

  __m128i src_reg, src_reg_shift_1, src_reg_shift_2, src_reg_shift_3;
  __m128i dst_first, dst_second;
  __m128i even, odd;

  // Start one pixel before as we need tap/2 - 1 = 1 sample from the past
  src_ptr -= 1;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg = _mm_srai_epi16(kernel_reg, 1);
  kernel_reg_23 = extract_quarter_2_epi16_sse2(&kernel_reg);
  kernel_reg_45 = extract_quarter_3_epi16_sse2(&kernel_reg);

  for (h = height; h > 0; --h) {
    // We will load multiple shifted versions of the row and shuffle them into
    // 16-bit words of the form
    // ... s[2] s[1] s[0] s[-1]
    // ... s[4] s[3] s[2] s[1]
    // Then we call multiply and add to get partial results
    // s[2]k[3]+s[1]k[2] s[0]k[3]s[-1]k[2]
    // s[4]k[5]+s[3]k[4] s[2]k[5]s[1]k[4]
    // The two results are then added together for the first half of even
    // output.
    // Repeat multiple times to get the whole outoput
    src_reg = _mm_loadu_si128((const __m128i *)src_ptr);
    src_reg_shift_1 = _mm_srli_si128(src_reg, 1);
    src_reg_shift_2 = _mm_srli_si128(src_reg, 2);
    src_reg_shift_3 = _mm_srli_si128(src_reg, 3);

    // Output 6 4 2 0
    even = mm_madd_add_epi8_sse2(&src_reg, &src_reg_shift_2, &kernel_reg_23,
                                 &kernel_reg_45);

    // Output 7 5 3 1
    odd = mm_madd_add_epi8_sse2(&src_reg_shift_1, &src_reg_shift_3,
                                &kernel_reg_23, &kernel_reg_45);

    // Combine to get the first half of the dst
    dst_first = mm_zip_epi32_sse2(&even, &odd);

    // Do again to get the second half of dst
    src_reg = _mm_loadu_si128((const __m128i *)(src_ptr + 8));
    src_reg_shift_1 = _mm_srli_si128(src_reg, 1);
    src_reg_shift_2 = _mm_srli_si128(src_reg, 2);
    src_reg_shift_3 = _mm_srli_si128(src_reg, 3);

    // Output 14 12 10 8
    even = mm_madd_add_epi8_sse2(&src_reg, &src_reg_shift_2, &kernel_reg_23,
                                 &kernel_reg_45);

    // Output 15 13 11 9
    odd = mm_madd_add_epi8_sse2(&src_reg_shift_1, &src_reg_shift_3,
                                &kernel_reg_23, &kernel_reg_45);

    // Combine to get the second half of the dst
    dst_second = mm_zip_epi32_sse2(&even, &odd);

    // Round each result
    dst_first = mm_round_epi16_sse2(&dst_first, &reg_32, 6);
    dst_second = mm_round_epi16_sse2(&dst_second, &reg_32, 6);

    // Finally combine to get the final dst
    dst_first = _mm_packus_epi16(dst_first, dst_second);
    _mm_store_si128((__m128i *)dst_ptr, dst_first);

    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

/* The macro used to generate functions shifts the src_ptr up by 3 rows already
 * */

static void vpx_filter_block1d16_v4_sse2(const uint8_t *src_ptr,
                                         ptrdiff_t src_stride, uint8_t *dst_ptr,
                                         ptrdiff_t dst_stride, uint32_t height,
                                         const int16_t *kernel) {
  // Register for source s[-1:3, :]
  __m128i src_reg_m1, src_reg_0, src_reg_1, src_reg_2, src_reg_3;
  // Interleaved rows of the source. lo is first half, hi second
  __m128i src_reg_m10_lo, src_reg_m10_hi, src_reg_01_lo, src_reg_01_hi;
  __m128i src_reg_12_lo, src_reg_12_hi, src_reg_23_lo, src_reg_23_hi;
  // Half of half of the interleaved rows
  __m128i src_reg_m10_lo_1, src_reg_m10_lo_2, src_reg_m10_hi_1,
      src_reg_m10_hi_2;
  __m128i src_reg_01_lo_1, src_reg_01_lo_2, src_reg_01_hi_1, src_reg_01_hi_2;
  __m128i src_reg_12_lo_1, src_reg_12_lo_2, src_reg_12_hi_1, src_reg_12_hi_2;
  __m128i src_reg_23_lo_1, src_reg_23_lo_2, src_reg_23_hi_1, src_reg_23_hi_2;

  __m128i kernel_reg;                    // Kernel
  __m128i kernel_reg_23, kernel_reg_45;  // Segments of the kernel used

  // Result after multiply and add
  __m128i res_reg_m10_lo, res_reg_01_lo, res_reg_12_lo, res_reg_23_lo;
  __m128i res_reg_m10_hi, res_reg_01_hi, res_reg_12_hi, res_reg_23_hi;
  __m128i res_reg_m1012, res_reg_0123;
  __m128i res_reg_m1012_lo, res_reg_0123_lo, res_reg_m1012_hi, res_reg_0123_hi;

  const __m128i reg_32 = _mm_set1_epi16(32);  // Used for rounding

  // We will compute the result two rows at a time
  const ptrdiff_t src_stride_unrolled = src_stride << 1;
  const ptrdiff_t dst_stride_unrolled = dst_stride << 1;
  int h;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg = _mm_srai_epi16(kernel_reg, 1);
  kernel_reg_23 = extract_quarter_2_epi16_sse2(&kernel_reg);
  kernel_reg_45 = extract_quarter_3_epi16_sse2(&kernel_reg);

  // We will load two rows of pixels as 8-bit words, rearrange them as 16-bit
  // words,
  // shuffle the data into the form
  // ... s[0,1] s[-1,1] s[0,0] s[-1,0]
  // ... s[0,7] s[-1,7] s[0,6] s[-1,6]
  // ... s[0,9] s[-1,9] s[0,8] s[-1,8]
  // ... s[0,13] s[-1,13] s[0,12] s[-1,12]
  // so that we can call multiply and add with the kernel to get 32-bit words of
  // the form
  // ... s[0,1]k[3]+s[-1,1]k[2] s[0,0]k[3]+s[-1,0]k[2]
  // Finally, we can add multiple rows together to get the desired output.

  // First shuffle the data
  src_reg_m1 = _mm_loadu_si128((const __m128i *)src_ptr);
  src_reg_0 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride));
  src_reg_m10_lo = _mm_unpacklo_epi8(src_reg_m1, src_reg_0);
  src_reg_m10_hi = _mm_unpackhi_epi8(src_reg_m1, src_reg_0);
  src_reg_m10_lo_1 = _mm_unpacklo_epi8(src_reg_m10_lo, _mm_setzero_si128());
  src_reg_m10_lo_2 = _mm_unpackhi_epi8(src_reg_m10_lo, _mm_setzero_si128());
  src_reg_m10_hi_1 = _mm_unpacklo_epi8(src_reg_m10_hi, _mm_setzero_si128());
  src_reg_m10_hi_2 = _mm_unpackhi_epi8(src_reg_m10_hi, _mm_setzero_si128());

  // More shuffling
  src_reg_1 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 2));
  src_reg_01_lo = _mm_unpacklo_epi8(src_reg_0, src_reg_1);
  src_reg_01_hi = _mm_unpackhi_epi8(src_reg_0, src_reg_1);
  src_reg_01_lo_1 = _mm_unpacklo_epi8(src_reg_01_lo, _mm_setzero_si128());
  src_reg_01_lo_2 = _mm_unpackhi_epi8(src_reg_01_lo, _mm_setzero_si128());
  src_reg_01_hi_1 = _mm_unpacklo_epi8(src_reg_01_hi, _mm_setzero_si128());
  src_reg_01_hi_2 = _mm_unpackhi_epi8(src_reg_01_hi, _mm_setzero_si128());

  for (h = height; h > 1; h -= 2) {
    src_reg_2 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 3));

    src_reg_12_lo = _mm_unpacklo_epi8(src_reg_1, src_reg_2);
    src_reg_12_hi = _mm_unpackhi_epi8(src_reg_1, src_reg_2);

    src_reg_3 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 4));

    src_reg_23_lo = _mm_unpacklo_epi8(src_reg_2, src_reg_3);
    src_reg_23_hi = _mm_unpackhi_epi8(src_reg_2, src_reg_3);

    // Partial output from first half
    res_reg_m10_lo = mm_madd_packs_epi16_sse2(
        &src_reg_m10_lo_1, &src_reg_m10_lo_2, &kernel_reg_23);

    res_reg_01_lo = mm_madd_packs_epi16_sse2(&src_reg_01_lo_1, &src_reg_01_lo_2,
                                             &kernel_reg_23);

    src_reg_12_lo_1 = _mm_unpacklo_epi8(src_reg_12_lo, _mm_setzero_si128());
    src_reg_12_lo_2 = _mm_unpackhi_epi8(src_reg_12_lo, _mm_setzero_si128());
    res_reg_12_lo = mm_madd_packs_epi16_sse2(&src_reg_12_lo_1, &src_reg_12_lo_2,
                                             &kernel_reg_45);

    src_reg_23_lo_1 = _mm_unpacklo_epi8(src_reg_23_lo, _mm_setzero_si128());
    src_reg_23_lo_2 = _mm_unpackhi_epi8(src_reg_23_lo, _mm_setzero_si128());
    res_reg_23_lo = mm_madd_packs_epi16_sse2(&src_reg_23_lo_1, &src_reg_23_lo_2,
                                             &kernel_reg_45);

    // Add to get first half of the results
    res_reg_m1012_lo = _mm_adds_epi16(res_reg_m10_lo, res_reg_12_lo);
    res_reg_0123_lo = _mm_adds_epi16(res_reg_01_lo, res_reg_23_lo);

    // Now repeat everything again for the second half
    // Partial output for second half
    res_reg_m10_hi = mm_madd_packs_epi16_sse2(
        &src_reg_m10_hi_1, &src_reg_m10_hi_2, &kernel_reg_23);

    res_reg_01_hi = mm_madd_packs_epi16_sse2(&src_reg_01_hi_1, &src_reg_01_hi_2,
                                             &kernel_reg_23);

    src_reg_12_hi_1 = _mm_unpacklo_epi8(src_reg_12_hi, _mm_setzero_si128());
    src_reg_12_hi_2 = _mm_unpackhi_epi8(src_reg_12_hi, _mm_setzero_si128());
    res_reg_12_hi = mm_madd_packs_epi16_sse2(&src_reg_12_hi_1, &src_reg_12_hi_2,
                                             &kernel_reg_45);

    src_reg_23_hi_1 = _mm_unpacklo_epi8(src_reg_23_hi, _mm_setzero_si128());
    src_reg_23_hi_2 = _mm_unpackhi_epi8(src_reg_23_hi, _mm_setzero_si128());
    res_reg_23_hi = mm_madd_packs_epi16_sse2(&src_reg_23_hi_1, &src_reg_23_hi_2,
                                             &kernel_reg_45);

    // Second half of the results
    res_reg_m1012_hi = _mm_adds_epi16(res_reg_m10_hi, res_reg_12_hi);
    res_reg_0123_hi = _mm_adds_epi16(res_reg_01_hi, res_reg_23_hi);

    // Round the words
    res_reg_m1012_lo = mm_round_epi16_sse2(&res_reg_m1012_lo, &reg_32, 6);
    res_reg_0123_lo = mm_round_epi16_sse2(&res_reg_0123_lo, &reg_32, 6);
    res_reg_m1012_hi = mm_round_epi16_sse2(&res_reg_m1012_hi, &reg_32, 6);
    res_reg_0123_hi = mm_round_epi16_sse2(&res_reg_0123_hi, &reg_32, 6);

    // Combine to get the result
    res_reg_m1012 = _mm_packus_epi16(res_reg_m1012_lo, res_reg_m1012_hi);
    res_reg_0123 = _mm_packus_epi16(res_reg_0123_lo, res_reg_0123_hi);

    _mm_store_si128((__m128i *)dst_ptr, res_reg_m1012);
    _mm_store_si128((__m128i *)(dst_ptr + dst_stride), res_reg_0123);

    // Update the source by two rows
    src_ptr += src_stride_unrolled;
    dst_ptr += dst_stride_unrolled;

    src_reg_m10_lo_1 = src_reg_12_lo_1;
    src_reg_m10_lo_2 = src_reg_12_lo_2;
    src_reg_m10_hi_1 = src_reg_12_hi_1;
    src_reg_m10_hi_2 = src_reg_12_hi_2;
    src_reg_01_lo_1 = src_reg_23_lo_1;
    src_reg_01_lo_2 = src_reg_23_lo_2;
    src_reg_01_hi_1 = src_reg_23_hi_1;
    src_reg_01_hi_2 = src_reg_23_hi_2;
    src_reg_1 = src_reg_3;
  }
}

static void vpx_filter_block1d8_h4_sse2(const uint8_t *src_ptr,
                                        ptrdiff_t src_stride, uint8_t *dst_ptr,
                                        ptrdiff_t dst_stride, uint32_t height,
                                        const int16_t *kernel) {
  __m128i kernel_reg;                         // Kernel
  __m128i kernel_reg_23, kernel_reg_45;       // Segments of the kernel used
  const __m128i reg_32 = _mm_set1_epi16(32);  // Used for rounding
  int h;

  __m128i src_reg, src_reg_shift_1, src_reg_shift_2, src_reg_shift_3;
  __m128i dst_first;
  __m128i even, odd;

  // Start one pixel before as we need tap/2 - 1 = 1 sample from the past
  src_ptr -= 1;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg = _mm_srai_epi16(kernel_reg, 1);
  kernel_reg_23 = extract_quarter_2_epi16_sse2(&kernel_reg);
  kernel_reg_45 = extract_quarter_3_epi16_sse2(&kernel_reg);

  for (h = height; h > 0; --h) {
    // We will load multiple shifted versions of the row and shuffle them into
    // 16-bit words of the form
    // ... s[2] s[1] s[0] s[-1]
    // ... s[4] s[3] s[2] s[1]
    // Then we call multiply and add to get partial results
    // s[2]k[3]+s[1]k[2] s[0]k[3]s[-1]k[2]
    // s[4]k[5]+s[3]k[4] s[2]k[5]s[1]k[4]
    // The two results are then added together to get the even output
    src_reg = _mm_loadu_si128((const __m128i *)src_ptr);
    src_reg_shift_1 = _mm_srli_si128(src_reg, 1);
    src_reg_shift_2 = _mm_srli_si128(src_reg, 2);
    src_reg_shift_3 = _mm_srli_si128(src_reg, 3);

    // Output 6 4 2 0
    even = mm_madd_add_epi8_sse2(&src_reg, &src_reg_shift_2, &kernel_reg_23,
                                 &kernel_reg_45);

    // Output 7 5 3 1
    odd = mm_madd_add_epi8_sse2(&src_reg_shift_1, &src_reg_shift_3,
                                &kernel_reg_23, &kernel_reg_45);

    // Combine to get the first half of the dst
    dst_first = mm_zip_epi32_sse2(&even, &odd);
    dst_first = mm_round_epi16_sse2(&dst_first, &reg_32, 6);

    // Saturate and convert to 8-bit words
    dst_first = _mm_packus_epi16(dst_first, _mm_setzero_si128());

    _mm_storel_epi64((__m128i *)dst_ptr, dst_first);

    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

static void vpx_filter_block1d8_v4_sse2(const uint8_t *src_ptr,
                                        ptrdiff_t src_stride, uint8_t *dst_ptr,
                                        ptrdiff_t dst_stride, uint32_t height,
                                        const int16_t *kernel) {
  // Register for source s[-1:3, :]
  __m128i src_reg_m1, src_reg_0, src_reg_1, src_reg_2, src_reg_3;
  // Interleaved rows of the source. lo is first half, hi second
  __m128i src_reg_m10_lo, src_reg_01_lo;
  __m128i src_reg_12_lo, src_reg_23_lo;
  // Half of half of the interleaved rows
  __m128i src_reg_m10_lo_1, src_reg_m10_lo_2;
  __m128i src_reg_01_lo_1, src_reg_01_lo_2;
  __m128i src_reg_12_lo_1, src_reg_12_lo_2;
  __m128i src_reg_23_lo_1, src_reg_23_lo_2;

  __m128i kernel_reg;                    // Kernel
  __m128i kernel_reg_23, kernel_reg_45;  // Segments of the kernel used

  // Result after multiply and add
  __m128i res_reg_m10_lo, res_reg_01_lo, res_reg_12_lo, res_reg_23_lo;
  __m128i res_reg_m1012, res_reg_0123;
  __m128i res_reg_m1012_lo, res_reg_0123_lo;

  const __m128i reg_32 = _mm_set1_epi16(32);  // Used for rounding

  // We will compute the result two rows at a time
  const ptrdiff_t src_stride_unrolled = src_stride << 1;
  const ptrdiff_t dst_stride_unrolled = dst_stride << 1;
  int h;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg = _mm_srai_epi16(kernel_reg, 1);
  kernel_reg_23 = extract_quarter_2_epi16_sse2(&kernel_reg);
  kernel_reg_45 = extract_quarter_3_epi16_sse2(&kernel_reg);

  // We will load two rows of pixels as 8-bit words, rearrange them as 16-bit
  // words,
  // shuffle the data into the form
  // ... s[0,1] s[-1,1] s[0,0] s[-1,0]
  // ... s[0,7] s[-1,7] s[0,6] s[-1,6]
  // ... s[0,9] s[-1,9] s[0,8] s[-1,8]
  // ... s[0,13] s[-1,13] s[0,12] s[-1,12]
  // so that we can call multiply and add with the kernel to get 32-bit words of
  // the form
  // ... s[0,1]k[3]+s[-1,1]k[2] s[0,0]k[3]+s[-1,0]k[2]
  // Finally, we can add multiple rows together to get the desired output.

  // First shuffle the data
  src_reg_m1 = _mm_loadu_si128((const __m128i *)src_ptr);
  src_reg_0 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride));
  src_reg_m10_lo = _mm_unpacklo_epi8(src_reg_m1, src_reg_0);
  src_reg_m10_lo_1 = _mm_unpacklo_epi8(src_reg_m10_lo, _mm_setzero_si128());
  src_reg_m10_lo_2 = _mm_unpackhi_epi8(src_reg_m10_lo, _mm_setzero_si128());

  // More shuffling
  src_reg_1 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 2));
  src_reg_01_lo = _mm_unpacklo_epi8(src_reg_0, src_reg_1);
  src_reg_01_lo_1 = _mm_unpacklo_epi8(src_reg_01_lo, _mm_setzero_si128());
  src_reg_01_lo_2 = _mm_unpackhi_epi8(src_reg_01_lo, _mm_setzero_si128());

  for (h = height; h > 1; h -= 2) {
    src_reg_2 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 3));

    src_reg_12_lo = _mm_unpacklo_epi8(src_reg_1, src_reg_2);

    src_reg_3 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 4));

    src_reg_23_lo = _mm_unpacklo_epi8(src_reg_2, src_reg_3);

    // Partial output
    res_reg_m10_lo = mm_madd_packs_epi16_sse2(
        &src_reg_m10_lo_1, &src_reg_m10_lo_2, &kernel_reg_23);

    res_reg_01_lo = mm_madd_packs_epi16_sse2(&src_reg_01_lo_1, &src_reg_01_lo_2,
                                             &kernel_reg_23);

    src_reg_12_lo_1 = _mm_unpacklo_epi8(src_reg_12_lo, _mm_setzero_si128());
    src_reg_12_lo_2 = _mm_unpackhi_epi8(src_reg_12_lo, _mm_setzero_si128());
    res_reg_12_lo = mm_madd_packs_epi16_sse2(&src_reg_12_lo_1, &src_reg_12_lo_2,
                                             &kernel_reg_45);

    src_reg_23_lo_1 = _mm_unpacklo_epi8(src_reg_23_lo, _mm_setzero_si128());
    src_reg_23_lo_2 = _mm_unpackhi_epi8(src_reg_23_lo, _mm_setzero_si128());
    res_reg_23_lo = mm_madd_packs_epi16_sse2(&src_reg_23_lo_1, &src_reg_23_lo_2,
                                             &kernel_reg_45);

    // Add to get results
    res_reg_m1012_lo = _mm_adds_epi16(res_reg_m10_lo, res_reg_12_lo);
    res_reg_0123_lo = _mm_adds_epi16(res_reg_01_lo, res_reg_23_lo);

    // Round the words
    res_reg_m1012_lo = mm_round_epi16_sse2(&res_reg_m1012_lo, &reg_32, 6);
    res_reg_0123_lo = mm_round_epi16_sse2(&res_reg_0123_lo, &reg_32, 6);

    // Convert to 8-bit words
    res_reg_m1012 = _mm_packus_epi16(res_reg_m1012_lo, _mm_setzero_si128());
    res_reg_0123 = _mm_packus_epi16(res_reg_0123_lo, _mm_setzero_si128());

    // Save only half of the register (8 words)
    _mm_storel_epi64((__m128i *)dst_ptr, res_reg_m1012);
    _mm_storel_epi64((__m128i *)(dst_ptr + dst_stride), res_reg_0123);

    // Update the source by two rows
    src_ptr += src_stride_unrolled;
    dst_ptr += dst_stride_unrolled;

    src_reg_m10_lo_1 = src_reg_12_lo_1;
    src_reg_m10_lo_2 = src_reg_12_lo_2;
    src_reg_01_lo_1 = src_reg_23_lo_1;
    src_reg_01_lo_2 = src_reg_23_lo_2;
    src_reg_1 = src_reg_3;
  }
}

static void vpx_filter_block1d4_h4_sse2(const uint8_t *src_ptr,
                                        ptrdiff_t src_stride, uint8_t *dst_ptr,
                                        ptrdiff_t dst_stride, uint32_t height,
                                        const int16_t *kernel) {
  __m128i kernel_reg;                         // Kernel
  __m128i kernel_reg_23, kernel_reg_45;       // Segments of the kernel used
  const __m128i reg_32 = _mm_set1_epi16(32);  // Used for rounding
  int h;

  __m128i src_reg, src_reg_shift_1, src_reg_shift_2, src_reg_shift_3;
  __m128i dst_first;
  __m128i tmp_0, tmp_1;

  // Start one pixel before as we need tap/2 - 1 = 1 sample from the past
  src_ptr -= 1;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg = _mm_srai_epi16(kernel_reg, 1);
  kernel_reg_23 = extract_quarter_2_epi16_sse2(&kernel_reg);
  kernel_reg_45 = extract_quarter_3_epi16_sse2(&kernel_reg);

  for (h = height; h > 0; --h) {
    // We will load multiple shifted versions of the row and shuffle them into
    // 16-bit words of the form
    // ... s[1] s[0] s[0] s[-1]
    // ... s[3] s[2] s[2] s[1]
    // Then we call multiply and add to get partial results
    // s[1]k[3]+s[0]k[2] s[0]k[3]s[-1]k[2]
    // s[3]k[5]+s[2]k[4] s[2]k[5]s[1]k[4]
    // The two results are then added together to get the output
    src_reg = _mm_loadu_si128((const __m128i *)src_ptr);
    src_reg_shift_1 = _mm_srli_si128(src_reg, 1);
    src_reg_shift_2 = _mm_srli_si128(src_reg, 2);
    src_reg_shift_3 = _mm_srli_si128(src_reg, 3);

    // Convert to 16-bit words
    src_reg = _mm_unpacklo_epi8(src_reg, _mm_setzero_si128());
    src_reg_shift_1 = _mm_unpacklo_epi8(src_reg_shift_1, _mm_setzero_si128());
    src_reg_shift_2 = _mm_unpacklo_epi8(src_reg_shift_2, _mm_setzero_si128());
    src_reg_shift_3 = _mm_unpacklo_epi8(src_reg_shift_3, _mm_setzero_si128());

    // Shuffle into the right format
    tmp_0 = _mm_unpacklo_epi32(src_reg, src_reg_shift_1);
    tmp_1 = _mm_unpacklo_epi32(src_reg_shift_2, src_reg_shift_3);

    // Partial output
    tmp_0 = _mm_madd_epi16(tmp_0, kernel_reg_23);
    tmp_1 = _mm_madd_epi16(tmp_1, kernel_reg_45);

    // Output
    dst_first = _mm_add_epi32(tmp_0, tmp_1);
    dst_first = _mm_packs_epi32(dst_first, _mm_setzero_si128());

    dst_first = mm_round_epi16_sse2(&dst_first, &reg_32, 6);

    // Saturate and convert to 8-bit words
    dst_first = _mm_packus_epi16(dst_first, _mm_setzero_si128());

    *((int *)(dst_ptr)) = _mm_cvtsi128_si32(dst_first);

    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

static void vpx_filter_block1d4_v4_sse2(const uint8_t *src_ptr,
                                        ptrdiff_t src_stride, uint8_t *dst_ptr,
                                        ptrdiff_t dst_stride, uint32_t height,
                                        const int16_t *kernel) {
  // Register for source s[-1:3, :]
  __m128i src_reg_m1, src_reg_0, src_reg_1, src_reg_2, src_reg_3;
  // Interleaved rows of the source. lo is first half, hi second
  __m128i src_reg_m10_lo, src_reg_01_lo;
  __m128i src_reg_12_lo, src_reg_23_lo;
  // Half of half of the interleaved rows
  __m128i src_reg_m10_lo_1;
  __m128i src_reg_01_lo_1;
  __m128i src_reg_12_lo_1;
  __m128i src_reg_23_lo_1;

  __m128i kernel_reg;                    // Kernel
  __m128i kernel_reg_23, kernel_reg_45;  // Segments of the kernel used

  // Result after multiply and add
  __m128i res_reg_m10_lo, res_reg_01_lo, res_reg_12_lo, res_reg_23_lo;
  __m128i res_reg_m1012, res_reg_0123;
  __m128i res_reg_m1012_lo, res_reg_0123_lo;

  const __m128i reg_32 = _mm_set1_epi16(32);  // Used for rounding
  const __m128i reg_zero = _mm_setzero_si128();

  // We will compute the result two rows at a time
  const ptrdiff_t src_stride_unrolled = src_stride << 1;
  const ptrdiff_t dst_stride_unrolled = dst_stride << 1;
  int h;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg = _mm_srai_epi16(kernel_reg, 1);
  kernel_reg_23 = extract_quarter_2_epi16_sse2(&kernel_reg);
  kernel_reg_45 = extract_quarter_3_epi16_sse2(&kernel_reg);

  // We will load two rows of pixels as 8-bit words, rearrange them as 16-bit
  // words,
  // shuffle the data into the form
  // ... s[0,1] s[-1,1] s[0,0] s[-1,0]
  // ... s[0,7] s[-1,7] s[0,6] s[-1,6]
  // ... s[0,9] s[-1,9] s[0,8] s[-1,8]
  // ... s[0,13] s[-1,13] s[0,12] s[-1,12]
  // so that we can call multiply and add with the kernel to get 32-bit words of
  // the form
  // ... s[0,1]k[3]+s[-1,1]k[2] s[0,0]k[3]+s[-1,0]k[2]
  // Finally, we can add multiple rows together to get the desired output.

  // First shuffle the data
  src_reg_m1 = _mm_loadu_si128((const __m128i *)src_ptr);
  src_reg_0 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride));
  src_reg_m10_lo = _mm_unpacklo_epi8(src_reg_m1, src_reg_0);
  src_reg_m10_lo_1 = _mm_unpacklo_epi8(src_reg_m10_lo, _mm_setzero_si128());

  // More shuffling
  src_reg_1 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 2));
  src_reg_01_lo = _mm_unpacklo_epi8(src_reg_0, src_reg_1);
  src_reg_01_lo_1 = _mm_unpacklo_epi8(src_reg_01_lo, _mm_setzero_si128());

  for (h = height; h > 1; h -= 2) {
    src_reg_2 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 3));

    src_reg_12_lo = _mm_unpacklo_epi8(src_reg_1, src_reg_2);

    src_reg_3 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 4));

    src_reg_23_lo = _mm_unpacklo_epi8(src_reg_2, src_reg_3);

    // Partial output
    res_reg_m10_lo =
        mm_madd_packs_epi16_sse2(&src_reg_m10_lo_1, &reg_zero, &kernel_reg_23);

    res_reg_01_lo =
        mm_madd_packs_epi16_sse2(&src_reg_01_lo_1, &reg_zero, &kernel_reg_23);

    src_reg_12_lo_1 = _mm_unpacklo_epi8(src_reg_12_lo, _mm_setzero_si128());
    res_reg_12_lo =
        mm_madd_packs_epi16_sse2(&src_reg_12_lo_1, &reg_zero, &kernel_reg_45);

    src_reg_23_lo_1 = _mm_unpacklo_epi8(src_reg_23_lo, _mm_setzero_si128());
    res_reg_23_lo =
        mm_madd_packs_epi16_sse2(&src_reg_23_lo_1, &reg_zero, &kernel_reg_45);

    // Add to get results
    res_reg_m1012_lo = _mm_adds_epi16(res_reg_m10_lo, res_reg_12_lo);
    res_reg_0123_lo = _mm_adds_epi16(res_reg_01_lo, res_reg_23_lo);

    // Round the words
    res_reg_m1012_lo = mm_round_epi16_sse2(&res_reg_m1012_lo, &reg_32, 6);
    res_reg_0123_lo = mm_round_epi16_sse2(&res_reg_0123_lo, &reg_32, 6);

    // Convert to 8-bit words
    res_reg_m1012 = _mm_packus_epi16(res_reg_m1012_lo, reg_zero);
    res_reg_0123 = _mm_packus_epi16(res_reg_0123_lo, reg_zero);

    // Save only half of the register (8 words)
    *((int *)(dst_ptr)) = _mm_cvtsi128_si32(res_reg_m1012);
    *((int *)(dst_ptr + dst_stride)) = _mm_cvtsi128_si32(res_reg_0123);

    // Update the source by two rows
    src_ptr += src_stride_unrolled;
    dst_ptr += dst_stride_unrolled;

    src_reg_m10_lo_1 = src_reg_12_lo_1;
    src_reg_01_lo_1 = src_reg_23_lo_1;
    src_reg_1 = src_reg_3;
  }
}

#if CONFIG_VP9_HIGHBITDEPTH && VPX_ARCH_X86_64
static void vpx_highbd_filter_block1d4_h4_sse2(
    const uint16_t *src_ptr, ptrdiff_t src_stride, uint16_t *dst_ptr,
    ptrdiff_t dst_stride, uint32_t height, const int16_t *kernel, int bd) {
  // We will load multiple shifted versions of the row and shuffle them into
  // 16-bit words of the form
  // ... s[2] s[1] s[0] s[-1]
  // ... s[4] s[3] s[2] s[1]
  // Then we call multiply and add to get partial results
  // s[2]k[3]+s[1]k[2] s[0]k[3]s[-1]k[2]
  // s[4]k[5]+s[3]k[4] s[2]k[5]s[1]k[4]
  // The two results are then added together to get the even output

  __m128i src_reg, src_reg_shift_1, src_reg_shift_2, src_reg_shift_3;
  __m128i res_reg;
  __m128i even, odd;

  __m128i kernel_reg;                    // Kernel
  __m128i kernel_reg_23, kernel_reg_45;  // Segments of the kernel used
  const __m128i reg_round =
      _mm_set1_epi32(CONV8_ROUNDING_NUM);  // Used for rounding
  const __m128i reg_max = _mm_set1_epi16((1 << bd) - 1);
  const __m128i reg_zero = _mm_setzero_si128();
  int h;

  // Start one pixel before as we need tap/2 - 1 = 1 sample from the past
  src_ptr -= 1;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg_23 = extract_quarter_2_epi16_sse2(&kernel_reg);
  kernel_reg_45 = extract_quarter_3_epi16_sse2(&kernel_reg);

  for (h = height; h > 0; --h) {
    src_reg = _mm_loadu_si128((const __m128i *)src_ptr);
    src_reg_shift_1 = _mm_srli_si128(src_reg, 2);
    src_reg_shift_2 = _mm_srli_si128(src_reg, 4);
    src_reg_shift_3 = _mm_srli_si128(src_reg, 6);

    // Output 2 0
    even = mm_madd_add_epi16_sse2(&src_reg, &src_reg_shift_2, &kernel_reg_23,
                                  &kernel_reg_45);

    // Output 3 1
    odd = mm_madd_add_epi16_sse2(&src_reg_shift_1, &src_reg_shift_3,
                                 &kernel_reg_23, &kernel_reg_45);

    // Combine to get the first half of the dst
    res_reg = _mm_unpacklo_epi32(even, odd);
    res_reg = mm_round_epi32_sse2(&res_reg, &reg_round, CONV8_ROUNDING_BITS);
    res_reg = _mm_packs_epi32(res_reg, reg_zero);

    // Saturate the result and save
    res_reg = _mm_min_epi16(res_reg, reg_max);
    res_reg = _mm_max_epi16(res_reg, reg_zero);
    _mm_storel_epi64((__m128i *)dst_ptr, res_reg);

    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

static void vpx_highbd_filter_block1d4_v4_sse2(
    const uint16_t *src_ptr, ptrdiff_t src_stride, uint16_t *dst_ptr,
    ptrdiff_t dst_stride, uint32_t height, const int16_t *kernel, int bd) {
  // We will load two rows of pixels as 16-bit words, and shuffle them into the
  // form
  // ... s[0,1] s[-1,1] s[0,0] s[-1,0]
  // ... s[0,7] s[-1,7] s[0,6] s[-1,6]
  // ... s[0,9] s[-1,9] s[0,8] s[-1,8]
  // ... s[0,13] s[-1,13] s[0,12] s[-1,12]
  // so that we can call multiply and add with the kernel to get 32-bit words of
  // the form
  // ... s[0,1]k[3]+s[-1,1]k[2] s[0,0]k[3]+s[-1,0]k[2]
  // Finally, we can add multiple rows together to get the desired output.

  // Register for source s[-1:3, :]
  __m128i src_reg_m1, src_reg_0, src_reg_1, src_reg_2, src_reg_3;
  // Interleaved rows of the source. lo is first half, hi second
  __m128i src_reg_m10, src_reg_01;
  __m128i src_reg_12, src_reg_23;

  __m128i kernel_reg;                    // Kernel
  __m128i kernel_reg_23, kernel_reg_45;  // Segments of the kernel used

  // Result after multiply and add
  __m128i res_reg_m10, res_reg_01, res_reg_12, res_reg_23;
  __m128i res_reg_m1012, res_reg_0123;

  const __m128i reg_round =
      _mm_set1_epi32(CONV8_ROUNDING_NUM);  // Used for rounding
  const __m128i reg_max = _mm_set1_epi16((1 << bd) - 1);
  const __m128i reg_zero = _mm_setzero_si128();

  // We will compute the result two rows at a time
  const ptrdiff_t src_stride_unrolled = src_stride << 1;
  const ptrdiff_t dst_stride_unrolled = dst_stride << 1;
  int h;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg_23 = extract_quarter_2_epi16_sse2(&kernel_reg);
  kernel_reg_45 = extract_quarter_3_epi16_sse2(&kernel_reg);

  // First shuffle the data
  src_reg_m1 = _mm_loadl_epi64((const __m128i *)src_ptr);
  src_reg_0 = _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride));
  src_reg_m10 = _mm_unpacklo_epi16(src_reg_m1, src_reg_0);

  // More shuffling
  src_reg_1 = _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride * 2));
  src_reg_01 = _mm_unpacklo_epi16(src_reg_0, src_reg_1);

  for (h = height; h > 1; h -= 2) {
    src_reg_2 = _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride * 3));

    src_reg_12 = _mm_unpacklo_epi16(src_reg_1, src_reg_2);

    src_reg_3 = _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride * 4));

    src_reg_23 = _mm_unpacklo_epi16(src_reg_2, src_reg_3);

    // Partial output
    res_reg_m10 = _mm_madd_epi16(src_reg_m10, kernel_reg_23);
    res_reg_01 = _mm_madd_epi16(src_reg_01, kernel_reg_23);
    res_reg_12 = _mm_madd_epi16(src_reg_12, kernel_reg_45);
    res_reg_23 = _mm_madd_epi16(src_reg_23, kernel_reg_45);

    // Add to get results
    res_reg_m1012 = _mm_add_epi32(res_reg_m10, res_reg_12);
    res_reg_0123 = _mm_add_epi32(res_reg_01, res_reg_23);

    // Round the words
    res_reg_m1012 =
        mm_round_epi32_sse2(&res_reg_m1012, &reg_round, CONV8_ROUNDING_BITS);
    res_reg_0123 =
        mm_round_epi32_sse2(&res_reg_0123, &reg_round, CONV8_ROUNDING_BITS);

    res_reg_m1012 = _mm_packs_epi32(res_reg_m1012, reg_zero);
    res_reg_0123 = _mm_packs_epi32(res_reg_0123, reg_zero);

    // Saturate according to bit depth
    res_reg_m1012 = _mm_min_epi16(res_reg_m1012, reg_max);
    res_reg_0123 = _mm_min_epi16(res_reg_0123, reg_max);
    res_reg_m1012 = _mm_max_epi16(res_reg_m1012, reg_zero);
    res_reg_0123 = _mm_max_epi16(res_reg_0123, reg_zero);

    // Save only half of the register (8 words)
    _mm_storel_epi64((__m128i *)dst_ptr, res_reg_m1012);
    _mm_storel_epi64((__m128i *)(dst_ptr + dst_stride), res_reg_0123);

    // Update the source by two rows
    src_ptr += src_stride_unrolled;
    dst_ptr += dst_stride_unrolled;

    src_reg_m10 = src_reg_12;
    src_reg_01 = src_reg_23;
    src_reg_1 = src_reg_3;
  }
}

static void vpx_highbd_filter_block1d8_h4_sse2(
    const uint16_t *src_ptr, ptrdiff_t src_stride, uint16_t *dst_ptr,
    ptrdiff_t dst_stride, uint32_t height, const int16_t *kernel, int bd) {
  // We will load multiple shifted versions of the row and shuffle them into
  // 16-bit words of the form
  // ... s[2] s[1] s[0] s[-1]
  // ... s[4] s[3] s[2] s[1]
  // Then we call multiply and add to get partial results
  // s[2]k[3]+s[1]k[2] s[0]k[3]s[-1]k[2]
  // s[4]k[5]+s[3]k[4] s[2]k[5]s[1]k[4]
  // The two results are then added together for the first half of even
  // output.
  // Repeat multiple times to get the whole outoput

  __m128i src_reg, src_reg_next, src_reg_shift_1, src_reg_shift_2,
      src_reg_shift_3;
  __m128i res_reg;
  __m128i even, odd;
  __m128i tmp_0, tmp_1;

  __m128i kernel_reg;                    // Kernel
  __m128i kernel_reg_23, kernel_reg_45;  // Segments of the kernel used
  const __m128i reg_round =
      _mm_set1_epi32(CONV8_ROUNDING_NUM);  // Used for rounding
  const __m128i reg_max = _mm_set1_epi16((1 << bd) - 1);
  const __m128i reg_zero = _mm_setzero_si128();
  int h;

  // Start one pixel before as we need tap/2 - 1 = 1 sample from the past
  src_ptr -= 1;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg_23 = extract_quarter_2_epi16_sse2(&kernel_reg);
  kernel_reg_45 = extract_quarter_3_epi16_sse2(&kernel_reg);

  for (h = height; h > 0; --h) {
    // We will put first half in the first half of the reg, and second half in
    // second half
    src_reg = _mm_loadu_si128((const __m128i *)src_ptr);
    src_reg_next = _mm_loadu_si128((const __m128i *)(src_ptr + 5));

    // Output 6 4 2 0
    tmp_0 = _mm_srli_si128(src_reg, 4);
    tmp_1 = _mm_srli_si128(src_reg_next, 2);
    src_reg_shift_2 = _mm_unpacklo_epi64(tmp_0, tmp_1);
    even = mm_madd_add_epi16_sse2(&src_reg, &src_reg_shift_2, &kernel_reg_23,
                                  &kernel_reg_45);

    // Output 7 5 3 1
    tmp_0 = _mm_srli_si128(src_reg, 2);
    tmp_1 = src_reg_next;
    src_reg_shift_1 = _mm_unpacklo_epi64(tmp_0, tmp_1);

    tmp_0 = _mm_srli_si128(src_reg, 6);
    tmp_1 = _mm_srli_si128(src_reg_next, 4);
    src_reg_shift_3 = _mm_unpacklo_epi64(tmp_0, tmp_1);

    odd = mm_madd_add_epi16_sse2(&src_reg_shift_1, &src_reg_shift_3,
                                 &kernel_reg_23, &kernel_reg_45);

    // Combine to get the first half of the dst
    even = mm_round_epi32_sse2(&even, &reg_round, CONV8_ROUNDING_BITS);
    odd = mm_round_epi32_sse2(&odd, &reg_round, CONV8_ROUNDING_BITS);
    res_reg = mm_zip_epi32_sse2(&even, &odd);

    // Saturate the result and save
    res_reg = _mm_min_epi16(res_reg, reg_max);
    res_reg = _mm_max_epi16(res_reg, reg_zero);

    _mm_store_si128((__m128i *)dst_ptr, res_reg);

    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
}

static void vpx_highbd_filter_block1d8_v4_sse2(
    const uint16_t *src_ptr, ptrdiff_t src_stride, uint16_t *dst_ptr,
    ptrdiff_t dst_stride, uint32_t height, const int16_t *kernel, int bd) {
  // We will load two rows of pixels as 16-bit words, and shuffle them into the
  // form
  // ... s[0,1] s[-1,1] s[0,0] s[-1,0]
  // ... s[0,7] s[-1,7] s[0,6] s[-1,6]
  // ... s[0,9] s[-1,9] s[0,8] s[-1,8]
  // ... s[0,13] s[-1,13] s[0,12] s[-1,12]
  // so that we can call multiply and add with the kernel to get 32-bit words of
  // the form
  // ... s[0,1]k[3]+s[-1,1]k[2] s[0,0]k[3]+s[-1,0]k[2]
  // Finally, we can add multiple rows together to get the desired output.

  // Register for source s[-1:3, :]
  __m128i src_reg_m1, src_reg_0, src_reg_1, src_reg_2, src_reg_3;
  // Interleaved rows of the source. lo is first half, hi second
  __m128i src_reg_m10_lo, src_reg_01_lo, src_reg_m10_hi, src_reg_01_hi;
  __m128i src_reg_12_lo, src_reg_23_lo, src_reg_12_hi, src_reg_23_hi;

  // Result after multiply and add
  __m128i res_reg_m10_lo, res_reg_01_lo, res_reg_12_lo, res_reg_23_lo;
  __m128i res_reg_m10_hi, res_reg_01_hi, res_reg_12_hi, res_reg_23_hi;
  __m128i res_reg_m1012, res_reg_0123;
  __m128i res_reg_m1012_lo, res_reg_0123_lo;
  __m128i res_reg_m1012_hi, res_reg_0123_hi;

  __m128i kernel_reg;                    // Kernel
  __m128i kernel_reg_23, kernel_reg_45;  // Segments of the kernel used

  const __m128i reg_round =
      _mm_set1_epi32(CONV8_ROUNDING_NUM);  // Used for rounding
  const __m128i reg_max = _mm_set1_epi16((1 << bd) - 1);
  const __m128i reg_zero = _mm_setzero_si128();

  // We will compute the result two rows at a time
  const ptrdiff_t src_stride_unrolled = src_stride << 1;
  const ptrdiff_t dst_stride_unrolled = dst_stride << 1;
  int h;

  // Load Kernel
  kernel_reg = _mm_loadu_si128((const __m128i *)kernel);
  kernel_reg_23 = extract_quarter_2_epi16_sse2(&kernel_reg);
  kernel_reg_45 = extract_quarter_3_epi16_sse2(&kernel_reg);

  // First shuffle the data
  src_reg_m1 = _mm_loadu_si128((const __m128i *)src_ptr);
  src_reg_0 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride));
  src_reg_m10_lo = _mm_unpacklo_epi16(src_reg_m1, src_reg_0);
  src_reg_m10_hi = _mm_unpackhi_epi16(src_reg_m1, src_reg_0);

  // More shuffling
  src_reg_1 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 2));
  src_reg_01_lo = _mm_unpacklo_epi16(src_reg_0, src_reg_1);
  src_reg_01_hi = _mm_unpackhi_epi16(src_reg_0, src_reg_1);

  for (h = height; h > 1; h -= 2) {
    src_reg_2 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 3));

    src_reg_12_lo = _mm_unpacklo_epi16(src_reg_1, src_reg_2);
    src_reg_12_hi = _mm_unpackhi_epi16(src_reg_1, src_reg_2);

    src_reg_3 = _mm_loadu_si128((const __m128i *)(src_ptr + src_stride * 4));

    src_reg_23_lo = _mm_unpacklo_epi16(src_reg_2, src_reg_3);
    src_reg_23_hi = _mm_unpackhi_epi16(src_reg_2, src_reg_3);

    // Partial output for first half
    res_reg_m10_lo = _mm_madd_epi16(src_reg_m10_lo, kernel_reg_23);
    res_reg_01_lo = _mm_madd_epi16(src_reg_01_lo, kernel_reg_23);
    res_reg_12_lo = _mm_madd_epi16(src_reg_12_lo, kernel_reg_45);
    res_reg_23_lo = _mm_madd_epi16(src_reg_23_lo, kernel_reg_45);

    // Add to get results
    res_reg_m1012_lo = _mm_add_epi32(res_reg_m10_lo, res_reg_12_lo);
    res_reg_0123_lo = _mm_add_epi32(res_reg_01_lo, res_reg_23_lo);

    // Round the words
    res_reg_m1012_lo =
        mm_round_epi32_sse2(&res_reg_m1012_lo, &reg_round, CONV8_ROUNDING_BITS);
    res_reg_0123_lo =
        mm_round_epi32_sse2(&res_reg_0123_lo, &reg_round, CONV8_ROUNDING_BITS);

    // Partial output for first half
    res_reg_m10_hi = _mm_madd_epi16(src_reg_m10_hi, kernel_reg_23);
    res_reg_01_hi = _mm_madd_epi16(src_reg_01_hi, kernel_reg_23);
    res_reg_12_hi = _mm_madd_epi16(src_reg_12_hi, kernel_reg_45);
    res_reg_23_hi = _mm_madd_epi16(src_reg_23_hi, kernel_reg_45);

    // Add to get results
    res_reg_m1012_hi = _mm_add_epi32(res_reg_m10_hi, res_reg_12_hi);
    res_reg_0123_hi = _mm_add_epi32(res_reg_01_hi, res_reg_23_hi);

    // Round the words
    res_reg_m1012_hi =
        mm_round_epi32_sse2(&res_reg_m1012_hi, &reg_round, CONV8_ROUNDING_BITS);
    res_reg_0123_hi =
        mm_round_epi32_sse2(&res_reg_0123_hi, &reg_round, CONV8_ROUNDING_BITS);

    // Combine the two halfs
    res_reg_m1012 = _mm_packs_epi32(res_reg_m1012_lo, res_reg_m1012_hi);
    res_reg_0123 = _mm_packs_epi32(res_reg_0123_lo, res_reg_0123_hi);

    // Saturate according to bit depth
    res_reg_m1012 = _mm_min_epi16(res_reg_m1012, reg_max);
    res_reg_0123 = _mm_min_epi16(res_reg_0123, reg_max);
    res_reg_m1012 = _mm_max_epi16(res_reg_m1012, reg_zero);
    res_reg_0123 = _mm_max_epi16(res_reg_0123, reg_zero);

    // Save only half of the register (8 words)
    _mm_store_si128((__m128i *)dst_ptr, res_reg_m1012);
    _mm_store_si128((__m128i *)(dst_ptr + dst_stride), res_reg_0123);

    // Update the source by two rows
    src_ptr += src_stride_unrolled;
    dst_ptr += dst_stride_unrolled;

    src_reg_m10_lo = src_reg_12_lo;
    src_reg_m10_hi = src_reg_12_hi;
    src_reg_01_lo = src_reg_23_lo;
    src_reg_01_hi = src_reg_23_hi;
    src_reg_1 = src_reg_3;
  }
}

static void vpx_highbd_filter_block1d16_h4_sse2(
    const uint16_t *src_ptr, ptrdiff_t src_stride, uint16_t *dst_ptr,
    ptrdiff_t dst_stride, uint32_t height, const int16_t *kernel, int bd) {
  vpx_highbd_filter_block1d8_h4_sse2(src_ptr, src_stride, dst_ptr, dst_stride,
                                     height, kernel, bd);
  vpx_highbd_filter_block1d8_h4_sse2(src_ptr + 8, src_stride, dst_ptr + 8,
                                     dst_stride, height, kernel, bd);
}

static void vpx_highbd_filter_block1d16_v4_sse2(
    const uint16_t *src_ptr, ptrdiff_t src_stride, uint16_t *dst_ptr,
    ptrdiff_t dst_stride, uint32_t height, const int16_t *kernel, int bd) {
  vpx_highbd_filter_block1d8_v4_sse2(src_ptr, src_stride, dst_ptr, dst_stride,
                                     height, kernel, bd);
  vpx_highbd_filter_block1d8_v4_sse2(src_ptr + 8, src_stride, dst_ptr + 8,
                                     dst_stride, height, kernel, bd);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH && VPX_ARCH_X86_64

// From vpx_subpixel_8t_sse2.asm.
filter8_1dfunction vpx_filter_block1d16_v8_sse2;
filter8_1dfunction vpx_filter_block1d16_h8_sse2;
filter8_1dfunction vpx_filter_block1d8_v8_sse2;
filter8_1dfunction vpx_filter_block1d8_h8_sse2;
filter8_1dfunction vpx_filter_block1d4_v8_sse2;
filter8_1dfunction vpx_filter_block1d4_h8_sse2;
filter8_1dfunction vpx_filter_block1d16_v8_avg_sse2;
filter8_1dfunction vpx_filter_block1d16_h8_avg_sse2;
filter8_1dfunction vpx_filter_block1d8_v8_avg_sse2;
filter8_1dfunction vpx_filter_block1d8_h8_avg_sse2;
filter8_1dfunction vpx_filter_block1d4_v8_avg_sse2;
filter8_1dfunction vpx_filter_block1d4_h8_avg_sse2;

// Use the [vh]8 version because there is no [vh]4 implementation.
#define vpx_filter_block1d16_v4_avg_sse2 vpx_filter_block1d16_v8_avg_sse2
#define vpx_filter_block1d16_h4_avg_sse2 vpx_filter_block1d16_h8_avg_sse2
#define vpx_filter_block1d8_v4_avg_sse2 vpx_filter_block1d8_v8_avg_sse2
#define vpx_filter_block1d8_h4_avg_sse2 vpx_filter_block1d8_h8_avg_sse2
#define vpx_filter_block1d4_v4_avg_sse2 vpx_filter_block1d4_v8_avg_sse2
#define vpx_filter_block1d4_h4_avg_sse2 vpx_filter_block1d4_h8_avg_sse2

// From vpx_dsp/x86/vpx_subpixel_bilinear_sse2.asm.
filter8_1dfunction vpx_filter_block1d16_v2_sse2;
filter8_1dfunction vpx_filter_block1d16_h2_sse2;
filter8_1dfunction vpx_filter_block1d8_v2_sse2;
filter8_1dfunction vpx_filter_block1d8_h2_sse2;
filter8_1dfunction vpx_filter_block1d4_v2_sse2;
filter8_1dfunction vpx_filter_block1d4_h2_sse2;
filter8_1dfunction vpx_filter_block1d16_v2_avg_sse2;
filter8_1dfunction vpx_filter_block1d16_h2_avg_sse2;
filter8_1dfunction vpx_filter_block1d8_v2_avg_sse2;
filter8_1dfunction vpx_filter_block1d8_h2_avg_sse2;
filter8_1dfunction vpx_filter_block1d4_v2_avg_sse2;
filter8_1dfunction vpx_filter_block1d4_h2_avg_sse2;

// void vpx_convolve8_horiz_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                               uint8_t *dst, ptrdiff_t dst_stride,
//                               const InterpKernel *filter, int x0_q4,
//                               int32_t x_step_q4, int y0_q4, int y_step_q4,
//                               int w, int h);
// void vpx_convolve8_vert_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                              uint8_t *dst, ptrdiff_t dst_stride,
//                              const InterpKernel *filter, int x0_q4,
//                              int32_t x_step_q4, int y0_q4, int y_step_q4,
//                              int w, int h);
// void vpx_convolve8_avg_horiz_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                                   uint8_t *dst, ptrdiff_t dst_stride,
//                                   const InterpKernel *filter, int x0_q4,
//                                   int32_t x_step_q4, int y0_q4,
//                                   int y_step_q4, int w, int h);
// void vpx_convolve8_avg_vert_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                                  uint8_t *dst, ptrdiff_t dst_stride,
//                                  const InterpKernel *filter, int x0_q4,
//                                  int32_t x_step_q4, int y0_q4, int y_step_q4,
//                                  int w, int h);
FUN_CONV_1D(horiz, x0_q4, x_step_q4, h, src, , sse2, 0)
FUN_CONV_1D(vert, y0_q4, y_step_q4, v, src - (num_taps / 2 - 1) * src_stride, ,
            sse2, 0)
FUN_CONV_1D(avg_horiz, x0_q4, x_step_q4, h, src, avg_, sse2, 1)
FUN_CONV_1D(avg_vert, y0_q4, y_step_q4, v,
            src - (num_taps / 2 - 1) * src_stride, avg_, sse2, 1)

// void vpx_convolve8_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                         uint8_t *dst, ptrdiff_t dst_stride,
//                         const InterpKernel *filter, int x0_q4,
//                         int32_t x_step_q4, int y0_q4, int y_step_q4,
//                         int w, int h);
// void vpx_convolve8_avg_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                             uint8_t *dst, ptrdiff_t dst_stride,
//                             const InterpKernel *filter, int x0_q4,
//                             int32_t x_step_q4, int y0_q4, int y_step_q4,
//                             int w, int h);
FUN_CONV_2D(, sse2, 0)
FUN_CONV_2D(avg_, sse2, 1)

#if CONFIG_VP9_HIGHBITDEPTH && VPX_ARCH_X86_64
// From vpx_dsp/x86/vpx_high_subpixel_8t_sse2.asm.
highbd_filter8_1dfunction vpx_highbd_filter_block1d16_v8_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d16_h8_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d8_v8_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d8_h8_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d4_v8_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d4_h8_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d16_v8_avg_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d16_h8_avg_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d8_v8_avg_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d8_h8_avg_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d4_v8_avg_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d4_h8_avg_sse2;

// Use the [vh]8 version because there is no [vh]4 implementation.
#define vpx_highbd_filter_block1d16_v4_avg_sse2 \
  vpx_highbd_filter_block1d16_v8_avg_sse2
#define vpx_highbd_filter_block1d16_h4_avg_sse2 \
  vpx_highbd_filter_block1d16_h8_avg_sse2
#define vpx_highbd_filter_block1d8_v4_avg_sse2 \
  vpx_highbd_filter_block1d8_v8_avg_sse2
#define vpx_highbd_filter_block1d8_h4_avg_sse2 \
  vpx_highbd_filter_block1d8_h8_avg_sse2
#define vpx_highbd_filter_block1d4_v4_avg_sse2 \
  vpx_highbd_filter_block1d4_v8_avg_sse2
#define vpx_highbd_filter_block1d4_h4_avg_sse2 \
  vpx_highbd_filter_block1d4_h8_avg_sse2

// From vpx_dsp/x86/vpx_high_subpixel_bilinear_sse2.asm.
highbd_filter8_1dfunction vpx_highbd_filter_block1d16_v2_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d16_h2_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d8_v2_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d8_h2_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d4_v2_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d4_h2_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d16_v2_avg_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d16_h2_avg_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d8_v2_avg_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d8_h2_avg_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d4_v2_avg_sse2;
highbd_filter8_1dfunction vpx_highbd_filter_block1d4_h2_avg_sse2;

// void vpx_highbd_convolve8_horiz_sse2(const uint8_t *src,
//                                      ptrdiff_t src_stride,
//                                      uint8_t *dst,
//                                      ptrdiff_t dst_stride,
//                                      const int16_t *filter_x,
//                                      int x_step_q4,
//                                      const int16_t *filter_y,
//                                      int y_step_q4,
//                                      int w, int h, int bd);
// void vpx_highbd_convolve8_vert_sse2(const uint8_t *src,
//                                     ptrdiff_t src_stride,
//                                     uint8_t *dst,
//                                     ptrdiff_t dst_stride,
//                                     const int16_t *filter_x,
//                                     int x_step_q4,
//                                     const int16_t *filter_y,
//                                     int y_step_q4,
//                                     int w, int h, int bd);
// void vpx_highbd_convolve8_avg_horiz_sse2(const uint8_t *src,
//                                          ptrdiff_t src_stride,
//                                          uint8_t *dst,
//                                          ptrdiff_t dst_stride,
//                                          const int16_t *filter_x,
//                                          int x_step_q4,
//                                          const int16_t *filter_y,
//                                          int y_step_q4,
//                                          int w, int h, int bd);
// void vpx_highbd_convolve8_avg_vert_sse2(const uint8_t *src,
//                                         ptrdiff_t src_stride,
//                                         uint8_t *dst,
//                                         ptrdiff_t dst_stride,
//                                         const int16_t *filter_x,
//                                         int x_step_q4,
//                                         const int16_t *filter_y,
//                                         int y_step_q4,
//                                         int w, int h, int bd);
HIGH_FUN_CONV_1D(horiz, x0_q4, x_step_q4, h, src, , sse2, 0)
HIGH_FUN_CONV_1D(vert, y0_q4, y_step_q4, v,
                 src - src_stride * (num_taps / 2 - 1), , sse2, 0)
HIGH_FUN_CONV_1D(avg_horiz, x0_q4, x_step_q4, h, src, avg_, sse2, 1)
HIGH_FUN_CONV_1D(avg_vert, y0_q4, y_step_q4, v,
                 src - src_stride * (num_taps / 2 - 1), avg_, sse2, 1)

// void vpx_highbd_convolve8_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                                uint8_t *dst, ptrdiff_t dst_stride,
//                                const InterpKernel *filter, int x0_q4,
//                                int32_t x_step_q4, int y0_q4, int y_step_q4,
//                                int w, int h, int bd);
// void vpx_highbd_convolve8_avg_sse2(const uint8_t *src, ptrdiff_t src_stride,
//                                    uint8_t *dst, ptrdiff_t dst_stride,
//                                    const InterpKernel *filter, int x0_q4,
//                                    int32_t x_step_q4, int y0_q4,
//                                    int y_step_q4, int w, int h, int bd);
HIGH_FUN_CONV_2D(, sse2, 0)
HIGH_FUN_CONV_2D(avg_, sse2, 1)
#endif  // CONFIG_VP9_HIGHBITDEPTH && VPX_ARCH_X86_64
